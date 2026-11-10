"""run_distractor_locality.py — mechanism probe (Batch-2).

The paper's open puzzle (C2 limitation): dataset-level MEAN relevant-doc
locality does NOT order the dataset-level paragraph penalties (FiQA has high
mean locality yet the biggest penalty).  The hypothesised resolution is a
DISTRACTOR-side effect: under max-aggregation a flat relevant document is only
punished when *competing* non-relevant documents contain a peaked,
locally-similar fragment that gets promoted past it.

This script computes, per query, distractor-side quantities from the SAME
cached E5 segment matrix the system scores (no re-encoding beyond indexing the
pool once), and tests whether they

  (a) predict the per-query granularity delta (within dataset), and
  (b) rank-order the six datasets' penalties that mean relevant-doc locality
      could not.

Definitions (per query, all from query-vs-segment cosine):
  For a document d: max_seg(d)=max segment sim, mean_seg(d)=mean segment sim,
  doc_vec_sim(d)=document-vector sim, peak(d)=max_seg-mean_seg (the locality
  metric on d), inflation(d)=max_seg(d)-doc_vec_sim(d) (how much
  max-aggregation lifts d above its single-vector score).

  Candidates = non-relevant docs in the union of the document-level and
  paragraph-level top-K (the docs that actually compete near rank 1).

  distractor_peak      = max over distractors of max_seg
  inflation_advantage  = mean inflation(distractors) - mean inflation(rel docs)
  competitive_gap_shift= (best_distractor max_seg - best_rel max_seg)
                         - (best_distractor doc_vec_sim - best_rel doc_vec_sim)

Usage:
    python scripts/run_distractor_locality.py --config configs/locality_e5.yaml [--seed 7]
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys

import numpy as np
from ir_measures import RR

try:
    import torch
except Exception:  # torch always present in this env, but stay defensive
    torch = None


def _free_mps() -> None:
    """Release the MPS caching allocator between datasets so memory pressure
    does not accumulate across the 6 pools (observed to stall dataset 4+)."""
    gc.collect()
    if torch is not None and getattr(torch.backends, "mps", None) is not None:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import load_beir_sample
from src.encoders import build_encoder
from src.locality import correlate, within_correlations
from src.metrics import evaluate, per_query
from src.retrieval import DocumentLevelRetriever, ParagraphLevelRetriever
from src.segmentation import build_segmenter

TOPK_COMPETE = 20   # how deep to look for competing docs
MAXK_DISTRACT = 10  # cap on distractors scored per query

DISTRACTOR_FEATURES = [
    "distractor_peak",
    "distractor_peak_mean",
    "rel_peak_mean",
    "inflation_advantage",
    "competitive_gap_shift",
    "n_seg_distractor_mean",
]


def _doc_sims(doc_ret: DocumentLevelRetriever, q_text: str) -> dict:
    """Full document-vector similarity for every pooled doc (one matmul)."""
    q = doc_ret.encoder.encode([q_text], is_query=True)[0]
    scores = doc_ret._matrix @ q  # research code: reuse cached matrix
    return {doc_ret._ids[i]: float(scores[i]) for i in range(len(doc_ret._ids))}


def _per_doc_stats(sims_by_doc: dict, doc_sim: dict) -> dict:
    """max_seg, mean_seg, peak, inflation, n_seg per doc."""
    out = {}
    for d, sims in sims_by_doc.items():
        a = np.asarray(sims, float)
        mx, mn = float(a.max()), float(a.mean())
        out[d] = {
            "max_seg": mx, "mean_seg": mn, "peak": mx - mn,
            "inflation": mx - doc_sim.get(d, mn), "n_seg": int(a.size),
        }
    return out


def _query_features(rel_ids, distractor_ids, stats) -> dict | None:
    rel = [stats[d] for d in rel_ids if d in stats]
    dis = [stats[d] for d in distractor_ids if d in stats]
    if not rel or not dis:
        return None
    rel_infl = np.mean([s["inflation"] for s in rel])
    dis_infl = np.mean([s["inflation"] for s in dis])
    best_dis = max(dis, key=lambda s: s["max_seg"])
    best_rel = max(rel, key=lambda s: s["max_seg"])
    best_dis_dv = max(dis, key=lambda s: s["mean_seg"])  # proxy unused; keep dv via max_seg-inflation
    # doc_vec_sim reconstructed as max_seg - inflation
    def dv(s):
        return s["max_seg"] - s["inflation"]
    gap_seg = best_dis["max_seg"] - best_rel["max_seg"]
    gap_dv = max(dv(s) for s in dis) - max(dv(s) for s in rel)
    return {
        "distractor_peak": max(s["max_seg"] for s in dis),
        "distractor_peak_mean": float(np.mean([s["peak"] for s in dis])),
        "rel_peak_mean": float(np.mean([s["peak"] for s in rel])),
        "inflation_advantage": float(dis_infl - rel_infl),
        "competitive_gap_shift": float(gap_seg - gap_dv),
        "n_seg_distractor_mean": float(np.mean([s["n_seg"] for s in dis])),
    }


def run(config_path: str, seed_override: int | None = None) -> None:
    cfg = load_config(config_path)
    name = "distractor_" + cfg.name
    if seed_override is not None:
        cfg = dataclasses.replace(cfg, seed=seed_override)
        name = f"{name}_seed{seed_override}"

    all_rows: list[dict] = []
    per_dataset: dict[str, dict] = {}
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    ckpt_path = os.path.join(results_dir, f"{name}_partial.json")

    for dataset in cfg.datasets:
        print(f"\n=== {dataset} ===", flush=True)
        data = load_beir_sample(dataset, cfg.n_docs, cfg.n_queries,
                                seed=cfg.seed, split=cfg.split, data_dir=cfg.data_dir)
        passages, queries, qrels = data["passages"], data["queries"], data["qrels"]
        text_by_id = {p["id"]: p["text"] for p in passages}

        doc_ret = DocumentLevelRetriever(build_encoder(cfg.encoder)).index(passages)
        doc_run = {q["id"]: dict(doc_ret.retrieve(q["text"], top_k=cfg.top_k)) for q in queries}
        doc_pq = per_query(doc_run, qrels, RR @ 10)
        doc_metrics = evaluate(doc_run, qrels)

        segmenter = build_segmenter(cfg.segmentation)
        par_ret = ParagraphLevelRetriever(build_encoder(cfg.encoder), segmenter,
                                           aggregation="max", k=cfg.agg_k).index(passages)
        par_run = {q["id"]: dict(par_ret.retrieve(q["text"], top_k=cfg.top_k)) for q in queries}
        par_pq = per_query(par_run, qrels, RR @ 10)
        par_metrics = evaluate(par_run, qrels)

        print(f"  indexed; scoring {len(queries)} queries...", flush=True)
        rows = []
        for qi, q in enumerate(queries):
            if qi and qi % 50 == 0:
                print(f"    ...{qi}/{len(queries)} queries", flush=True)
            qid = q["id"]
            if qid not in doc_pq:
                continue
            rel_ids = [d for d, r in qrels.get(qid, {}).items() if r > 0 and d in text_by_id]
            if not rel_ids:
                continue
            doc_sim = _doc_sims(doc_ret, q["text"])
            compete = list(doc_run[qid])[:TOPK_COMPETE] + list(par_run[qid])[:TOPK_COMPETE]
            seen, distractor_ids = set(), []
            for d in compete:
                if d not in rel_ids and d not in seen and d in text_by_id:
                    seen.add(d)
                    distractor_ids.append(d)
                if len(distractor_ids) >= MAXK_DISTRACT:
                    break
            if not distractor_ids:
                continue
            sims = par_ret.doc_segment_sims(q["text"], rel_ids + distractor_ids)
            stats = _per_doc_stats(sims, doc_sim)
            feats = _query_features(rel_ids, distractor_ids, stats)
            if feats is None:
                continue
            feats.update({"dataset": dataset, "qid": qid,
                          "delta_max": par_pq[qid] - doc_pq[qid],
                          "n_distractors": len(distractor_ids)})
            rows.append(feats)

        all_rows.extend(rows)
        delta = doc_metrics["RR@10"]
        pen = par_metrics["RR@10"] - doc_metrics["RR@10"]
        corr = {f: correlate([r[f] for r in rows], [r["delta_max"] for r in rows])
                for f in DISTRACTOR_FEATURES}
        per_dataset[dataset] = {
            "n_rows": len(rows),
            "doc_rr10": doc_metrics["RR@10"],
            "penalty_max": pen,
            "mean_features": {f: float(np.mean([r[f] for r in rows])) for f in DISTRACTOR_FEATURES} if rows else {},
            "within_correlations": corr,
        }
        print(f"  penalty(max)={pen:+.4f}  n={len(rows)}  "
              f"mean inflation_adv={per_dataset[dataset]['mean_features'].get('inflation_advantage', float('nan')):+.4f}  "
              f"mean comp_gap_shift={per_dataset[dataset]['mean_features'].get('competitive_gap_shift', float('nan')):+.4f}",
              flush=True)
        for f in ("inflation_advantage", "competitive_gap_shift", "distractor_peak"):
            c = corr[f]
            if c["spearman_rho"] is not None:
                print(f"    {f:>22} vs delta_max: rho={c['spearman_rho']:+.3f} (p={c['spearman_p']:.4f}) n={c['n']}")
        # checkpoint after every dataset so a stall never loses completed work
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"done": list(per_dataset), "per_dataset": per_dataset, "rows": all_rows}, f)
        print(f"  [checkpoint saved: {len(per_dataset)} datasets]", flush=True)
        del doc_ret, par_ret, doc_run, par_run
        _free_mps()

    # pooled within-dataset correlation
    within = within_correlations(all_rows, "delta_max", DISTRACTOR_FEATURES)
    # dataset-level ordering: does the mean feature rank-correlate with the penalty?
    non_ceiling = [d for d in per_dataset if d not in ("trec-covid", "webis-touche2020")]
    order = {}
    for f in DISTRACTOR_FEATURES:
        xs = [per_dataset[d]["mean_features"].get(f) for d in non_ceiling]
        ys = [per_dataset[d]["penalty_max"] for d in non_ceiling]
        order[f] = correlate(xs, ys)

    print("\n=== pooled within-dataset correlation (distractor features vs delta_max) ===")
    for f in DISTRACTOR_FEATURES:
        c = within[f]
        if c["spearman_rho"] is not None:
            print(f"  {f:>22}: rho={c['spearman_rho']:+.3f} (p={c['spearman_p']:.2e}) n={c['n']}")
    print(f"\n=== dataset-level ordering vs penalty (non-ceiling n={len(non_ceiling)}: {non_ceiling}) ===")
    print("  (penalty is negative; a feature that CAUSES penalty should be NEGATIVELY correlated with penalty)")
    for f in DISTRACTOR_FEATURES:
        c = order[f]
        r = c["spearman_rho"]
        print(f"  {f:>22}: dataset-level spearman(mean_feat, penalty)={r if r is None else f'{r:+.3f}'}")

    out = {
        "config_name": name, "encoder_model": cfg.encoder.model,
        "topk_compete": TOPK_COMPETE, "maxk_distract": MAXK_DISTRACT,
        "per_dataset": per_dataset,
        "pooled_within_correlations": within,
        "dataset_level_ordering_vs_penalty": order,
        "rows": all_rows,
    }
    out_path = os.path.join(results_dir, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args.config, seed_override=args.seed)
