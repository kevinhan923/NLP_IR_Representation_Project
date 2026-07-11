"""run_colbert.py — the granularity spectrum under ONE ColBERT model.

Three systems sharing the same token embeddings, differing only in how much
they compress before scoring:

  doc_single       mean-pool all tokens -> one vector          (max compression)
  paragraph[agg]   per-segment pooled vectors -> aggregate     (middle)
  late_interaction no pooling, MaxSim over tokens              (no compression)

Per dataset: RR@10/nDCG@10/R@100 for every system, paired sign-flip
significance for paragraph-vs-doc (the C1 delta under a ColBERT encoder),
late-vs-doc and late-vs-paragraph. Per-query RR@10 saved for reuse.

Usage:
    python scripts/run_colbert.py --config configs/colbert.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_measures import RR

from src.config import load_config
from src.data import load_beir_sample
from src.lateinteraction import (
    ColbertTokenEncoder,
    LateInteractionRetriever,
    PooledDocRetriever,
    PooledSegmentRetriever,
    pool_token_matrix,
)
from src.metrics import bootstrap_ci, evaluate, paired_randomization_test, per_query
from src.segmentation import build_segmenter


def _sig_block(base_pq, sys_pq, seed):
    sig = paired_randomization_test(base_pq, sys_pq, seed=seed)
    ci = bootstrap_ci(base_pq, sys_pq, seed=seed)
    return {**sig, **ci}


def run(config_path: str, seed_override: int | None = None) -> None:
    cfg = load_config(config_path)
    if seed_override is not None:
        import dataclasses
        cfg = dataclasses.replace(
            cfg, seed=seed_override, name=f"{cfg.name}_seed{seed_override}"
        )
    primary_agg = cfg.aggregations[0]

    enc = ColbertTokenEncoder(
        model_name=cfg.encoder.model or "lightonai/colbertv2.0",
        batch_size=cfg.encoder.batch_size,
    )
    segmenter = build_segmenter(cfg.segmentation)

    results = {}
    all_pq = {}
    for dataset in cfg.datasets:
        print(f"\n=== {dataset} ===", flush=True)
        data = load_beir_sample(
            dataset, cfg.n_docs, cfg.n_queries,
            seed=cfg.seed, split=cfg.split, data_dir=cfg.data_dir,
        )
        passages, queries, qrels = data["passages"], data["queries"], data["qrels"]
        ids = [p["id"] for p in passages]

        # Encode document tokens ONCE; shared by doc_single and late systems.
        doc_mats = enc.encode_tokens([p["text"] for p in passages])
        late_ret = LateInteractionRetriever().index(ids, doc_mats)
        doc_ret = PooledDocRetriever().index(ids, doc_mats)

        # Segment side: segments encoded independently (same convention as the
        # dense paragraph systems), pooled per segment.
        seg_texts, doc_seg_idx = [], {}
        for p in passages:
            segs = segmenter(p["text"]) or [p["text"]]
            start = len(seg_texts)
            seg_texts.extend(segs)
            doc_seg_idx[p["id"]] = list(range(start, start + len(segs)))
        seg_mats = enc.encode_tokens(seg_texts)
        par_ret = PooledSegmentRetriever(aggregation=primary_agg, k=cfg.agg_k).index(
            ids, seg_mats, doc_seg_idx
        )
        print(f"  encoded {len(doc_mats)} docs, {len(seg_mats)} segments", flush=True)

        # Query encodings: token matrices for late, pooled vectors for the rest.
        q_mats = enc.encode_tokens([q["text"] for q in queries], is_query=True)

        runs = {"doc_single": {}, "late_interaction": {}}
        for agg in cfg.aggregations:
            runs[f"paragraph_{agg}"] = {}
        for q, qm in zip(queries, q_mats):
            qid = q["id"]
            q_vec = pool_token_matrix(qm)
            runs["doc_single"][qid] = dict(doc_ret.retrieve(q_vec, top_k=cfg.top_k))
            runs["late_interaction"][qid] = dict(late_ret.retrieve(qm, top_k=cfg.top_k))
            for agg in cfg.aggregations:
                par_ret.aggregation = agg
                runs[f"paragraph_{agg}"][qid] = dict(par_ret.retrieve(q_vec, top_k=cfg.top_k))

        metrics = {name: evaluate(run, qrels) for name, run in runs.items()}
        pqs = {name: per_query(run, qrels, RR @ 10) for name, run in runs.items()}
        for name, pq in pqs.items():
            all_pq.setdefault(name, {}).update({f"{dataset}::{k}": v for k, v in pq.items()})

        par_primary = f"paragraph_{primary_agg}"
        sig = {
            f"{par_primary}_vs_doc": _sig_block(pqs["doc_single"], pqs[par_primary], cfg.seed),
            "late_vs_doc": _sig_block(pqs["doc_single"], pqs["late_interaction"], cfg.seed),
            "late_vs_paragraph": _sig_block(pqs[par_primary], pqs["late_interaction"], cfg.seed),
        }
        for agg in cfg.aggregations[1:]:
            sig[f"paragraph_{agg}_vs_doc"] = _sig_block(
                pqs["doc_single"], pqs[f"paragraph_{agg}"], cfg.seed
            )

        results[dataset] = {
            "n_docs": len(passages),
            "n_queries_judged": len(qrels),
            "metrics": metrics,
            "significance_rr10": sig,
        }

        d = metrics["doc_single"]
        print(f"  doc_single       RR@10={d['RR@10']:.4f}  nDCG@10={d['nDCG@10']:.4f}")
        for agg in cfg.aggregations:
            m = metrics[f"paragraph_{agg}"]
            s = sig.get(f"paragraph_{agg}_vs_doc") or sig[f"{par_primary}_vs_doc"]
            print(f"  paragraph[{agg:>9}] RR@10={m['RR@10']:.4f}  Δdoc={m['RR@10']-d['RR@10']:+.4f}"
                  f" (p={s['p_value']:.4f})")
        li = metrics["late_interaction"]
        print(f"  late_interaction RR@10={li['RR@10']:.4f}"
              f"  vs_doc Δ={li['RR@10']-d['RR@10']:+.4f} (p={sig['late_vs_doc']['p_value']:.4f})"
              f"  vs_par Δ={li['RR@10']-metrics[par_primary]['RR@10']:+.4f}"
              f" (p={sig['late_vs_paragraph']['p_value']:.4f})", flush=True)

    out = {
        "config_name": cfg.name,
        "encoder_model": cfg.encoder.model or "lightonai/colbertv2.0",
        "primary_agg": primary_agg,
        "results": results,
        "per_query_rr10": all_pq,
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{cfg.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args.config, seed_override=args.seed)
