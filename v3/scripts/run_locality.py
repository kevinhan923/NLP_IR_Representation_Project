"""run_locality.py — M2: does evidence locality predict the granularity delta?

For each dataset:
  1. run the controlled document- vs paragraph-level comparison (same encoder,
     only granularity differs) and keep PER-QUERY RR@10 for both sides;
  2. compute evidence-locality features for every query from its relevant
     documents, using the paragraph-side encoder (the exact signal that
     system sees);
  3. correlate each locality feature with the per-query delta
     (paragraph RR@10 − document RR@10).

Methodological guards (from the M2 design review):
  * If the config sets ``locality_encoder``, features are ALSO computed with
    that independent encoder (``x_``-prefixed) — a same-vs-cross-encoder
    comparison that defuses the "mechanical correlation" objection (under
    agg=max the paragraph score IS the max of the same similarity vector).
  * Pooled correlations are reported raw AND dataset-demeaned (within-dataset)
    so a cross-dataset composition effect (Simpson's paradox) cannot pose as
    per-query signal.
  * The "multiseg" subset requires n_segments_min >= 2: EVERY relevant doc of
    the query must actually split, otherwise 1-segment convention values
    (entropy_locality = 1.0) would leak in.
  * Per-dataset drop counts (unjudged / no relevant text in pool) are reported.

Pooled analysis across datasets + quantile-binned delta curves are saved to
results/<config-name>.json together with all per-query rows, so paper plots
can be regenerated without re-running retrieval.

Usage:
    python scripts/run_locality.py --config configs/locality_e5.yaml
"""

import argparse
import json
import os
import sys

# Make src importable when run from v3/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_measures import RR

from src.config import load_config
from src.data import load_beir_sample
from src.encoders import build_encoder
from src.locality import (
    FEATURE_KEYS,
    binned_means,
    feature_correlations,
    query_locality,
    within_correlations,
)
from src.metrics import evaluate, per_query
from src.retrieval import DocumentLevelRetriever, ParagraphLevelRetriever
from src.segmentation import build_segmenter

# Query-level covariates reported alongside the shape features.
ANALYSIS_KEYS = FEATURE_KEYS + ["doc_words"]
# Shape features recomputed under the independent locality encoder.
X_KEYS = [f"x_{k}" for k in FEATURE_KEYS if k != "n_segments"]


def _run_dict(retriever, queries, top_k):
    return {
        q["id"]: {doc_id: score for doc_id, score in retriever.retrieve(q["text"], top_k=top_k)}
        for q in queries
    }


_feature_table = feature_correlations
_within_table = within_correlations


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    primary_agg = cfg.aggregations[0]
    delta_key = f"delta_{primary_agg}"

    all_rows = []
    per_dataset = {}

    for dataset in cfg.datasets:
        print(f"\n=== {dataset} ===")
        data = load_beir_sample(
            dataset, cfg.n_docs, cfg.n_queries,
            seed=cfg.seed, split=cfg.split, data_dir=cfg.data_dir,
        )
        passages, queries, qrels = data["passages"], data["queries"], data["qrels"]
        text_by_id = {p["id"]: p["text"] for p in passages}

        # --- controlled comparison, per-query RR@10 on both sides ---------
        doc_ret = DocumentLevelRetriever(build_encoder(cfg.encoder)).index(passages)
        doc_run = _run_dict(doc_ret, queries, cfg.top_k)
        doc_metrics = evaluate(doc_run, qrels)
        doc_pq = per_query(doc_run, qrels, RR @ 10)

        segmenter = build_segmenter(cfg.segmentation)
        par_ret = ParagraphLevelRetriever(
            build_encoder(cfg.encoder), segmenter,
            aggregation=primary_agg, k=cfg.agg_k,
        ).index(passages)

        par_pq_by_agg = {}
        par_rr10 = {}
        for agg in cfg.aggregations:
            par_ret.aggregation = agg
            par_run = _run_dict(par_ret, queries, cfg.top_k)
            par_pq_by_agg[agg] = per_query(par_run, qrels, RR @ 10)
            par_rr10[agg] = evaluate(par_run, qrels)["RR@10"]

        # --- optional independent encoder for the circularity check -------
        x_encoder = None
        if cfg.locality_encoder is not None:
            x_encoder = build_encoder(cfg.locality_encoder)
            # TF-IDF variants must be fit; fit on the same segment units the
            # paragraph retriever indexed (no-op for pretrained dense models).
            seg_texts = [s for p in passages for s in (segmenter(p["text"]) or [p["text"]])]
            x_encoder.fit(seg_texts)

        # --- locality features from each query's relevant docs ------------
        # Primary features use the paragraph-side encoder: for dense models it
        # is the same pretrained weights; for TF-IDF the segment-fit instance.
        rows = []
        n_unjudged = 0
        n_no_rel_text = 0
        for q in queries:
            qid = q["id"]
            if qid not in doc_pq:
                n_unjudged += 1  # query lost all judged docs in sampling
                continue
            rel_texts = [
                text_by_id[did]
                for did, rel in qrels.get(qid, {}).items()
                if rel > 0 and did in text_by_id
            ]
            if not rel_texts:
                n_no_rel_text += 1
                continue
            feats = query_locality(q["text"], rel_texts, par_ret.encoder, segmenter)
            row = {"dataset": dataset, "qid": qid, "doc_rr10": doc_pq[qid], **feats}
            if x_encoder is not None:
                x_feats = query_locality(q["text"], rel_texts, x_encoder, segmenter)
                row.update({f"x_{k}": x_feats[k] for k in FEATURE_KEYS if k != "n_segments"})
            for agg in cfg.aggregations:
                # Direct indexing on purpose: both per_query dicts come from the
                # same qrels and full-collection runs, so a missing key means a
                # broken invariant — fail loudly instead of faking RR = 0.
                row[f"delta_{agg}"] = par_pq_by_agg[agg][qid] - doc_pq[qid]
            rows.append(row)

        all_rows.extend(rows)

        keys = ANALYSIS_KEYS + (X_KEYS if x_encoder is not None else [])
        corr_all = _feature_table(rows, delta_key, keys)
        multiseg = [r for r in rows if r["n_segments_min"] >= 2]
        corr_multiseg = _feature_table(multiseg, delta_key, keys)

        per_dataset[dataset] = {
            "n_queries_sampled": len(queries),
            "n_dropped_unjudged": n_unjudged,
            "n_dropped_no_rel_text": n_no_rel_text,
            "n_rows": len(rows),
            "n_rows_multiseg": len(multiseg),
            "doc_rr10": doc_metrics["RR@10"],
            "par_rr10_by_agg": par_rr10,
            "delta_rr10_by_agg": {a: par_rr10[a] - doc_metrics["RR@10"] for a in par_rr10},
            "mean_features": {
                k: float(sum(r[k] for r in rows) / len(rows)) for k in ANALYSIS_KEYS
            } if rows else {},
            "correlations_vs_" + delta_key: corr_all,
            "correlations_vs_" + delta_key + "_multiseg": corr_multiseg,
        }

        d = per_dataset[dataset]
        print(f"  doc RR@10={d['doc_rr10']:.4f}  "
              + "  ".join(f"Δ[{a}]={v:+.4f}" for a, v in d["delta_rr10_by_agg"].items())
              + f"  rows={len(rows)} (multiseg={len(multiseg)},"
              f" dropped: unjudged={n_unjudged}, no_rel_text={n_no_rel_text})")
        print(f"  mean entropy_locality={d['mean_features'].get('entropy_locality', float('nan')):.4f}"
              f"  mean n_segments={d['mean_features'].get('n_segments', float('nan')):.2f}")
        show = ["entropy_locality", "top1_share", "peakedness"]
        if x_encoder is not None:
            show.append("x_entropy_locality")
        for k in show:
            c = corr_multiseg[k]
            if c["spearman_rho"] is not None:
                print(f"  {k:>20} vs {delta_key}: spearman={c['spearman_rho']:+.3f}"
                      f" (p={c['spearman_p']:.4f})  pearson={c['pearson_r']:+.3f}"
                      f" (p={c['pearson_p']:.4f})  n={c['n']}")

    # --- pooled analysis across datasets -----------------------------------
    has_x = any("x_entropy_locality" in r for r in all_rows)
    pooled_keys = ANALYSIS_KEYS + (X_KEYS if has_x else [])
    pooled_multiseg = [r for r in all_rows if r["n_segments_min"] >= 2]
    pooled = {
        "n_rows": len(all_rows),
        "n_rows_multiseg": len(pooled_multiseg),
        "correlations_vs_" + delta_key: _feature_table(all_rows, delta_key, pooled_keys),
        "correlations_vs_" + delta_key + "_multiseg": _feature_table(
            pooled_multiseg, delta_key, pooled_keys
        ),
        "within_dataset_correlations_vs_" + delta_key + "_multiseg": _within_table(
            pooled_multiseg, delta_key, pooled_keys
        ),
        "binned_" + delta_key + "_by_entropy_locality": binned_means(
            [r["entropy_locality"] for r in pooled_multiseg],
            [r[delta_key] for r in pooled_multiseg],
            n_bins=4,
        ),
    }

    print("\n=== pooled, dataset-demeaned (within-dataset; n_segments_min >= 2) ===")
    for k in pooled_keys:
        c = pooled["within_dataset_correlations_vs_" + delta_key + "_multiseg"][k]
        if c["spearman_rho"] is not None:
            print(f"  {k:>20}: spearman={c['spearman_rho']:+.3f} (p={c['spearman_p']:.2e})"
                  f"  pearson={c['pearson_r']:+.3f} (p={c['pearson_p']:.2e})  n={c['n']}")
    print("  binned Δ by entropy_locality quartile (raw pooled):")
    for b in pooled["binned_" + delta_key + "_by_entropy_locality"]:
        print(f"    [{b['x_low']:.3f}, {b['x_high']:.3f}]  meanΔ={b['mean_y']:+.4f}  n={b['count']}")

    out = {
        "config_name": cfg.name,
        "encoder": cfg.encoder.name,
        "encoder_model": cfg.encoder.model,
        "locality_encoder_model": cfg.locality_encoder.model if cfg.locality_encoder else None,
        "primary_agg": primary_agg,
        "per_dataset": per_dataset,
        "pooled": pooled,
        "rows": all_rows,
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
    args = parser.parse_args()
    run(args.config)
