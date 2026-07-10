"""run_adaptive.py — M3: adaptive granularity vs fixed document/paragraph.

Protocol (leakage-safe):
  1. Per dataset: build both retrievers (same encoder, only granularity
     differs), get full runs + query-time router features.
  2. Split queries per dataset into train/test (fixed seed). POOL the train
     queries of all datasets: fit ONE router (and tune ONE global alpha) —
     the stronger "works across domains" claim.
  3. Evaluate on test queries only, per dataset and pooled:
       document / paragraph (fixed) / global-alpha fusion /
       router-hard / router-soft / oracle (upper bound).
     Paired sign-flip significance of each adaptive system vs BOTH fixed
     granularities on per-query nDCG@10 (the plan's primary metric).

Labels (per-query delta) use qrels on TRAIN queries only; every feature the
router sees is computable at query time without judgments.

Usage:
    python scripts/run_adaptive.py --config configs/adaptive_e5.yaml
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_measures import nDCG

from src.adaptive import (
    GranularityRouter,
    fuse_runs,
    oracle_per_query,
    query_features,
    tune_global_alpha,
)
from src.config import AdaptiveConfig, load_config
from src.data import load_beir_sample
from src.encoders import build_encoder
from src.metrics import bootstrap_ci, evaluate, paired_randomization_test, per_query
from src.retrieval import DocumentLevelRetriever, ParagraphLevelRetriever
from src.segmentation import build_segmenter

PRIMARY = nDCG @ 10


def _run_dict(retriever, queries, top_k):
    return {
        q["id"]: {doc_id: score for doc_id, score in retriever.retrieve(q["text"], top_k=top_k)}
        for q in queries
    }


def _split_qids(qids, test_ratio, seed):
    rng = random.Random(seed)
    shuffled = sorted(qids)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    return set(shuffled[n_test:]), set(shuffled[:n_test])  # train, test


def _sig_block(base_pq, sys_pq, seed):
    sig = paired_randomization_test(base_pq, sys_pq, seed=seed)
    ci = bootstrap_ci(base_pq, sys_pq, seed=seed)
    return {**sig, **ci}


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    ada = cfg.adaptive or AdaptiveConfig()
    agg = cfg.aggregations[0]

    # ---- pass 1: per-dataset retrieval runs + features -----------------------
    ds_state = {}
    for dataset in cfg.datasets:
        print(f"=== indexing + retrieval: {dataset} ===", flush=True)
        data = load_beir_sample(
            dataset, cfg.n_docs, cfg.n_queries,
            seed=cfg.seed, split=cfg.split, data_dir=cfg.data_dir,
        )
        passages, queries, qrels = data["passages"], data["queries"], data["qrels"]

        doc_ret = DocumentLevelRetriever(build_encoder(cfg.encoder)).index(passages)
        segmenter = build_segmenter(cfg.segmentation)
        par_ret = ParagraphLevelRetriever(
            build_encoder(cfg.encoder), segmenter, aggregation=agg, k=cfg.agg_k
        ).index(passages)

        doc_run = _run_dict(doc_ret, queries, cfg.top_k)
        par_run = _run_dict(par_ret, queries, cfg.top_k)
        doc_pq = per_query(doc_run, qrels, PRIMARY)
        par_pq = per_query(par_run, qrels, PRIMARY)

        features = {}
        for q in queries:
            qid = q["id"]
            if qid not in doc_pq:
                continue  # unjudged after sampling -> excluded everywhere
            doc_ranked = sorted(doc_run[qid].items(), key=lambda kv: -kv[1])
            par_ranked = sorted(par_run[qid].items(), key=lambda kv: -kv[1])
            top_ids = [d for d, _ in doc_ranked[:ada.feature_top_k]]
            seg_sims = par_ret.doc_segment_sims(q["text"], top_ids)
            features[qid] = query_features(
                q["text"], doc_ranked, par_ranked, seg_sims, ada.feature_top_k
            )

        split_seed = ada.split_seed + sum(ord(c) for c in dataset)
        train_qids, test_qids = _split_qids(features.keys(), ada.test_ratio, split_seed)
        ds_state[dataset] = {
            "qrels": qrels, "doc_run": doc_run, "par_run": par_run,
            "doc_pq": doc_pq, "par_pq": par_pq, "features": features,
            "train": train_qids, "test": test_qids,
        }
        print(f"  judged={len(features)}  train={len(train_qids)}  test={len(test_qids)}"
              f"  doc nDCG@10={np.mean([doc_pq[q] for q in doc_pq]):.4f}"
              f"  par nDCG@10={np.mean([par_pq[q] for q in par_pq]):.4f}", flush=True)

    # ---- pass 2: fit router + global alpha on POOLED train queries -----------
    train_rows, train_deltas = [], []
    for dataset, st in ds_state.items():
        for qid in st["train"]:
            train_rows.append(st["features"][qid])
            train_deltas.append(st["par_pq"].get(qid, 0.0) - st["doc_pq"].get(qid, 0.0))
    router = GranularityRouter(ridge_alpha=ada.ridge_alpha).fit(train_rows, train_deltas)

    # Global alpha on the pooled train pool. BEIR qids can COLLIDE across
    # datasets (scifact and fiqa both use bare numeric ids), so every pooled
    # structure namespaces qids as "<dataset>::<qid>".
    merged_doc, merged_par, merged_qrels, merged_train = {}, {}, {}, []
    for dataset, st in ds_state.items():
        merged_doc.update({f"{dataset}::{q}": r for q, r in st["doc_run"].items()})
        merged_par.update({f"{dataset}::{q}": r for q, r in st["par_run"].items()})
        merged_qrels.update({f"{dataset}::{q}": r for q, r in st["qrels"].items()})
        merged_train.extend(f"{dataset}::{q}" for q in st["train"])
    global_alpha, _ = tune_global_alpha(
        merged_doc, merged_par, merged_qrels, merged_train, PRIMARY
    )
    print(f"\nrouter fit on {len(train_rows)} pooled train queries; "
          f"global_alpha={global_alpha:.2f}", flush=True)

    # ---- pass 3: evaluate all systems on test queries -------------------------
    def evaluate_systems(st):
        test = st["test"]
        # ir_measures scores EVERY qrels query (missing-from-run -> 0), so the
        # qrels must be restricted to the test split too — otherwise train
        # queries enter every metric as zeros and halve the aggregates.
        qrels = {q: st["qrels"][q] for q in test if q in st["qrels"]}
        doc_run = {q: st["doc_run"][q] for q in test if q in st["doc_run"]}
        par_run = {q: st["par_run"][q] for q in test if q in st["par_run"]}
        rows = [st["features"][q] for q in sorted(test)]
        qid_order = sorted(test)

        hard = dict(zip(qid_order, router.hard_alphas(rows)))
        soft = dict(zip(qid_order, router.soft_alphas(rows, tau=ada.soft_tau)))
        glob = {q: global_alpha for q in qid_order}

        systems = {
            "document": doc_run,
            "paragraph": par_run,
            "global_alpha": fuse_runs(st["doc_run"], st["par_run"], glob),
            "router_hard": fuse_runs(st["doc_run"], st["par_run"], hard),
            "router_soft": fuse_runs(st["doc_run"], st["par_run"], soft),
        }
        pqs = {name: per_query(run, qrels, PRIMARY) for name, run in systems.items()}
        pqs["oracle"] = oracle_per_query(pqs["document"], pqs["paragraph"], qid_order)
        metrics = {name: evaluate(run, qrels) for name, run in systems.items()}
        metrics["oracle"] = {"nDCG@10": float(np.mean(list(pqs["oracle"].values())))}
        return systems, pqs, metrics, hard

    per_dataset_out = {}
    pooled_pqs = {}
    routing_stats = {}
    for dataset, st in ds_state.items():
        systems, pqs, metrics, hard = evaluate_systems(st)
        for name, pq in pqs.items():
            pooled_pqs.setdefault(name, {}).update(
                {f"{dataset}::{q}": v for q, v in pq.items()}
            )
        n_para = int(sum(1 for a in hard.values() if a == 0.0))
        routing_stats[dataset] = {"n_test": len(hard), "routed_to_paragraph": n_para}

        sig = {}
        for name in ("global_alpha", "router_hard", "router_soft", "oracle"):
            sig[name] = {
                "vs_document": _sig_block(pqs["document"], pqs[name], cfg.seed),
                "vs_paragraph": _sig_block(pqs["paragraph"], pqs[name], cfg.seed),
            }
            if name != "global_alpha":
                # The decisive comparison for the "adaptive" claim: does the
                # per-query router beat one static fusion weight?
                sig[name]["vs_global_alpha"] = _sig_block(
                    pqs["global_alpha"], pqs[name], cfg.seed
                )
        per_dataset_out[dataset] = {
            "n_test": len(st["test"]),
            "metrics": metrics,
            "significance_ndcg10": sig,
            "routing": routing_stats[dataset],
        }

        print(f"\n=== {dataset} (test n={len(st['test'])}) ===")
        for name in ("document", "paragraph", "global_alpha", "router_hard", "router_soft", "oracle"):
            nd = metrics[name].get("nDCG@10", float("nan"))
            line = f"  {name:>13}: nDCG@10={nd:.4f}"
            if name in sig:
                d, p = sig[name]["vs_document"], sig[name]["vs_paragraph"]
                line += (f"  vs_doc Δ={d['mean_diff']:+.4f} (p={d['p_value']:.3f})"
                         f"  vs_par Δ={p['mean_diff']:+.4f} (p={p['p_value']:.3f})")
            print(line, flush=True)

    # ---- pooled test analysis -------------------------------------------------
    pooled_sig = {}
    for name in ("global_alpha", "router_hard", "router_soft", "oracle"):
        pooled_sig[name] = {
            "vs_document": _sig_block(pooled_pqs["document"], pooled_pqs[name], cfg.seed),
            "vs_paragraph": _sig_block(pooled_pqs["paragraph"], pooled_pqs[name], cfg.seed),
        }
        if name != "global_alpha":
            pooled_sig[name]["vs_global_alpha"] = _sig_block(
                pooled_pqs["global_alpha"], pooled_pqs[name], cfg.seed
            )
    pooled_means = {name: float(np.mean(list(pq.values()))) for name, pq in pooled_pqs.items()}

    print(f"\n=== pooled test ({len(pooled_pqs['document'])} queries) ===")
    for name in ("document", "paragraph", "global_alpha", "router_hard", "router_soft", "oracle"):
        line = f"  {name:>13}: mean nDCG@10={pooled_means[name]:.4f}"
        if name in pooled_sig:
            d, p = pooled_sig[name]["vs_document"], pooled_sig[name]["vs_paragraph"]
            line += (f"  vs_doc Δ={d['mean_diff']:+.4f} (p={d['p_value']:.4f})"
                     f"  vs_par Δ={p['mean_diff']:+.4f} (p={p['p_value']:.4f})")
            g = pooled_sig[name].get("vs_global_alpha")
            if g:
                line += f"  vs_glob Δ={g['mean_diff']:+.4f} (p={g['p_value']:.4f})"
        print(line, flush=True)

    coefs = router.model.named_steps["ridge"].coef_
    from src.adaptive import FEATURE_ORDER
    out = {
        "config_name": cfg.name,
        "encoder_model": cfg.encoder.model,
        "primary_measure": str(PRIMARY),
        "aggregation": agg,
        "global_alpha": global_alpha,
        "router_coefficients": dict(zip(FEATURE_ORDER, map(float, coefs))),
        "n_train_pooled": len(train_rows),
        "per_dataset": per_dataset_out,
        "pooled": {
            "n_test": len(pooled_pqs["document"]),
            "mean_ndcg10": pooled_means,
            "significance_ndcg10": pooled_sig,
        },
        # Per-query nDCG@10 for every system ("<dataset>::<qid>" keys) so paper
        # plots and extra significance tests never require re-running retrieval.
        "per_query_ndcg10": pooled_pqs,
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
