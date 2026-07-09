"""experiment.py — Orchestrate the controlled comparison.

For each dataset: load real data -> build document-level and paragraph-level
retrievers with the SAME encoder config -> evaluate both with ir_measures ->
paired significance test on per-query RR@10. The paragraph-side segment encoding
is computed once and reused across aggregation methods.
"""

from __future__ import annotations

from typing import Dict, List

from ir_measures import RR

from .config import ExperimentConfig
from .data import load_beir_sample
from .encoders import build_encoder
from .metrics import bootstrap_ci, evaluate, paired_randomization_test, per_query
from .retrieval import DocumentLevelRetriever, ParagraphLevelRetriever
from .segmentation import build_segmenter


def _run_dict(retriever, queries: List[Dict], top_k: int) -> Dict[str, Dict[str, float]]:
    return {
        q["id"]: {doc_id: score for doc_id, score in retriever.retrieve(q["text"], top_k=top_k)}
        for q in queries
    }


def run_experiment(cfg: ExperimentConfig) -> Dict:
    results: Dict[str, Dict] = {}

    for dataset in cfg.datasets:
        data = load_beir_sample(
            dataset, cfg.n_docs, cfg.n_queries,
            seed=cfg.seed, split=cfg.split, data_dir=cfg.data_dir,
        )
        passages, queries, qrels = data["passages"], data["queries"], data["qrels"]

        # Document-level: encode whole docs once.
        doc_ret = DocumentLevelRetriever(build_encoder(cfg.encoder)).index(passages)
        doc_run = _run_dict(doc_ret, queries, cfg.top_k)
        doc_metrics = evaluate(doc_run, qrels)
        doc_pq = per_query(doc_run, qrels, RR @ 10)

        # Paragraph-level: encode segments once, vary only the aggregation.
        segmenter = build_segmenter(cfg.segmentation)
        par_ret = ParagraphLevelRetriever(
            build_encoder(cfg.encoder), segmenter,
            aggregation=cfg.aggregations[0], k=cfg.agg_k,
        ).index(passages)

        per_agg: Dict[str, Dict] = {}
        for agg in cfg.aggregations:
            par_ret.aggregation = agg
            par_run = _run_dict(par_ret, queries, cfg.top_k)
            par_metrics = evaluate(par_run, qrels)
            par_pq = per_query(par_run, qrels, RR @ 10)
            sig = paired_randomization_test(doc_pq, par_pq, seed=cfg.seed)
            ci = bootstrap_ci(doc_pq, par_pq, seed=cfg.seed)
            per_agg[agg] = {
                "paragraph": par_metrics,
                "delta_rr10": par_metrics["RR@10"] - doc_metrics["RR@10"],
                "significance_rr10": {**sig, **ci},
            }

        results[dataset] = {
            "n_docs": len(passages),
            "n_queries_judged": len(qrels),
            "document": doc_metrics,
            "paragraph_by_agg": per_agg,
        }

    return {"config_name": cfg.name, "encoder": cfg.encoder.name, "results": results}
