from __future__ import annotations

import json
from json import encoder
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from .aggregation import AggregationSpec, DEFAULT_AGGREGATIONS
from .analysis import build_query_level_comparison, grouped_analysis
from .datasets import build_doc_lookup, msmarco_load_documents, msmarco_load_qrels, msmarco_load_queries
from .encoders import build_encoder
from .metrics import metric_bundle
from .rerank import TwoStageReranker
from .segmentation import DEFAULT_SEGMENTATIONS, SegmentationSpec, segment_text
from .stage1 import LocalTfidfRetriever
from .types import CandidateResult, DocumentExample, QueryExample


def split_queries(queries: List[QueryExample], seed: int, train_ratio: float = 0.6, dev_ratio: float = 0.2):
    rng = random.Random(seed)
    qs = list(queries)
    rng.shuffle(qs)
    n = len(qs)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    return qs[:train_end], qs[train_end:dev_end], qs[dev_end:]


class GridExperiment:
    def __init__(self, config: Dict[str, object]):
        self.config = config

    def run(self) -> Dict[str, object]:
        data_cfg = self.config["data"]
        exp_cfg = self.config["experiment"]
        encoder_cfg = self.config["encoder"]

        docs = msmarco_load_documents(data_cfg["doc_dataset"], max_docs=data_cfg.get("max_docs"))
        queries = msmarco_load_queries(data_cfg["query_dataset"], max_queries=data_cfg.get("max_queries"))
        qrels = msmarco_load_qrels(data_cfg["qrels_dataset"], positive_only=True)
        queries = [q for q in queries if q.query_id in qrels]

        train_q, dev_q, test_q = split_queries(queries, seed=exp_cfg.get("seed", 13), train_ratio=exp_cfg.get("train_ratio", 0.6), dev_ratio=exp_cfg.get("dev_ratio", 0.2))
        docs_lookup = build_doc_lookup(docs)

        stage1 = LocalTfidfRetriever(max_features=exp_cfg.get("stage1_max_features", 300000))
        stage1.fit(docs)
        stage1_dev = stage1.retrieve(dev_q, top_k=exp_cfg.get("candidate_pool", 100))
        stage1_test = stage1.retrieve(test_q, top_k=exp_cfg.get("candidate_pool", 100))

        encoder = build_encoder(encoder_cfg["name"], **encoder_cfg.get("params", {}))
        train_texts = [docs_lookup[d].text for qid, rels in qrels.items() for d in list(rels)[:1] if d in docs_lookup]
        if len(train_texts) > exp_cfg.get("fit_text_cap", 50000):
            train_texts = train_texts[: exp_cfg.get("fit_text_cap", 50000)]
        fit_texts = train_texts if len(train_texts) >= 20 else [d.text for d in docs[:1000]]
        encoder.fit(fit_texts)
        segmentations = DEFAULT_SEGMENTATIONS if exp_cfg.get("segmentations") == "default" else [SegmentationSpec(**x) for x in exp_cfg["segmentations"]]
        aggs = DEFAULT_AGGREGATIONS if exp_cfg.get("aggregations") == "default" else [AggregationSpec(**x) for x in exp_cfg["aggregations"]]
        ks = exp_cfg.get("metric_ks", [10])

        grid_results = []
        best = None
        best_dev = -1.0
        for seg in segmentations:
            for agg in aggs:
                reranker = TwoStageReranker(encoder=encoder, segmentation=seg, aggregation=agg)
                dev_out = reranker.rerank(dev_q, stage1_dev.runs)
                dev_run = {qid: [c.doc_id for c in ranked] for qid, ranked in dev_out.runs.items()}
                dev_metrics = metric_bundle(dev_run, qrels, ks=ks)
                row = {
                    "segmentation": seg.name,
                    "aggregation": agg.name,
                    "dev_metrics": dev_metrics,
                    "dev_latency_seconds": dev_out.latency_seconds,
                    "avg_segments_per_candidate": dev_out.avg_segments_per_candidate,
                    "avg_scored_segments_per_query": dev_out.avg_scored_segments_per_query,
                }
                grid_results.append(row)
                if dev_metrics["mrr@10"] > best_dev:
                    best_dev = dev_metrics["mrr@10"]
                    best = (seg, agg)

        if best is None:
            raise RuntimeError("No valid grid result found.")
        best_seg, best_agg = best
        test_reranker = TwoStageReranker(encoder=encoder, segmentation=best_seg, aggregation=best_agg)
        para_test_out = test_reranker.rerank(test_q, stage1_test.runs)
        para_run = {qid: [c.doc_id for c in ranked] for qid, ranked in para_test_out.runs.items()}
        para_metrics = metric_bundle(para_run, qrels, ks=ks)

        single_run = {qid: [c.doc_id for c in ranked] for qid, ranked in stage1_test.runs.items()}
        single_metrics = metric_bundle(single_run, qrels, ks=ks)

        doc_segment_counts = {d.doc_id: len(segment_text(d.text, best_seg)) for d in docs}
        out = {
            "config": self.config,
            "dataset_stats": {
                "num_documents": len(docs),
                "num_queries": len(queries),
                "num_test_queries": len(test_q),
            },
            "grid_results": sorted(grid_results, key=lambda x: x["dev_metrics"]["mrr@10"], reverse=True),
            "selected_setting": {"segmentation": best_seg.name, "aggregation": best_agg.name},
            "test_results": {
                "single_vector": single_metrics,
                "paragraph_level": para_metrics,
                "delta_paragraph_minus_single": {k: para_metrics.get(k, 0.0) - single_metrics.get(k, 0.0) for k in para_metrics},
                "stage1_latency_seconds": stage1_test.latency_seconds,
                "paragraph_rerank_latency_seconds": para_test_out.latency_seconds,
                "avg_segments_per_candidate": para_test_out.avg_segments_per_candidate,
                "avg_scored_segments_per_query": para_test_out.avg_scored_segments_per_query,
            },
            "query_level_comparison": build_query_level_comparison(test_q, qrels, single_run, para_run, para_test_out.best_paragraphs),
            "grouped_analysis": grouped_analysis(test_q, qrels, docs_lookup, doc_segment_counts, single_run, para_run),
        }
        return out


def save_json(obj: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
