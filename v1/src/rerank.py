from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .aggregation import AggregationSpec, aggregate_scores
from .encoders import BaseEncoder
from .segmentation import SegmentationSpec, segment_text
from .types import CandidateResult, ComparisonRow, DocumentExample, QueryExample


@dataclass
class RerankOutput:
    runs: Dict[str, List[CandidateResult]]
    latency_seconds: float
    avg_segments_per_candidate: float
    avg_scored_segments_per_query: float
    best_paragraphs: Dict[str, Dict[str, object]]


class TwoStageReranker:
    def __init__(self, encoder: BaseEncoder, segmentation: SegmentationSpec, aggregation: AggregationSpec):
        self.encoder = encoder
        self.segmentation = segmentation
        self.aggregation = aggregation

    def fit(self, train_texts: List[str]) -> "TwoStageReranker":
        self.encoder.fit(train_texts)
        return self

    def rerank(self, queries: List[QueryExample], stage1_runs: Dict[str, List[CandidateResult]]) -> RerankOutput:
        start = time.time()
        query_texts = [q.text for q in queries]
        query_ids = [q.query_id for q in queries]
        query_vecs = self.encoder.encode(query_texts)

        runs: Dict[str, List[CandidateResult]] = {}
        best_paragraphs: Dict[str, Dict[str, object]] = {}
        total_segments = 0
        total_candidates = 0
        total_scored_segments = 0

        for qid, qvec in tqdm(list(zip(query_ids, query_vecs)), desc=f"Reranking: {self.segmentation.name} + {self.aggregation.name}"):
            candidates = stage1_runs.get(qid, [])
            rescored: List[Tuple[CandidateResult, float, int, float, str]] = []
            for cand in candidates:
                segs = segment_text(cand.text or "", self.segmentation)
                total_segments += len(segs)
                total_candidates += 1
                total_scored_segments += len(segs)
                seg_vecs = self.encoder.encode(segs)
                scores = np.dot(seg_vecs, qvec)
                agg = aggregate_scores(scores.tolist(), self.aggregation)
                best_idx = int(np.argmax(scores)) if len(scores) else 0
                best_score = float(scores[best_idx]) if len(scores) else float("-inf")
                best_text = segs[best_idx] if segs else ""
                rescored.append((cand, float(agg), best_idx, best_score, best_text))
            rescored.sort(key=lambda x: x[1], reverse=True)
            ranked: List[CandidateResult] = []
            for cand, agg, _, _, _ in rescored:
                ranked.append(CandidateResult(doc_id=cand.doc_id, score=agg, text=cand.text, title=cand.title, metadata=cand.metadata))
            runs[qid] = ranked
            if rescored:
                top = rescored[0]
                best_paragraphs[qid] = {
                    "top_doc_id": top[0].doc_id,
                    "top_doc_score": top[1],
                    "best_para_id": top[2],
                    "best_para_score": top[3],
                    "best_para_text": top[4],
                }

        latency = time.time() - start
        return RerankOutput(
            runs=runs,
            latency_seconds=latency,
            avg_segments_per_candidate=(total_segments / total_candidates) if total_candidates else 0.0,
            avg_scored_segments_per_query=(total_scored_segments / len(queries)) if queries else 0.0,
            best_paragraphs=best_paragraphs,
        )
