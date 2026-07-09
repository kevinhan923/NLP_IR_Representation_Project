"""retrieval.py — Controlled document-level vs paragraph-level retrieval.

The heart of the paper's question. Both retrievers use the SAME encoder
(class + config); the ONLY difference is the unit of representation:

  * DocumentLevelRetriever  — whole document -> one vector.
  * ParagraphLevelRetriever — document -> segments -> aggregate segment scores.

Both score the FULL collection (no shared stage-1 candidate pool) and rank by
cosine similarity on L2-normalized vectors. retrieve() returns (doc_id, score)
pairs so results plug directly into ir_measures runs.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .aggregation import aggregate


class DocumentLevelRetriever:
    def __init__(self, encoder):
        self.encoder = encoder
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None

    def index(self, passages: List[Dict]) -> "DocumentLevelRetriever":
        self._ids = [p["id"] for p in passages]
        texts = [p["text"] for p in passages]
        self.encoder.fit(texts)
        self._matrix = self.encoder.encode(texts)
        return self

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        if self._matrix is None:
            raise RuntimeError("Call index() before retrieve().")
        q = self.encoder.encode([query], is_query=True)[0]
        scores = self._matrix @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [(self._ids[i], float(scores[i])) for i in order]


class ParagraphLevelRetriever:
    """Same encoder config as DocumentLevelRetriever, fit on segments instead."""

    def __init__(
        self,
        encoder,
        segmenter: Callable[[str], List[str]],
        aggregation: str = "topk_mean",
        k: int = 3,
        temperature: float = 1.0,
        threshold: float = 0.2,
    ):
        self.encoder = encoder
        self.segmenter = segmenter
        self.aggregation = aggregation
        self.k = k
        self.temperature = temperature
        self.threshold = threshold
        self._ids: List[str] = []
        self._seg_matrix: Optional[np.ndarray] = None
        self._doc_seg_idx: Dict[str, List[int]] = {}

    def index(self, passages: List[Dict]) -> "ParagraphLevelRetriever":
        self._ids = [p["id"] for p in passages]
        seg_texts: List[str] = []
        self._doc_seg_idx = {}
        for p in passages:
            segs = self.segmenter(p["text"]) or [p["text"]]
            start = len(seg_texts)
            seg_texts.extend(segs)
            self._doc_seg_idx[p["id"]] = list(range(start, start + len(segs)))
        self.encoder.fit(seg_texts)
        self._seg_matrix = self.encoder.encode(seg_texts)
        return self

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        if self._seg_matrix is None:
            raise RuntimeError("Call index() before retrieve().")
        q = self.encoder.encode([query], is_query=True)[0]
        seg_scores = self._seg_matrix @ q
        scored: List[Tuple[str, float]] = []
        for doc_id in self._ids:
            idx = self._doc_seg_idx[doc_id]
            score = aggregate(
                seg_scores[idx],
                method=self.aggregation,
                k=self.k,
                temperature=self.temperature,
                threshold=self.threshold,
            )
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
