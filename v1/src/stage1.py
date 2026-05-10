from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from .types import CandidateResult, DocumentExample, QueryExample


@dataclass
class Stage1Output:
    runs: Dict[str, List[CandidateResult]]
    latency_seconds: float
    method: str


class LocalTfidfRetriever:
    def __init__(self, max_features: int = 300000, ngram_range: tuple = (1, 2), stop_words: str | None = "english"):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words, sublinear_tf=True)
        self.doc_matrix = None
        self.doc_ids: List[str] = []
        self.docs: Dict[str, DocumentExample] = {}

    def fit(self, docs: List[DocumentExample]) -> "LocalTfidfRetriever":
        self.doc_ids = [d.doc_id for d in docs]
        self.docs = {d.doc_id: d for d in docs}
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in docs])
        return self

    def retrieve(self, queries: List[QueryExample], top_k: int = 100) -> Stage1Output:
        if self.doc_matrix is None:
            raise RuntimeError("Retriever must be fit before retrieve().")
        start = time.time()
        runs: Dict[str, List[CandidateResult]] = {}
        for q in tqdm(queries, desc="Stage1 TF-IDF retrieval"):
            qv = self.vectorizer.transform([q.text])
            sims = linear_kernel(qv, self.doc_matrix).ravel()
            top_idx = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            ranked: List[CandidateResult] = []
            for i in top_idx:
                doc_id = self.doc_ids[int(i)]
                doc = self.docs[doc_id]
                ranked.append(CandidateResult(doc_id=doc_id, score=float(sims[int(i)]), text=doc.text, title=doc.title, metadata=doc.metadata))
            runs[q.query_id] = ranked
        return Stage1Output(runs=runs, latency_seconds=time.time() - start, method="local_tfidf")
