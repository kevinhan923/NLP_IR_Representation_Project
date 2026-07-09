"""
granularity.py — Controlled document-level vs paragraph-level retrieval.

This is the module that most faithfully matches the paper's research question:
"Can a single vector represent a whole document, or do paragraph-level
representations preserve meaning better for retrieval?"

The paper's central methodological claim (Sec. 5.2) is that the comparison must
isolate **granularity** as the only difference: both systems share the *same*
encoder pipeline, and only the unit of representation changes (whole document
vs. paragraph). Earlier code (v1 experiment_beir, v2 pipeline) violated this:
the document-level "single-vector" side used one retriever while the
paragraph-level side used a *different* encoder, so the comparison conflated
"granularity" with "encoder". This module fixes that.

Both retrievers here:
  * use the SAME encoder type/config (a fresh instance each, fit on their own
    unit — documents vs. segments — which is exactly what "granularity" means);
  * score the FULL document collection (no shared stage-1 candidate pool);
  * use cosine similarity on L2-normalized vectors (paper Eq. 3-4).

Encoder default = TF-IDF -> Gaussian Random Projection -> L2 (paper Eq. 1-3),
which runs with only numpy/scikit-learn (no torch). Swap in a dense encoder for
the stronger-encoder experiments from the research plan.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


# ---------------------------------------------------------------------------
# Shared encoder (paper Eq. 1-3): TF-IDF -> Gaussian Random Projection -> L2
# ---------------------------------------------------------------------------

class TfidfProjectionEncoder:
    """TF-IDF -> Gaussian Random Projection -> L2 normalization.

    Mirrors the paper's encoder (Eq. 1-3). ``r`` is the projected dimension
    (paper uses r=192). The SAME class/config is used for both the document-
    level and paragraph-level retriever; only the text unit it is fit/encoded
    on differs, which is precisely the granularity variable under study.
    """

    def __init__(self, r: int = 192, max_features: int = 50000, seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.random_projection import GaussianRandomProjection

        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.projector = GaussianRandomProjection(n_components=r, random_state=seed)
        self._fitted = False

    def fit(self, texts: List[str]) -> "TfidfProjectionEncoder":
        sparse = self.vectorizer.fit_transform(texts)
        self.projector.fit(sparse)
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        sparse = self.vectorizer.transform(texts)
        dense = self.projector.transform(sparse).astype(np.float32)
        return _l2_normalize(dense)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization so that dot product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def default_encoder_factory(r: int = 192, seed: int = 42) -> Callable[[], TfidfProjectionEncoder]:
    """Return a zero-arg factory producing fresh, identically-configured encoders."""
    return lambda: TfidfProjectionEncoder(r=r, seed=seed)


# ---------------------------------------------------------------------------
# Aggregation (paper Sec. 4.2 / Eq. 5): collapse segment scores to a doc score
# ---------------------------------------------------------------------------

def aggregate(
    scores: np.ndarray,
    method: str = "topk_mean",
    k: int = 3,
    temperature: float = 1.0,
    threshold: float = 0.2,
) -> float:
    """Aggregate per-segment similarity scores into one document score.

    Implements the five strategies discussed in the paper:
    max, topk_mean, weighted_topk, softmax_weighted, threshold_mean.
    """
    if scores.size == 0:
        return float("-inf")
    s = np.sort(scores)[::-1]  # descending

    if method == "max":
        return float(s[0])
    if method == "topk_mean":
        kk = min(k, s.size)
        return float(s[:kk].mean())
    if method == "weighted_topk":
        kk = min(k, s.size)
        top = s[:kk]
        weights = np.arange(kk, 0, -1, dtype=np.float32)  # k, k-1, ..., 1
        return float((weights * top).sum() / weights.sum())
    if method == "softmax_weighted":
        t = max(temperature, 1e-8)
        e = np.exp(s / t)
        return float((e * s).sum() / e.sum())
    if method == "threshold_mean":
        kept = s[s >= threshold]
        if kept.size == 0:
            kept = s[:1]
        return float(kept.mean())
    raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# The two retrieval systems — identical encoder, different granularity
# ---------------------------------------------------------------------------

class DocumentLevelRetriever:
    """Whole document -> one vector; rank all documents by cosine to the query."""

    def __init__(self, encoder: TfidfProjectionEncoder):
        self.encoder = encoder
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None

    def index(self, passages: List[Dict]) -> "DocumentLevelRetriever":
        self._ids = [p["id"] for p in passages]
        texts = [p["text"] for p in passages]
        self.encoder.fit(texts)
        self._matrix = self.encoder.encode(texts)
        return self

    def retrieve(self, query: str, top_k: int = 100) -> List[str]:
        if self._matrix is None:
            raise RuntimeError("Call index() before retrieve().")
        q = self.encoder.encode([query])[0]
        scores = self._matrix @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [self._ids[i] for i in order]


class ParagraphLevelRetriever:
    """Each document -> segments; score every segment, aggregate to a doc score.

    Uses the SAME encoder class/config as DocumentLevelRetriever, but fit and
    encoded on segments instead of whole documents — the granularity variable.
    """

    def __init__(
        self,
        encoder: TfidfProjectionEncoder,
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

    def retrieve(self, query: str, top_k: int = 100) -> List[str]:
        if self._seg_matrix is None:
            raise RuntimeError("Call index() before retrieve().")
        q = self.encoder.encode([query])[0]
        seg_scores = self._seg_matrix @ q
        doc_scores = []
        for doc_id in self._ids:
            idx = self._doc_seg_idx[doc_id]
            score = aggregate(
                seg_scores[idx],
                method=self.aggregation,
                k=self.k,
                temperature=self.temperature,
                threshold=self.threshold,
            )
            doc_scores.append((doc_id, score))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in doc_scores[:top_k]]


# ---------------------------------------------------------------------------
# Controlled comparison driver
# ---------------------------------------------------------------------------

def run_granularity_comparison(
    data: Dict,
    make_encoder: Callable[[], TfidfProjectionEncoder],
    segmenter: Callable[[str], List[str]],
    aggregation: str = "topk_mean",
    k: int = 3,
    top_k: int = 100,
) -> Dict:
    """Evaluate document-level vs paragraph-level on the SAME data + encoder.

    Args:
        data:         dict from data.load_* with keys passages/queries/qrels
                      (qrels maps qid -> {doc_id: relevance}).
        make_encoder: zero-arg factory returning a fresh, identically-configured
                      encoder (one for each retriever — same config, separate fit).
        segmenter:    callable text -> list[str] of segments.
        aggregation:  segment-score aggregation method.
        k:            top-k for aggregation.
        top_k:        ranking depth.

    Returns:
        dict with single-vector / paragraph-level metrics, deltas, and per-query
        MRR for downstream "when does granularity help" analysis.
    """
    passages = data["passages"]
    queries = data["queries"]
    qrels = data["qrels"]

    doc_ret = DocumentLevelRetriever(make_encoder()).index(passages)
    par_ret = ParagraphLevelRetriever(
        make_encoder(), segmenter=segmenter, aggregation=aggregation, k=k
    ).index(passages)

    rows = []
    s_mrr = s_ndcg = s_rec = 0.0
    p_mrr = p_ndcg = p_rec = 0.0
    n = 0
    for query in queries:
        qid = query["id"]
        q_qrels = qrels.get(qid, {})
        if not q_qrels:
            continue
        s_ranked = doc_ret.retrieve(query["text"], top_k=top_k)
        p_ranked = par_ret.retrieve(query["text"], top_k=top_k)

        sm = mrr_at_k(s_ranked, q_qrels, k=10)
        pm = mrr_at_k(p_ranked, q_qrels, k=10)
        s_mrr += sm
        p_mrr += pm
        s_ndcg += ndcg_at_k(s_ranked, q_qrels, k=10)
        p_ndcg += ndcg_at_k(p_ranked, q_qrels, k=10)
        s_rec += recall_at_k(s_ranked, q_qrels, k=100)
        p_rec += recall_at_k(p_ranked, q_qrels, k=100)
        rows.append({"qid": qid, "single_mrr10": sm, "para_mrr10": pm, "delta": pm - sm})
        n += 1

    if n == 0:
        raise RuntimeError("No judged queries to evaluate.")

    single = {"mrr@10": s_mrr / n, "ndcg@10": s_ndcg / n, "recall@100": s_rec / n}
    paragraph = {"mrr@10": p_mrr / n, "ndcg@10": p_ndcg / n, "recall@100": p_rec / n}
    return {
        "n_queries": n,
        "single_vector": single,
        "paragraph_level": paragraph,
        "delta": {m: paragraph[m] - single[m] for m in single},
        "per_query": rows,
        "config": {"aggregation": aggregation, "k": k, "top_k": top_k},
    }
