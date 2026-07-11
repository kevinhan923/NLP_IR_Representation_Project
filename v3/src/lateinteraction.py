"""lateinteraction.py — ColBERT late-interaction and the granularity spectrum.

The paper's question is "how much can you compress a document before retrieval
suffers?". Dense encoders give two points on that spectrum (one vector; one
vector per segment). ColBERT supplies the third, uncompressed end: EVERY token
keeps its embedding and relevance is MaxSim over tokens. Crucially, all three
systems here share the SAME ColBERT token embeddings — only the pooling
granularity differs — so this is the purest version of the controlled design:

  * PooledDocRetriever      — mean-pool ALL tokens -> one vector per document.
  * PooledSegmentRetriever  — segment the text, encode each segment separately,
                              mean-pool per segment, aggregate segment scores
                              (mirrors ParagraphLevelRetriever).
  * LateInteractionRetriever— no pooling at all: MaxSim(query tokens, doc
                              tokens), the standard ColBERT scoring.

If late interaction removes the paragraph penalty on holistic datasets (fiqa,
scidocs) while matching document-level elsewhere, the "compression loses
holistic context" story gets its final column.

Token encoding is done once in the driver script and passed in, so the model
loads once and the doc token matrices are shared between systems.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aggregation import aggregate


def pool_token_matrix(mat: np.ndarray) -> np.ndarray:
    """Mean-pool one (n_tokens, dim) matrix into a unit-norm vector."""
    v = np.asarray(mat, dtype=np.float32).mean(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def pool_token_matrices(mats: Sequence[np.ndarray]) -> np.ndarray:
    """Stack pooled vectors: list of (n_i, dim) -> (len(mats), dim), L2 rows."""
    return np.stack([pool_token_matrix(m) for m in mats])


class ColbertTokenEncoder:
    """pylate ColBERT wrapper returning per-text token-embedding matrices.

    Rows are L2-normalized by the model, so token dot products are cosines.
    Queries are padded/augmented to the model's fixed query length ([MASK]
    expansion), the standard ColBERT behavior.
    """

    def __init__(
        self,
        model_name: str = "lightonai/colbertv2.0",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        from pylate import models
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = models.ColBERT(model_name_or_path=model_name, device=device)
        self.batch_size = batch_size

    def encode_tokens(self, texts: List[str], is_query: bool = False) -> List[np.ndarray]:
        return self.model.encode(
            texts,
            is_query=is_query,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


class LateInteractionRetriever:
    """Token-granularity retrieval: score(q, d) = Σ_i max_j (q_i · d_j)."""

    def __init__(self):
        self._ids: List[str] = []
        self._tokens: Optional[np.ndarray] = None  # (total_tokens, dim)
        self._starts: Optional[np.ndarray] = None  # row offsets per doc

    def index(self, ids: List[str], token_mats: Sequence[np.ndarray]) -> "LateInteractionRetriever":
        if len(ids) != len(token_mats) or not ids:
            raise ValueError("ids and token_mats must be equal-length and non-empty")
        if any(m.shape[0] == 0 for m in token_mats):
            raise ValueError("every document needs at least one token embedding")
        self._ids = list(ids)
        self._tokens = np.concatenate([np.asarray(m, dtype=np.float32) for m in token_mats])
        lengths = [m.shape[0] for m in token_mats]
        self._starts = np.cumsum([0] + lengths[:-1]).astype(np.int64)
        return self

    def retrieve(self, q_tokens: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        if self._tokens is None:
            raise RuntimeError("Call index() before retrieve().")
        sims = self._tokens @ np.asarray(q_tokens, dtype=np.float32).T  # (total, n_q)
        per_doc_max = np.maximum.reduceat(sims, self._starts, axis=0)   # (n_docs, n_q)
        scores = per_doc_max.sum(axis=1)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self._ids[i], float(scores[i])) for i in order]


class PooledDocRetriever:
    """Document-level single vector from pooled ColBERT tokens (cosine)."""

    def __init__(self):
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None

    def index(self, ids: List[str], token_mats: Sequence[np.ndarray]) -> "PooledDocRetriever":
        self._ids = list(ids)
        self._matrix = pool_token_matrices(token_mats)
        return self

    def retrieve(self, q_vec: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        if self._matrix is None:
            raise RuntimeError("Call index() before retrieve().")
        scores = self._matrix @ np.asarray(q_vec, dtype=np.float32)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self._ids[i], float(scores[i])) for i in order]


class PooledSegmentRetriever:
    """Paragraph-level from pooled per-segment ColBERT tokens + aggregation."""

    def __init__(self, aggregation: str = "max", k: int = 3):
        self.aggregation = aggregation
        self.k = k
        self._ids: List[str] = []
        self._seg_matrix: Optional[np.ndarray] = None
        self._doc_seg_idx: Dict[str, List[int]] = {}

    def index(
        self,
        ids: List[str],
        seg_token_mats: Sequence[np.ndarray],
        doc_seg_idx: Dict[str, List[int]],
    ) -> "PooledSegmentRetriever":
        if set(ids) != set(doc_seg_idx):
            raise ValueError("doc_seg_idx must cover exactly the given ids")
        self._ids = list(ids)
        self._seg_matrix = pool_token_matrices(seg_token_mats)
        self._doc_seg_idx = doc_seg_idx
        return self

    def retrieve(self, q_vec: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        if self._seg_matrix is None:
            raise RuntimeError("Call index() before retrieve().")
        seg_scores = self._seg_matrix @ np.asarray(q_vec, dtype=np.float32)
        scored: List[Tuple[str, float]] = []
        for doc_id in self._ids:
            idx = self._doc_seg_idx[doc_id]
            scored.append((doc_id, aggregate(seg_scores[idx], method=self.aggregation, k=self.k)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
