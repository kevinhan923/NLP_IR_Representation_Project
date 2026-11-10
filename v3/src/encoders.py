"""encoders.py — Unified encoder interface.

All encoders return L2-normalized vectors so dot product == cosine similarity
(paper Eq. 3-4). The SAME encoder config is used for both the document-level
and paragraph-level retriever — only the unit it is fit/encoded on differs.

`encode(texts, is_query=...)` supports asymmetric models (E5/BGE) that need a
different prefix for queries vs. passages; symmetric encoders ignore the flag.

  * M0: TfidfProjectionEncoder (TF-IDF -> Gaussian Random Projection -> L2).
  * M1: DenseEncoder (sentence-transformers: BGE / E5 / MiniLM).
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np


class Encoder(Protocol):
    def fit(self, texts: List[str]) -> "Encoder": ...
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero (no NaN)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfProjectionEncoder:
    """TF-IDF -> Gaussian Random Projection (dim r) -> L2 (paper Eq. 1-3).

    Symmetric: ignores is_query.
    """

    def __init__(
        self,
        r: int = 192,
        max_features: int = 50000,
        seed: int = 42,
        ngram_max: int = 1,
        min_df: int = 1,
        sublinear_tf: bool = False,
        fit_cap: int = 0,
    ):
        """Defaults reproduce the simplified vectorizer of the controlled runs.

        ngram_max / min_df / sublinear_tf / fit_cap let a run match the
        two-stage paragraph encoder instead (v1: 1-2-grams, min_df=2,
        sublinear TF, fit on the first `fit_cap` texts); fit_cap=0 means fit
        on every text.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.random_projection import GaussianRandomProjection

        self.r = r
        self.fit_cap = fit_cap
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, ngram_max),
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )
        self.projector = GaussianRandomProjection(n_components=r, random_state=seed)
        self._fitted = False

    def fit(self, texts: List[str]) -> "TfidfProjectionEncoder":
        # Like v1's fit_text_cap: fit on the FIRST fit_cap texts, in order.
        fit_texts = texts[: self.fit_cap] if self.fit_cap > 0 else texts
        sparse = self.vectorizer.fit_transform(fit_texts)
        self.projector.fit(sparse)
        self._fitted = True
        return self

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before encode().")
        sparse = self.vectorizer.transform(texts)
        dense = self.projector.transform(sparse).astype(np.float32)
        return l2_normalize(dense)


def _dense_prefixes(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, doc_prefix) for known asymmetric dense models."""
    m = model_name.lower()
    if "e5" in m:
        return ("query: ", "passage: ")
    if "bge" in m and "-en" in m:
        return ("Represent this sentence for searching relevant passages: ", "")
    return ("", "")  # symmetric (e.g. all-MiniLM)


class DenseEncoder:
    """sentence-transformers bi-encoder. Pretrained -> fit() is a no-op.

    Used identically for document-level and paragraph-level retrieval (same
    model); only the unit encoded differs. Queries vs passages get the model's
    recommended prefix so E5/BGE perform as intended.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 64,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size

    def fit(self, texts: List[str]) -> "DenseEncoder":
        return self  # pretrained; nothing to fit

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.doc_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # dot product == cosine
            show_progress_bar=False,
        ).astype(np.float32)


def build_encoder(cfg) -> Encoder:
    """Construct an encoder from an EncoderConfig (or duck-typed equivalent)."""
    if cfg.name == "tfidf_projection":
        return TfidfProjectionEncoder(
            r=cfg.r, max_features=cfg.max_features, seed=cfg.seed,
            ngram_max=getattr(cfg, "ngram_max", 1),
            min_df=getattr(cfg, "min_df", 1),
            sublinear_tf=getattr(cfg, "sublinear_tf", False),
            fit_cap=getattr(cfg, "fit_cap", 0),
        )
    if cfg.name == "dense":
        if not getattr(cfg, "model", ""):
            raise ValueError("dense encoder requires cfg.model (e.g. 'intfloat/e5-base-v2')")
        qp, dp = _dense_prefixes(cfg.model)
        return DenseEncoder(cfg.model, query_prefix=qp, doc_prefix=dp,
                            batch_size=getattr(cfg, "batch_size", 64))
    raise ValueError(f"Unknown encoder '{cfg.name}'. Use 'tfidf_projection' or 'dense'.")
