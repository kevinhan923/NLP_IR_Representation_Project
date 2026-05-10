from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize


class BaseEncoder:
    def fit(self, texts: List[str]) -> 'BaseEncoder':
        raise NotImplementedError

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class TfidfProjectionEncoder(BaseEncoder):
    max_features: int = 200000
    projected_dim: int = 256
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    sublinear_tf: bool = True
    stop_words: Optional[str] = 'english'

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=1.0,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        self.projector = GaussianRandomProjection(n_components=self.projected_dim, random_state=42)
        self._fit_done = False

    def fit(self, texts: List[str]) -> 'TfidfProjectionEncoder':
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            raise ValueError("No valid texts provided to encoder.fit().")

        # small-corpus safety
        n_texts = len(texts)
        if n_texts < 3:
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0

        sparse = self.vectorizer.fit_transform(texts)
        dense = self.projector.fit_transform(sparse)
        _ = normalize(dense)
        self._fit_done = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fit_done:
            raise RuntimeError('Encoder must be fit before encode().')
        sparse = self.vectorizer.transform(texts)
        dense = self.projector.transform(sparse)
        dense = normalize(dense)
        return dense.astype(np.float32)


@dataclass
class TfidfEncoder(BaseEncoder):
    max_features: int = 200000
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    sublinear_tf: bool = True
    stop_words: Optional[str] = 'english'

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=1.0,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        self._fit_done = False

    def fit(self, texts: List[str]) -> 'TfidfEncoder':
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            raise ValueError("No valid texts provided to encoder.fit().")

        n_texts = len(texts)
        if n_texts < 3:
            self.vectorizer.min_df = 1
            self.vectorizer.max_df = 1.0

        self.vectorizer.fit(texts)
        self._fit_done = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fit_done:
            raise RuntimeError('Encoder must be fit before encode().')
        sparse = self.vectorizer.transform(texts)
        dense = normalize(sparse)
        return dense.astype(np.float32)


class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 64):
        raise RuntimeError(
            'SentenceTransformerEncoder is disabled in the no-torch version of this project. '
            'Use encoder.name = tfidf_projection or tfidf_dense_sparse instead.'
        )

    def fit(self, texts: List[str]) -> 'SentenceTransformerEncoder':
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        raise RuntimeError('SentenceTransformerEncoder is disabled in the no-torch version.')


def build_encoder(name: str = 'tfidf_projection', **kwargs) -> BaseEncoder:
    if 'ngram_range' in kwargs and isinstance(kwargs['ngram_range'], list):
        kwargs['ngram_range'] = tuple(kwargs['ngram_range'])
    if name == 'tfidf_projection':
        return TfidfProjectionEncoder(**kwargs)
    if name == 'tfidf_dense_sparse':
        return TfidfEncoder(**kwargs)
    if name == 'sentence_transformer':
        raise RuntimeError('sentence_transformer is not available in the base-environment version of this project.')
    raise ValueError(f'Unsupported encoder: {name}')