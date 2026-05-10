import re
import numpy as np
from typing import List, Dict


def _tokenize(text: str) -> list:
    """Lowercase and split on non-alphanumeric characters (strips punctuation)."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """
    Stage 1 retriever using Okapi BM25.
    Principle: probabilistic term-frequency ranking with IDF and doc-length
    normalization.  Term frequency saturates (k1=1.5) and doc length is
    normalized (b=0.75) — both improvements over raw TF-IDF.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._passages: List[Dict] = []

    def index(self, passages: List[Dict]):
        from rank_bm25 import BM25Okapi
        self._passages = passages
        tokenized = [_tokenize(p["text"]) for p in passages]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve()")
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Sort descending; only include passages with a positive score so
        # documents with zero BM25 overlap are excluded from results.
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._passages[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]


class DenseRetriever:
    """
    Stage 1 retriever using a dense bi-encoder + FAISS index.
    Principle: encode all docs offline into L2-normalised vectors, then at
    query time compute dot product (= cosine similarity) via FAISS
    IndexFlatIP for exact nearest-neighbour search.
    """

    def __init__(self, model_name: str = "msmarco-distilbert-base-v4"):
        from src.encoders import DenseEncoder
        self.encoder = DenseEncoder(model_name)
        self._index = None
        self._passages: List[Dict] = []

    def index(self, passages: List[Dict], batch_size: int = 64):
        import faiss
        self._passages = passages
        texts = [p["text"] for p in passages]
        embeddings = self.encoder.encode(texts, batch_size=batch_size, show_progress=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        if self._index is None:
            raise RuntimeError("Call index() before retrieve()")
        q_emb = self.encoder.encode([query]).astype(np.float32)
        scores, indices = self._index.search(q_emb, top_k)
        return [
            {**self._passages[idx], "score": float(scores[0][rank])}
            for rank, idx in enumerate(indices[0])
            if idx != -1
        ]
