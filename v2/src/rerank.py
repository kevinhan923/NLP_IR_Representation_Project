import numpy as np
from typing import List, Dict


class BiEncoderReranker:
    """
    Paragraph reranker using a sentence-transformers bi-encoder.
    For each candidate document:
      1. Segment into paragraphs.
      2. Encode all paragraphs with the bi-encoder.
      3. Score = cosine similarity between query embedding and each paragraph embedding.
      4. Aggregate paragraph scores to a single document score.
    """

    def __init__(
        self,
        model_name: str = "msmarco-distilbert-base-v4",
        segmenter=None,
        aggregation: str = "max",
    ):
        from sentence_transformers import SentenceTransformer
        from .segmentation import StructuralSegmenter
        self.model = SentenceTransformer(model_name)
        self.segmenter = segmenter or StructuralSegmenter()
        self.aggregation = aggregation

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scored = []
        for doc in candidates:
            paragraphs = self.segmenter.segment(doc["text"])
            para_texts = [p["text"] for p in paragraphs]
            para_embs = self.model.encode(para_texts, normalize_embeddings=True)
            para_scores = para_embs @ q_emb
            doc_score = self._aggregate(para_scores, paragraphs)
            scored.append({**doc, "score": float(doc_score), "para_scores": para_scores.tolist()})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _aggregate(self, scores: np.ndarray, paragraphs: list) -> float:
        if self.aggregation == "max":
            return float(scores.max())
        elif self.aggregation == "topk_mean":
            k = min(3, len(scores))
            return float(np.sort(scores)[::-1][:k].mean())
        else:
            return float(scores.max())


class CrossEncoderReranker:
    """
    Paragraph reranker using a cross-encoder.
    Cross-encoder principle: concatenate [query, paragraph] into a single
    transformer input -> full cross-attention -> scalar relevance score.
    More accurate than bi-encoder because query and paragraph tokens attend to each other.
    Expensive: O(Q x P) forward passes, not pre-computable.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        segmenter=None,
        aggregation: str = "max",
    ):
        from sentence_transformers import CrossEncoder
        from .segmentation import StructuralSegmenter
        self.model = CrossEncoder(model_name, max_length=512)
        self.segmenter = segmenter or StructuralSegmenter()
        self.aggregation = aggregation

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        scored = []
        for doc in candidates:
            paragraphs = self.segmenter.segment(doc["text"])
            pairs = [(query, p["text"]) for p in paragraphs]
            para_scores = np.array(self.model.predict(pairs))
            doc_score = float(para_scores.max()) if self.aggregation == "max" else float(para_scores.mean())
            scored.append({**doc, "score": doc_score, "para_scores": para_scores.tolist()})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
