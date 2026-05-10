import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection


class TfidfEncoder:
    """V1-compatible sparse TF-IDF encoder. Used only as ablation baseline."""

    def __init__(self, max_features=50000, n_components=256, seed=42):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
        self._fitted = False

    def fit(self, texts: list[str]):
        sparse = self.vectorizer.fit_transform(texts)
        self.projector.fit(sparse)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before encode()")
        sparse = self.vectorizer.transform(texts)
        return self.projector.transform(sparse).astype(np.float32)


class DenseEncoder:
    """
    Wraps a sentence-transformers bi-encoder.
    Principle: maps text -> fixed-dim dense vector via a fine-tuned transformer.
    Similarity = cosine / dot product in the embedding space.
    """

    def __init__(self, model_name: str = "msmarco-distilbert-base-v4"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # ensures dot product == cosine similarity
        )
