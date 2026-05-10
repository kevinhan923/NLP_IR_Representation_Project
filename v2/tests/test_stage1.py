import pytest
from src.stage1 import BM25Retriever, DenseRetriever

PASSAGES = [
    {"id": "p1", "text": "The capital of France is Paris."},
    {"id": "p2", "text": "Python is a programming language."},
    {"id": "p3", "text": "Paris has many museums and art galleries."},
]


def test_bm25_retriever_returns_ranked_ids():
    retriever = BM25Retriever()
    retriever.index(PASSAGES)
    results = retriever.retrieve("Paris museums", top_k=2)
    assert len(results) == 2
    # p3 contains BOTH "Paris" AND "museums", should rank higher than p1
    assert results[0]["id"] == "p3"
    ids = [r["id"] for r in results]
    assert "p2" not in ids


def test_bm25_retriever_scores_are_descending():
    retriever = BM25Retriever()
    retriever.index(PASSAGES)
    results = retriever.retrieve("Python programming", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_retriever_returns_ranked_ids():
    retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
    retriever.index(PASSAGES)
    results = retriever.retrieve("French capital city", top_k=2)
    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert "p1" in ids or "p3" in ids


def test_dense_retriever_scores_are_descending():
    retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
    retriever.index(PASSAGES)
    results = retriever.retrieve("museum art gallery", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_raises_before_index():
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError, match="index"):
        retriever.retrieve("query")


def test_dense_raises_before_index():
    retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
    with pytest.raises(RuntimeError, match="index"):
        retriever.retrieve("query")
