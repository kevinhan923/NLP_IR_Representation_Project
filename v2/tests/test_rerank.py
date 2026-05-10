import pytest
from src.segmentation import StructuralSegmenter, SentenceWindowSegmenter
from src.rerank import BiEncoderReranker, CrossEncoderReranker

DOC = {
    "id": "d1",
    "text": (
        "Climate change affects global temperatures.\n\n"
        "Greenhouse gases trap heat in the atmosphere.\n\n"
        "Carbon dioxide is the most significant contributor."
    )
}


def test_structural_segmenter_splits_on_double_newline():
    seg = StructuralSegmenter(min_chars=10)
    paragraphs = seg.segment(DOC["text"])
    assert len(paragraphs) == 3
    assert "Greenhouse gases" in paragraphs[1]["text"]


def test_sentence_window_segmenter_produces_overlapping_windows():
    seg = SentenceWindowSegmenter(window=2, stride=1)
    text = "A. B. C. D."
    paragraphs = seg.segment(text)
    # 4 sentences, window=2, stride=1 -> 4 windows (trailing sentence no longer dropped)
    assert len(paragraphs) == 4


def test_biencoder_reranker_scores_passages():
    reranker = BiEncoderReranker(model_name="all-MiniLM-L6-v2")
    query = "greenhouse gas emissions"
    candidates = [DOC]
    results = reranker.rerank(query, candidates, top_k=1)
    assert len(results) == 1
    assert "score" in results[0]
    assert results[0]["score"] > 0


def test_crossencoder_reranker_scores_passages():
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    query = "greenhouse gas emissions"
    candidates = [DOC]
    results = reranker.rerank(query, candidates, top_k=1)
    assert len(results) == 1
    assert "score" in results[0]


def test_unknown_aggregation_raises():
    reranker = BiEncoderReranker(model_name="all-MiniLM-L6-v2", aggregation="bad_agg")
    with pytest.raises(ValueError, match="Unknown aggregation"):
        reranker.rerank("query", [DOC], top_k=1)


def test_biencoder_reranker_ranks_relevant_doc_first():
    reranker = BiEncoderReranker(model_name="all-MiniLM-L6-v2")
    query = "greenhouse gas climate change"
    relevant_doc = {"id": "d1", "text": "Greenhouse gases trap heat and cause climate change."}
    irrelevant_doc = {"id": "d2", "text": "Python is a popular programming language for data science."}
    results = reranker.rerank(query, [irrelevant_doc, relevant_doc], top_k=2)
    assert results[0]["id"] == "d1"  # relevant doc should rank first


# --- Aggregation tests ---

from src.aggregation import MaxAggregator, TopKMeanAggregator, LearnedMLPAggregator
import numpy as np


def test_max_aggregator():
    agg = MaxAggregator()
    scores = np.array([0.3, 0.8, 0.5])
    features = [{"position": 0, "length": 50}, {"position": 1, "length": 80}, {"position": 2, "length": 40}]
    assert agg.aggregate(scores, features) == 0.8


def test_topk_mean_aggregator():
    agg = TopKMeanAggregator(k=2)
    scores = np.array([0.3, 0.8, 0.5])
    features = [{"position": 0, "length": 50}, {"position": 1, "length": 80}, {"position": 2, "length": 40}]
    result = agg.aggregate(scores, features)
    assert abs(result - (0.8 + 0.5) / 2) < 1e-5


def test_learned_mlp_aggregator_predict_before_train():
    agg = LearnedMLPAggregator()
    scores = np.array([0.3, 0.8])
    features = [{"position": 0, "length": 50, "overlap": 0.2}, {"position": 1, "length": 80, "overlap": 0.5}]
    # Should return a float without crashing (uses random weights before training)
    result = agg.aggregate(scores, features)
    assert isinstance(result, float)
