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
    # 4 sentences, window=2, stride=1 -> 3 windows
    assert len(paragraphs) == 3


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
