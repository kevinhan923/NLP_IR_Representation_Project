# Run from v2/: cd v2 && python -m pytest tests/test_granularity.py -v
"""
test_granularity.py — Tests for the controlled document- vs paragraph-level
retrieval module (src/granularity.py).

Focus: the comparison isolates granularity (same encoder config on both sides),
aggregation methods are numerically correct, and both retrievers rank a clearly
relevant document first on a small hand-built corpus.
"""

import numpy as np
import pytest

from src.granularity import (
    aggregate,
    _l2_normalize,
    default_encoder_factory,
    DocumentLevelRetriever,
    ParagraphLevelRetriever,
    run_granularity_comparison,
)


def _split_sentences(text):
    return [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_max():
    assert aggregate(np.array([0.1, 0.9, 0.5]), method="max") == pytest.approx(0.9)


def test_aggregate_topk_mean():
    # top-2 of [0.9, 0.5, 0.1] = (0.9+0.5)/2 = 0.7
    assert aggregate(np.array([0.1, 0.9, 0.5]), method="topk_mean", k=2) == pytest.approx(0.7)


def test_aggregate_weighted_topk():
    # top-2 weighted: (2*0.9 + 1*0.5) / 3 = 2.3/3
    got = aggregate(np.array([0.9, 0.5, 0.1]), method="weighted_topk", k=2)
    assert got == pytest.approx((2 * 0.9 + 1 * 0.5) / 3)


def test_aggregate_threshold_mean():
    # threshold 0.4 keeps [0.9, 0.5]; mean = 0.7
    got = aggregate(np.array([0.1, 0.9, 0.5, 0.2]), method="threshold_mean", threshold=0.4)
    assert got == pytest.approx(0.7)


def test_aggregate_threshold_mean_empty_falls_back_to_top():
    # nothing passes threshold -> falls back to single best
    got = aggregate(np.array([0.1, 0.2]), method="threshold_mean", threshold=0.9)
    assert got == pytest.approx(0.2)


def test_aggregate_unknown_raises():
    with pytest.raises(ValueError):
        aggregate(np.array([0.1]), method="nope")


def test_l2_normalize_unit_rows():
    m = _l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.linalg.norm(m[0]) == pytest.approx(1.0)
    assert np.allclose(m[1], 0.0)  # zero row stays zero, no NaN


# ---------------------------------------------------------------------------
# Retrievers on a small hand-built corpus
# ---------------------------------------------------------------------------

_PASSAGES = [
    {"id": "d1", "text": "Photosynthesis converts sunlight into chemical energy in plants. Chlorophyll absorbs light."},
    {"id": "d2", "text": "The stock market fell sharply today. Investors worried about interest rates and inflation."},
    {"id": "d3", "text": "Volcanoes erupt when magma rises. Lava flows cover the surrounding land with ash."},
    {"id": "d4", "text": "Dogs are loyal pets. They need exercise, food, and regular veterinary care."},
]


def test_document_retriever_ranks_relevant_first():
    enc = default_encoder_factory(r=64)()
    ret = DocumentLevelRetriever(enc).index(_PASSAGES)
    ranked = ret.retrieve("sunlight chlorophyll plants photosynthesis", top_k=4)
    assert ranked[0] == "d1"


def test_paragraph_retriever_ranks_relevant_first_and_shape():
    enc = default_encoder_factory(r=64)()
    ret = ParagraphLevelRetriever(enc, segmenter=_split_sentences, aggregation="max").index(_PASSAGES)
    ranked = ret.retrieve("magma lava volcano eruption", top_k=4)
    assert ranked[0] == "d3"
    assert len(ranked) == 4
    assert set(ranked) == {"d1", "d2", "d3", "d4"}


# ---------------------------------------------------------------------------
# Controlled comparison driver
# ---------------------------------------------------------------------------

def test_run_comparison_shape_and_isolation():
    data = {
        "passages": _PASSAGES,
        "queries": [
            {"id": "q1", "text": "photosynthesis sunlight chlorophyll"},
            {"id": "q2", "text": "volcano magma lava eruption"},
        ],
        "qrels": {"q1": {"d1": 1}, "q2": {"d3": 1}},
    }
    out = run_granularity_comparison(
        data,
        make_encoder=default_encoder_factory(r=64),
        segmenter=_split_sentences,
        aggregation="topk_mean",
        k=2,
        top_k=4,
    )
    assert out["n_queries"] == 2
    for side in ("single_vector", "paragraph_level", "delta"):
        assert {"mrr@10", "ndcg@10", "recall@100"} <= set(out[side])
    assert len(out["per_query"]) == 2
    # delta == paragraph - single, consistency check
    assert out["delta"]["mrr@10"] == pytest.approx(
        out["paragraph_level"]["mrr@10"] - out["single_vector"]["mrr@10"]
    )
