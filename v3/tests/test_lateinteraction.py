# Run from v3/: cd v3 && python -m pytest tests/test_lateinteraction.py -v
"""Unit tests for the ColBERT granularity-spectrum retrievers.

The retrieval math is tested with hand-built token matrices (no model), so
these tests are fast and deterministic; one optional smoke test exercises the
real pylate model when installed.
"""

import importlib.util

import numpy as np
import pytest

from src.lateinteraction import (
    LateInteractionRetriever,
    PooledDocRetriever,
    PooledSegmentRetriever,
    pool_token_matrix,
    pool_token_matrices,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# Two orthogonal "topics" in a 4-dim toy space.
_VOLCANO = _unit([1, 0, 0, 0])
_FINANCE = _unit([0, 1, 0, 0])
_NOISE = _unit([0, 0, 1, 0])


def test_pool_token_matrix_is_unit_norm_mean():
    mat = np.stack([_VOLCANO, _NOISE])
    pooled = pool_token_matrix(mat)
    assert np.linalg.norm(pooled) == pytest.approx(1.0)
    expected = _unit((_VOLCANO + _NOISE) / 2)
    assert np.allclose(pooled, expected, atol=1e-6)


def test_late_interaction_maxsim_exact_value():
    """MaxSim must equal the hand-computed sum of per-query-token maxima."""
    d1 = np.stack([_VOLCANO, _NOISE])           # volcano doc
    d2 = np.stack([_FINANCE, _NOISE, _NOISE])   # finance doc
    ret = LateInteractionRetriever().index(["d1", "d2"], [d1, d2])

    q = np.stack([_VOLCANO, _NOISE])            # 2-token query
    out = ret.retrieve(q, top_k=2)
    assert out[0][0] == "d1"
    # d1: max(vol·vol, noise·vol)=1 + max(vol·noise, noise·noise)=1 -> 2.0
    assert out[0][1] == pytest.approx(2.0)
    # d2: max over tokens for vol query token = 0, for noise token = 1 -> 1.0
    assert dict(out)["d2"] == pytest.approx(1.0)


def test_late_interaction_dilution_immunity_vs_pooling():
    """The discriminating case: a long doc with ONE on-topic token among many
    off-topic ones. Pooling dilutes the signal; late interaction keeps it."""
    focused = np.stack([_VOLCANO])                        # short, on-topic
    diluted = np.stack([_VOLCANO] + [_NOISE] * 9)          # 1 signal, 9 noise

    late = LateInteractionRetriever().index(["f", "d"], [focused, diluted])
    q = np.stack([_VOLCANO])
    late_scores = dict(late.retrieve(q, top_k=2))
    assert late_scores["f"] == pytest.approx(late_scores["d"])  # both hit max=1

    pooled = PooledDocRetriever().index(["f", "d"], [focused, diluted])
    pooled_scores = dict(pooled.retrieve(_VOLCANO, top_k=2))
    assert pooled_scores["f"] > pooled_scores["d"]  # pooling diluted "d"


def test_pooled_segment_retriever_max_aggregation():
    seg_mats = [np.stack([_VOLCANO]), np.stack([_NOISE]),   # d1: 2 segments
                np.stack([_FINANCE])]                        # d2: 1 segment
    idx = {"d1": [0, 1], "d2": [2]}
    ret = PooledSegmentRetriever(aggregation="max").index(["d1", "d2"], seg_mats, idx)
    out = dict(ret.retrieve(_VOLCANO, top_k=2))
    assert out["d1"] == pytest.approx(1.0)   # best segment wins
    assert out["d2"] == pytest.approx(0.0)


def test_index_validation_errors():
    with pytest.raises(ValueError):
        LateInteractionRetriever().index([], [])
    with pytest.raises(ValueError):
        LateInteractionRetriever().index(["a"], [np.zeros((0, 4), dtype=np.float32)])
    with pytest.raises(ValueError):
        PooledSegmentRetriever().index(["a"], [np.stack([_NOISE])], {"b": [0]})
    with pytest.raises(RuntimeError):
        LateInteractionRetriever().retrieve(np.stack([_VOLCANO]))


def test_reduceat_grouping_matches_naive_loop():
    """The vectorized reduceat path must equal a naive per-doc max loop."""
    rng = np.random.default_rng(0)
    mats = [rng.normal(size=(n, 8)).astype(np.float32) for n in (3, 1, 5, 2)]
    mats = [m / np.linalg.norm(m, axis=1, keepdims=True) for m in mats]
    ids = [f"d{i}" for i in range(len(mats))]
    q = rng.normal(size=(4, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    fast = dict(LateInteractionRetriever().index(ids, mats).retrieve(q, top_k=4))
    naive = {i: float((m @ q.T).max(axis=0).sum()) for i, m in zip(ids, mats)}
    for i in ids:
        assert fast[i] == pytest.approx(naive[i], abs=1e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("pylate") is None,
    reason="pylate not installed",
)
def test_colbert_encoder_smoke_cpu():
    """Real-model smoke: shapes, normalization, and relevance ordering."""
    from src.lateinteraction import ColbertTokenEncoder

    enc = ColbertTokenEncoder(device="cpu", batch_size=4)
    docs = enc.encode_tokens(
        ["Volcanoes erupt when magma rises.", "The stock market fell today."]
    )
    q = enc.encode_tokens(["volcano lava eruption"], is_query=True)[0]
    assert docs[0].ndim == 2 and docs[0].shape[1] == q.shape[1]
    assert np.allclose(np.linalg.norm(docs[0], axis=1), 1.0, atol=1e-2)

    ret = LateInteractionRetriever().index(["vol", "fin"], docs)
    assert ret.retrieve(q, top_k=2)[0][0] == "vol"
