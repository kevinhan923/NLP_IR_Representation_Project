# Run from v3/: cd v3 && python -m pytest tests/test_adaptive.py -v
"""Tests for adaptive granularity retrieval (M3)."""

import numpy as np
import pytest

from src.adaptive import (
    FEATURE_ORDER,
    GranularityRouter,
    fuse_query,
    fuse_runs,
    oracle_per_query,
    query_features,
    tune_global_alpha,
    z_normalize_scores,
)
from src.config import EncoderConfig
from src.encoders import build_encoder
from src.retrieval import ParagraphLevelRetriever


# --------------------------------------------------------------------------- #
# z-normalization + fusion
# --------------------------------------------------------------------------- #

def _ranking(scores):
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def test_z_normalize_preserves_order_and_standardizes():
    z = z_normalize_scores({"a": 3.0, "b": 2.0, "c": 1.0})
    assert _ranking(z) == ["a", "b", "c"]
    vals = np.array(list(z.values()))
    assert vals.mean() == pytest.approx(0.0, abs=1e-9)
    assert vals.std() == pytest.approx(1.0, abs=1e-9)
    # constant scores -> all zero, no division blowup
    assert set(z_normalize_scores({"a": 0.5, "b": 0.5}).values()) == {0.0}


DOC_Q = {"d1": 0.9, "d2": 0.8, "d3": 0.7}
PAR_Q = {"d3": 0.95, "d2": 0.5, "d4": 0.4}


def test_fuse_alpha_one_reproduces_doc_ranking():
    fused = fuse_query(DOC_Q, PAR_Q, alpha=1.0)
    # d4 only exists on the paragraph side -> ranks below every doc-side entry
    assert _ranking(fused) == ["d1", "d2", "d3", "d4"]


def test_fuse_alpha_zero_reproduces_par_ranking():
    fused = fuse_query(DOC_Q, PAR_Q, alpha=0.0)
    assert _ranking(fused) == ["d3", "d2", "d4", "d1"]


def test_fuse_blends_between_extremes():
    fused = fuse_query(DOC_Q, PAR_Q, alpha=0.5)
    assert set(fused) == {"d1", "d2", "d3", "d4"}
    # d3 is strong on both sides -> must beat d4 (weak) and d1 (missing on par)
    assert fused["d3"] > fused["d4"]
    assert fused["d3"] > fused["d1"]


def test_fuse_runs_per_query_alpha():
    doc_run = {"q1": DOC_Q, "q2": DOC_Q}
    par_run = {"q1": PAR_Q, "q2": PAR_Q}
    fused = fuse_runs(doc_run, par_run, {"q1": 1.0, "q2": 0.0})
    assert _ranking(fused["q1"])[0] == "d1"
    assert _ranking(fused["q2"])[0] == "d3"


# --------------------------------------------------------------------------- #
# query-time features
# --------------------------------------------------------------------------- #

def test_query_features_keys_and_values():
    doc_ranked = [("d1", 0.9), ("d2", 0.5), ("d3", 0.4)]
    par_ranked = [("d2", 0.8), ("d1", 0.7), ("d4", 0.2)]
    seg_sims = {
        "d1": np.array([0.9, 0.2, 0.1]),   # peaked -> high mmm
        "d2": np.array([0.5, 0.5]),        # flat -> mmm 0
        "d3": np.array([0.4]),             # single segment
    }
    f = query_features("what is photosynthesis", doc_ranked, par_ranked, seg_sims, feature_top_k=3)
    assert set(f) == set(FEATURE_ORDER)
    assert f["top1_mmm"] == pytest.approx(0.9 - np.mean([0.9, 0.2, 0.1]))
    assert f["frac_multiseg"] == pytest.approx(2 / 3)
    assert f["overlap10"] == pytest.approx(2 / 10)  # d1, d2 shared
    assert f["qlen"] == 3.0
    assert f["doc_margin"] > 0.0


def test_query_features_single_segment_candidates_are_zero_mmm():
    f = query_features(
        "q", [("d1", 1.0)], [("d1", 1.0)], {"d1": np.array([0.7])}, feature_top_k=1
    )
    assert f["top1_mmm"] == 0.0
    assert f["frac_multiseg"] == 0.0


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #

def _feature_row(mmm):
    row = {k: 0.0 for k in FEATURE_ORDER}
    row["top1_mmm"] = mmm
    row["mean_mmm"] = mmm
    return row


def test_router_learns_feature_driven_delta():
    rng = np.random.default_rng(0)
    mmms = rng.uniform(0.0, 0.1, size=200)
    rows = [_feature_row(m) for m in mmms]
    # ground truth: high locality -> paragraph wins (positive delta)
    deltas = [(m - 0.05) * 2 + rng.normal(scale=0.005) for m in mmms]
    router = GranularityRouter().fit(rows, deltas)

    hard = router.hard_alphas([_feature_row(0.09), _feature_row(0.01)])
    assert hard[0] == 0.0  # high mmm -> paragraph (doc weight 0)
    assert hard[1] == 1.0  # low mmm -> document

    soft = router.soft_alphas([_feature_row(0.09), _feature_row(0.01)], tau=0.02)
    assert soft[0] < 0.5 < soft[1]
    assert np.all((soft >= 0) & (soft <= 1))


def test_router_requires_fit_and_valid_input():
    router = GranularityRouter()
    with pytest.raises(RuntimeError):
        router.predict_delta([_feature_row(0.1)])
    with pytest.raises(ValueError):
        router.fit([], [])


# --------------------------------------------------------------------------- #
# global alpha + oracle
# --------------------------------------------------------------------------- #

def test_tune_global_alpha_picks_winning_side():
    from ir_measures import RR

    # d_rel is relevant; doc side ranks it first, par side buries it (3 docs
    # per side so z-scores are asymmetric and alpha=0.5 cannot tie).
    doc_q = {"d_rel": 0.9, "d_x": 0.5, "d_y": 0.4}
    par_q = {"d_x": 0.9, "d_y": 0.8, "d_rel": 0.1}
    doc_run = {"q1": dict(doc_q), "q2": dict(doc_q)}
    par_run = {"q1": dict(par_q), "q2": dict(par_q)}
    qrels = {"q1": {"d_rel": 1}, "q2": {"d_rel": 1}}
    alpha, score = tune_global_alpha(doc_run, par_run, qrels, ["q1", "q2"], RR @ 10)
    assert alpha > 0.5  # must lean document
    assert score == pytest.approx(1.0)


def test_oracle_takes_per_query_max():
    doc_pq = {"q1": 1.0, "q2": 0.2}
    par_pq = {"q1": 0.5, "q2": 0.8, "q3": 0.3}
    oracle = oracle_per_query(doc_pq, par_pq, ["q1", "q2", "q3"])
    assert oracle == {"q1": 1.0, "q2": 0.8, "q3": 0.3}


# --------------------------------------------------------------------------- #
# retrieval integration: doc_segment_sims
# --------------------------------------------------------------------------- #

def test_doc_segment_sims_matches_manual_computation():
    passages = [
        {"id": "d1", "text": "Volcanoes erupt when magma rises. Lava flows cover the land with ash."},
        {"id": "d2", "text": "The stock market fell today. Investors worried about interest rates."},
    ]
    enc = build_encoder(EncoderConfig(r=64))
    seg = lambda t: [s.strip() for s in t.split(".") if s.strip()]
    ret = ParagraphLevelRetriever(enc, seg, aggregation="max").index(passages)

    sims = ret.doc_segment_sims("volcano magma lava", ["d1", "d2"])
    assert set(sims) == {"d1", "d2"}
    assert sims["d1"].shape == (2,)
    # manual recomputation must match exactly
    q = ret.encoder.encode(["volcano magma lava"], is_query=True)[0]
    manual = ret.encoder.encode(seg(passages[0]["text"])) @ q
    assert np.allclose(sims["d1"], manual)
