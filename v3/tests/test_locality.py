# Run from v3/: cd v3 && python -m pytest tests/test_locality.py -v
"""Tests for the evidence-locality metric (M2, Contribution 2).

The metric must capture the *shape* of the query-vs-segment similarity
distribution: one segment standing out (local evidence) scores high; a flat
distribution (holistic relevance, every segment looks alike) scores low.
"""

import numpy as np
import pytest

from src.locality import (
    FEATURE_KEYS,
    binned_means,
    correlate,
    locality_features,
    query_locality,
)


# --------------------------------------------------------------------------- #
# locality_features — shape statistics on one similarity vector
# --------------------------------------------------------------------------- #

def test_concentrated_scores_high_uniform_scores_low():
    concentrated = locality_features(np.array([0.9, 0.1, 0.1, 0.1]))
    uniform = locality_features(np.array([0.5, 0.5, 0.5, 0.5]))

    assert concentrated["entropy_locality"] > uniform["entropy_locality"]
    assert concentrated["peakedness"] > uniform["peakedness"]
    assert concentrated["top1_share"] > uniform["top1_share"]
    # uniform distribution is the defined floor
    assert uniform["entropy_locality"] == pytest.approx(0.0, abs=1e-9)
    assert uniform["peakedness"] == pytest.approx(0.0)
    assert uniform["gini"] == pytest.approx(0.0)
    assert uniform["max_margin"] == pytest.approx(0.0)


def test_ordering_more_concentrated_is_higher():
    """Monotone: sharper peak -> higher locality (entropy + top1_share)."""
    sharp = locality_features(np.array([0.95, 0.10, 0.10, 0.10, 0.10]))
    mild = locality_features(np.array([0.60, 0.40, 0.35, 0.30, 0.25]))
    flat = locality_features(np.array([0.42, 0.41, 0.40, 0.39, 0.38]))
    for key in ("entropy_locality", "top1_share"):
        assert sharp[key] > mild[key] > flat[key]


def test_scale_and_shift_invariance_of_z_based_features():
    """Dense vs TF-IDF cosines live on different scales; z-score-based
    features must not change under affine rescaling s -> a*s + b (a>0)."""
    s = np.array([0.82, 0.71, 0.70, 0.66])
    t = 0.05 * s + 0.3
    a, b = locality_features(s), locality_features(t)
    for key in ("entropy_locality", "top1_share", "peakedness"):
        assert a[key] == pytest.approx(b[key], abs=1e-9)


def test_single_segment_conventions():
    """A 1-segment doc cannot be 'cut'; features must be defined, flagged."""
    f = locality_features(np.array([0.7]))
    assert f["n_segments"] == 1
    assert f["entropy_locality"] == pytest.approx(1.0)
    assert f["top1_share"] == pytest.approx(1.0)
    assert f["peakedness"] == pytest.approx(0.0)
    assert f["max_margin"] == pytest.approx(0.0)
    assert f["gini"] == pytest.approx(0.0)


def test_raw_margin_features():
    f = locality_features(np.array([0.9, 0.6, 0.5]))
    assert f["max_margin"] == pytest.approx(0.3)
    assert f["max_minus_mean"] == pytest.approx(0.9 - (0.9 + 0.6 + 0.5) / 3)


def test_all_feature_keys_present_and_finite():
    for sims in (np.array([0.9, 0.1]), np.array([0.5]), np.array([0.3, 0.3, 0.3])):
        f = locality_features(sims)
        assert set(FEATURE_KEYS) <= set(f)
        assert all(np.isfinite(f[k]) for k in FEATURE_KEYS)


def test_empty_sims_raises():
    with pytest.raises(ValueError):
        locality_features(np.array([]))


# --------------------------------------------------------------------------- #
# correlate / binned_means — the validation tooling
# --------------------------------------------------------------------------- #

def test_correlate_recovers_positive_relationship():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 2.0 * x + rng.normal(scale=0.5, size=200)
    out = correlate(x.tolist(), y.tolist())
    assert out["n"] == 200
    assert out["pearson_r"] > 0.9 and out["pearson_p"] < 0.001
    assert out["spearman_rho"] > 0.9 and out["spearman_p"] < 0.001


def test_correlate_degenerate_inputs_are_graceful():
    # constant x -> undefined correlation, must not crash
    out = correlate([1.0, 1.0, 1.0, 1.0], [0.1, 0.2, 0.3, 0.4])
    assert out["n"] == 4
    assert out["pearson_r"] is None
    # too few points
    out2 = correlate([1.0], [2.0])
    assert out2["pearson_r"] is None


def test_correlate_drops_nonfinite_pairs():
    out = correlate([1.0, 2.0, 3.0, float("nan")], [1.0, 2.0, 3.0, 4.0])
    assert out["n"] == 3


def test_binned_means_monotone_signal():
    x = list(np.linspace(0, 1, 100))
    y = [2 * v for v in x]
    bins = binned_means(x, y, n_bins=4)
    assert len(bins) == 4
    means = [b["mean_y"] for b in bins]
    assert means == sorted(means)
    assert sum(b["count"] for b in bins) == 100


# --------------------------------------------------------------------------- #
# query_locality — integration with an encoder + segmenter
# --------------------------------------------------------------------------- #

class _FakeBowEncoder:
    """Deterministic bag-of-words encoder over a tiny vocab (L2-normalized)."""

    _VOCAB = ["volcano", "magma", "lava", "cat", "sky", "dog", "tree", "rock"]

    def fit(self, texts):
        return self

    def encode(self, texts, is_query: bool = False):
        out = np.zeros((len(texts), len(self._VOCAB)), dtype=np.float32)
        for i, t in enumerate(texts):
            words = t.lower().split()
            for j, v in enumerate(self._VOCAB):
                out[i, j] = float(words.count(v))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def _sentence_segmenter(text):
    return [s.strip() for s in text.split(".") if s.strip()]


def test_query_locality_orders_local_vs_spread_evidence():
    enc = _FakeBowEncoder()
    query = "volcano magma lava"
    # evidence concentrated in ONE sentence
    local_doc = "volcano magma lava rock. cat cat sky. dog tree sky."
    # the same evidence smeared over EVERY sentence
    spread_doc = "volcano magma cat. lava volcano sky. magma lava tree."

    f_local = query_locality(query, [local_doc], enc, _sentence_segmenter)
    f_spread = query_locality(query, [spread_doc], enc, _sentence_segmenter)

    assert f_local["entropy_locality"] > f_spread["entropy_locality"]
    assert f_local["top1_share"] > f_spread["top1_share"]
    assert f_local["n_rel_docs"] == 1
    assert f_local["n_segments"] == pytest.approx(3)
    assert f_local["doc_words"] > 0


def test_query_locality_averages_over_relevant_docs():
    enc = _FakeBowEncoder()
    query = "volcano magma lava"
    d1 = "volcano magma lava rock. cat cat sky. dog tree sky."
    d2 = "volcano magma cat. lava volcano sky. magma lava tree."
    f1 = query_locality(query, [d1], enc, _sentence_segmenter)
    f2 = query_locality(query, [d2], enc, _sentence_segmenter)
    both = query_locality(query, [d1, d2], enc, _sentence_segmenter)
    assert both["n_rel_docs"] == 2
    assert both["entropy_locality"] == pytest.approx(
        (f1["entropy_locality"] + f2["entropy_locality"]) / 2
    )


def test_query_locality_n_segments_min_flags_unsplittable_docs():
    """A 1-segment rel doc must be detectable even when averaged with a
    3-segment one (mean n_segments = 2 would sneak past a mean-based filter)."""
    enc = _FakeBowEncoder()
    query = "volcano magma lava"
    multi = "volcano magma lava rock. cat cat sky. dog tree sky."  # 3 segments
    single = "volcano magma lava rock"                             # 1 segment
    both = query_locality(query, [multi, single], enc, _sentence_segmenter)
    assert both["n_segments"] == pytest.approx(2.0)      # mean
    assert both["n_segments_min"] == pytest.approx(1.0)  # min flags the leak


def test_query_locality_requires_relevant_docs():
    with pytest.raises(ValueError):
        query_locality("q", [], _FakeBowEncoder(), _sentence_segmenter)


def test_query_locality_from_sims_equivalent_to_encode_path():
    """The fast precomputed-sims path must reproduce query_locality exactly."""
    from src.locality import query_locality_from_sims

    enc = _FakeBowEncoder()
    query = "volcano magma lava"
    docs = [
        "volcano magma lava rock. cat cat sky. dog tree sky.",
        "volcano magma cat. lava volcano sky. magma lava tree.",
    ]
    slow = query_locality(query, docs, enc, _sentence_segmenter)

    q_vec = enc.encode([query], is_query=True)[0]
    sims = [enc.encode(_sentence_segmenter(t)) @ q_vec for t in docs]
    words = [float(len(t.split())) for t in docs]
    fast = query_locality_from_sims(sims, words)

    assert set(slow) == set(fast)
    for k in slow:
        assert slow[k] == pytest.approx(fast[k], abs=1e-9), k


def test_query_locality_from_sims_validation():
    from src.locality import query_locality_from_sims

    with pytest.raises(ValueError):
        query_locality_from_sims([], [])
    with pytest.raises(ValueError):
        query_locality_from_sims([np.array([0.5])], [10.0, 20.0])
