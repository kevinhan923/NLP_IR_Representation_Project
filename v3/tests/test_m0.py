# Run from v3/: cd v3 && python -m pytest tests/ -v
"""M0 unit + smoke tests for the controlled retrieval pipeline."""

import numpy as np
import pytest

from src.aggregation import aggregate
from src.config import load_config, EncoderConfig, SegmentationConfig
from src.segmentation import build_segmenter
from src.encoders import build_encoder, l2_normalize
from src.retrieval import DocumentLevelRetriever, ParagraphLevelRetriever
from src.metrics import evaluate, per_query, paired_randomization_test, bootstrap_ci
from src.data import _DEFAULT_BEIR_DIR


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def test_aggregate_methods():
    s = np.array([0.1, 0.9, 0.5])
    assert aggregate(s, "max") == pytest.approx(0.9)
    assert aggregate(s, "topk_mean", k=2) == pytest.approx(0.7)
    assert aggregate(np.array([0.9, 0.5, 0.1]), "weighted_topk", k=2) == pytest.approx((2 * 0.9 + 0.5) / 3)
    assert aggregate(np.array([0.1, 0.9, 0.5, 0.2]), "threshold_mean", threshold=0.4) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        aggregate(s, "nope")


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #

def test_segmenter_modes_and_fallback():
    text = "First sentence here. Second sentence follows. Third one closes it."
    sw = build_segmenter(SegmentationConfig(mode="sentence_window", window=2, stride=1, min_words=2))
    assert len(sw(text)) >= 2
    tok = build_segmenter(SegmentationConfig(mode="token_chunk", token_size=3, min_words=1))
    assert len(tok(text)) >= 2
    # No paragraph breaks -> natural falls back to whole text (never empty).
    nat = build_segmenter(SegmentationConfig(mode="natural", min_words=5))
    assert nat(text) and all(seg.strip() for seg in nat(text))


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #

def test_l2_normalize_and_encoder():
    m = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.linalg.norm(m[0]) == pytest.approx(1.0)
    assert np.allclose(m[1], 0.0)
    enc = build_encoder(EncoderConfig(r=8)).fit(["alpha beta gamma", "delta epsilon zeta"])
    v = enc.encode(["alpha beta"])
    assert v.shape[0] == 1 and np.isfinite(v).all()


# --------------------------------------------------------------------------- #
# retrieval (small hand-built corpus)
# --------------------------------------------------------------------------- #

_PASSAGES = [
    {"id": "d1", "text": "Photosynthesis converts sunlight into chemical energy. Chlorophyll absorbs light."},
    {"id": "d2", "text": "The stock market fell today. Investors worried about interest rates."},
    {"id": "d3", "text": "Volcanoes erupt when magma rises. Lava flows cover the land with ash."},
]


def test_document_and_paragraph_retrieval_rank_and_scores():
    doc = DocumentLevelRetriever(build_encoder(EncoderConfig(r=64))).index(_PASSAGES)
    out = doc.retrieve("sunlight chlorophyll photosynthesis", top_k=3)
    assert out[0][0] == "d1"
    assert all(isinstance(s, float) for _, s in out)
    assert [s for _, s in out] == sorted([s for _, s in out], reverse=True)

    par = ParagraphLevelRetriever(
        build_encoder(EncoderConfig(r=64)),
        build_segmenter(SegmentationConfig(mode="sentence_window", window=1, stride=1, min_words=1)),
        aggregation="max",
    ).index(_PASSAGES)
    pout = par.retrieve("magma lava volcano eruption", top_k=3)
    assert pout[0][0] == "d3"
    assert {doc_id for doc_id, _ in pout} == {"d1", "d2", "d3"}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def test_metrics_and_significance():
    qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
    perfect = {"q1": {"d1": 2.0, "d2": 1.0}, "q2": {"d3": 2.0, "d1": 1.0}}
    m = evaluate(perfect, qrels)
    assert m["RR@10"] == pytest.approx(1.0)

    a = {"q1": 0.5, "q2": 0.5, "q3": 0.5}
    b = {"q1": 1.0, "q2": 1.0, "q3": 1.0}
    sig = paired_randomization_test(a, b, n_iter=2000, seed=0)
    assert sig["mean_diff"] == pytest.approx(0.5)
    assert 0.0 <= sig["p_value"] <= 1.0
    ci = bootstrap_ci(a, b, n_iter=2000, seed=0)
    assert ci["ci_low"] <= 0.5 <= ci["ci_high"]
    # identical inputs -> no difference, p == 1
    same = paired_randomization_test(a, a)
    assert same["p_value"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def test_load_config_and_validation(tmp_path):
    yaml_text = """
name: t
datasets: [scifact]
n_docs: 100
n_queries: 10
seed: 1
split: test
top_k: 50
aggregations: [max]
agg_k: 3
encoder: {name: tfidf_projection, r: 64}
segmentation: {mode: sentence_window, window: 2, stride: 1}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.datasets == ["scifact"] and cfg.encoder.r == 64

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml_text.replace("n_docs: 100", "n_docs: 0"))
    with pytest.raises(ValueError):
        load_config(bad)


# --------------------------------------------------------------------------- #
# smoke test on real local scifact (skips if data absent)
# --------------------------------------------------------------------------- #

import importlib.util

@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed (M1)",
)
def test_dense_encoder_interface_minilm():
    """DenseEncoder returns L2-normalized vectors and honors is_query (M1)."""
    from src.config import EncoderConfig
    enc = build_encoder(EncoderConfig(name="dense", model="sentence-transformers/all-MiniLM-L6-v2"))
    dv = enc.encode(["a small cat sat on the mat"], is_query=False)
    qv = enc.encode(["a small cat sat on the mat"], is_query=True)
    assert dv.shape[0] == 1 and qv.shape[0] == 1
    assert abs(float(np.linalg.norm(dv[0])) - 1.0) < 1e-3  # normalized


def test_dense_requires_model():
    from src.config import EncoderConfig
    with pytest.raises(ValueError):
        EncoderConfig(name="dense", model="")


@pytest.mark.skipif(
    not (_DEFAULT_BEIR_DIR / "scifact" / "corpus.jsonl").exists(),
    reason="local BEIR scifact data not present",
)
def test_smoke_real_scifact_experiment():
    from src.config import ExperimentConfig
    from src.experiment import run_experiment

    cfg = ExperimentConfig(
        name="smoke", datasets=["scifact"], n_docs=300, n_queries=30,
        seed=42, split="test", top_k=100, aggregations=["max"], agg_k=3,
        encoder=EncoderConfig(r=128), segmentation=SegmentationConfig(mode="sentence_window"),
    )
    out = run_experiment(cfg)
    r = out["results"]["scifact"]
    assert r["document"]["RR@10"] > 0.0
    assert "max" in r["paragraph_by_agg"]
    assert "p_value" in r["paragraph_by_agg"]["max"]["significance_rr10"]
