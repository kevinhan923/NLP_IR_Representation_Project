"""Vectorizer options that let a controlled run match the two-stage encoder."""

import numpy as np
import pytest

from src.config import EncoderConfig
from src.encoders import TfidfProjectionEncoder, build_encoder


def test_defaults_are_the_simplified_vectorizer():
    enc = TfidfProjectionEncoder(r=4)
    assert enc.vectorizer.ngram_range == (1, 1)
    assert enc.vectorizer.min_df == 1
    assert enc.vectorizer.sublinear_tf is False
    assert enc.fit_cap == 0


def test_matched_options_reach_the_vectorizer_and_fit_cap_binds():
    enc = TfidfProjectionEncoder(r=4, ngram_max=2, min_df=1, sublinear_tf=True, fit_cap=2)
    texts = ["alpha beta gamma", "alpha beta delta", "zeta zeta zeta"]
    enc.fit(texts)
    vocab = set(enc.vectorizer.vocabulary_)
    assert enc.vectorizer.ngram_range == (1, 2)
    assert enc.vectorizer.sublinear_tf is True
    assert "alpha beta" in vocab           # a bigram was learned
    assert "zeta" not in vocab             # third text is beyond the cap
    vecs = enc.encode(texts)
    assert vecs.shape == (3, 4)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms[:2], 1.0)     # L2-normalized
    assert norms[2] == 0.0                 # no known terms -> zero row, no NaN


def test_build_encoder_passes_the_options_through():
    cfg = EncoderConfig(name="tfidf_projection", r=8, ngram_max=2, min_df=2,
                        sublinear_tf=True, fit_cap=15000)
    enc = build_encoder(cfg)
    assert enc.vectorizer.ngram_range == (1, 2)
    assert enc.vectorizer.min_df == 2
    assert enc.vectorizer.sublinear_tf is True
    assert enc.fit_cap == 15000


def test_invalid_options_fail_fast():
    with pytest.raises(ValueError):
        EncoderConfig(ngram_max=0)
    with pytest.raises(ValueError):
        EncoderConfig(fit_cap=-1)
