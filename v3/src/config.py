"""config.py — Load and validate YAML experiment configs into dataclasses.

Config-driven experiments keep runs reproducible and self-documenting. All
fields have defaults so a minimal YAML still works; invalid values fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class EncoderConfig:
    name: str = "tfidf_projection"
    r: int = 192
    max_features: int = 50000
    seed: int = 42
    model: str = ""        # for name == "dense", e.g. "intfloat/e5-base-v2"
    batch_size: int = 64
    # TF-IDF vectorizer knobs (tfidf_projection only). Defaults = the simplified
    # controlled vectorizer; set ngram_max=2, min_df=2, sublinear_tf=true and a
    # fit_cap to match the two-stage paragraph encoder of v1.
    ngram_max: int = 1
    min_df: int = 1
    sublinear_tf: bool = False
    fit_cap: int = 0       # 0 = fit on every text; N = fit on the first N texts

    def __post_init__(self) -> None:
        if self.name == "dense" and not self.model:
            raise ValueError("dense encoder requires 'model'")
        if self.ngram_max < 1 or self.min_df < 1 or self.fit_cap < 0:
            raise ValueError("encoder.ngram_max/min_df must be >= 1 and fit_cap >= 0")


@dataclass(frozen=True)
class SegmentationConfig:
    mode: str = "sentence_window"  # natural | sentence_window | token_chunk
    window: int = 2
    stride: int = 1
    token_size: int = 128
    min_words: int = 5


@dataclass(frozen=True)
class AdaptiveConfig:
    """M3 settings: router features, query split, and model hyperparameters."""
    feature_top_k: int = 10    # doc-level candidates used for router features
    test_ratio: float = 0.5    # per-dataset query holdout for evaluation
    split_seed: int = 7        # shuffle seed for the train/test query split
    ridge_alpha: float = 1.0   # router regularization
    soft_tau: float = 0.05     # logistic temperature for soft per-query alpha

    def __post_init__(self) -> None:
        if not (0.0 < self.test_ratio < 1.0):
            raise ValueError("adaptive.test_ratio must be in (0, 1)")
        if self.feature_top_k <= 0:
            raise ValueError("adaptive.feature_top_k must be > 0")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    datasets: List[str]
    n_docs: int
    n_queries: int
    seed: int
    split: str
    top_k: int
    aggregations: List[str]
    agg_k: int
    encoder: EncoderConfig
    segmentation: SegmentationConfig
    data_dir: Optional[str] = None  # None -> default v1/data/beir
    # Optional INDEPENDENT encoder for the M2 locality metric. When set,
    # locality features are additionally computed with this encoder so the
    # locality-vs-delta correlation cannot be dismissed as mechanical reuse
    # of the same similarity scores that produced the delta.
    locality_encoder: Optional[EncoderConfig] = None
    # Optional M3 block; defaults apply when the YAML has an empty `adaptive:`.
    adaptive: Optional[AdaptiveConfig] = None

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("config.datasets must be non-empty")
        if not self.aggregations:
            raise ValueError("config.aggregations must be non-empty")
        for n, v in (("n_docs", self.n_docs), ("n_queries", self.n_queries),
                     ("top_k", self.top_k), ("agg_k", self.agg_k)):
            if v <= 0:
                raise ValueError(f"config.{n} must be > 0, got {v}")
        if self.encoder.r <= 0:
            raise ValueError("encoder.r must be > 0")


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML file into a validated ExperimentConfig."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    enc = EncoderConfig(**(raw.pop("encoder", {}) or {}))
    seg = SegmentationConfig(**(raw.pop("segmentation", {}) or {}))
    loc_raw = raw.pop("locality_encoder", None)
    loc = EncoderConfig(**loc_raw) if loc_raw else None
    has_adaptive = "adaptive" in raw
    ada_raw = raw.pop("adaptive", None)
    ada = AdaptiveConfig(**(ada_raw or {})) if has_adaptive else None
    return ExperimentConfig(
        encoder=enc, segmentation=seg, locality_encoder=loc, adaptive=ada, **raw
    )
