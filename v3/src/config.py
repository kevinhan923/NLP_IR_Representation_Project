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

    def __post_init__(self) -> None:
        if self.name == "dense" and not self.model:
            raise ValueError("dense encoder requires 'model'")


@dataclass(frozen=True)
class SegmentationConfig:
    mode: str = "sentence_window"  # natural | sentence_window | token_chunk
    window: int = 2
    stride: int = 1
    token_size: int = 128
    min_words: int = 5


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
    return ExperimentConfig(encoder=enc, segmentation=seg, **raw)
