"""aggregation.py — Collapse per-segment scores into one document score.

Implements the five strategies discussed in the paper (Sec. 4.2 / Eq. 5):
max, topk_mean, weighted_topk, softmax_weighted, threshold_mean.
"""

from __future__ import annotations

import numpy as np


def aggregate(
    scores: np.ndarray,
    method: str = "topk_mean",
    k: int = 3,
    temperature: float = 1.0,
    threshold: float = 0.2,
) -> float:
    if scores.size == 0:
        return float("-inf")
    s = np.sort(scores)[::-1]  # descending

    if method == "max":
        return float(s[0])
    if method == "topk_mean":
        kk = min(k, s.size)
        return float(s[:kk].mean())
    if method == "weighted_topk":
        kk = min(k, s.size)
        top = s[:kk]
        weights = np.arange(kk, 0, -1, dtype=np.float32)  # k, k-1, ..., 1
        return float((weights * top).sum() / weights.sum())
    if method == "softmax_weighted":
        t = max(temperature, 1e-8)
        e = np.exp(s / t)
        return float((e * s).sum() / e.sum())
    if method == "threshold_mean":
        kept = s[s >= threshold]
        if kept.size == 0:
            kept = s[:1]
        return float(kept.mean())
    raise ValueError(f"Unknown aggregation method: {method}")
