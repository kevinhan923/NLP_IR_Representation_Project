from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class AggregationSpec:
    name: str
    mode: str
    top_k: int | None = None
    temperature: float | None = None
    threshold: float | None = None


DEFAULT_AGGREGATIONS = [
    AggregationSpec(name="max", mode="max"),
    AggregationSpec(name="topk_mean_k3", mode="topk_mean", top_k=3),
    AggregationSpec(name="topk_mean_k5", mode="topk_mean", top_k=5),
    AggregationSpec(name="weighted_topk_k3", mode="weighted_topk", top_k=3),
    AggregationSpec(name="weighted_topk_k5", mode="weighted_topk", top_k=5),
    AggregationSpec(name="softmax_t1", mode="softmax_weighted", temperature=1.0),
    AggregationSpec(name="softmax_t05", mode="softmax_weighted", temperature=0.5),
    AggregationSpec(name="threshold_02", mode="threshold_mean", threshold=0.2),
    AggregationSpec(name="threshold_03", mode="threshold_mean", threshold=0.3),
]


def _sorted(scores: Iterable[float]) -> List[float]:
    return sorted(scores, reverse=True)


def aggregate_scores(scores: Iterable[float], spec: AggregationSpec) -> float:
    vals = _sorted(scores)
    if not vals:
        return float("-inf")
    if spec.mode == "max":
        return vals[0]
    if spec.mode == "topk_mean":
        k = min(spec.top_k or 1, len(vals))
        return sum(vals[:k]) / k
    if spec.mode == "weighted_topk":
        k = min(spec.top_k or 1, len(vals))
        top = vals[:k]
        weights = [k - i for i in range(k)]
        denom = sum(weights)
        return sum(w * s for w, s in zip(weights, top)) / denom
    if spec.mode == "softmax_weighted":
        t = spec.temperature or 1.0
        exps = [math.exp(v / max(t, 1e-8)) for v in vals]
        denom = sum(exps)
        return sum(e * v for e, v in zip(exps, vals)) / denom
    if spec.mode == "threshold_mean":
        threshold = spec.threshold if spec.threshold is not None else 0.0
        kept = [v for v in vals if v >= threshold]
        if not kept:
            kept = vals[:1]
        return sum(kept) / len(kept)
    raise ValueError(f"Unknown aggregation mode: {spec.mode}")
