"""metrics.py — Evaluation via ir_measures + paired significance testing.

Retrieval metrics go through ir_measures (BEIR/TREC-standard), never hand-rolled.
Significance uses a paired sign-flip randomization test plus a bootstrap CI on the
per-query difference — required for any main-conference claim.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import ir_measures
from ir_measures import RR, nDCG, R

# Default reported measures.
MEASURES = [RR @ 10, nDCG @ 10, R @ 100]

Run = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]


def evaluate(run: Run, qrels: Qrels, measures: List = MEASURES) -> Dict[str, float]:
    """Aggregate metrics over all queries (e.g. {'RR@10': .., 'nDCG@10': ..})."""
    agg = ir_measures.calc_aggregate(measures, qrels, run)
    return {str(m): float(v) for m, v in agg.items()}


def per_query(run: Run, qrels: Qrels, measure=RR @ 10) -> Dict[str, float]:
    """Return {query_id: metric_value} for one measure."""
    return {m.query_id: float(m.value) for m in ir_measures.iter_calc([measure], qrels, run)}


def _aligned_diffs(a: Dict[str, float], b: Dict[str, float]) -> np.ndarray:
    """b - a over the shared query ids (sorted for reproducibility)."""
    qids = sorted(set(a) & set(b))
    return np.array([b[q] - a[q] for q in qids], dtype=np.float64)


def paired_randomization_test(
    a: Dict[str, float], b: Dict[str, float], n_iter: int = 10000, seed: int = 0
) -> Dict[str, float]:
    """Two-sided paired sign-flip test for mean(b - a) != 0.

    Returns observed mean difference, p-value, and n (paired queries).
    """
    diffs = _aligned_diffs(a, b)
    n = diffs.size
    if n == 0:
        return {"mean_diff": 0.0, "p_value": 1.0, "n": 0}
    obs = float(diffs.mean())
    if np.allclose(diffs, 0.0):
        return {"mean_diff": 0.0, "p_value": 1.0, "n": n}
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_iter, n))
    perm_means = (signs * diffs).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(obs) - 1e-12).mean())
    return {"mean_diff": obs, "p_value": p, "n": n}


def bootstrap_ci(
    a: Dict[str, float], b: Dict[str, float], n_iter: int = 10000, seed: int = 0, alpha: float = 0.05
) -> Dict[str, float]:
    """Bootstrap CI for mean(b - a)."""
    diffs = _aligned_diffs(a, b)
    if diffs.size == 0:
        return {"ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_iter, diffs.size))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci_low": float(lo), "ci_high": float(hi)}
