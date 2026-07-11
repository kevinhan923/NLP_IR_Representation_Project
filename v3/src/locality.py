"""locality.py — Evidence-locality metric (M2, Contribution 2).

M1 established *that* strong encoders reveal a dataset-dependent granularity
effect (fiqa/scidocs: paragraph-level significantly worse; scifact: no
difference) and *that* document length does not explain it. The working
hypothesis is "holistic vs local relevance": when relevance requires judging
the whole document, cutting it up loses context and max-aggregation is fooled
by locally-similar fragments.

This module turns that hypothesis into a measurable quantity. For a (query,
relevant document) pair we segment the document, score every segment against
the query with the SAME encoder the paragraph-level retriever uses, and
summarize the *shape* of the similarity distribution:

  * peaked  (one segment stands out)  -> evidence is LOCAL  -> high locality
  * flat    (all segments look alike) -> relevance is HOLISTIC -> low locality

Several candidate operationalizations are computed side by side (research
plan stage 2: "择优 + 消融"); the correlation analysis picks the winner by
predictive power against the per-query granularity delta:

  entropy_locality  1 - normalized entropy of softmax(z-scores). Scale- and
                    shift-invariant, in [0, 1].
  top1_share        max share of the softmax(z-score) mass. Scale-invariant.
  peakedness        (max - mean) / std of raw similarities. Scale-invariant.
  max_margin        raw gap between best and second-best segment.
  max_minus_mean    raw gap between best segment and the mean.
  gini              Gini coefficient of min-shifted similarities.

Structural covariates: n_segments, doc_words (and n_rel_docs at query level).

Degenerate-case conventions (flagged via n_segments for filtering):
  * single segment: trivially "concentrated" -> entropy_locality = 1,
    top1_share = 1, all margins/peakedness = 0.
  * all segments equal: perfectly flat -> entropy_locality = 0,
    top1_share = 1/n, peakedness = margins = gini = 0.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Feature keys produced by locality_features(), in reporting order.
FEATURE_KEYS = [
    "entropy_locality",
    "top1_share",
    "peakedness",
    "max_margin",
    "max_minus_mean",
    "gini",
    "n_segments",
]

_EPS = 1e-12


def _z_softmax(sims: np.ndarray) -> np.ndarray:
    """Softmax over z-scored similarities — an affine-invariant distribution.

    Raw cosine scales differ wildly across encoders (dense ~[0.6, 0.95],
    TF-IDF ~[0, 0.4]), so any shape statistic must not depend on scale/offset.
    All-equal input has no shape: return the uniform distribution.
    """
    std = float(sims.std())
    if std < _EPS:
        return np.full(sims.size, 1.0 / sims.size)
    z = (sims - sims.mean()) / std
    e = np.exp(z - z.max())  # subtract max for numerical stability
    return e / e.sum()


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of min-shifted values (0 = perfectly equal)."""
    v = np.sort(values - values.min())
    total = v.sum()
    if total < _EPS:
        return 0.0
    n = v.size
    # G = (2 * sum(i * v_i) / (n * sum(v))) - (n + 1) / n, with i = 1..n
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * total) - (n + 1.0) / n)


def locality_features(sims: np.ndarray) -> Dict[str, float]:
    """Shape features of one query-vs-segments similarity vector."""
    sims = np.asarray(sims, dtype=np.float64).ravel()
    if sims.size == 0:
        raise ValueError("locality_features requires at least one similarity")

    n = sims.size
    if n == 1:
        return {
            "entropy_locality": 1.0,
            "top1_share": 1.0,
            "peakedness": 0.0,
            "max_margin": 0.0,
            "max_minus_mean": 0.0,
            "gini": 0.0,
            "n_segments": 1.0,
        }

    p = _z_softmax(sims)
    entropy = float(-(p * np.log(p + _EPS)).sum())
    entropy_locality = 1.0 - entropy / np.log(n)

    s_desc = np.sort(sims)[::-1]
    std = float(sims.std())
    peakedness = float((s_desc[0] - sims.mean()) / std) if std >= _EPS else 0.0

    return {
        "entropy_locality": float(entropy_locality),
        "top1_share": float(p.max()),
        "peakedness": peakedness,
        "max_margin": float(s_desc[0] - s_desc[1]),
        "max_minus_mean": float(s_desc[0] - sims.mean()),
        "gini": _gini(sims),
        "n_segments": float(n),
    }


def query_locality(
    query: str,
    rel_doc_texts: Sequence[str],
    encoder,
    segmenter: Callable[[str], List[str]],
) -> Dict[str, float]:
    """Locality of a query's relevant documents, averaged over those docs.

    Uses the same encoder/segmenter as the paragraph-level retriever so the
    metric measures exactly the signal that system sees. The encoder must
    already be fit (dense encoders are pretrained; TF-IDF should be the
    segment-fit instance from ParagraphLevelRetriever).
    """
    if not rel_doc_texts:
        raise ValueError("query_locality requires at least one relevant doc")

    q_vec = encoder.encode([query], is_query=True)[0]
    sims_list: List[np.ndarray] = []
    words: List[float] = []
    for text in rel_doc_texts:
        segs = segmenter(text) or [text]
        sims_list.append(encoder.encode(segs) @ q_vec)
        words.append(float(len(text.split())))
    return query_locality_from_sims(sims_list, words)


def query_locality_from_sims(
    sims_per_doc: Sequence[np.ndarray],
    doc_words: Sequence[float],
) -> Dict[str, float]:
    """Same aggregation as query_locality, from PRECOMPUTED per-doc sims.

    This is the fast path for experiment scripts: segment embeddings are
    encoded once per collection (they already exist inside the paragraph
    retriever), so per-query locality is one matmul + slicing instead of one
    encoder call per relevant document (which made dense-qrels datasets like
    trec-covid pathologically slow).
    """
    if not sims_per_doc:
        raise ValueError("query_locality_from_sims requires at least one doc")
    if len(sims_per_doc) != len(doc_words):
        raise ValueError("sims_per_doc and doc_words must be equal-length")

    per_doc: List[Dict[str, float]] = []
    for sims, words in zip(sims_per_doc, doc_words):
        feats = locality_features(np.asarray(sims))
        feats["doc_words"] = float(words)
        per_doc.append(feats)

    keys = list(per_doc[0].keys())
    out = {k: float(np.mean([f[k] for f in per_doc])) for k in keys}
    # Min over rel docs, so "every relevant doc really splits" is filterable —
    # the mean alone would let 1-segment docs leak their convention values
    # (entropy_locality = 1.0) into a supposedly multi-segment subset.
    out["n_segments_min"] = float(min(f["n_segments"] for f in per_doc))
    out["n_rel_docs"] = float(len(per_doc))
    return out


def correlate(x: Sequence[float], y: Sequence[float]) -> Dict[str, Optional[float]]:
    """Pearson + Spearman correlation with p-values (None when undefined).

    Non-finite pairs are dropped; degenerate inputs (n < 3 or zero variance)
    return None statistics instead of crashing, so callers can report tables
    with holes rather than dying mid-experiment.
    """
    from scipy import stats

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    n = int(xa.size)

    out: Dict[str, Optional[float]] = {
        "n": n,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
    }
    if n < 3 or xa.std() < _EPS or ya.std() < _EPS:
        return out

    pr = stats.pearsonr(xa, ya)
    sr = stats.spearmanr(xa, ya)
    out.update(
        pearson_r=float(pr.statistic),
        pearson_p=float(pr.pvalue),
        spearman_rho=float(sr.statistic),
        spearman_p=float(sr.pvalue),
    )
    return out


def feature_correlations(rows: Sequence[Dict], delta_key: str, keys: Sequence[str]) -> Dict[str, Dict]:
    """{feature: correlation-dict} of each feature against the per-query delta."""
    deltas = [r[delta_key] for r in rows]
    return {k: correlate([r[k] for r in rows], deltas) for k in keys}


def demean_by_dataset(rows: Sequence[Dict], key: str) -> List[float]:
    """Subtract the per-dataset mean — the within-dataset (fixed-effects) view."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        sums[r["dataset"]] = sums.get(r["dataset"], 0.0) + r[key]
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    means = {ds: sums[ds] / counts[ds] for ds in sums}
    return [r[key] - means[r["dataset"]] for r in rows]


def within_correlations(rows: Sequence[Dict], delta_key: str, keys: Sequence[str]) -> Dict[str, Dict]:
    """Dataset-demeaned correlations: immune to cross-dataset composition
    (a pooled correlation could otherwise be pure Simpson's paradox)."""
    deltas = demean_by_dataset(rows, delta_key)
    return {k: correlate(demean_by_dataset(rows, k), deltas) for k in keys}


def binned_means(
    x: Sequence[float], y: Sequence[float], n_bins: int = 4
) -> List[Dict[str, float]]:
    """Mean of y per quantile bin of x — the '分箱收益曲线' from the plan.

    Returns one dict per bin: {x_low, x_high, mean_x, mean_y, count}.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size == 0:
        return []

    edges = np.quantile(xa, np.linspace(0, 1, n_bins + 1))
    bins: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin closed on the right so the max point is included
        sel = (xa >= lo) & (xa <= hi) if i == n_bins - 1 else (xa >= lo) & (xa < hi)
        if not sel.any():
            continue
        bins.append(
            {
                "x_low": float(lo),
                "x_high": float(hi),
                "mean_x": float(xa[sel].mean()),
                "mean_y": float(ya[sel].mean()),
                "count": int(sel.sum()),
            }
        )
    return bins
