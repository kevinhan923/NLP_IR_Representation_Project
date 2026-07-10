"""adaptive.py — Adaptive granularity retrieval (M3, Contribution 3).

M2 showed the granularity delta is predictable per query from the shape of
query-vs-segment similarity distributions. This module exploits that: choose
(or blend) document-level and paragraph-level retrieval PER QUERY, using only
features computable at query time — no relevance judgments at inference.

Systems, from weakest claim to strongest:
  * global-alpha fusion  — one fusion weight tuned on train queries. If this
    already beats both fixed granularities, fusion helps but is not "adaptive".
  * router (this module) — a ridge regression predicts the per-query delta
    from query-time features; the prediction drives a hard route (pick a side)
    or a soft per-query alpha. Must beat global-alpha to justify "adaptive".
  * oracle              — per-query best side by the actual metric; the upper
    bound on what any router could achieve.

Score fusion uses per-query z-normalized scores (document and paragraph
cosines live on different effective scales); a document missing from one
side's top-k gets that side's observed minimum minus a fixed penalty, so
absence ranks below every observed score. alpha is the DOCUMENT-side weight:
alpha=1 reproduces the document ranking, alpha=0 the paragraph ranking.

Features (per query, all query-time computable):
  top1_mmm / mean_mmm — max-minus-mean of segment sims for the top document-
                        level candidate / averaged over top-K candidates (the
                        M2-validated locality signal, applied to candidates).
  frac_multiseg       — fraction of top-K candidates that actually split.
  mean_log_nseg       — mean log(1 + n_segments) over top-K candidates.
  doc_margin/par_margin — top-1 vs top-2 score gap, scaled by the top-K score
                        std (how decisive each side is about its winner).
  overlap10           — |top-10 doc ∩ top-10 para| / 10 (side agreement).
  qlen                — query length in words.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Run = Dict[str, Dict[str, float]]

FEATURE_ORDER = [
    "top1_mmm",
    "mean_mmm",
    "frac_multiseg",
    "mean_log_nseg",
    "doc_margin",
    "par_margin",
    "overlap10",
    "qlen",
]

_EPS = 1e-9
_MISSING_PENALTY = 1.0  # in z-score units, applied below the observed minimum


# --------------------------------------------------------------------------- #
# score fusion
# --------------------------------------------------------------------------- #

def z_normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Per-query z-normalization; constant score lists map to all-zero."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    std = float(vals.std())
    if std < _EPS:
        return {d: 0.0 for d in scores}
    mean = float(vals.mean())
    return {d: (s - mean) / std for d, s in scores.items()}


def fuse_query(
    doc_scores: Dict[str, float],
    par_scores: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    """Fuse one query's two candidate lists; alpha is the document weight."""
    zd = z_normalize_scores(doc_scores)
    zp = z_normalize_scores(par_scores)
    floor_d = (min(zd.values()) if zd else 0.0) - _MISSING_PENALTY
    floor_p = (min(zp.values()) if zp else 0.0) - _MISSING_PENALTY
    return {
        d: alpha * zd.get(d, floor_d) + (1.0 - alpha) * zp.get(d, floor_p)
        for d in set(zd) | set(zp)
    }


def fuse_runs(doc_run: Run, par_run: Run, alpha_by_qid: Dict[str, float]) -> Run:
    """Fuse two runs with a per-query alpha (use a constant dict for global)."""
    return {
        qid: fuse_query(doc_run.get(qid, {}), par_run.get(qid, {}), alpha_by_qid[qid])
        for qid in alpha_by_qid
        if qid in doc_run or qid in par_run
    }


# --------------------------------------------------------------------------- #
# query-time features
# --------------------------------------------------------------------------- #

def _margin(ranked: Sequence[Tuple[str, float]]) -> float:
    """Top-1 vs top-2 gap scaled by the candidate-score std (decisiveness)."""
    if len(ranked) < 2:
        return 0.0
    scores = np.array([s for _, s in ranked], dtype=np.float64)
    std = float(scores.std())
    if std < _EPS:
        return 0.0
    return float((scores[0] - scores[1]) / std)


def query_features(
    query: str,
    doc_ranked: Sequence[Tuple[str, float]],
    par_ranked: Sequence[Tuple[str, float]],
    seg_sims_by_doc: Dict[str, np.ndarray],
    feature_top_k: int = 10,
) -> Dict[str, float]:
    """Features for one query. seg_sims_by_doc must cover the top-K document-
    level candidates (from ParagraphLevelRetriever.doc_segment_sims)."""
    top_doc_ids = [d for d, _ in doc_ranked[:feature_top_k]]
    mmms: List[float] = []
    nsegs: List[float] = []
    for d in top_doc_ids:
        sims = np.asarray(seg_sims_by_doc[d], dtype=np.float64)
        mmms.append(float(sims.max() - sims.mean()) if sims.size > 1 else 0.0)
        nsegs.append(float(sims.size))

    doc_top10 = {d for d, _ in doc_ranked[:10]}
    par_top10 = {d for d, _ in par_ranked[:10]}

    return {
        "top1_mmm": mmms[0] if mmms else 0.0,
        "mean_mmm": float(np.mean(mmms)) if mmms else 0.0,
        "frac_multiseg": float(np.mean([n > 1 for n in nsegs])) if nsegs else 0.0,
        "mean_log_nseg": float(np.mean(np.log1p(nsegs))) if nsegs else 0.0,
        "doc_margin": _margin(doc_ranked),
        "par_margin": _margin(par_ranked),
        "overlap10": len(doc_top10 & par_top10) / 10.0,
        "qlen": float(len(query.split())),
    }


# --------------------------------------------------------------------------- #
# the router
# --------------------------------------------------------------------------- #

class GranularityRouter:
    """Ridge regression: query-time features -> predicted granularity delta.

    Positive predicted delta = paragraph-level expected to win. Hard routing
    picks a side per query; soft routing maps the prediction to a per-query
    document weight alpha via a logistic squash.
    """

    def __init__(self, ridge_alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        self._fitted = False

    @staticmethod
    def _matrix(rows: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.array([[r[k] for k in FEATURE_ORDER] for r in rows], dtype=np.float64)

    def fit(self, rows: Sequence[Dict[str, float]], deltas: Sequence[float]) -> "GranularityRouter":
        if len(rows) != len(deltas) or not rows:
            raise ValueError("rows and deltas must be equal-length and non-empty")
        self.model.fit(self._matrix(rows), np.asarray(deltas, dtype=np.float64))
        self._fitted = True
        return self

    def predict_delta(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_delta().")
        return self.model.predict(self._matrix(rows))

    def hard_alphas(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        """Per-query document weight: 0 when paragraph predicted to win, else 1."""
        return (self.predict_delta(rows) <= 0.0).astype(np.float64)

    def soft_alphas(self, rows: Sequence[Dict[str, float]], tau: float = 0.05) -> np.ndarray:
        """Logistic squash of the predicted PARAGRAPH advantage into a document
        weight: strongly positive prediction -> alpha near 0 (trust paragraph)."""
        pred = self.predict_delta(rows)
        return 1.0 / (1.0 + np.exp(pred / max(tau, _EPS)))


def tune_global_alpha(
    doc_run: Run,
    par_run: Run,
    qrels: Dict[str, Dict[str, int]],
    train_qids: Sequence[str],
    measure,
    grid: Sequence[float] = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 2)),
) -> Tuple[float, float]:
    """Grid-search one fusion weight on the train queries; returns (alpha, score)."""
    from .metrics import per_query

    best_alpha, best_score = 1.0, -1.0
    train_set = set(train_qids)
    doc_sub = {q: r for q, r in doc_run.items() if q in train_set}
    par_sub = {q: r for q, r in par_run.items() if q in train_set}
    for alpha in grid:
        fused = fuse_runs(doc_sub, par_sub, {q: float(alpha) for q in train_set})
        pq = per_query(fused, qrels, measure)
        score = float(np.mean(list(pq.values()))) if pq else 0.0
        if score > best_score + 1e-12:
            best_alpha, best_score = float(alpha), score
    return best_alpha, best_score


def oracle_per_query(
    doc_pq: Dict[str, float], par_pq: Dict[str, float], qids: Sequence[str]
) -> Dict[str, float]:
    """Upper bound: the better side's metric value for every query."""
    return {
        q: max(doc_pq.get(q, 0.0), par_pq.get(q, 0.0))
        for q in qids
        if q in doc_pq or q in par_pq
    }
