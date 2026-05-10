from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set

from .types import QueryRun


def reciprocal_rank(doc_ids: Sequence[str], relevant: Set[str]) -> float:
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int = 10) -> float:
    vals = [reciprocal_rank(run[qid][:k], qrels.get(qid, set())) for qid in run]
    return sum(vals) / len(vals) if vals else 0.0


def recall_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int = 10) -> float:
    vals = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        vals.append(len(set(ranked[:k]) & rel) / len(rel))
    return sum(vals) / len(vals) if vals else 0.0


def precision_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int = 10) -> float:
    vals = []
    for qid, ranked in run.items():
        if not ranked:
            vals.append(0.0)
            continue
        rel = qrels.get(qid, set())
        vals.append(len(set(ranked[:k]) & rel) / k)
    return sum(vals) / len(vals) if vals else 0.0


def hit_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int = 10) -> float:
    vals = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, set())
        vals.append(1.0 if set(ranked[:k]) & rel else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def ndcg_at_k(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int = 10) -> float:
    vals = []
    for qid, ranked in run.items():
        rel = qrels.get(qid, set())
        dcg = 0.0
        for rank, doc_id in enumerate(ranked[:k], start=1):
            if doc_id in rel:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(rel), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def metric_bundle(run: Dict[str, List[str]], qrels: Dict[str, Set[str]], ks: Iterable[int] = (10,)) -> Dict[str, float]:
    out: Dict[str, float] = {"mrr@10": mrr_at_k(run, qrels, k=10)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(run, qrels, k=k)
        out[f"precision@{k}"] = precision_at_k(run, qrels, k=k)
        out[f"hit@{k}"] = hit_at_k(run, qrels, k=k)
        out[f"ndcg@{k}"] = ndcg_at_k(run, qrels, k=k)
    return out
