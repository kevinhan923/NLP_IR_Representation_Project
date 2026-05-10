from typing import List, Dict
import numpy as np
from .metrics import mrr_at_k


def error_analysis(
    queries: List[Dict],
    single_rankings: Dict[str, List[str]],
    para_rankings: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    passages: List[Dict],
) -> Dict:
    """
    For each query, compute delta MRR@10 (para - single).
    Group by:
      - query_length_bucket: short (<5 tokens), medium (5-10), long (>10)
      - doc_length_bucket: by length of top-ranked document
    Returns dict with per-bucket mean deltas.
    """
    passage_len = {p["id"]: len(p["text"].split()) for p in passages}
    results = []
    for q in queries:
        qid = q["id"]
        q_qrels = qrels.get(qid, {})
        single_mrr = mrr_at_k(single_rankings.get(qid, []), q_qrels)
        para_mrr = mrr_at_k(para_rankings.get(qid, []), q_qrels)
        q_len = len(q["text"].split())
        top_doc = single_rankings.get(qid, [None])[0]
        doc_len = passage_len.get(top_doc, 0) if top_doc else 0
        results.append({
            "qid": qid,
            "delta": para_mrr - single_mrr,
            "q_len": q_len,
            "doc_len": doc_len,
        })

    if not results:
        return {
            "overall_delta": 0.0,
            "by_query_length": {"short": None, "medium": None, "long": None},
            "by_doc_length": {"short": None, "medium": None, "long": None},
            "n_queries_para_wins": 0,
            "n_queries_para_loses": 0,
        }

    def bucket_mean(key, bins, labels):
        groups = {lbl: [] for lbl in labels}
        for r in results:
            val = r[key]
            for i, (lo, hi) in enumerate(bins):
                if lo <= val < hi:
                    groups[labels[i]].append(r["delta"])
                    break
        return {lbl: float(np.mean(v)) if v else None for lbl, v in groups.items()}

    return {
        "overall_delta": float(np.mean([r["delta"] for r in results])),
        "by_query_length": bucket_mean(
            "q_len",
            bins=[(0, 5), (5, 10), (10, 9999)],
            labels=["short", "medium", "long"]
        ),
        "by_doc_length": bucket_mean(
            "doc_len",
            bins=[(0, 100), (100, 300), (300, 9999)],
            labels=["short", "medium", "long"]
        ),
        "n_queries_para_wins": sum(1 for r in results if r["delta"] > 0),
        "n_queries_para_loses": sum(1 for r in results if r["delta"] < 0),
    }
