from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from .metrics import reciprocal_rank
from .types import ComparisonRow, DocumentExample, QueryExample


def build_query_level_comparison(
    queries: List[QueryExample],
    qrels: Dict[str, Set[str]],
    single_run: Dict[str, List[str]],
    paragraph_run: Dict[str, List[str]],
    best_paragraphs: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    rows: List[ComparisonRow] = []
    for q in queries:
        rel = qrels.get(q.query_id, set())
        s_rr = reciprocal_rank(single_run.get(q.query_id, []), rel)
        p_rr = reciprocal_rank(paragraph_run.get(q.query_id, []), rel)
        meta = best_paragraphs.get(q.query_id, {})
        rows.append(
            ComparisonRow(
                query_id=q.query_id,
                query=q.text,
                single_rr=s_rr,
                paragraph_rr=p_rr,
                rr_gain=p_rr - s_rr,
                single_top_doc=(single_run.get(q.query_id, [""])[:1] or [""])[0],
                paragraph_top_doc=(paragraph_run.get(q.query_id, [""])[:1] or [""])[0],
                paragraph_top_doc_score=float(meta.get("top_doc_score", 0.0)),
                paragraph_top_doc_is_relevant=((meta.get("top_doc_id") in rel) if meta else False),
                best_para_id=int(meta.get("best_para_id", -1)),
                best_para_score=float(meta.get("best_para_score", 0.0)),
                best_para_text=str(meta.get("best_para_text", "")),
            )
        )
    better = [r for r in rows if r.rr_gain > 0]
    worse = [r for r in rows if r.rr_gain < 0]
    ties = [r for r in rows if r.rr_gain == 0]
    better.sort(key=lambda r: r.rr_gain, reverse=True)
    worse.sort(key=lambda r: r.rr_gain)
    return {
        "counts": {
            "paragraph_better": len(better),
            "single_better": len(worse),
            "tie": len(ties),
        },
        "best_paragraph_wins": [r.__dict__ for r in better[:10]],
        "largest_single_vector_wins": [r.__dict__ for r in worse[:10]],
    }


def grouped_analysis(
    queries: List[QueryExample],
    qrels: Dict[str, Set[str]],
    docs: Dict[str, DocumentExample],
    doc_segment_counts: Dict[str, int],
    single_run: Dict[str, List[str]],
    paragraph_run: Dict[str, List[str]],
) -> Dict[str, object]:
    lengths = []
    seg_counts = []
    query_to_rel_doc = {}
    for q in queries:
        rel = qrels.get(q.query_id, set())
        if not rel:
            continue
        first_rel = next(iter(rel))
        doc = docs.get(first_rel)
        if doc is None:
            continue
        query_to_rel_doc[q.query_id] = first_rel
        lengths.append(len(doc.text.split()))
        seg_counts.append(doc_segment_counts.get(first_rel, 1))

    def bucket(value: float, vals: List[float]) -> str:
        if not vals:
            return "mid"
        sorted_vals = sorted(vals)
        lo = sorted_vals[len(sorted_vals) // 3]
        hi = sorted_vals[(2 * len(sorted_vals)) // 3]
        if value <= lo:
            return "low"
        if value <= hi:
            return "mid"
        return "high"

    out = {
        "by_relevant_doc_length": defaultdict(lambda: {"query_count": 0.0, "single_mrr": 0.0, "paragraph_mrr": 0.0}),
        "by_relevant_doc_segment_count": defaultdict(lambda: {"query_count": 0.0, "single_mrr": 0.0, "paragraph_mrr": 0.0}),
    }
    for q in queries:
        doc_id = query_to_rel_doc.get(q.query_id)
        if not doc_id:
            continue
        rel = qrels[q.query_id]
        doc = docs[doc_id]
        s_rr = reciprocal_rank(single_run.get(q.query_id, []), rel)
        p_rr = reciprocal_rank(paragraph_run.get(q.query_id, []), rel)
        len_bucket = bucket(len(doc.text.split()), lengths)
        seg_bucket = bucket(doc_segment_counts.get(doc_id, 1), seg_counts)
        for key, b in [("by_relevant_doc_length", len_bucket), ("by_relevant_doc_segment_count", seg_bucket)]:
            out[key][b]["query_count"] += 1
            out[key][b]["single_mrr"] += s_rr
            out[key][b]["paragraph_mrr"] += p_rr

    final = {}
    for key, groups in out.items():
        final[key] = {}
        for name, stats in groups.items():
            count = stats["query_count"] or 1.0
            s = stats["single_mrr"] / count
            p = stats["paragraph_mrr"] / count
            final[key][name] = {
                "query_count": stats["query_count"],
                "single_mrr": s,
                "paragraph_mrr": p,
                "mrr_delta": p - s,
            }
    return final
