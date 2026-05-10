import numpy as np
from typing import List, Dict


def mrr_at_k(ranked_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: List[str], qrels: Dict[str, int], k: int = 100) -> float:
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def ndcg_at_k(ranked_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    dcg = sum(
        qrels.get(doc_id, 0) / np.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
