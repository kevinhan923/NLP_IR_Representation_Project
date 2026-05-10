from typing import List, Dict
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


class TwoStagePipeline:
    """
    Wires together: Stage1Retriever -> Reranker -> Aggregator -> Evaluation.
    Produces both single-vector (Stage 1 only) and paragraph-reranked rankings.
    """

    def __init__(self, retriever, reranker, top_k_stage1: int = 100, top_k_stage2: int = 10):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k_stage1 = top_k_stage1
        self.top_k_stage2 = top_k_stage2

    def index(self, passages: List[Dict]):
        self.retriever.index(passages)

    def evaluate(self, queries: List[Dict], qrels: Dict[str, Dict[str, int]]) -> Dict:
        if not queries:
            return {
                "single_mrr10": 0.0,
                "single_ndcg10": 0.0,
                "single_recall100": 0.0,
                "para_mrr10": 0.0,
                "para_ndcg10": 0.0,
                "para_recall100": 0.0,
                "delta_mrr10": 0.0,
                "single_rankings": {},
                "para_rankings": {},
            }

        single_mrrs, para_mrrs = [], []
        single_ndcgs, para_ndcgs = [], []
        single_recalls, para_recalls = [], []
        single_rankings, para_rankings = {}, {}

        for q in queries:
            qid = q["id"]
            q_qrels = qrels.get(qid, {})

            # Stage 1: BM25 or dense retrieval
            stage1_results = self.retriever.retrieve(q["text"], top_k=self.top_k_stage1)
            single_ids = [r["id"] for r in stage1_results]
            single_rankings[qid] = single_ids
            single_mrrs.append(mrr_at_k(single_ids, q_qrels))
            single_ndcgs.append(ndcg_at_k(single_ids, q_qrels))
            single_recalls.append(recall_at_k(single_ids, q_qrels, k=100))

            # Stage 2: paragraph reranking
            para_results = self.reranker.rerank(q["text"], stage1_results, top_k=self.top_k_stage2)
            para_ids = [r["id"] for r in para_results]
            para_rankings[qid] = para_ids
            para_mrrs.append(mrr_at_k(para_ids, q_qrels))
            para_ndcgs.append(ndcg_at_k(para_ids, q_qrels))
            para_recalls.append(recall_at_k(para_ids, q_qrels, k=100))

        n = len(queries)
        return {
            "single_mrr10": sum(single_mrrs) / n,
            "single_ndcg10": sum(single_ndcgs) / n,
            "single_recall100": sum(single_recalls) / n,
            "para_mrr10": sum(para_mrrs) / n,
            "para_ndcg10": sum(para_ndcgs) / n,
            "para_recall100": sum(para_recalls) / n,
            "delta_mrr10": sum(para_mrrs) / n - sum(single_mrrs) / n,
            "single_rankings": single_rankings,
            "para_rankings": para_rankings,
        }
