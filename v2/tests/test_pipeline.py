from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.pipeline import TwoStagePipeline
from src.stage1 import BM25Retriever
from src.rerank import BiEncoderReranker

# --- Metric tests ---

def test_mrr_at_k_first_relevant_at_rank1():
    assert mrr_at_k(["a", "b", "c"], {"a": 1}, k=10) == 1.0


def test_mrr_at_k_first_relevant_at_rank2():
    assert mrr_at_k(["b", "a", "c"], {"a": 1}, k=10) == 0.5


def test_mrr_at_k_no_relevant():
    assert mrr_at_k(["b", "c"], {"a": 1}, k=10) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a": 1, "d": 1}, k=3) == 0.5


def test_ndcg_at_k_perfect_ranking():
    # Perfect ranking: relevant doc at rank 1
    score = ndcg_at_k(["a", "b"], {"a": 1}, k=10)
    assert abs(score - 1.0) < 1e-9


def test_ndcg_at_k_no_relevant():
    assert ndcg_at_k(["b", "c"], {"a": 1}, k=10) == 0.0


# --- Pipeline smoke test ---

PASSAGES = [
    {"id": "p1", "text": "The Eiffel Tower is in Paris."},
    {"id": "p2", "text": "Python is a programming language used in data science."},
    {"id": "p3", "text": "Paris is known for its cuisine and culture."},
]
QUERIES = [{"id": "q1", "text": "Paris France"}]
QRELS = {"q1": {"p1": 1, "p3": 1}}


def test_pipeline_runs_and_returns_metrics():
    retriever = BM25Retriever()
    reranker = BiEncoderReranker(model_name="all-MiniLM-L6-v2")
    pipeline = TwoStagePipeline(retriever=retriever, reranker=reranker, top_k_stage1=3, top_k_stage2=3)
    pipeline.index(PASSAGES)
    results = pipeline.evaluate(QUERIES, QRELS)
    assert "single_mrr10" in results
    assert "para_mrr10" in results
    assert isinstance(results["single_mrr10"], float)
