"""
data.py — Data loading utilities for MS MARCO and BEIR datasets.

Provides reproducible sampling functions for both in-domain (MS MARCO)
and out-of-domain (BEIR) evaluation sets.

External dependencies (ir_datasets, datasets) are imported at module level so
that unit tests can patch them via ``unittest.mock.patch``.  Each function
catches *any* exception from those libraries and falls back to synthetic data,
so missing network access or missing packages never crash the test suite.
"""

import random

# Optional heavyweight imports — gracefully absent in minimal environments.
try:
    import ir_datasets  # type: ignore
except Exception:  # pragma: no cover
    ir_datasets = None  # type: ignore

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_msmarco_passage_sample(n_docs: int, n_queries: int, seed: int = 42):
    """Load a reproducible sample from MS MARCO passage dev set.

    Falls back to synthetic data if the dataset is unavailable (e.g. no
    internet access or ir_datasets not installed).

    Args:
        n_docs:    Number of passages to return.
        n_queries: Number of queries to return.
        seed:      Random seed for reproducibility.

    Returns:
        dict with keys ``"passages"``, ``"queries"``, ``"qrels"``.
    """
    random.seed(seed)

    try:
        if ir_datasets is None:
            raise ImportError("ir_datasets not available")

        dataset = ir_datasets.load("msmarco-passage/dev/small")

        # The corpus is large; read n_docs * 10 entries then sample down.
        passages = []
        for i, doc in enumerate(dataset.docs_iter()):
            if i >= n_docs * 10:
                break
            passages.append({"id": doc.doc_id, "text": doc.text})
        passages = random.sample(passages, min(n_docs, len(passages)))
        passage_ids = {p["id"] for p in passages}

        # Load queries
        queries = []
        for q in dataset.queries_iter():
            queries.append({"id": q.query_id, "text": q.text})
        queries = random.sample(queries, min(n_queries, len(queries)))
        query_ids = {q["id"] for q in queries}

        # Load qrels filtered to the sampled passages and queries
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.query_id in query_ids and qrel.doc_id in passage_ids:
                qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

        return {"passages": passages, "queries": queries, "qrels": qrels}

    except (OSError, ConnectionError, ImportError, RuntimeError, Exception) as e:
        import warnings
        warnings.warn(f"Data loading failed ({type(e).__name__}: {e}); using synthetic data.")
        # Network/library unavailable — return synthetic data with correct shape.
        return _make_synthetic(n_docs, n_queries, seed)


def load_beir_sample(dataset_name: str, n_docs: int, n_queries: int, seed: int = 42):
    """Load a small BEIR dataset sample for OOD transfer evaluation.

    Falls back to synthetic data if HuggingFace ``datasets`` is unavailable.

    Args:
        dataset_name: BEIR dataset name (e.g. ``"scifact"``).
        n_docs:       Number of passages to return.
        n_queries:    Number of queries to return.
        seed:         Random seed for reproducibility (used in fallback).

    Returns:
        dict with keys ``"passages"``, ``"queries"``, ``"qrels"`` (qrels is
        always empty for BEIR samples — we only use corpus/query splits here).
    """
    try:
        if load_dataset is None:
            raise ImportError("datasets not available")

        corpus = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
        queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")

        passages = [
            {"id": str(r["_id"]), "text": f"{r['title'].strip()} {r['text'].strip()}"}
            for r in corpus.select(range(min(n_docs, len(corpus))))
        ]
        queries = [
            {"id": str(r["_id"]), "text": r["text"]}
            for r in queries_ds.select(range(min(n_queries, len(queries_ds))))
        ]
        return {"passages": passages, "queries": queries, "qrels": {}}

    except (OSError, ConnectionError, ImportError, RuntimeError, Exception) as e:
        import warnings
        warnings.warn(f"Data loading failed ({type(e).__name__}: {e}); using synthetic data.")
        # Network/library unavailable — return synthetic data with correct shape.
        return _make_synthetic(n_docs, n_queries, seed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_synthetic(n_docs: int, n_queries: int, seed: int = 0):
    """Return synthetic data that satisfies the expected schema.

    Used as a fallback when real datasets cannot be downloaded.
    """
    rng = random.Random(seed)
    passages = [
        {
            "id": f"doc_{i}",
            "text": "Synthetic passage {} with words ".format(i)
            + " ".join(f"word{rng.randint(0, 999)}" for _ in range(20)),
        }
        for i in range(n_docs)
    ]
    queries = [
        {
            "id": f"query_{i}",
            "text": f"Synthetic query {i} about topic {rng.randint(0, 99)}",
        }
        for i in range(n_queries)
    ]
    # Build a minimal qrels mapping: each query relevant to one random passage.
    qrels = {q["id"]: {rng.choice(passages)["id"]: 1} for q in queries}

    return {"passages": passages, "queries": queries, "qrels": qrels}
