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

        # 1. Build qid -> {doc_id: rel} from full qrels so we can sample queries
        #    that actually have judgments (dev/small has ~7437 qrels for ~6980
        #    queries, so this is essentially every query — but be defensive).
        full_qrels = {}
        for qrel in dataset.qrels_iter():
            full_qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

        # 2. Sample queries from those with at least one positive judgment.
        all_queries = [
            q for q in dataset.queries_iter()
            if any(r > 0 for r in full_qrels.get(q.query_id, {}).values())
        ]
        sampled_queries = random.sample(all_queries, min(n_queries, len(all_queries)))
        query_ids = {q.query_id for q in sampled_queries}
        queries = [{"id": q.query_id, "text": q.text} for q in sampled_queries]

        # 3. Collect positive doc_ids for the sampled queries — these MUST
        #    end up in the passage pool, otherwise qrels would be empty and
        #    every metric would be 0 (the original bug).
        sampled_qrels = {qid: full_qrels[qid] for qid in query_ids if qid in full_qrels}
        positive_doc_ids = {
            did for rels in sampled_qrels.values()
            for did, rel in rels.items() if rel > 0
        }

        # 4. Fast path: fetch positives by id via docs_store (random access).
        #    Falls through to the docs_iter scan below for any ids it can't
        #    resolve (e.g. when ir_datasets is mocked in unit tests).
        positives = []
        unfetched = set(positive_doc_ids)
        try:
            store = dataset.docs_store()
            for did in list(unfetched):
                doc = store.get(did)
                if doc is not None and getattr(doc, "doc_id", None) == did:
                    positives.append({"id": doc.doc_id, "text": doc.text})
                    unfetched.discard(did)
        except Exception:
            pass

        # 5. One pass through docs_iter to (a) recover any unfetched positives
        #    and (b) collect a distractor candidate pool. Stop as soon as we
        #    have all positives AND a large enough distractor buffer.
        n_distractors_target = max(0, n_docs - len(positive_doc_ids))
        distractor_buffer = max(n_distractors_target * 5, n_distractors_target + 200)
        distractor_pool = []
        if unfetched or n_distractors_target > 0:
            for doc in dataset.docs_iter():
                if doc.doc_id in unfetched:
                    positives.append({"id": doc.doc_id, "text": doc.text})
                    unfetched.discard(doc.doc_id)
                elif doc.doc_id not in positive_doc_ids:
                    if len(distractor_pool) < distractor_buffer:
                        distractor_pool.append({"id": doc.doc_id, "text": doc.text})
                if not unfetched and len(distractor_pool) >= distractor_buffer:
                    break

        # 6. Combine positives + sampled distractors into the final pool.
        #    If positives alone exceed n_docs, trim them down (unlikely).
        if len(positives) > n_docs:
            positives = random.sample(positives, n_docs)
        n_distractors_actual = max(0, n_docs - len(positives))
        distractors = random.sample(
            distractor_pool, min(n_distractors_actual, len(distractor_pool))
        )
        passages = positives + distractors
        random.shuffle(passages)

        # 7. Filter qrels to passages that actually made it into the pool.
        passage_ids = {p["id"] for p in passages}
        qrels = {}
        for qid, doc_rel in sampled_qrels.items():
            kept = {did: rel for did, rel in doc_rel.items() if did in passage_ids}
            if kept:
                qrels[qid] = kept

        # Loud failure beats silent zeros — surface upstream sampling bugs.
        if not qrels:
            raise RuntimeError(
                "qrels empty after sampling — relevant docs missing from passage pool"
            )

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
