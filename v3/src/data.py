"""
data.py — Data loading utilities for MS MARCO and BEIR datasets.

Provides reproducible sampling functions for both in-domain (MS MARCO) and
out-of-domain (BEIR) evaluation sets.

Design rules (after the bug fixes):
  * NO silent fallback to synthetic data. If a real dataset cannot be loaded,
    the loader RAISES. Synthetic data is only returned when the caller passes
    ``allow_synthetic=True`` explicitly (used by unit tests to check the shape
    contract without network/data). This prevents experiments from silently
    running on fake data and reporting meaningless numbers.
  * BEIR loading uses the REAL local qrels shipped under ``v1/data/beir`` so the
    returned ``qrels`` is never empty. The previous version returned
    ``qrels={}``, which made every BEIR metric collapse to 0.
  * For both loaders the relevant ("positive") documents are guaranteed to be in
    the returned passage pool — otherwise qrels would point at missing docs and
    metrics would collapse to 0 (the original sampling bug).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

# Optional heavyweight import — only needed for MS MARCO via ir_datasets.
try:
    import ir_datasets  # type: ignore
except Exception:  # pragma: no cover
    ir_datasets = None  # type: ignore


# Default location of the local BEIR datasets shipped in the repo (v1/data/beir).
# data.py lives at <repo>/v2/src/data.py -> parents[2] == <repo>.
_DEFAULT_BEIR_DIR = Path(__file__).resolve().parents[2] / "v1" / "data" / "beir"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_msmarco_passage_sample(
    n_docs: int,
    n_queries: int,
    seed: int = 42,
    allow_synthetic: bool = False,
):
    """Load a reproducible sample from MS MARCO passage dev/small.

    Args:
        n_docs:          Number of passages to return.
        n_queries:       Number of queries to return.
        seed:            Random seed for reproducibility.
        allow_synthetic: If True, return synthetic data when the real dataset
                         is unavailable. If False (default), RAISE instead of
                         silently faking the data.

    Returns:
        dict with keys ``"passages"``, ``"queries"``, ``"qrels"`` where
        ``qrels`` maps ``qid -> {doc_id: relevance}``.

    Raises:
        RuntimeError: real dataset unavailable and ``allow_synthetic`` is False.
    """
    random.seed(seed)

    try:
        if ir_datasets is None:
            raise ImportError("ir_datasets not available")

        dataset = ir_datasets.load("msmarco-passage/dev/small")

        # 1. Full qrels so we can sample queries that actually have judgments.
        full_qrels: Dict[str, Dict[str, int]] = {}
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

        # 3. Positive doc_ids for the sampled queries — these MUST end up in the
        #    passage pool, else qrels would be empty and metrics 0 (the bug).
        sampled_qrels = {qid: full_qrels[qid] for qid in query_ids if qid in full_qrels}
        positive_doc_ids = {
            did for rels in sampled_qrels.values()
            for did, rel in rels.items() if rel > 0
        }

        # 4. Fast path: fetch positives by id via docs_store (random access).
        positives: List[Dict] = []
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

        # 5. One pass through docs_iter to recover unfetched positives and to
        #    collect a distractor candidate pool.
        n_distractors_target = max(0, n_docs - len(positive_doc_ids))
        distractor_buffer = max(n_distractors_target * 5, n_distractors_target + 200)
        distractor_pool: List[Dict] = []
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

        passages = _assemble_pool(positives, distractor_pool, n_docs, seed)
        qrels = _filter_qrels_to_pool(sampled_qrels, passages)

        if not qrels:
            raise RuntimeError(
                "qrels empty after sampling — relevant docs missing from passage pool"
            )

        return {"passages": passages, "queries": queries, "qrels": qrels}

    except Exception as e:
        if allow_synthetic:
            import warnings
            warnings.warn(
                f"MS MARCO load failed ({type(e).__name__}: {e}); using synthetic data "
                "because allow_synthetic=True."
            )
            return _make_synthetic(n_docs, n_queries, seed)
        raise RuntimeError(
            f"MS MARCO load failed ({type(e).__name__}: {e}). "
            "Refusing to fall back to synthetic data silently. "
            "Pass allow_synthetic=True only for shape tests."
        ) from e


def load_beir_sample(
    dataset_name: str,
    n_docs: int,
    n_queries: int,
    seed: int = 42,
    split: str = "test",
    data_dir: Optional[Path] = None,
    allow_synthetic: bool = False,
):
    """Load a BEIR sample from the LOCAL files under ``v1/data/beir``.

    Unlike the previous version, this returns the REAL qrels (never empty) and
    guarantees the relevant documents are in the passage pool.

    Args:
        dataset_name:    BEIR dataset name (e.g. ``"scifact"``, ``"nfcorpus"``,
                         ``"fiqa"``) — must exist under ``data_dir``.
        n_docs:          Number of passages to return.
        n_queries:       Number of queries to return.
        seed:            Random seed for reproducibility.
        split:           Which qrels split to use (``"test"``/``"dev"``/``"train"``).
        data_dir:        Root of local BEIR datasets. Defaults to ``v1/data/beir``.
        allow_synthetic: If True, return synthetic data when the local dataset is
                         unavailable. If False (default), RAISE instead.

    Returns:
        dict with keys ``"passages"``, ``"queries"``, ``"qrels"`` where
        ``qrels`` maps ``qid -> {doc_id: relevance}`` (real judgments).

    Raises:
        RuntimeError: local dataset unavailable and ``allow_synthetic`` is False.
    """
    random.seed(seed)
    root = Path(data_dir) if data_dir is not None else _DEFAULT_BEIR_DIR
    ds_dir = root / dataset_name

    try:
        qrels_path = ds_dir / "qrels" / f"{split}.tsv"
        queries_path = ds_dir / "queries.jsonl"
        corpus_path = ds_dir / "corpus.jsonl"
        for p in (qrels_path, queries_path, corpus_path):
            if not p.exists():
                raise FileNotFoundError(f"missing BEIR file: {p}")

        # 1. Read real qrels: qid -> {corpus_id: relevance}.
        full_qrels = _read_beir_qrels(qrels_path)
        if not full_qrels:
            raise RuntimeError(f"no qrels parsed from {qrels_path}")

        # 2. Read query texts and sample queries that have a positive judgment.
        query_text = _read_beir_jsonl_text(queries_path)
        candidate_qids = [
            qid for qid in full_qrels
            if qid in query_text and any(r > 0 for r in full_qrels[qid].values())
        ]
        sampled_qids = random.sample(candidate_qids, min(n_queries, len(candidate_qids)))
        queries = [{"id": qid, "text": query_text[qid]} for qid in sampled_qids]
        sampled_qrels = {qid: full_qrels[qid] for qid in sampled_qids}

        # 3. Positive corpus ids must end up in the pool.
        positive_doc_ids = {
            did for rels in sampled_qrels.values()
            for did, rel in rels.items() if rel > 0
        }

        # 4. Stream corpus once: collect positives + a distractor buffer.
        n_distractors_target = max(0, n_docs - len(positive_doc_ids))
        distractor_buffer = max(n_distractors_target * 5, n_distractors_target + 200)
        positives: List[Dict] = []
        distractor_pool: List[Dict] = []
        unfetched = set(positive_doc_ids)
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cid = str(rec["_id"])
                text = f"{rec.get('title', '').strip()} {rec.get('text', '').strip()}".strip()
                if cid in unfetched:
                    positives.append({"id": cid, "text": text})
                    unfetched.discard(cid)
                elif cid not in positive_doc_ids and len(distractor_pool) < distractor_buffer:
                    distractor_pool.append({"id": cid, "text": text})
                if not unfetched and len(distractor_pool) >= distractor_buffer:
                    break

        passages = _assemble_pool(positives, distractor_pool, n_docs, seed)
        qrels = _filter_qrels_to_pool(sampled_qrels, passages)

        if not qrels:
            raise RuntimeError(
                "qrels empty after sampling — relevant docs missing from passage pool"
            )

        return {"passages": passages, "queries": queries, "qrels": qrels}

    except Exception as e:
        if allow_synthetic:
            import warnings
            warnings.warn(
                f"BEIR load failed ({type(e).__name__}: {e}); using synthetic data "
                "because allow_synthetic=True."
            )
            return _make_synthetic(n_docs, n_queries, seed)
        raise RuntimeError(
            f"BEIR '{dataset_name}' load failed ({type(e).__name__}: {e}). "
            "Refusing to fall back to synthetic data silently. "
            "Pass allow_synthetic=True only for shape tests."
        ) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_beir_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Parse a BEIR qrels TSV (header: query-id, corpus-id, score)."""
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        header = f.readline()  # skip header line (query-id\tcorpus-id\tscore)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], parts[2]
            try:
                rel = int(score)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[cid] = rel
    return qrels


def _read_beir_jsonl_text(path: Path) -> Dict[str, str]:
    """Read a BEIR queries/corpus jsonl into ``{_id: text}``."""
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[str(rec["_id"])] = rec.get("text", "")
    return out


def _assemble_pool(
    positives: List[Dict],
    distractor_pool: List[Dict],
    n_docs: int,
    seed: int,
) -> List[Dict]:
    """Combine positives + sampled distractors into a shuffled pool of size n_docs."""
    rng = random.Random(seed)
    if len(positives) > n_docs:
        positives = rng.sample(positives, n_docs)
    n_distractors = max(0, n_docs - len(positives))
    distractors = rng.sample(distractor_pool, min(n_distractors, len(distractor_pool)))
    passages = positives + distractors
    rng.shuffle(passages)
    return passages


def _filter_qrels_to_pool(
    sampled_qrels: Dict[str, Dict[str, int]],
    passages: List[Dict],
) -> Dict[str, Dict[str, int]]:
    """Keep only judgments whose doc actually made it into the passage pool."""
    passage_ids = {p["id"] for p in passages}
    qrels: Dict[str, Dict[str, int]] = {}
    for qid, doc_rel in sampled_qrels.items():
        kept = {did: rel for did, rel in doc_rel.items() if did in passage_ids}
        if kept:
            qrels[qid] = kept
    return qrels


def _make_synthetic(n_docs: int, n_queries: int, seed: int = 0):
    """Return synthetic data that satisfies the expected schema.

    Only used when a caller explicitly passes ``allow_synthetic=True`` (e.g.
    shape tests). Never returned silently from a failed real load.
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
    qrels = {q["id"]: {rng.choice(passages)["id"]: 1} for q in queries}
    return {"passages": passages, "queries": queries, "qrels": qrels}
