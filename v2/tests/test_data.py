# Run tests from the v2/ directory: cd v2 && python -m pytest tests/
"""
test_data.py — Tests for v2/src/data.py

After the bug fixes:
  * MS MARCO is still loaded via ir_datasets (mocked here for speed).
  * BEIR is loaded from the REAL local files under v1/data/beir, returning
    real (non-empty) qrels. Tests use the local scifact dataset when present
    and skip otherwise.
  * Neither loader falls back to synthetic data silently: by default a failed
    load RAISES; synthetic data is only returned when allow_synthetic=True.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.data import load_msmarco_passage_sample, load_beir_sample, _DEFAULT_BEIR_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_ir_dataset(n_docs=500, n_queries=50):
    """Return a minimal mock of an ir_datasets dataset."""
    docs = [
        MagicMock(doc_id=f"doc_{i}", text=f"Passage text number {i}")
        for i in range(n_docs)
    ]
    queries = [
        MagicMock(query_id=f"q_{i}", text=f"Query text {i}")
        for i in range(n_queries)
    ]
    qrels = [
        MagicMock(query_id=f"q_{i}", doc_id=f"doc_{i}", relevance=1)
        for i in range(min(n_queries, n_docs))
    ]
    ds = MagicMock()
    ds.docs_iter.return_value = iter(docs)
    ds.queries_iter.return_value = iter(queries)
    ds.qrels_iter.return_value = iter(qrels)
    # docs_store() raising forces the docs_iter recovery path (covered code).
    ds.docs_store.side_effect = Exception("no docs_store in mock")
    return ds


_HAS_LOCAL_SCIFACT = (_DEFAULT_BEIR_DIR / "scifact" / "corpus.jsonl").exists()
_skip_no_scifact = pytest.mark.skipif(
    not _HAS_LOCAL_SCIFACT, reason="local BEIR scifact data not present"
)


# ---------------------------------------------------------------------------
# Unit tests for load_msmarco_passage_sample (mocked)
# ---------------------------------------------------------------------------

@patch("src.data.ir_datasets")
def test_load_msmarco_passage_returns_correct_shapes(mock_ir):
    mock_ir.load.return_value = _make_fake_ir_dataset(n_docs=500, n_queries=50)
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)

    assert set(data) == {"passages", "queries", "qrels"}
    assert len(data["passages"]) == 50
    assert len(data["queries"]) == 10
    p = data["passages"][0]
    assert "id" in p and "text" in p


@patch("src.data.ir_datasets")
def test_load_msmarco_passage_query_shape(mock_ir):
    mock_ir.load.return_value = _make_fake_ir_dataset(n_docs=500, n_queries=50)
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)
    q = data["queries"][0]
    assert "id" in q and "text" in q


@patch("src.data.ir_datasets")
def test_load_msmarco_qrels_nonempty_and_nested(mock_ir):
    mock_ir.load.return_value = _make_fake_ir_dataset(n_docs=500, n_queries=50)
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)
    assert isinstance(data["qrels"], dict)
    assert data["qrels"], "qrels must not be empty"
    inner = next(iter(data["qrels"].values()))
    assert isinstance(inner, dict), "qrels values should be {doc_id: relevance} dicts"
    # Every judged doc must be present in the passage pool (the original bug).
    pool = {p["id"] for p in data["passages"]}
    for rels in data["qrels"].values():
        assert set(rels).issubset(pool)


@patch("src.data.ir_datasets")
def test_load_msmarco_reproducible(mock_ir):
    """Same seed must produce the same passage order."""
    mock_ir.load.side_effect = [
        _make_fake_ir_dataset(n_docs=500, n_queries=50),
        _make_fake_ir_dataset(n_docs=500, n_queries=50),
    ]
    data1 = load_msmarco_passage_sample(n_docs=5, n_queries=3, seed=7)
    data2 = load_msmarco_passage_sample(n_docs=5, n_queries=3, seed=7)
    assert [p["id"] for p in data1["passages"]] == [p["id"] for p in data2["passages"]]


# ---------------------------------------------------------------------------
# Unit tests for load_beir_sample (real local data)
# ---------------------------------------------------------------------------

@_skip_no_scifact
def test_load_beir_scifact_shapes_and_real_qrels():
    data = load_beir_sample("scifact", n_docs=200, n_queries=5, seed=42)

    assert set(data) == {"passages", "queries", "qrels"}
    assert len(data["passages"]) == 200
    assert len(data["queries"]) == 5

    p = data["passages"][0]
    assert "id" in p and "text" in p
    q = data["queries"][0]
    assert "id" in q and "text" in q

    # The core bug fix: qrels must be REAL and non-empty.
    assert data["qrels"], "BEIR qrels must not be empty (was the bug)"
    pool = {p["id"] for p in data["passages"]}
    for rels in data["qrels"].values():
        assert isinstance(rels, dict)
        assert set(rels).issubset(pool), "relevant docs must be in the passage pool"


@_skip_no_scifact
def test_load_beir_scifact_reproducible():
    d1 = load_beir_sample("scifact", n_docs=100, n_queries=5, seed=13)
    d2 = load_beir_sample("scifact", n_docs=100, n_queries=5, seed=13)
    assert [p["id"] for p in d1["passages"]] == [p["id"] for p in d2["passages"]]
    assert [q["id"] for q in d1["queries"]] == [q["id"] for q in d2["queries"]]


# ---------------------------------------------------------------------------
# No-silent-fallback tests
# ---------------------------------------------------------------------------

@patch("src.data.ir_datasets")
def test_msmarco_raises_by_default_on_error(mock_ir):
    """A failed real load must RAISE, not silently return synthetic data."""
    mock_ir.load.side_effect = RuntimeError("no network")
    with pytest.raises(RuntimeError):
        load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)


@patch("src.data.ir_datasets")
def test_msmarco_synthetic_only_when_allowed(mock_ir):
    """Synthetic data is returned only with allow_synthetic=True."""
    mock_ir.load.side_effect = RuntimeError("no network")
    data = load_msmarco_passage_sample(
        n_docs=50, n_queries=10, seed=42, allow_synthetic=True
    )
    assert len(data["passages"]) == 50
    assert len(data["queries"]) == 10


def test_beir_raises_by_default_on_missing_data():
    """Missing local BEIR data must RAISE by default."""
    with pytest.raises(RuntimeError):
        load_beir_sample(
            "does_not_exist", n_docs=30, n_queries=5, data_dir="/no/such/dir"
        )


def test_beir_synthetic_only_when_allowed():
    data = load_beir_sample(
        "does_not_exist",
        n_docs=30,
        n_queries=5,
        data_dir="/no/such/dir",
        allow_synthetic=True,
    )
    assert len(data["passages"]) == 30
    assert len(data["queries"]) == 5
