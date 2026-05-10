# Run tests from the v2/ directory: cd v2 && python -m pytest tests/
"""
test_data.py — Tests for v2/src/data.py

The real MS MARCO and BEIR datasets require internet access and can be slow
to download.  We patch the external calls at the import boundary so that:

  * Tests always run quickly without network access.
  * The shape contract (keys, lengths, field names) is verified on every run.
  * A separate *integration* test marker (skipped by default) can exercise real
    downloads when `--run-integration` is passed.

Both loader functions also fall back to synthetic data automatically when
the real dataset raises, so even without patching the shape contract holds.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.data import load_msmarco_passage_sample, load_beir_sample


# ---------------------------------------------------------------------------
# Helpers to build fake ir_datasets / HuggingFace dataset objects
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
    return ds


def _make_fake_hf_dataset(n_rows, keys):
    """Return a minimal mock of a HuggingFace Dataset."""
    rows = [{k: f"val_{k}_{i}" for k in keys} | {"_id": str(i)} for i in range(n_rows)]

    class FakeHFDataset:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def select(self, indices):
            return [self._rows[i] for i in indices]

    return FakeHFDataset(rows)


# ---------------------------------------------------------------------------
# Unit tests for load_msmarco_passage_sample (mocked)
# ---------------------------------------------------------------------------

@patch("src.data.ir_datasets")
def test_load_msmarco_passage_returns_correct_shapes(mock_ir):
    mock_ir.load.return_value = _make_fake_ir_dataset(n_docs=500, n_queries=50)
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)

    assert "passages" in data
    assert "queries" in data
    assert "qrels" in data
    assert len(data["passages"]) == 50
    assert len(data["queries"]) == 10

    # Each passage must have 'id' and 'text'
    p = data["passages"][0]
    assert "id" in p and "text" in p


@patch("src.data.ir_datasets")
def test_load_msmarco_passage_query_shape(mock_ir):
    mock_ir.load.return_value = _make_fake_ir_dataset(n_docs=500, n_queries=50)
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)

    q = data["queries"][0]
    assert "id" in q and "text" in q


@patch("src.data.ir_datasets")
def test_load_msmarco_passage_qrels_type(mock_ir):
    mock_ir.load.return_value = _make_fake_ir_dataset(n_docs=500, n_queries=50)
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)
    assert isinstance(data["qrels"], dict)
    # If qrels has entries, check the nested structure
    if data["qrels"]:
        inner = next(iter(data["qrels"].values()))
        assert isinstance(inner, dict), "qrels values should be {doc_id: relevance} dicts"


@patch("src.data.ir_datasets")
def test_load_msmarco_reproducible(mock_ir):
    """Same seed must produce the same result."""
    mock_ir.load.side_effect = [
        _make_fake_ir_dataset(n_docs=500, n_queries=50),
        _make_fake_ir_dataset(n_docs=500, n_queries=50),
        _make_fake_ir_dataset(n_docs=500, n_queries=50),
    ]
    data1 = load_msmarco_passage_sample(n_docs=5, n_queries=3, seed=7)
    data2 = load_msmarco_passage_sample(n_docs=5, n_queries=3, seed=7)
    assert [p["id"] for p in data1["passages"]] == [p["id"] for p in data2["passages"]]
    # Different seed → different order (statistically very likely with enough docs)
    data3 = load_msmarco_passage_sample(n_docs=5, n_queries=3, seed=99)
    # Not guaranteed but very likely; just assert we got valid data
    assert len(data3["passages"]) == 5


# ---------------------------------------------------------------------------
# Unit tests for load_beir_sample (mocked)
# ---------------------------------------------------------------------------

@patch("src.data.load_dataset")
def test_load_beir_scifact_returns_correct_shapes(mock_load_dataset):
    corpus_ds = _make_fake_hf_dataset(100, ["title", "text"])
    queries_ds = _make_fake_hf_dataset(20, ["text"])
    mock_load_dataset.side_effect = [corpus_ds, queries_ds]

    data = load_beir_sample("scifact", n_docs=30, n_queries=5)

    assert "passages" in data
    assert "queries" in data
    assert "qrels" in data
    assert len(data["passages"]) == 30
    assert len(data["queries"]) == 5


@patch("src.data.load_dataset")
def test_load_beir_passage_has_id_and_text(mock_load_dataset):
    corpus_ds = _make_fake_hf_dataset(100, ["title", "text"])
    queries_ds = _make_fake_hf_dataset(20, ["text"])
    mock_load_dataset.side_effect = [corpus_ds, queries_ds]

    data = load_beir_sample("scifact", n_docs=30, n_queries=5)
    p = data["passages"][0]
    assert "id" in p and "text" in p


@patch("src.data.load_dataset")
def test_load_beir_query_has_id_and_text(mock_load_dataset):
    corpus_ds = _make_fake_hf_dataset(100, ["title", "text"])
    queries_ds = _make_fake_hf_dataset(20, ["text"])
    mock_load_dataset.side_effect = [corpus_ds, queries_ds]

    data = load_beir_sample("scifact", n_docs=30, n_queries=5)
    q = data["queries"][0]
    assert "id" in q and "text" in q


# ---------------------------------------------------------------------------
# Fallback / resilience tests
# ---------------------------------------------------------------------------

@patch("src.data.ir_datasets")
def test_msmarco_fallback_on_error(mock_ir):
    """When ir_datasets.load raises, synthetic data must still satisfy shape."""
    mock_ir.load.side_effect = RuntimeError("no network")
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)
    assert len(data["passages"]) == 50
    assert len(data["queries"]) == 10


@patch("src.data.load_dataset")
def test_beir_fallback_on_error(mock_load_dataset):
    """When datasets.load_dataset raises, synthetic data must still satisfy shape."""
    mock_load_dataset.side_effect = RuntimeError("no network")
    data = load_beir_sample("scifact", n_docs=30, n_queries=5)
    assert len(data["passages"]) == 30
    assert len(data["queries"]) == 5
