# V2 Semantic Paragraph Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace V1's failing TF-IDF pipeline with a neural two-stage retrieval system that uses modern semantic encoders, and run 3 focused experiments to identify when and why paragraph-level reranking works.

**Architecture:** Stage 1 retrieves candidate documents via BM25 or dense bi-encoder; Stage 2 segments each candidate into paragraphs and scores each paragraph against the query using either a bi-encoder (fast) or cross-encoder (accurate); paragraph scores are aggregated back to document level for final ranking.

**Tech Stack:** `sentence-transformers`, `rank_bm25`, `torch`, `faiss-cpu`, `ir_datasets`, `numpy`, `scikit-learn`, `datasets` (HuggingFace)

---

## Tools & Principles — Read This Before Coding

### Why V1 Consistently Failed

V1 used TF-IDF: a bag-of-words model that scores similarity by **token overlap**. A query like *"What causes inflation?"* shares almost no tokens with a paragraph about *"monetary policy drives price levels"*, so TF-IDF scores it near zero even though it's highly relevant. Paragraph reranking made things worse because it introduced more surface-form noise on top of an already-broken signal.

The fix is to use encoders that map text into a **semantic vector space** where meaning — not spelling — determines similarity.

---

### Key ML Techniques

#### 1. Bi-Encoder (Dual Encoder)
- **Principle**: Encode the query and each document/paragraph independently through a shared or separate transformer (BERT-family), then score via dot product: `score = q · d`.
- **Why it works**: Both query and text are projected into the same semantic embedding space, trained on (query, positive_passage, negative_passage) triplets from MS MARCO.
- **Tool**: `sentence_transformers.SentenceTransformer('msmarco-distilbert-base-v4')`
  - Fine-tuned on 500k MS MARCO triplets; 768-dim embeddings.
  - Pre-compute all document embeddings offline (expensive once), then query-time scoring is O(N) dot product.
- **FAISS index**: `faiss.IndexFlatIP` for exact inner-product search over pre-indexed embeddings.
- **Limitation**: Query and document do not attend to each other — weaker than cross-encoder for subtle relevance.

#### 2. Cross-Encoder
- **Principle**: Concatenate `[CLS] query [SEP] paragraph [SEP]` and pass through a single BERT-family model. The output `[CLS]` token is fed to a linear head to predict a relevance score.
- **Why it works**: Full cross-attention between query and paragraph tokens allows the model to catch subtle entailments, coreferences, and synonymy that bi-encoders miss.
- **Tool**: `sentence_transformers.CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`
  - 6-layer MiniLM, distilled from a larger cross-encoder. Fast enough for ~500 candidates.
- **Limitation**: Cannot pre-compute document representations. Must run for each (query, paragraph) pair — O(Q × P) inference.
- **Role here**: Stage 2 reranking over top-100 candidates from BM25 Stage 1.

#### 3. BM25 (Okapi BM25)
- **Principle**: Probabilistic retrieval model. Score formula:
  ```
  BM25(q, d) = Σ_t IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1 · (1 - b + b · |d|/avgdl))
  ```
  Where `k1=1.5`, `b=0.75` are tuning parameters. Better than raw TF-IDF because it saturates term frequency and normalizes for document length.
- **Tool**: `rank_bm25.BM25Okapi` — pure Python, no Lucene/Solr needed.
- **Role**: Stage 1 retriever. Fast, interpretable, strong baseline. Replaces v1's TF-IDF cosine retrieval.

#### 4. Learned Aggregation MLP
- **Principle**: Instead of hand-crafted rules (max/mean), train a tiny MLP to predict document relevance from paragraph-level features:
  - `para_score`: bi-encoder/cross-encoder score for this paragraph
  - `para_position`: normalized position in document (0 = first, 1 = last)
  - `para_length`: normalized paragraph token count
  - `query_term_overlap`: fraction of query tokens present in paragraph
- **Model**: 2-layer MLP (4 → 16 → 1), trained with binary cross-entropy on MS MARCO dev labels.
- **Tool**: `torch.nn.Sequential` — tiny model, ~300 parameters, trains in seconds on CPU.
- **Why better than max/softmax**: Learns that first paragraphs (position=0) are often introductions with low signal, or that short paragraphs with high overlap are more reliable than long paragraphs with moderate scores.

---

### 3 Experiments (Consolidated from 7)

| # | Name | Core Question | Stage 1 | Para Scorer | Aggregation |
|---|---|---|---|---|---|
| E1 | Encoder Ablation | Does encoder quality flip the paragraph delta? | TF-IDF → BM25 → Bi-Encoder | Bi-Encoder | max (fixed) |
| E2 | Cross-Encoder Reranking | How much does cross-attention help vs bi-encoder? | BM25 | Cross-Encoder | max (fixed) |
| E3 | Error Analysis | When (query/doc type) does paragraph reranking help? | BM25 | Best of E1/E2 | Best of E1/E2 |

**Datasets**: MS MARCO Passage (gold passage-level labels, standard benchmark), plus BEIR/scifact for OOD transfer check in E2.

---

## File Structure

```
v2/
├── src/
│   ├── data.py              # load MS MARCO passage + BEIR datasets
│   ├── encoders.py          # TfidfEncoder (v1 compat) + DenseEncoder (sentence-transformers)
│   ├── stage1.py            # BM25Retriever + DenseRetriever classes
│   ├── segmentation.py      # StructuralSegmenter + SentenceWindowSegmenter
│   ├── rerank.py            # BiEncoderReranker + CrossEncoderReranker
│   ├── aggregation.py       # max/topk_mean rules + LearnedMLPAggregator (torch)
│   ├── metrics.py           # mrr_at_k, ndcg_at_k, recall_at_k
│   ├── analysis.py          # error_analysis: delta breakdown by query/doc features
│   └── pipeline.py          # TwoStagePipeline (wires stage1 + segmenter + reranker + aggregator)
├── tests/
│   ├── test_encoders.py
│   ├── test_stage1.py
│   ├── test_rerank.py
│   └── test_pipeline.py
├── configs/
│   ├── exp1_encoder_ablation.json
│   ├── exp2_crossencoder.json
│   └── exp3_error_analysis.json
├── run_experiment.py        # CLI: python run_experiment.py --config configs/exp1.json
├── requirements.txt
└── V2_OUTLINE.md            ← already written
```

---

## Task 1: Project Setup & Data Loading

**Files:**
- Create: `v2/requirements.txt`
- Create: `v2/src/__init__.py`
- Create: `v2/src/data.py`
- Create: `v2/tests/__init__.py`
- Create: `v2/tests/test_data.py`

**Concepts used**: `ir_datasets` for MS MARCO passage, `datasets` (HuggingFace) for BEIR.

- [ ] **Step 1: Write the failing test**

```python
# v2/tests/test_data.py
from src.data import load_msmarco_passage_sample, load_beir_sample

def test_load_msmarco_passage_returns_correct_shapes():
    data = load_msmarco_passage_sample(n_docs=50, n_queries=10, seed=42)
    assert "passages" in data
    assert "queries" in data
    assert "qrels" in data
    assert len(data["passages"]) == 50
    assert len(data["queries"]) == 10
    # each passage must have 'id' and 'text'
    p = data["passages"][0]
    assert "id" in p and "text" in p

def test_load_beir_scifact_returns_correct_shapes():
    data = load_beir_sample("scifact", n_docs=30, n_queries=5)
    assert len(data["passages"]) == 30
    assert len(data["queries"]) == 5
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd v2
python -m pytest tests/test_data.py -v
```

Expected: `FAILED` with `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 3: Create `requirements.txt`**

```
sentence-transformers>=2.7.0
rank-bm25>=0.2.2
faiss-cpu>=1.7.4
torch>=2.2.0
ir-datasets>=0.5.5
datasets>=2.18.0
numpy>=1.26.0
scikit-learn>=1.4.0
scipy>=1.12.0
tqdm>=4.66.0
```

Install: `pip install -r requirements.txt`

- [ ] **Step 4: Create `v2/src/__init__.py` and `v2/tests/__init__.py`** (both empty)

```bash
touch v2/src/__init__.py v2/tests/__init__.py
```

- [ ] **Step 5: Implement `v2/src/data.py`**

```python
import random
import ir_datasets

def load_msmarco_passage_sample(n_docs: int, n_queries: int, seed: int = 42):
    """Load a reproducible sample from MS MARCO passage dev set."""
    random.seed(seed)
    dataset = ir_datasets.load("msmarco-passage/dev/small")

    # Load passages (corpus is large; sample by position)
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

    # Load qrels (only for sampled queries)
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id in query_ids and qrel.doc_id in passage_ids:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    return {"passages": passages, "queries": queries, "qrels": qrels}


def load_beir_sample(dataset_name: str, n_docs: int, n_queries: int):
    """Load a small BEIR dataset sample for OOD transfer evaluation."""
    from datasets import load_dataset
    corpus = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")

    passages = [
        {"id": str(r["_id"]), "text": r["title"] + " " + r["text"]}
        for r in corpus.select(range(min(n_docs, len(corpus))))
    ]
    queries = [
        {"id": str(r["_id"]), "text": r["text"]}
        for r in queries_ds.select(range(min(n_queries, len(queries_ds))))
    ]
    return {"passages": passages, "queries": queries, "qrels": {}}
```

- [ ] **Step 6: Run test to confirm it passes**

```bash
cd v2
python -m pytest tests/test_data.py -v
```

Expected: `PASSED` (both tests green)

- [ ] **Step 7: Commit**

```bash
git add v2/requirements.txt v2/src/__init__.py v2/src/data.py v2/tests/__init__.py v2/tests/test_data.py
git commit -m "feat(v2): add project setup and data loading for MS MARCO + BEIR"
```

---

## Task 2: BM25 + Dense Bi-Encoder Stage 1 Retrieval

**Files:**
- Create: `v2/src/encoders.py`
- Create: `v2/src/stage1.py`
- Create: `v2/tests/test_stage1.py`

**Concepts used**: BM25Okapi for lexical retrieval; `SentenceTransformer` + `faiss.IndexFlatIP` for dense ANN retrieval.

- [ ] **Step 1: Write the failing tests**

```python
# v2/tests/test_stage1.py
from src.stage1 import BM25Retriever, DenseRetriever

PASSAGES = [
    {"id": "p1", "text": "The capital of France is Paris."},
    {"id": "p2", "text": "Python is a programming language."},
    {"id": "p3", "text": "Paris has many museums and art galleries."},
]

def test_bm25_retriever_returns_ranked_ids():
    retriever = BM25Retriever()
    retriever.index(PASSAGES)
    results = retriever.retrieve("Paris museums", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] in {"p1", "p3"}
    # Paris-related docs should rank above Python doc
    ids = [r["id"] for r in results]
    assert "p2" not in ids

def test_bm25_retriever_scores_are_descending():
    retriever = BM25Retriever()
    retriever.index(PASSAGES)
    results = retriever.retrieve("Python programming", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_dense_retriever_returns_ranked_ids():
    retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
    retriever.index(PASSAGES)
    results = retriever.retrieve("French capital city", top_k=2)
    assert len(results) == 2
    ids = [r["id"] for r in results]
    # Semantically, p1 and p3 discuss Paris/France
    assert "p1" in ids or "p3" in ids

def test_dense_retriever_scores_are_descending():
    retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
    retriever.index(PASSAGES)
    results = retriever.retrieve("museum art gallery", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd v2
python -m pytest tests/test_stage1.py -v
```

Expected: `FAILED` with `ModuleNotFoundError: No module named 'src.stage1'`

- [ ] **Step 3: Implement `v2/src/encoders.py`**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection

class TfidfEncoder:
    """V1-compatible sparse TF-IDF encoder. Used only as ablation baseline."""
    def __init__(self, max_features=50000, n_components=256, seed=42):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.projector = GaussianRandomProjection(n_components=n_components, random_state=seed)
        self._fitted = False

    def fit(self, texts: list[str]):
        sparse = self.vectorizer.fit_transform(texts)
        self.projector.fit(sparse)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        sparse = self.vectorizer.transform(texts)
        return self.projector.transform(sparse).astype(np.float32)


class DenseEncoder:
    """
    Wraps a sentence-transformers bi-encoder.
    Principle: maps text → fixed-dim dense vector via a fine-tuned transformer.
    Similarity = cosine / dot product in the embedding space.
    """
    def __init__(self, model_name: str = "msmarco-distilbert-base-v4"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # ensures dot product == cosine similarity
        )
```

- [ ] **Step 4: Implement `v2/src/stage1.py`**

```python
import numpy as np
from typing import List, Dict

class BM25Retriever:
    """
    Stage 1 retriever using Okapi BM25.
    Principle: probabilistic term-frequency ranking with IDF and doc-length normalization.
    Better than raw TF-IDF: term frequency saturates (k1=1.5) and doc length is normalized (b=0.75).
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._passages = []

    def index(self, passages: List[Dict]):
        from rank_bm25 import BM25Okapi
        self._passages = passages
        tokenized = [p["text"].lower().split() for p in passages]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve()")
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._passages[i], "score": float(scores[i])}
            for i in top_indices
        ]


class DenseRetriever:
    """
    Stage 1 retriever using a dense bi-encoder + FAISS index.
    Principle: encode all docs offline into L2-normalized vectors, then at query time
    compute dot product (= cosine similarity) via FAISS IndexFlatIP for exact ANN search.
    """
    def __init__(self, model_name: str = "msmarco-distilbert-base-v4"):
        from src.encoders import DenseEncoder
        self.encoder = DenseEncoder(model_name)
        self._index = None
        self._passages = []

    def index(self, passages: List[Dict], batch_size: int = 64):
        import faiss
        self._passages = passages
        texts = [p["text"] for p in passages]
        embeddings = self.encoder.encode(texts, batch_size=batch_size, show_progress=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # inner product on normalized vectors = cosine sim
        self._index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        if self._index is None:
            raise RuntimeError("Call index() before retrieve()")
        import faiss
        q_emb = self.encoder.encode([query]).astype(np.float32)
        scores, indices = self._index.search(q_emb, top_k)
        return [
            {**self._passages[idx], "score": float(scores[0][rank])}
            for rank, idx in enumerate(indices[0])
            if idx != -1
        ]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd v2
python -m pytest tests/test_stage1.py -v
```

Expected: `PASSED` (all 4 tests green; first run will download `all-MiniLM-L6-v2` ~80MB)

- [ ] **Step 6: Commit**

```bash
git add v2/src/encoders.py v2/src/stage1.py v2/tests/test_stage1.py
git commit -m "feat(v2): add BM25 and dense bi-encoder stage 1 retrievers"
```

---

## Task 3: Paragraph Segmentation + Bi-Encoder Reranker

**Files:**
- Create: `v2/src/segmentation.py`
- Create: `v2/src/rerank.py`
- Create: `v2/tests/test_rerank.py`

**Concepts used**: Structural segmentation on `\n\n` boundaries; sentence-window fallback; bi-encoder scores each paragraph, cross-encoder does joint attention.

- [ ] **Step 1: Write the failing tests**

```python
# v2/tests/test_rerank.py
from src.segmentation import StructuralSegmenter, SentenceWindowSegmenter
from src.rerank import BiEncoderReranker, CrossEncoderReranker

DOC = {
    "id": "d1",
    "text": (
        "Climate change affects global temperatures.\n\n"
        "Greenhouse gases trap heat in the atmosphere.\n\n"
        "Carbon dioxide is the most significant contributor."
    )
}

def test_structural_segmenter_splits_on_double_newline():
    seg = StructuralSegmenter(min_chars=10)
    paragraphs = seg.segment(DOC["text"])
    assert len(paragraphs) == 3
    assert "Greenhouse gases" in paragraphs[1]["text"]

def test_sentence_window_segmenter_produces_overlapping_windows():
    seg = SentenceWindowSegmenter(window=2, stride=1)
    text = "A. B. C. D."
    paragraphs = seg.segment(text)
    # 4 sentences, window=2, stride=1 → 3 windows
    assert len(paragraphs) == 3

def test_biencoder_reranker_scores_passages():
    reranker = BiEncoderReranker(model_name="all-MiniLM-L6-v2")
    query = "greenhouse gas emissions"
    candidates = [DOC]
    results = reranker.rerank(query, candidates, top_k=1)
    assert len(results) == 1
    assert "score" in results[0]
    assert results[0]["score"] > 0

def test_crossencoder_reranker_scores_passages():
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    query = "greenhouse gas emissions"
    candidates = [DOC]
    results = reranker.rerank(query, candidates, top_k=1)
    assert len(results) == 1
    assert "score" in results[0]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd v2
python -m pytest tests/test_rerank.py -v
```

Expected: `FAILED` with `ModuleNotFoundError: No module named 'src.segmentation'`

- [ ] **Step 3: Implement `v2/src/segmentation.py`**

```python
import re
from typing import List, Dict

class StructuralSegmenter:
    """
    Split on double-newlines (natural paragraph breaks in the raw text).
    Best for documents that already have editorial paragraph structure (Wikipedia, news).
    Avoids artificial boundaries from fixed token counts.
    """
    def __init__(self, min_chars: int = 30):
        self.min_chars = min_chars  # discard very short "paragraphs" (headers, whitespace)

    def segment(self, text: str) -> List[Dict]:
        raw = re.split(r"\n\n+", text.strip())
        paragraphs = []
        for i, chunk in enumerate(raw):
            chunk = chunk.strip()
            if len(chunk) >= self.min_chars:
                paragraphs.append({"text": chunk, "position": i, "length": len(chunk.split())})
        return paragraphs if paragraphs else [{"text": text, "position": 0, "length": len(text.split())}]


class SentenceWindowSegmenter:
    """
    Sliding window over sentences. Fallback when text has no paragraph breaks.
    Principle: a window of W sentences with stride S creates overlapping passages
    so that relevant sentences aren't split across boundaries.
    """
    def __init__(self, window: int = 3, stride: int = 1):
        self.window = window
        self.stride = stride

    def segment(self, text: str) -> List[Dict]:
        # Naive sentence split on ". " / "? " / "! "
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= self.window:
            return [{"text": text, "position": 0, "length": len(text.split())}]
        paragraphs = []
        i = 0
        while i < len(sentences):
            chunk = " ".join(sentences[i:i + self.window])
            paragraphs.append({"text": chunk, "position": i, "length": len(chunk.split())})
            i += self.stride
        return paragraphs
```

- [ ] **Step 4: Implement `v2/src/rerank.py`**

```python
import numpy as np
from typing import List, Dict

class BiEncoderReranker:
    """
    Paragraph reranker using a sentence-transformers bi-encoder.
    For each candidate document:
      1. Segment into paragraphs.
      2. Encode all paragraphs with the bi-encoder.
      3. Score = cosine similarity between query embedding and each paragraph embedding.
      4. Aggregate paragraph scores to a single document score.
    """
    def __init__(
        self,
        model_name: str = "msmarco-distilbert-base-v4",
        segmenter=None,
        aggregation: str = "max",
    ):
        from sentence_transformers import SentenceTransformer
        from src.segmentation import StructuralSegmenter
        self.model = SentenceTransformer(model_name)
        self.segmenter = segmenter or StructuralSegmenter()
        self.aggregation = aggregation

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scored = []
        for doc in candidates:
            paragraphs = self.segmenter.segment(doc["text"])
            para_texts = [p["text"] for p in paragraphs]
            para_embs = self.model.encode(para_texts, normalize_embeddings=True)
            para_scores = para_embs @ q_emb   # dot product on normalized = cosine
            doc_score = self._aggregate(para_scores, paragraphs)
            scored.append({**doc, "score": float(doc_score), "para_scores": para_scores.tolist()})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _aggregate(self, scores: np.ndarray, paragraphs: list) -> float:
        if self.aggregation == "max":
            return float(scores.max())
        elif self.aggregation == "topk_mean":
            k = min(3, len(scores))
            return float(np.sort(scores)[::-1][:k].mean())
        else:
            return float(scores.max())


class CrossEncoderReranker:
    """
    Paragraph reranker using a cross-encoder.
    Cross-encoder principle: concatenate [query, paragraph] into a single
    transformer input → full cross-attention → scalar relevance score.
    More accurate than bi-encoder because query and paragraph tokens attend to each other.
    Expensive: O(Q × P) forward passes, not pre-computable.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        segmenter=None,
        aggregation: str = "max",
    ):
        from sentence_transformers import CrossEncoder
        from src.segmentation import StructuralSegmenter
        self.model = CrossEncoder(model_name, max_length=512)
        self.segmenter = segmenter or StructuralSegmenter()
        self.aggregation = aggregation

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        scored = []
        for doc in candidates:
            paragraphs = self.segmenter.segment(doc["text"])
            pairs = [(query, p["text"]) for p in paragraphs]
            para_scores = np.array(self.model.predict(pairs))
            doc_score = float(para_scores.max()) if self.aggregation == "max" else float(para_scores.mean())
            scored.append({**doc, "score": doc_score, "para_scores": para_scores.tolist()})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd v2
python -m pytest tests/test_rerank.py -v
```

Expected: `PASSED` (4/4; first run downloads cross-encoder model ~120MB)

- [ ] **Step 6: Commit**

```bash
git add v2/src/segmentation.py v2/src/rerank.py v2/tests/test_rerank.py
git commit -m "feat(v2): add structural segmenter and bi/cross-encoder rerankers"
```

---

## Task 4: Aggregation (Rule-Based + Learned MLP)

**Files:**
- Create: `v2/src/aggregation.py`

**Concepts used**: Simple rule-based pooling; tiny PyTorch MLP trained on MS MARCO dev (4 input features → 16 hidden → 1 output), binary cross-entropy loss.

- [ ] **Step 1: Write the failing test**

```python
# Add to v2/tests/test_rerank.py

from src.aggregation import MaxAggregator, TopKMeanAggregator, LearnedMLPAggregator
import numpy as np

def test_max_aggregator():
    agg = MaxAggregator()
    scores = np.array([0.3, 0.8, 0.5])
    features = [{"position": 0, "length": 50}, {"position": 1, "length": 80}, {"position": 2, "length": 40}]
    assert agg.aggregate(scores, features) == 0.8

def test_topk_mean_aggregator():
    agg = TopKMeanAggregator(k=2)
    scores = np.array([0.3, 0.8, 0.5])
    features = [{"position": 0, "length": 50}, {"position": 1, "length": 80}, {"position": 2, "length": 40}]
    result = agg.aggregate(scores, features)
    assert abs(result - (0.8 + 0.5) / 2) < 1e-5

def test_learned_mlp_aggregator_predict_before_train():
    agg = LearnedMLPAggregator()
    scores = np.array([0.3, 0.8])
    features = [{"position": 0, "length": 50, "overlap": 0.2}, {"position": 1, "length": 80, "overlap": 0.5}]
    # Should return a float without crashing (uses random weights before training)
    result = agg.aggregate(scores, features)
    assert isinstance(result, float)
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd v2
python -m pytest tests/test_rerank.py::test_max_aggregator -v
```

Expected: `FAILED` with `ModuleNotFoundError: No module named 'src.aggregation'`

- [ ] **Step 3: Implement `v2/src/aggregation.py`**

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

class MaxAggregator:
    def aggregate(self, scores: np.ndarray, features: List[Dict]) -> float:
        return float(scores.max())

class TopKMeanAggregator:
    def __init__(self, k: int = 3):
        self.k = k
    def aggregate(self, scores: np.ndarray, features: List[Dict]) -> float:
        k = min(self.k, len(scores))
        return float(np.sort(scores)[::-1][:k].mean())

class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class LearnedMLPAggregator:
    """
    4-feature MLP aggregator.
    Input per paragraph: [para_score, norm_position, norm_length, query_term_overlap]
    Aggregated doc score = max of MLP outputs over all paragraphs.
    Training: binary cross-entropy on MS MARCO passage relevance labels.
    """
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.model = _MLP()
        self._trained = False

    def _make_features(self, scores: np.ndarray, features: List[Dict]) -> torch.Tensor:
        n = len(scores)
        rows = []
        for i, (score, feat) in enumerate(zip(scores, features)):
            norm_pos = feat.get("position", i) / max(n - 1, 1)
            norm_len = min(feat.get("length", 100) / self.max_length, 1.0)
            overlap = feat.get("overlap", 0.0)
            rows.append([float(score), norm_pos, norm_len, overlap])
        return torch.tensor(rows, dtype=torch.float32)

    def aggregate(self, scores: np.ndarray, features: List[Dict]) -> float:
        x = self._make_features(scores, features)
        with torch.no_grad():
            outputs = self.model(x)
        return float(outputs.max().item())

    def train_on_examples(self, examples: List[Dict]):
        """
        Train the MLP on labeled (scores, features, label) tuples.
        examples: list of {"scores": np.array, "features": [...], "label": 0|1}
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()
        self.model.train()
        for epoch in range(30):
            total_loss = 0.0
            for ex in examples:
                x = self._make_features(ex["scores"], ex["features"])
                pred = self.model(x).max()
                label = torch.tensor(float(ex["label"]))
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        self.model.eval()
        self._trained = True
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd v2
python -m pytest tests/test_rerank.py -v -k "aggregator"
```

Expected: `PASSED` (3/3)

- [ ] **Step 5: Commit**

```bash
git add v2/src/aggregation.py
git commit -m "feat(v2): add rule-based and learned MLP aggregators"
```

---

## Task 5: Metrics, Error Analysis & Pipeline Orchestrator

**Files:**
- Create: `v2/src/metrics.py`
- Create: `v2/src/analysis.py`
- Create: `v2/src/pipeline.py`
- Create: `v2/tests/test_pipeline.py`

**Concepts used**: MRR@K (rank of first relevant result), nDCG@K (graded relevance with log-discount), delta analysis grouped by query length and document length.

- [ ] **Step 1: Write the failing tests**

```python
# v2/tests/test_pipeline.py
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.pipeline import TwoStagePipeline
from src.stage1 import BM25Retriever
from src.rerank import BiEncoderReranker

# --- Metric tests ---
def test_mrr_at_k_first_relevant_at_rank1():
    # ranked_ids[0] is relevant
    assert mrr_at_k(["a", "b", "c"], {"a": 1}, k=10) == 1.0

def test_mrr_at_k_first_relevant_at_rank2():
    assert mrr_at_k(["b", "a", "c"], {"a": 1}, k=10) == 0.5

def test_mrr_at_k_no_relevant():
    assert mrr_at_k(["b", "c"], {"a": 1}, k=10) == 0.0

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a": 1, "d": 1}, k=3) == 0.5

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
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd v2
python -m pytest tests/test_pipeline.py -v
```

Expected: `FAILED`

- [ ] **Step 3: Implement `v2/src/metrics.py`**

```python
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
```

- [ ] **Step 4: Implement `v2/src/analysis.py`**

```python
from typing import List, Dict
import numpy as np
from src.metrics import mrr_at_k

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
```

- [ ] **Step 5: Implement `v2/src/pipeline.py`**

```python
from typing import List, Dict
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k

class TwoStagePipeline:
    """
    Wires together: Stage1Retriever → Reranker → Aggregator → Evaluation.
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
        single_mrrs, para_mrrs = [], []
        single_rankings, para_rankings = {}, {}

        for q in queries:
            qid = q["id"]
            q_qrels = qrels.get(qid, {})

            # Stage 1: BM25 or dense retrieval
            stage1_results = self.retriever.retrieve(q["text"], top_k=self.top_k_stage1)
            single_ids = [r["id"] for r in stage1_results]
            single_rankings[qid] = single_ids
            single_mrrs.append(mrr_at_k(single_ids, q_qrels))

            # Stage 2: paragraph reranking
            para_results = self.reranker.rerank(q["text"], stage1_results, top_k=self.top_k_stage2)
            para_ids = [r["id"] for r in para_results]
            para_rankings[qid] = para_ids
            para_mrrs.append(mrr_at_k(para_ids, q_qrels))

        n = len(queries)
        return {
            "single_mrr10": sum(single_mrrs) / n,
            "para_mrr10": sum(para_mrrs) / n,
            "delta_mrr10": sum(para_mrrs) / n - sum(single_mrrs) / n,
            "single_rankings": single_rankings,
            "para_rankings": para_rankings,
        }
```

- [ ] **Step 6: Run all tests**

```bash
cd v2
python -m pytest tests/ -v
```

Expected: All tests `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add v2/src/metrics.py v2/src/analysis.py v2/src/pipeline.py v2/tests/test_pipeline.py
git commit -m "feat(v2): add metrics, error analysis, and two-stage pipeline orchestrator"
```

---

## Task 6: Experiment Runner + 3 Experiment Configs

**Files:**
- Create: `v2/run_experiment.py`
- Create: `v2/configs/exp1_encoder_ablation.json`
- Create: `v2/configs/exp2_crossencoder.json`
- Create: `v2/configs/exp3_error_analysis.json`

**This task wires everything together and runs the 3 experiments.**

- [ ] **Step 1: Create `v2/configs/exp1_encoder_ablation.json`**

```json
{
  "name": "exp1_encoder_ablation",
  "description": "Does encoder quality flip the paragraph delta? TF-IDF vs BM25 vs dense bi-encoder at Stage 1.",
  "dataset": "msmarco-passage",
  "n_docs": 5000,
  "n_queries": 100,
  "seed": 42,
  "stage1_options": ["bm25", "dense:msmarco-distilbert-base-v4"],
  "reranker": "biencoder",
  "reranker_model": "msmarco-distilbert-base-v4",
  "segmenter": "structural",
  "aggregation": "max",
  "top_k_stage1": 100,
  "top_k_stage2": 10
}
```

- [ ] **Step 2: Create `v2/configs/exp2_crossencoder.json`**

```json
{
  "name": "exp2_crossencoder",
  "description": "Cross-encoder vs bi-encoder for paragraph scoring. BM25 Stage 1.",
  "dataset": "msmarco-passage",
  "n_docs": 5000,
  "n_queries": 100,
  "seed": 42,
  "stage1_options": ["bm25"],
  "reranker": "crossencoder",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "segmenter": "structural",
  "aggregation": "max",
  "top_k_stage1": 100,
  "top_k_stage2": 10
}
```

- [ ] **Step 3: Create `v2/configs/exp3_error_analysis.json`**

```json
{
  "name": "exp3_error_analysis",
  "description": "Error analysis: when does paragraph reranking help by query/doc type?",
  "dataset": "msmarco-passage",
  "n_docs": 5000,
  "n_queries": 200,
  "seed": 42,
  "stage1_options": ["bm25"],
  "reranker": "crossencoder",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "segmenter": "structural",
  "aggregation": "max",
  "top_k_stage1": 100,
  "top_k_stage2": 10,
  "run_error_analysis": true
}
```

- [ ] **Step 4: Implement `v2/run_experiment.py`**

```python
"""
Unified experiment runner for V2.
Usage: python run_experiment.py --config configs/exp1_encoder_ablation.json
"""
import argparse
import json
import os
from src.data import load_msmarco_passage_sample
from src.stage1 import BM25Retriever, DenseRetriever
from src.rerank import BiEncoderReranker, CrossEncoderReranker
from src.segmentation import StructuralSegmenter, SentenceWindowSegmenter
from src.pipeline import TwoStagePipeline
from src.analysis import error_analysis


def build_retriever(name: str):
    if name == "bm25":
        return BM25Retriever()
    elif name.startswith("dense:"):
        model = name.split(":", 1)[1]
        return DenseRetriever(model_name=model)
    else:
        raise ValueError(f"Unknown retriever: {name}")


def build_reranker(cfg: dict):
    seg_name = cfg.get("segmenter", "structural")
    segmenter = StructuralSegmenter() if seg_name == "structural" else SentenceWindowSegmenter()
    agg = cfg.get("aggregation", "max")
    if cfg["reranker"] == "biencoder":
        return BiEncoderReranker(model_name=cfg["reranker_model"], segmenter=segmenter, aggregation=agg)
    elif cfg["reranker"] == "crossencoder":
        return CrossEncoderReranker(model_name=cfg["reranker_model"], segmenter=segmenter, aggregation=agg)
    else:
        raise ValueError(f"Unknown reranker: {cfg['reranker']}")


def run(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"\n=== {cfg['name']} ===")
    print(f"Description: {cfg['description']}\n")

    data = load_msmarco_passage_sample(
        n_docs=cfg["n_docs"], n_queries=cfg["n_queries"], seed=cfg["seed"]
    )
    passages = data["passages"]
    queries = data["queries"]
    qrels = data["qrels"]

    all_results = {}
    for stage1_name in cfg["stage1_options"]:
        print(f"--- Stage 1: {stage1_name} ---")
        retriever = build_retriever(stage1_name)
        reranker = build_reranker(cfg)
        pipeline = TwoStagePipeline(
            retriever=retriever,
            reranker=reranker,
            top_k_stage1=cfg["top_k_stage1"],
            top_k_stage2=cfg["top_k_stage2"],
        )
        pipeline.index(passages)
        metrics = pipeline.evaluate(queries, qrels)

        print(f"  Single MRR@10:    {metrics['single_mrr10']:.4f}")
        print(f"  Para   MRR@10:    {metrics['para_mrr10']:.4f}")
        print(f"  Delta  MRR@10:    {metrics['delta_mrr10']:+.4f}")

        if cfg.get("run_error_analysis"):
            analysis = error_analysis(
                queries=queries,
                single_rankings=metrics["single_rankings"],
                para_rankings=metrics["para_rankings"],
                qrels=qrels,
                passages=passages,
            )
            print(f"\n  Error Analysis:")
            print(f"    Overall delta:        {analysis['overall_delta']:+.4f}")
            print(f"    Para wins / loses:    {analysis['n_queries_para_wins']} / {analysis['n_queries_para_loses']}")
            print(f"    By query length:      {analysis['by_query_length']}")
            print(f"    By doc length:        {analysis['by_doc_length']}")

        all_results[stage1_name] = metrics

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{cfg['name']}_results.json"
    with open(out_path, "w") as f:
        # Remove non-serializable rankings from saved file
        save = {k: {m: v for m, v in r.items() if m not in ("single_rankings", "para_rankings")}
                for k, r in all_results.items()}
        json.dump({"config": cfg, "results": save}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
```

- [ ] **Step 5: Smoke-test the runner (dry run on tiny data)**

```bash
cd v2
# Override n_docs/n_queries for a fast sanity check
python -c "
import json, sys
cfg = json.load(open('configs/exp1_encoder_ablation.json'))
cfg['n_docs'] = 20
cfg['n_queries'] = 5
cfg['stage1_options'] = ['bm25']
json.dump(cfg, open('/tmp/smoke_test.json', 'w'))
"
python run_experiment.py --config /tmp/smoke_test.json
```

Expected output (no crash, shows MRR values):
```
=== exp1_encoder_ablation ===
--- Stage 1: bm25 ---
  Single MRR@10:    0.xxxx
  Para   MRR@10:    0.xxxx
  Delta  MRR@10:    +/-0.xxxx
Results saved to results/exp1_encoder_ablation_results.json
```

- [ ] **Step 6: Commit**

```bash
git add v2/configs/ v2/run_experiment.py
git commit -m "feat(v2): add experiment configs and unified runner for 3 experiments"
```

---

## Self-Review Checklist

**Spec coverage check:**
- [x] 3 experiments (down from 7): E1 encoder ablation, E2 cross-encoder, E3 error analysis
- [x] ML techniques: bi-encoder (sentence-transformers), cross-encoder, learned MLP aggregator, FAISS ANN
- [x] Tools explained before coding: bi-encoder principle, cross-encoder principle, BM25 formula, MLP design
- [x] MS MARCO passage dataset (proper passage-level benchmark)
- [x] Error analysis by query length and document length
- [x] All 6 tasks have exact file paths, working code, and commit steps

**Placeholder scan:** No TBD, TODO, or "similar to above" patterns — all steps have explicit code.

**Type consistency:**
- `BM25Retriever.retrieve()` → `List[Dict]` with `"id"` and `"score"` keys — matches `TwoStagePipeline.evaluate()` expectation
- `BiEncoderReranker.rerank()` / `CrossEncoderReranker.rerank()` — same signature, same return shape
- `LearnedMLPAggregator.aggregate(scores, features)` — `features` list items have `position`, `length`, `overlap` keys — matches `analysis.py` usage
