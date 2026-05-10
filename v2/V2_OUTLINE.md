# V2 Outline: Semantic Paragraph Retrieval

## What V1 Proved (and Didn't)

### V1 Findings Summary

V1 ran a large-scale grid search (63 configurations × 10 seeds) across MS MARCO and 3 BEIR datasets (nfcorpus, scifact, fiqa), testing a **two-stage pipeline**:
- Stage 1: TF-IDF sparse retrieval → 100–500 candidate documents
- Stage 2: Paragraph-level reranking via TF-IDF + random projection

**Core result**: Paragraph reranking **consistently underperformed** the single-vector baseline across every configuration, dataset, and seed.

| Setting | Single MRR@10 | Para MRR@10 | Delta |
|---|---|---|---|
| MS MARCO (mean over 10 seeds) | 0.3257 | 0.1891 | **-0.1366** |
| nfcorpus | 0.4932 | 0.4363 | -0.0569 |
| scifact | 0.5756 | 0.4015 | **-0.1742** |
| fiqa | 0.1587 | 0.1301 | -0.0285 |

---

## What V1 Failed to Do

### 1. No Semantic Encoders
V1 used only TF-IDF (bag-of-words). Paragraph-level matching requires understanding **meaning**, not just token overlap. Queries like "What causes inflation?" share few tokens with a paragraph that discusses "monetary policy and price levels." Dense retrieval models close this gap.

### 2. Weak Stage 1 Retriever
If Stage 1 (TF-IDF) fails to retrieve the truly relevant document into the candidate pool, Stage 2 can never recover it. A poor recall ceiling at Stage 1 dooms paragraph reranking before it starts.

### 3. No Cross-Encoder Reranking
V1 aggregated paragraph similarity scores (max, mean, softmax). It never used a **cross-encoder** that can jointly encode query + paragraph for a relevance judgment — the approach proven most effective in the literature.

### 4. No Learned Aggregation
Aggregation (max, topk_mean, softmax) was entirely rule-based. There was no learned function that weighs paragraphs by their position, length, or semantic coherence with the query.

### 5. Tiny Test Sets
MS MARCO evaluations used **10 test queries** — statistically unreliable. Results could shift significantly with more queries.

### 6. No Error Analysis
V1 did not diagnose *why* individual queries failed. No breakdown by query type, document length, domain, or relevance annotation density. Without this, it's impossible to know when paragraphs could help.

### 7. Syntactic Segmentation Only
All 7 segmentation strategies were purely syntactic (sentence windows, fixed token chunks). They don't respect semantic paragraph boundaries or discourse structure.

### 8. No Ablation Studies
V1 ran a grid but didn't isolate the contribution of individual components (encoder quality, candidate pool size, segmentation granularity). It's unclear which factor hurts most.

### 9. No Dense Retrieval Baseline
V1 compared paragraph reranking only to TF-IDF single-vector. There's no comparison to state-of-the-art dense retrievers (DPR, ColBERT), which would reveal whether the failure is specific to TF-IDF or fundamental.

### 10. No Passage-Level Dataset
MS MARCO has a **passage** version that is ideally suited for paragraph retrieval research. V1 used the document version and segmented manually — the passage version has gold-labeled passage relevance judgments.

---

## V2 Goals

1. Determine whether paragraph-level retrieval can **beat** document-level retrieval given proper semantic encoding.
2. Identify the conditions (encoder quality, candidate pool, segmentation, aggregation) under which paragraph reranking helps.
3. Produce a reproducible, publication-quality experiment comparing multiple retrieval paradigms.

---

## V2 Architecture Overview

```
Query
  │
  ▼
Stage 1: Bi-encoder Dense Retrieval (BM25 or DPR)
  │  → Top-K candidate documents (K = 100, 200, 500)
  ▼
Stage 2: Paragraph Segmentation
  │  → Semantic: topic-based, discourse-aware
  │  → Syntactic: sentence windows, token chunks (for ablation)
  ▼
Stage 3: Paragraph Scoring
  │  Option A: Bi-encoder dot product (fast, scalable)
  │  Option B: Cross-encoder (slower, more accurate)
  │  Option C: ColBERT-style late interaction
  ▼
Stage 4: Score Aggregation
  │  → Rule-based: max, topk_mean, softmax
  │  → Learned: MLP over paragraph scores + position features
  ▼
Final Document Ranking
  │
  ▼
Evaluation: MRR@10, nDCG@10, Recall@100 (and passage-level metrics)
```

---

## Detailed V2 Experiment Plan

### Experiment 1: Stage 1 Retriever Quality vs. Paragraph Benefit

**Hypothesis**: Paragraph reranking only helps when Stage 1 has high recall.

**Setup**:
- Stage 1 options: BM25 (via pyserini), DPR, TF-IDF (v1 baseline)
- Stage 2: Fixed bi-encoder (sentence-transformers/msmarco-distilbert-base-v4)
- Measure: Recall@100 at Stage 1, MRR@10 at Stage 2
- Datasets: MS MARCO passage, nfcorpus, scifact

**Expected output**: Curve of MRR@10 vs. Stage-1 recall ceiling — shows the "blame" for paragraph underperformance.

---

### Experiment 2: Encoder Quality Ablation

**Hypothesis**: TF-IDF projection is too weak; semantic encoders will flip the paragraph vs. document delta.

**Encoders to compare**:
| Encoder | Type | Dim |
|---|---|---|
| TF-IDF + random projection (v1) | Sparse → dense | 256 |
| BM25 score only | Sparse | 1 |
| sentence-transformers/all-MiniLM-L6-v2 | Dense bi-encoder | 384 |
| sentence-transformers/msmarco-distilbert-base-v4 | Dense bi-encoder (IR-tuned) | 768 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder | N/A (scalar output) |

**Metric**: MRR@10 for document-level ranking vs. paragraph-reranked ranking.

**Expected output**: Table showing which encoder is the tipping point where paragraphs start to help.

---

### Experiment 3: Segmentation Strategy Comparison

**Hypothesis**: Semantic segmentation outperforms syntactic windows.

**Segmentation methods**:
- V1 baseline: sent2_stride1, tok128 (syntactic)
- New: `nltk` topic segmentation (TextTiling)
- New: `spacy` sentence boundary + coreference-aware grouping
- New: Recursive character text splitter (LangChain-style, overlap-aware)
- New: Structural segmentation (respect `\n\n` paragraph breaks in raw text)

**Metric**: Paragraph-level hit@1 (does the highest-scoring paragraph contain the gold answer span?), then document MRR@10.

---

### Experiment 4: Aggregation Strategy — Rule-Based vs. Learned

**Hypothesis**: A learned aggregation function outperforms hand-crafted rules.

**Rule-based (from v1)**: max, topk_mean_k3, softmax_t1

**New learned aggregation**:
- Input: paragraph scores + [position index, paragraph length, overlap with query tokens]
- Model: small MLP (2 layers, trained on MS MARCO dev set)
- Training signal: binary relevance label per document

**Expected output**: Does a 3-feature MLP beat max-pooling? Quantifies the value of learned aggregation.

---

### Experiment 5: MS MARCO Passage Retrieval (Proper Setup)

V1 used the MS MARCO **document** corpus with manual segmentation. V2 will use MS MARCO **passage** — the standard benchmark where:
- Each passage is already a ~100-word unit
- Gold relevance is at the passage level
- Evaluation is on passage MRR@10 (standard leaderboard metric)

This gives a clean, apples-to-apples evaluation of paragraph/passage retrieval.

---

### Experiment 6: Cross-Encoder Reranking Baseline

Add a proper cross-encoder reranker (the current SOTA approach) as an upper-bound reference:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` on top of BM25 Stage 1
- Compare: paragraph bi-encoder vs. cross-encoder on identical candidate pools
- Goal: Quantify the gap between v2's best configuration and the SOTA.

---

### Experiment 7: Error Analysis & When Does Paragraph Help?

For the best-performing configuration, run a systematic analysis:
- **Query length**: Do longer, more specific queries benefit more from paragraphs?
- **Document length**: Do longer documents (more paragraphs) benefit more?
- **Query type**: Factoid vs. exploratory vs. navigational?
- **Relevance density**: Does paragraph reranking help when only part of a document is relevant?

**Implementation**: Segment query set by the above dimensions; compute delta(MRR@10) per segment.

---

## V2 Dataset Plan

| Dataset | Task | Why |
|---|---|---|
| MS MARCO Passage | Passage retrieval | Standard benchmark, passage-level labels |
| MS MARCO Document | Document retrieval | V1 baseline for comparison |
| BEIR/nfcorpus | Out-of-domain transfer | Short, dense medical abstracts |
| BEIR/scifact | OOD transfer | Claim verification, precise paragraphs matter |
| BEIR/fiqa | OOD transfer | Forum posts, noisy text |
| BEIR/arguana | OOD transfer | Argument retrieval — paragraph structure matters |

---

## V2 File Structure (Planned)

```
v2/
├── V2_OUTLINE.md              ← this file
├── docs/
│   ├── V1_GAPS.md             ← detailed v1 failure analysis
│   └── V2_HYPOTHESES.md       ← falsifiable hypotheses per experiment
├── src/
│   ├── types.py               ← extended data types
│   ├── encoders.py            ← + dense bi-encoders, cross-encoders
│   ├── segmentation.py        ← + semantic segmentation strategies
│   ├── aggregation.py         ← + learned MLP aggregator
│   ├── stage1.py              ← + BM25/DPR retrieval
│   ├── rerank.py              ← refactored reranking pipeline
│   ├── metrics.py             ← + passage-level metrics
│   └── analysis.py            ← + error analysis tools
├── configs/
│   ├── exp1_stage1_quality.json
│   ├── exp2_encoder_ablation.json
│   ├── exp3_segmentation.json
│   ├── exp4_aggregation.json
│   ├── exp5_msmarco_passage.json
│   ├── exp6_crossencoder.json
│   └── exp7_error_analysis.json
├── results/                   ← populated after experiments run
├── scripts/
│   └── download_v2_data.py
├── run_experiment.py          ← unified entry point
└── requirements.txt           ← + sentence-transformers, torch, pyserini
```

---

## Key Hypotheses to Falsify

| # | Hypothesis | How to test |
|---|---|---|
| H1 | TF-IDF is the bottleneck; dense encoders will flip the delta | Experiment 2 |
| H2 | Stage 1 recall ceiling explains paragraph underperformance | Experiment 1 |
| H3 | Cross-encoders make paragraph reranking unnecessary | Experiment 6 |
| H4 | Semantic segmentation > syntactic windows | Experiment 3 |
| H5 | Learned aggregation > rule-based aggregation | Experiment 4 |
| H6 | Paragraph reranking helps when documents are long and queries are specific | Experiment 7 |

---

## Success Criteria for V2

- At least one configuration achieves **positive delta (MRR@10 > 0)** for paragraph vs. document baseline.
- Identify the minimal set of components (encoder + segmentation + aggregation) that makes paragraphs beneficial.
- Reproduce and exceed MS MARCO passage leaderboard competitive range with dense retrieval.
- Produce a clean ablation table publishable in a course report or workshop paper.
