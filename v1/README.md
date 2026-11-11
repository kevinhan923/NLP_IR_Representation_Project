> This is the original README of the first-generation (two-stage) framework. Commands are run from inside `v1/`, and some of them refer to helper scripts that now live under `v1/scripts/`. The setup has known evaluation defects, documented in the top-level `README.md`; it is kept unchanged because the paper's two-stage numbers come from it.

# Large-Scale Paragraph Retrieval Experiments

This package upgrades the earlier Cranfield paragraph-level project into a larger experimental framework.

## What is included

- `run_grid_experiment.py`
  - Large-scale segmentation × aggregation grid study.
- `run_multiseed_experiment.py`
  - Multi-seed repeated-run stability study.
- `run_beir_transfer.py`
  - Cross-dataset transfer study on selected BEIR datasets.
- `download_experiment_data.sh`
  - One-command helper for downloading or warming the required datasets locally.
- `scripts/download_experiment_data.py`
  - Python downloader / warm-up script for MS MARCO through `ir_datasets` and BEIR through official dataset zips.
- `src/`
  - Shared code for segmentation, aggregation, local stage-1 retrieval, reranking, metrics, grouped analysis, and query-level comparison.
- `configs/`
  - Default configs for fast and full runs.

## Main design choices

### Main large-scale study
The package is set up around a large retrieval study with:
- a **stage-1 retriever**
- a **stage-2 paragraph reranker**
- a **segmentation × aggregation grid**
- a **multi-seed repeated-run study**

### Why MS MARCO document by default
The original large-scale plan mentioned MS MARCO passage, but paragraph-level segmentation is more meaningful on document-like text than on already short passages. Because of that, the default configs use an **MS MARCO document** dataset identifier.

### Stage-1 retrieval
The default implementation uses a **local TF-IDF retriever** for stage 1. This keeps the project runnable in plain Python. It is computationally heavy on larger corpora, which is useful for long-running experiments.

### Stage-2 reranking
The reranker compares:
- a **single-vector baseline** from stage-1 scores
- a **paragraph-level reranker** that splits each candidate into segments, scores all segments, and aggregates the segment scores back to the document level

## Default experiments

### 1. Grid experiment
This tests multiple segmentation and aggregation settings.

Default segmentations:
- natural paragraphs
- 2-sentence windows, stride 1
- 3-sentence windows, stride 1
- 2-sentence windows, stride 2
- 64-token chunks
- 128-token chunks
- 256-token chunks

Default aggregations:
- max
- top-k mean (k=3,5)
- weighted top-k (k=3,5)
- softmax-weighted
- threshold mean

### 2. Multi-seed experiment
This repeats the full pipeline with multiple random seeds, so the results do not depend on one split only.

### 3. BEIR transfer experiment
This reuses a selected setting on several BEIR datasets.

## Quick start

### Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

### Download or warm the data
```bash
bash download_experiment_data.sh ./data
```

### Run the fast grid experiment
```bash
python3 run_grid_experiment.py --fast
```

### Run the full grid experiment
```bash
python3 run_grid_experiment.py
```

### Run the fast multi-seed experiment
```bash
python3 run_multiseed_experiment.py --fast
```

### Run the full multi-seed experiment
```bash
python3 run_multiseed_experiment.py
```

### Run the BEIR transfer experiment
```bash
python3 run_beir_transfer.py
```

## Runtime expectations
These are rough CPU-side expectations only. Real times depend heavily on your machine and the dataset cache state.

### Fast configs
- Grid experiment: around **20 minutes to 2 hours**
- Multi-seed experiment: around **1 to 6 hours**

### Full configs
- Grid experiment: around **2 to 10+ hours**
- Multi-seed experiment: around **many hours to multiple days**

If you want truly long runs, increase:
- `max_docs`
- `max_queries`
- `candidate_pool`
- the number of seeds

## Important note
Because this package depends on external public datasets and `ir_datasets` identifiers, local download behavior may vary by environment. If an MS MARCO dataset identifier changes in your local installation, run:

```bash
python3 -c "import ir_datasets; print([x for x in ir_datasets.registry._registered if 'msmarco-document' in x][:50])"
```

and then update the config JSON accordingly.


## Paragraph-Hypothesis Transfer Runs

To test settings that may legitimately help paragraph-level reranking on BEIR, use:

```bash
python3 run_beir_paragraph_hypothesis.py
```

To sweep multiple paragraph-friendly settings:

```bash
python3 run_beir_paragraph_sweep.py --limit 12
```

The sweep compares multiple segmentation, aggregation, encoder, and candidate-pool choices and ranks them by average paragraph minus single-vector MRR@10 delta across the configured BEIR datasets.
