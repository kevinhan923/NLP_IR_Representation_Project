# One Vector Is Enough: document- vs. paragraph-level retrieval under a controlled comparison

Code and result files for:

> Kevin Han. *One Vector Is Enough: A Controlled Comparison of Document- and Paragraph-Level Retrieval.* NLPVR 2026 (accepted).

## The question

A retrieval system can encode a whole document as one vector, or it can score smaller units (paragraphs, sentence windows) and aggregate their scores back to the document. Two-stage pipelines usually report that the paragraph-level route is worse. This project asks whether that penalty comes from paragraph-level representation itself or from how the comparison was set up.

## What the comparison shows

Both settings use a TF-IDF encoder followed by a Gaussian random projection and cosine scoring.

- **Two-stage setting** (`v1/`): a first stage supplies a candidate pool and only the paragraph-level system re-scores it. The two sides end up with different encoders, and recall is capped by the pool.
- **Controlled setting** (`v3/`): both systems share one encoder configuration and rank the full corpus, so granularity is the only difference left.

RR@10 (reported as MRR@10 in the two-stage runs). Δ is paragraph minus document; *p* is from a paired randomization test.

| Corpus | Two-stage Doc. | Two-stage Par. | Two-stage Δ | Controlled Doc. | Controlled Par. | Controlled Δ | *p* |
|---|---|---|---|---|---|---|---|
| MS MARCO (TREC DL 2019)† | 0.3601 | 0.1333 | −0.2268 | 0.6787 | 0.7952 | +0.1165 | 0.047 |
| NFCorpus | 0.4932 | 0.4363 | −0.0569 | 0.3927 | 0.4136 | +0.0210 | 0.21 |
| SciFact | 0.5756 | 0.4015 | −0.1742 | 0.4076 | 0.4172 | +0.0096 | 0.63 |
| FiQA | 0.1587 | 0.1301 | −0.0285 | 0.1552 | 0.1437 | −0.0115 | 0.25 |

† The two MS MARCO settings share no query set, pool or relevance threshold (10 test queries against 43), so that row is not a like-for-like pair.

The two-stage deficit does not survive the control: the gap closes on every corpus and changes sign on three of the four. No controlled difference survives Holm correction across the paper's nine significance tests.

The two settings also differ in their TF-IDF vectorizer, so the controlled comparison was rerun with the two-stage vectorizer (RR@10, Δ is paragraph minus document):

| Corpus | Doc. | Δ `max` | *p* | Δ `topk_mean` | *p* |
|---|---|---|---|---|---|
| MS MARCO | 0.5788 | −0.0468 | 0.62 | −0.0164 | 0.83 |
| NFCorpus | 0.2817 | +0.0658 | 0.003 | +0.0641 | 0.001 |
| SciFact | 0.2664 | +0.0529 | 0.023 | +0.0725 | 0.002 |
| FiQA | 0.1104 | −0.0133 | 0.21 | +0.0092 | 0.39 |

The document-level advantage still does not come back. Three of these eight tests survive Holm correction, and all three favor paragraph-level retrieval (NFCorpus with both aggregations, SciFact with `topk_mean`).

In short: the penalty attributed to paragraph-level representation is a property of the evaluation setup. Under no controlled configuration is a single vector clearly better than many; whether it is fully sufficient depends on the vectorizer.

Where the numbers come from:

| Numbers | File |
|---|---|
| Two-stage, BEIR corpora | `v1/results/beir_transfer_results.json` |
| Two-stage, MS MARCO | `v1/results/grid_full_results.json`, `v1/results/multiseed_full_results.json` |
| Controlled | `v3/results/controlled_tfidf_paper1_{small,fiqa,msmarco_hard}.json` |
| Controlled, two-stage vectorizer | `v3/results/controlled_tfidf_paper1_{small,fiqa,msmarco_hard}_matched.json` |

## Evaluation defects found along the way

1. **Two representation spaces inside one comparison.** In the two-stage code the "document-level" run is the raw sparse first-stage ranking, while only the paragraph side goes through the projected encoder, and it re-ranks only the first stage's top 100 (`v1/src/experiment_beir.py`).
2. **An MS MARCO pool that could not contain the answers.** The two-stage runs indexed the first 50,000 documents in corpus order. That pool holds 110 of the 6,524 relevant documents (1.69%), which leaves 13 of 43 queries scoring zero for any system. `v3/scripts/export_msmarco_beir.py` rebuilds a pool of the same size with every relevant document forced in.
3. **A query-numbering mismatch in the Cranfield pilot.** Queries were matched to relevance judgments by ID value although the two files number queries differently, so 152 queries were scored against the wrong judgments. After the repair, document-level MRR moves from 0.056 to 0.398. The Cranfield pilot code is not in this repository.

## Repository layout

| Path | What it is |
|---|---|
| `v1/` | The original two-stage framework (grid, multi-seed and BEIR transfer runs). Kept as it was because the paper's two-stage numbers come from it, defects included. See `v1/README.md`. |
| `v2/` | A transitional rewrite: fixed data loading and the first controlled-comparison prototype, with tests. |
| `v3/` | The controlled-comparison framework used for the paper's controlled results: config-driven runs, `ir_measures` metrics, paired randomization tests. |
| `v3/docs/FINDINGS.md` | Running record of the v3 experiments. |
| `LARGE_FILES.md` | Four BEIR corpus files over GitHub's 100 MB limit that are not in the repository. |

`v3/` also holds follow-up experiments that are **not** part of the NLPVR paper (dense encoders, a late-interaction baseline, an evidence-locality measure, adaptive granularity). They are work in progress.

## Reproducing the controlled results

Developed with Python 3.12.

```bash
# 1. BEIR data (NFCorpus, SciFact, FiQA) goes to v1/data/beir/
cd v1
python3 -m pip install ir_datasets          # the downloader imports it
python3 scripts/download_experiment_data.py --root data --download-beir --beir-datasets nfcorpus scifact fiqa

# 2. Controlled runs
cd ../v3
python3 -m pip install -r requirements.txt
python3 scripts/run_controlled.py --config configs/controlled_tfidf_paper1_small.yaml   # NFCorpus, SciFact
python3 scripts/run_controlled.py --config configs/controlled_tfidf_paper1_fiqa.yaml    # FiQA

# 3. Unit tests (no data needed)
python3 -m pytest tests/ -q
```

Each run prints the document- and paragraph-level scores with the paired test, and writes `v3/results/<config name>.json`. The result files are tracked, so `git diff` after a run shows whether the numbers reproduce. Use the `*_matched.yaml` configs for the two-stage-vectorizer rerun. The MS MARCO run first needs the rebuilt pool from `v3/scripts/export_msmarco_beir.py`, which downloads TREC DL 2019 through `ir_datasets`.

## Citation

```bibtex
@inproceedings{han2026onevector,
  author    = {Han, Kevin},
  title     = {One Vector Is Enough: A Controlled Comparison of Document- and Paragraph-Level Retrieval},
  booktitle = {International Conference on Natural Language Processing, Vision and Robotics (NLPVR 2026)},
  address   = {Hong Kong, China},
  year      = {2026},
  note      = {Accepted}
}
```
