# v3 Experimental Findings (working draft for the paper)

> Controlled document-level vs paragraph-level retrieval. Both systems share the
> SAME encoder; only granularity differs. Evaluation via ir_measures with paired
> sign-flip significance tests. All numbers from real local BEIR data.

## Setup

- **Datasets**: scifact (short abstracts), nfcorpus (medical), fiqa (financial QA, long docs).
- **Pool / queries**: n_docs=2000, n_queries=200 per dataset (small-pool caveat below).
- **Encoders (same config on both granularities)**:
  - TF-IDF → Gaussian Random Projection (r=192) → L2  (the paper's original encoder)
  - E5-base-v2 (dense bi-encoder, asymmetric query/passage prefixes)
  - BGE-base-en-v1.5 (dense bi-encoder)
- **Segmentation**: 2-sentence windows, stride 1. **Aggregation**: max (and topk_mean, k=3).
- **Metrics**: RR@10, nDCG@10, R@100. **Significance**: paired sign-flip randomization on per-query RR@10 + bootstrap CI.

## Main result (agg = max, ΔRR@10 = paragraph − document)

| Dataset | TF-IDF Δ (p) | E5 Δ (p) | BGE Δ (p) | Consistent reading |
|---|---|---|---|---|
| scifact | +0.020 (0.49) | −0.000 (0.98) | +0.010 (0.52) | no significant difference |
| nfcorpus | +0.043 (0.07) | +0.032 (0.11) | +0.021 (0.27) | consistently positive, never significant |
| fiqa | −0.004 (0.87) | **−0.083 (0.000)** | **−0.061 (0.002)** | **document significantly better (strong encoders)** |

Absolute quality (document-level RR@10) rises sharply with encoder strength, e.g.
fiqa: TF-IDF 0.37 → E5 0.77 → BGE 0.76.

### Generalization across 6 datasets (E5, ordered by avg document length)

| Dataset | avg words | doc RR@10 | ΔRR@10 (max) | p | n | verdict |
|---|---|---|---|---|---|---|
| fiqa | 132 | 0.766 | −0.083 | 0.000 | 200 | paragraph WORSE |
| scidocs | 174 | 0.636 | −0.084 | 0.000 | 200 | paragraph WORSE |
| scifact | 213 | 0.776 | −0.000 | 0.98 | 200 | no diff |
| trec-covid | 225 | 0.922 | −0.003 | 1.00 | 50 | no diff (ceiling) |
| nfcorpus | 235 | 0.497 | +0.032 | 0.11 | 197 | no diff |
| webis-touche2020 | 316 | 0.990 | +0.000 | 1.00 | 49 | no diff (ceiling) |

**Length is conclusively NOT the driver**: the two *shortest* corpora (fiqa 132,
scidocs 174) are exactly the ones with a significant paragraph penalty; the
longest (touche 316) shows none.

**The two penalized datasets share a "holistic relevance" character**: fiqa (does
this whole forum answer resolve the question?) and scidocs (would this paper cite
that paper — a whole-topic judgment). Relevance needs the *whole* document, so
splitting into sentences + max aggregation gets misled by a locally-similar but
off-topic fragment. The no-difference datasets (scifact claim-check, trec-covid
topical match) have more *local* relevance — a single passage can decide it.

⚠️ **Ceiling caveat**: trec-covid (doc RR@10=0.92) and touche (0.99) are too easy
in this small pool (and have only ~50 queries), so their "no difference" is
unreliable — there is no headroom to detect a granularity effect. The trustworthy
contrasts are fiqa/scidocs (sig. negative) vs scifact/nfcorpus (mid-difficulty, no sig).

## Findings

1. **Encoder strength gates the granularity effect.** Under TF-IDF, all three
   datasets show no significant difference — the weak encoder flattens the signal.
   This is exactly the regime of the original paper, and explains why it could not
   draw a reliable granularity conclusion.
2. **Strong encoders reveal a dataset-dependent pattern.** With E5 and BGE,
   fiqa (financial QA) shows a *significant* paragraph penalty, reproduced
   independently across both encoders. scifact shows none; nfcorpus leans positive
   but never reaches significance.
   ⚠️ **Document length is NOT the driver.** fiqa averages only ~132 words — among
   the *shortest* corpora here (scifact 213, nfcorpus 235, trec-covid 225,
   webis-touche2020 316) — yet shows the strongest penalty. So the earlier
   "long documents penalize paragraphs" framing is WRONG. The real variable is
   more likely whether relevance needs *holistic* judgment (does this whole answer
   address the question?) vs *local* term match. This is exactly what the M2
   evidence-locality metric should capture; a generalization run over scidocs /
   trec-covid / webis-touche2020 (currently in progress) will test it.
3. **Internal consistency.** On fiqa the penalty is smaller for topk_mean than for
   max (e.g. E5: −0.071 vs −0.083), consistent with max being more easily misled by
   a single high-scoring but out-of-context segment.

## Proposed paper thesis (stronger than the original "it depends")

> The document- vs paragraph-level granularity effect is **conditional on encoder
> strength and on whether relevance is holistic or local** (NOT on document length).
> A weak encoder (TF-IDF, as in the original study) masks the effect entirely;
> strong encoders reveal that some datasets (fiqa) are significantly hurt by
> paragraph segmentation while others (scifact) are unaffected — and this does not
> track document length. The pattern reproduces across E5 and BGE and passes paired
> significance testing. M2 will quantify the true driver (evidence locality).

## M2: evidence locality predicts the per-query granularity delta (2026-07-10)

> Setup: same pools/seed as M1 (aggregates reproduce M1 exactly — built-in sanity
> check passed). For every judged query, locality features are computed from the
> query-vs-segment similarity distribution of its relevant docs (E5, same as
> retrieval) AND independently with MiniLM (`x_` features) to rule out mechanical
> same-encoder correlation. Analysis on n_segments_min ≥ 2 rows (every rel doc
> actually splits), within-dataset (dataset-demeaned) correlations only.
> Code: `src/locality.py`, `scripts/run_locality.py`, `scripts/analyze_locality.py`;
> data: `results/locality_e5.json` (per-query rows included).

**Protocol against feature-shopping**: winner selected on the three original
datasets only (scifact/nfcorpus/fiqa, n=541), then ONE confirmatory test on the
held-out datasets; full family reported with Holm correction as supplement.

- **Winner: `max_minus_mean`** — how far the best segment stands above the doc's
  mean segment similarity (raw cosine gap). Selection: ρ=+0.248, p=4.7e-9.
- **Held-out confirmation (scidocs)**: ρ=+0.168, p=0.054 — directionally
  consistent, marginal (n=132; trec-covid/touche unusable: 2/9 rows survive the
  multiseg filter). Honest reading: replicates in direction, needs more data for
  a clean p<0.05.
- **Direction as hypothesized**: LOW gap (flat distribution = holistic relevance)
  → paragraph penalty; HIGH gap (one segment stands out = local evidence) →
  paragraph-safe. Quartile-binned demeaned Δ: bottom −0.091 vs top +0.108
  (~0.2 RR@10 swing), middle quartiles noisy (not a clean dose-response).
- **Not mechanical**: the independent-encoder versions stay significant —
  `x_max_minus_mean` ρ=+0.136 (p=3.8e-4, Holm 4.9e-3), `x_peakedness` ρ=+0.109
  (Holm 0.046). Since under agg=max the same-encoder max IS the paragraph score,
  **cite the `x_` numbers as the headline** to preempt the circularity objection.
- **Holm-surviving family** (within-dataset, n=684): max_minus_mean, peakedness,
  and both `x_` counterparts. **Failed candidates**: entropy_locality (ρ=−0.03!),
  gini, top1_share — the z-softmax entropy is too coarse and n-dependent; the
  plan's original "熵/基尼" operationalization is refuted, the simple raw-gap
  family wins.
- **Length/segment-count again not the driver**: doc_words and n_segments have
  significant Pearson but n.s. Spearman (rank-fragile) and lose to the winner.

⚠️ **Open puzzle for the paper**: dataset-level MEAN locality does NOT order the
dataset-level penalties (fiqa has the *highest* mean gap of the selection set yet
the biggest penalty). The metric predicts *which queries within a dataset* get
hurt, but the cross-dataset mechanism (why fiqa/scidocs as a whole) is NOT just
"lower average locality". Candidate explanations: distractor-side effects (a
flat rel-doc distribution hurts only when competing docs have peaked ones), or
qrels character. Do not claim "holistic datasets have lower mean locality" — the
data contradicts it; claim per-query predictive power.

## M3: adaptive granularity (2026-07-11)

> Setup: E5, same pools/seed as M1/M2 (fixed-baseline numbers reproduce M1
> bit-for-bit). Per-dataset 50/50 query split; ONE ridge router and ONE global
> fusion weight fit on the pooled 449 train queries (features: candidate-side
> peakedness = M2's signal applied to top-10 doc-level candidates, side margins,
> rank agreement, query length — computable at query time, no qrels). Evaluation
> on 447 held-out test queries, paired sign-flip significance on nDCG@10.
> Code: `src/adaptive.py`, `scripts/run_adaptive.py`; results:
> `results/adaptive_e5.json` (per-query nDCG for all systems included).

**Gate 3 verdict — PASSED, with an honest asterisk.** Pooled test (n=447):

| system | nDCG@10 | vs document | vs paragraph | vs global fusion |
|---|---|---|---|---|
| document (fixed) | 0.5865 | — | +0.0178* | |
| paragraph (fixed) | 0.5687 | −0.0178* | — | |
| global-α fusion (α=0.55) | 0.6014 | **+0.0150 (p=.005)** | **+0.0327 (p<1e-4)** | — |
| router_soft | **0.6042** | **+0.0178 (p=.0003)** | **+0.0355 (p<1e-4)** | +0.0028 (p=.28) |
| oracle (upper bound) | 0.6299 | +0.0435 (p<1e-4) | +0.0612 (p<1e-4) | **+0.0285 (p<1e-4)** |

1. **Granularity complementarity is real and exploitable (the headline).** Even
   ONE static fusion weight significantly beats BOTH fixed granularities pooled;
   router_soft does too, with slightly larger margins. This satisfies the plan's
   Gate 3 criterion (同时 ≥ 两个固定粒度, 显著).
2. **"Adaptive beats static fusion" is NOT established** (+0.0028, p=0.28). Do
   not claim it. The oracle's +0.0285 (p<1e-4) over static fusion shows the
   per-query headroom exists — the current 8-feature router captures ~10% of it.
   Better routers/features are the obvious future-work lever.
3. **Where routing DOES add value: worst-case robustness.** Static fusion loses
   significantly to the document side on scidocs (−0.019, p=0.026); both router
   variants remove that significant loss (router_hard −0.001 p=1.0). On touche
   it shrinks the loss vs paragraph (−0.051 sig → −0.038). Frame the router as
   insurance against fusion's per-dataset failure modes, not as a mean-lift.
4. Per-dataset router_soft: significantly better than document on scifact and
   nfcorpus, than paragraph on fiqa/scidocs; never significantly below BOTH
   fixed sides anywhere; weakest spots: touche vs paragraph (−0.038, p=0.03,
   n=24) and scidocs vs document (−0.010, n.s.).
5. **Cost note for the paper**: fusion/routing requires running both systems
   (2× index + 2× query encode); the router itself is negligible (ridge on 8
   features). Oracle-vs-fusion headroom may justify the cost story.

## Multi-seed stability (2026-07-12; seeds 42/7/123, full rerun of M1+M2+M3)

> The single-seed objection is now answered. All numbers:
> `results/*_seed{7,123}.json` + `results/multiseed_summary.json`
> (per-seed paired sign-flip p computed from saved per-query rows).

**Controlled ΔRR@10 (paragraph − document), mean ± std over 3 seeds:**

| dataset | agg=max | sig seeds | verdict |
|---|---|---|---|
| fiqa | −0.066 ± 0.013 | **3/3** | paragraph penalty: rock solid |
| scidocs | −0.058 ± 0.020 | 2/3 (direction 3/3) | real, seed-sensitive at n=200 |
| scifact | −0.007 ± 0.009 | 0/3 (topk: 1/3 neg) | no consistent effect |
| nfcorpus | +0.016 ± 0.016 | 0/3 | positive lean, never significant |
| trec-covid | +0.019 ± 0.029 | 0/3 (topk 1/3 pos) | ceiling noise — unusable |
| touche | +0.003 ± 0.005 | 0/3 | no effect (ceiling) |

**M2 metric stability**: selection-set within-dataset spearman of
`max_minus_mean` = +0.198 / +0.248 / +0.210, all p<1e-4 — very stable.
Held-out scidocs: +0.134 / +0.168 / +0.251, all positive, seed123 p=0.003 —
the earlier single-seed p=0.054 concern is resolved (report per-seed values;
do NOT Fisher-combine, query samples overlap across seeds). Independent-encoder
x_ version: selection 3/3 positive (2/3 sig); holdout positive but weak (n.s.).

**M3 stability (pooled test nDCG@10 across seeds)**:

- **router_soft is the ONLY system that significantly beats BOTH fixed
  granularities in ALL 3 seeds** (vs doc p=0.0007/0.0003/0.0039; vs par all
  p<1e-4; mean 0.6122 ± 0.0095 — also the best non-oracle mean in every seed).
- global_alpha fails vs document in 1/3 seeds (p=0.09); RRF fails in 2/3.
  → The per-query router earns its keep as *reliability*, not mean-lift:
  static fusions are seed/dataset-fragile, the router is not.
- router_soft vs global_alpha: n.s. in 3/3 (unchanged honest caveat);
  vs RRF: significant in 2/3. oracle headroom over fusion: 3/3 significant.

## Limitations / open items

- **Small-pool caveat — CONFIRMED ROBUST (2026-06-29)**: n_docs=2000 does inflate
  absolute scores. At a 5× larger pool (n_docs=10000), fiqa document-level RR@10
  drops 0.77→0.63 (harder, more realistic), but the paragraph penalty *strengthens*
  and stays significant: max Δ=−0.089 (p=0.000), topk_mean Δ=−0.054 (p=0.028). The
  fiqa penalty is **not** a small-pool artifact — it is larger under harder retrieval.
- The significant signal currently rests mainly on **one dataset (fiqa)**. More
  long-document datasets are needed to establish "long docs penalize paragraphs"
  as a general rule.
- Only one segmentation (2-sentence windows) and two aggregations tested so far.
- nfcorpus's consistent-but-insignificant positive lean is unresolved (needs larger n).

## Status log

- 2026-06-29: M0 (TF-IDF) + M1 (E5, BGE) complete on 3 datasets, n_docs=2000.
- 2026-06-29: large-pool fiqa (E5, n_docs=10000) DONE — paragraph penalty robust
  and slightly stronger (max Δ=−0.089, p<0.001; topk_mean Δ=−0.054, p=0.028).
- 2026-06-29: +3 datasets (E5): scidocs sig. negative (Δ=−0.084, p<0.001) like
  fiqa; trec-covid & touche no diff (but ceiling-limited). Length disproven as
  driver; "holistic vs local relevance" is the candidate driver → M2.
- 2026-07-10: M2 DONE (E5 + independent MiniLM check, 6 datasets). Winner
  `max_minus_mean` selected on original 3 datasets (ρ=+0.25, p=5e-9), held-out
  scidocs confirms directionally (ρ=+0.17, p=0.054); cross-encoder `x_` versions
  survive Holm — correlation not mechanical. Entropy/gini candidates refuted.
  Dataset-level mean locality does NOT explain which datasets get penalized
  (per-query signal only) → open question for M3/paper framing.
- 2026-07-11: M3 DONE (E5, 6 datasets, pooled router). Gate 3 passed: fusion and
  router_soft both significantly beat the two fixed granularities pooled
  (router_soft +0.018/+0.036, p≤3e-4). Router does NOT significantly beat static
  fusion (+0.003, p=0.28) but removes fusion's significant scidocs failure;
  oracle shows +0.029 (p<1e-4) headroom above fusion for better routers.
