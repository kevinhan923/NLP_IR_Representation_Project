# Pre-registration — out-of-sample confirmation on ArguAna

**Committed BEFORE running any ArguAna retrieval.** Purpose: the paragraph
penalty and the holistic-vs-local mechanism were *discovered* on FiQA and
SciDocs; a confirmation dataset whose prediction is locked in advance is immune
to the "2 hits out of a ~40-test grid" multiplicity objection.

## Why ArguAna is a holistic-relevance dataset (the a priori label)

ArguAna is counter-argument retrieval: given an argument, find the best
counter-argument. Judging whether a passage *counters the whole argument* is a
whole-document (holistic) judgment, not a single-sentence match — the same
character as FiQA (does this whole answer resolve the question?) and SciDocs
(would this whole paper cite that one?). It is a **genuinely different domain**
(debate/argumentation) from the two discovery datasets (finance QA, citation).

## Locked predictions (E5 & BGE, agg=max, 2000-doc/200-query pool, 3 seeds)

1. **P1 — paragraph penalty.** Under E5 and BGE, ArguAna shows a *negative*
   ΔRR@10 (paragraph − document), ideally significant (paired sign-flip).
2. **P2 — encoder gate.** Under TF-IDF, ArguAna's Δ is not significant (the
   weak encoder flattens the effect, as on every other dataset).
3. **P3 — per-query locality.** `max_minus_mean` (relevant-side) positively
   predicts the per-query Δ (within-dataset Spearman > 0).
4. **P4 — distractor mechanism, out of sample.** The distractor-side
   `competitive_gap_shift` *negatively* predicts the per-query Δ (within-dataset
   Spearman < 0), replicating the mechanism found on the 6 local datasets on a
   dataset it was never fit on.

## Known risk (declared up front)

ArguAna is idiosyncratic for dense retrievers: relevant counter-arguments are
often near-duplicates of the query argument (unusually high surface overlap),
and ArguAna is atypically strong for lexical/BM25 methods. So P1 could come out
**null or even positive**, which would be a genuine boundary condition on the
holistic-penalty claim — and, because these predictions are locked, such a miss
**must be reported** as evidence against (or bounding) the mechanism, not
hidden. P4 (the per-query mechanism) is expected to be more robust than P1 (the
dataset-level sign), since the mechanism predicts *which* queries are hurt
regardless of the net sign.

## Scoring

Report a 2x2 of predicted-sign vs observed-sign for P1/P2, and the actual
within-dataset Spearman for P3/P4, in the paper's generalization section.
