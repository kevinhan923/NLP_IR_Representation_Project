"""analyze_locality.py — Held-out validation of the locality metric (M2).

Reads the per-query rows saved by run_locality.py and answers the
multiple-comparison objection head-on: with ~14 candidate features x several
tables, the best in-sample p-value is meaningless on its own. So:

  1. SELECT the winning feature on the selection datasets only
     (default: the three original M1 datasets), ranked by |spearman| of the
     dataset-demeaned (within-dataset) correlation;
  2. CONFIRM that single pre-chosen feature on the held-out datasets —
     one confirmatory test, no forking paths;
  3. report Holm-Bonferroni-adjusted significance for the full in-sample
     family as a supplementary table;
  4. print dataset-level means of the winner (does it also order the
     dataset-level penalties?) and its quartile-binned delta curve.

Usage:
    python scripts/analyze_locality.py --results results/locality_e5.json \
        --select-on scifact,nfcorpus,fiqa \
        --confirm-on scidocs,trec-covid,webis-touche2020
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.locality import binned_means, demean_by_dataset, within_correlations


def _fmt(c):
    if c["spearman_rho"] is None:
        return f"n={c['n']} (undefined)"
    return (f"spearman={c['spearman_rho']:+.3f} (p={c['spearman_p']:.2e})  "
            f"pearson={c['pearson_r']:+.3f} (p={c['pearson_p']:.2e})  n={c['n']}")


def holm(table):
    """Holm-Bonferroni over the spearman p-values of a correlation table."""
    tested = [(k, c["spearman_p"]) for k, c in table.items() if c["spearman_p"] is not None]
    tested.sort(key=lambda kv: kv[1])
    m = len(tested)
    out, running_max = {}, 0.0
    for i, (k, p) in enumerate(tested):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        out[k] = running_max
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--select-on", default="scifact,nfcorpus,fiqa")
    parser.add_argument("--confirm-on", default="scidocs,trec-covid,webis-touche2020")
    parser.add_argument("--delta-key", default=None, help="default: delta_<primary_agg>")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        res = json.load(f)
    delta_key = args.delta_key or f"delta_{res['primary_agg']}"
    select_ds = [d.strip() for d in args.select_on.split(",") if d.strip()]
    confirm_ds = [d.strip() for d in args.confirm_on.split(",") if d.strip()]

    rows = [r for r in res["rows"] if r["n_segments_min"] >= 2]
    feature_keys = [k for k in rows[0] if k not in
                    ("dataset", "qid", "doc_rr10", "n_rel_docs", "n_segments_min")
                    and not k.startswith("delta_")]

    sel_rows = [r for r in rows if r["dataset"] in select_ds]
    con_rows = [r for r in rows if r["dataset"] in confirm_ds]
    print(f"rows (n_segments_min>=2): total={len(rows)}  "
          f"selection={len(sel_rows)} {select_ds}  confirmation={len(con_rows)} {confirm_ds}")

    # --- 1. selection: rank features on the selection datasets only ---------
    sel_table = within_correlations(sel_rows, delta_key, feature_keys)
    ranked = sorted(
        (kv for kv in sel_table.items() if kv[1]["spearman_rho"] is not None),
        key=lambda kv: -abs(kv[1]["spearman_rho"]),
    )
    print(f"\n=== selection (within-dataset, {'+'.join(select_ds)}) — top 5 by |spearman| ===")
    for k, c in ranked[:5]:
        print(f"  {k:>20}: {_fmt(c)}")
    winner = ranked[0][0]
    print(f"\n  WINNER (pre-registered for confirmation): {winner}")

    # --- 2. one confirmatory test on held-out datasets ----------------------
    print(f"\n=== confirmation of '{winner}' on held-out datasets ===")
    if con_rows:
        con_table = within_correlations(con_rows, delta_key, [winner])
        print(f"  pooled within-dataset: {_fmt(con_table[winner])}")
    for ds in confirm_ds:
        ds_rows = [r for r in con_rows if r["dataset"] == ds]
        if len(ds_rows) < 10:
            print(f"  {ds:>18}: only {len(ds_rows)} usable rows — skipped")
            continue
        c = within_correlations(ds_rows, delta_key, [winner])[winner]
        print(f"  {ds:>18}: {_fmt(c)}")

    # --- 3. supplementary: full family with Holm correction (all datasets) --
    full_table = within_correlations(rows, delta_key, feature_keys)
    adj = holm(full_table)
    print("\n=== full family, all datasets, within-dataset (Holm-adjusted p) ===")
    for k, c in sorted(full_table.items(), key=lambda kv: kv[1]["spearman_p"] or 1.0):
        if c["spearman_rho"] is None:
            continue
        star = "*" if adj.get(k, 1.0) < 0.05 else " "
        print(f"  {k:>20}: {_fmt(c)}  holm_p={adj.get(k, 1.0):.2e}{star}")

    # --- 4. winner descriptives ---------------------------------------------
    print(f"\n=== '{winner}' dataset-level means vs dataset delta ===")
    for ds in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        mean_w = sum(r[winner] for r in ds_rows) / len(ds_rows)
        mean_d = sum(r[delta_key] for r in ds_rows) / len(ds_rows)
        print(f"  {ds:>18}: mean {winner}={mean_w:+.4f}  mean {delta_key}={mean_d:+.4f}  n={len(ds_rows)}")

    print(f"\n=== quartile-binned {delta_key} by '{winner}' (dataset-demeaned delta) ===")
    demeaned_delta = demean_by_dataset(rows, delta_key)
    for b in binned_means([r[winner] for r in rows], demeaned_delta, n_bins=4):
        print(f"  [{b['x_low']:+.3f}, {b['x_high']:+.3f}]  mean demeaned Δ={b['mean_y']:+.4f}  n={b['count']}")


if __name__ == "__main__":
    main()
