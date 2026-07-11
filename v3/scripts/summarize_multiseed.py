"""summarize_multiseed.py — Aggregate stability across sampling seeds.

Reviewers' first objection to the v3 results is "everything hangs on one
seed=42 sample". This script reads the per-seed result files and reports:

  A. Controlled granularity deltas (per dataset x aggregation): mean +- std of
     ΔRR@10 across seeds, plus a per-seed paired sign-flip p — computed from
     the saved per-query rows, no retrieval re-run needed.
  B. M2 winner stability: within-dataset spearman of max_minus_mean (and its
     independent-encoder x_ twin) vs the per-query delta, per seed, on the
     selection datasets and the scidocs holdout separately. Per-seed values
     are reported WITHOUT a combined p: query samples overlap across seeds
     (drawn from the same finite pools), so independence-based meta-p-values
     (Fisher etc.) would be invalid.
  C. M3 system stability: pooled test nDCG@10 per system per seed and how many
     seeds each key significance survives.

Usage:
    python scripts/summarize_multiseed.py \
        --locality results/locality_e5.json results/locality_e5_seed7.json ... \
        --adaptive results/adaptive_e5_seed42.json ...
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.locality import within_correlations
from src.metrics import paired_randomization_test

WINNERS = ["max_minus_mean", "x_max_minus_mean"]
SELECT_DS = ["scifact", "nfcorpus", "fiqa"]
HOLDOUT_DS = ["scidocs"]


def _seed_of(config_name: str) -> int:
    m = re.search(r"_seed(\d+)$", config_name)
    return int(m.group(1)) if m else 42


def _delta_sig_from_rows(rows, delta_key, seed):
    """Paired sign-flip p on the saved per-query deltas (doc side as zeros)."""
    deltas = [r[delta_key] for r in rows]
    a = {i: 0.0 for i in range(len(deltas))}
    b = {i: d for i, d in enumerate(deltas)}
    return paired_randomization_test(a, b, seed=seed)["p_value"]


def summarize_locality(paths):
    by_seed = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        by_seed[_seed_of(res["config_name"])] = res
    seeds = sorted(by_seed)

    # --- A. controlled deltas ------------------------------------------------
    print("=== A. controlled ΔRR@10 (paragraph − document) across seeds ===")
    delta_summary = {}
    datasets = list(next(iter(by_seed.values()))["per_dataset"].keys())
    aggs = sorted(
        k.removeprefix("delta_")
        for k in next(iter(by_seed.values()))["rows"][0]
        if k.startswith("delta_")
    )
    for ds in datasets:
        delta_summary[ds] = {}
        for agg in aggs:
            vals, ps = [], []
            for s in seeds:
                res = by_seed[s]
                vals.append(res["per_dataset"][ds]["delta_rr10_by_agg"][agg])
                rows = [r for r in res["rows"] if r["dataset"] == ds]
                ps.append(_delta_sig_from_rows(rows, f"delta_{agg}", seed=s))
            delta_summary[ds][agg] = {
                "per_seed_delta": dict(zip(map(str, seeds), vals)),
                "per_seed_p": dict(zip(map(str, seeds), ps)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_seeds_significant": int(sum(p < 0.05 for p in ps)),
            }
            d = delta_summary[ds][agg]
            flags = "".join("*" if p < 0.05 else "." for p in ps)
            print(f"  {ds:>18} [{agg:>9}]: mean Δ={d['mean']:+.4f} ± {d['std']:.4f}"
                  f"  sig {d['n_seeds_significant']}/{len(seeds)} seeds [{flags}]")

    # --- B. M2 winner stability ---------------------------------------------
    print("\n=== B. M2 locality-metric stability (within-dataset spearman vs delta_max) ===")
    m2_summary = {}
    for label, ds_list in (("selection(3ds)", SELECT_DS), ("holdout(scidocs)", HOLDOUT_DS)):
        m2_summary[label] = {}
        for feat in WINNERS:
            entries = {}
            for s in seeds:
                rows = [
                    r for r in by_seed[s]["rows"]
                    if r["dataset"] in ds_list and r.get("n_segments_min", r["n_segments"]) >= 2
                ]
                c = within_correlations(rows, "delta_max", [feat])[feat]
                entries[str(s)] = {
                    "rho": c["spearman_rho"], "p": c["spearman_p"], "n": c["n"],
                }
            m2_summary[label][feat] = entries
            line = "  ".join(
                f"seed{s}: ρ={e['rho']:+.3f} (p={e['p']:.4f}, n={e['n']})"
                if e["rho"] is not None else f"seed{s}: undefined"
                for s, e in entries.items()
            )
            print(f"  {label:>16} {feat:>18}: {line}")
    return {"seeds": seeds, "controlled_delta": delta_summary, "m2_stability": m2_summary}


def summarize_adaptive(paths):
    by_seed = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        by_seed[_seed_of(res["config_name"])] = res
    seeds = sorted(by_seed)

    print("\n=== C. M3 pooled test nDCG@10 across seeds ===")
    systems = list(next(iter(by_seed.values()))["pooled"]["mean_ndcg10"].keys())
    m3 = {"pooled_ndcg10": {}, "significance_survival": {}}
    for name in systems:
        vals = [by_seed[s]["pooled"]["mean_ndcg10"][name] for s in seeds]
        m3["pooled_ndcg10"][name] = {
            "per_seed": dict(zip(map(str, seeds), vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
        print(f"  {name:>13}: " + "  ".join(f"{v:.4f}" for v in vals)
              + f"   mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    checks = [
        ("global_alpha", "vs_document"), ("global_alpha", "vs_paragraph"),
        ("rrf", "vs_document"), ("rrf", "vs_paragraph"),
        ("router_soft", "vs_document"), ("router_soft", "vs_paragraph"),
        ("router_soft", "vs_global_alpha"), ("router_soft", "vs_rrf"),
        ("oracle", "vs_global_alpha"),
    ]
    print("\n  significance survival (pooled, p<0.05):")
    for name, versus in checks:
        ps = []
        for s in seeds:
            sig = by_seed[s]["pooled"]["significance_ndcg10"].get(name, {}).get(versus)
            ps.append(sig["p_value"] if sig else None)
        flags = "".join("*" if (p is not None and p < 0.05) else "." for p in ps)
        m3["significance_survival"][f"{name}.{versus}"] = {
            "per_seed_p": dict(zip(map(str, seeds), ps)),
            "n_seeds_significant": int(sum(1 for p in ps if p is not None and p < 0.05)),
        }
        shown = "  ".join("None" if p is None else f"{p:.4f}" for p in ps)
        print(f"    {name:>12} {versus:<16} p: {shown}   [{flags}]")
    return {"seeds": seeds, **m3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locality", nargs="+", default=[])
    parser.add_argument("--adaptive", nargs="+", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summary = {}
    if args.locality:
        summary["locality"] = summarize_locality(args.locality)
    if args.adaptive:
        summary["adaptive"] = summarize_adaptive(args.adaptive)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "multiseed_summary.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
