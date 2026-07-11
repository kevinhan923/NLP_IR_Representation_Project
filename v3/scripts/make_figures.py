"""make_figures.py — Paper figures from saved result JSONs (no re-running).

Reads results/ and writes PNG+PDF into figures/:

  fig1_controlled_deltas   ΔRR@10 (paragraph − document) per dataset, one bar
                           group per encoder, bootstrap CI whiskers.
  fig2_locality_binned     mean dataset-demeaned Δ per quartile of the M2
                           winner feature (max_minus_mean), pooled multiseg rows.
  fig3_adaptive_systems    pooled test nDCG@10 per M3 system, mean ± std
                           across seeds.

Usage:
    python scripts/make_figures.py
"""

import json
import os
import sys
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.locality import binned_means, demean_by_dataset

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(V3, "results")
FIGURES = os.path.join(V3, "figures")

DATASET_ORDER = ["scifact", "nfcorpus", "fiqa", "scidocs", "trec-covid", "webis-touche2020"]
ENCODER_FILES = [
    ("TF-IDF", "controlled_tfidf.json"),
    ("E5", "controlled_dense_e5.json"),
    ("BGE", "controlled_dense_bge.json"),
    ("MiniLM", "controlled_dense_minilm.json"),
]


def _load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{stem}.png/.pdf")


def fig1_controlled_deltas() -> None:
    encoders = [(label, _load(fname)) for label, fname in ENCODER_FILES]
    encoders = [(l, r) for l, r in encoders if r]
    # extra E5 datasets live in the longdocs file
    extra = _load("controlled_dense_e5_longdocs.json")

    def cells(res):
        out = {}
        for ds, r in res["results"].items():
            a = r["paragraph_by_agg"]["max"]
            s = a["significance_rr10"]
            out[ds] = (a["delta_rr10"], s["ci_low"], s["ci_high"], s["p_value"])
        return out

    data = {label: cells(res) for label, res in encoders}
    if extra and "E5" in data:
        data["E5"].update(cells(extra))

    datasets = [d for d in DATASET_ORDER if any(d in v for v in data.values())]
    labels = list(data)
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, label in enumerate(labels):
        xs, ys, lo, hi, stars = [], [], [], [], []
        for j, ds in enumerate(datasets):
            if ds not in data[label]:
                continue
            d, cl, ch, p = data[label][ds]
            xs.append(j + (i - (len(labels) - 1) / 2) * width)
            ys.append(d)
            lo.append(d - cl)
            hi.append(ch - d)
            stars.append(p < 0.05)
        bars = ax.bar(xs, ys, width=width * 0.9, label=label,
                      yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
        for x, y, s in zip(xs, ys, stars):
            if s:
                ax.annotate("*", (x, y + (0.012 if y >= 0 else -0.03)),
                            ha="center", fontsize=11)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylabel("ΔRR@10 (paragraph − document)")
    ax.set_title("Granularity effect by dataset and encoder (agg=max; * p<0.05)")
    ax.legend(ncols=len(labels), fontsize=9)
    _save(fig, "fig1_controlled_deltas")


def fig2_locality_binned() -> None:
    res = _load("locality_e5.json")
    if not res:
        print("  skip fig2 (locality_e5.json missing)")
        return
    rows = [r for r in res["rows"] if r.get("n_segments_min", r["n_segments"]) >= 2]
    demeaned = demean_by_dataset(rows, "delta_max")
    bins = binned_means([r["max_minus_mean"] for r in rows], demeaned, n_bins=4)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    xs = np.arange(len(bins))
    ax.bar(xs, [b["mean_y"] for b in bins], width=0.62)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"Q{i+1}\n[{b['x_low']:.3f},{b['x_high']:.3f}]" for i, b in enumerate(bins)],
        fontsize=8,
    )
    for x, b in zip(xs, bins):
        ax.annotate(f"n={b['count']}", (x, b["mean_y"]),
                    ha="center", va="bottom" if b["mean_y"] >= 0 else "top", fontsize=8)
    ax.set_xlabel("max_minus_mean quartile (evidence locality)")
    ax.set_ylabel("mean dataset-demeaned ΔRR@10")
    ax.set_title("Low evidence locality → paragraph penalty (E5, pooled)")
    _save(fig, "fig2_locality_binned")


def fig3_adaptive_systems() -> None:
    files = sorted(glob(os.path.join(RESULTS, "adaptive_e5_seed*.json")))
    if not files:
        print("  skip fig3 (no adaptive seed files)")
        return
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    systems = ["document", "paragraph", "rrf", "global_alpha",
               "router_hard", "router_soft", "oracle"]
    systems = [s for s in systems if s in runs[0]["pooled"]["mean_ndcg10"]]
    means = {s: [r["pooled"]["mean_ndcg10"][s] for r in runs] for s in systems}

    fig, ax = plt.subplots(figsize=(7, 3.8))
    xs = np.arange(len(systems))
    ys = [float(np.mean(means[s])) for s in systems]
    errs = [float(np.std(means[s])) for s in systems]
    colors = ["#888" if s in ("document", "paragraph") else
              "#d62728" if s == "oracle" else "#1f77b4" for s in systems]
    ax.bar(xs, ys, yerr=errs, capsize=3, color=colors, width=0.62)
    ax.set_xticks(xs)
    ax.set_xticklabels(systems, rotation=15, fontsize=9)
    ax.set_ylabel("pooled test nDCG@10")
    lo = min(ys) - (0.05 if len(runs) == 1 else 3 * max(errs))
    ax.set_ylim(max(0.0, lo), None)
    ax.set_title(f"M3 systems, mean ± std over {len(runs)} seed(s)")
    _save(fig, "fig3_adaptive_systems")


if __name__ == "__main__":
    print("generating figures ...")
    fig1_controlled_deltas()
    fig2_locality_binned()
    fig3_adaptive_systems()
