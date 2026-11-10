"""make_figures.py — publication-quality paper figures from saved result JSONs.

Reads results/ and writes PDF+PNG into figures/ at final size with a shared
professional style (no re-running retrieval):

  fig1_controlled_deltas   ΔRR@10 (paragraph − document) per dataset, grouped by
                           encoder. Encoders use an ORDINAL blue ramp (light=weak
                           TF-IDF → dark=strong BGE), so the gating story reads
                           off the color: stronger encoders (darker) go more
                           negative on the holistic datasets.
  fig2_locality_binned     two panels — dataset-demeaned ΔRR@10 binned by the
                           relevant-side metric (max_minus_mean) vs the stronger
                           distractor-side metric (competitive_gap_shift).
  fig3_adaptive_systems    pooled test nDCG@10 per system, mean ± std over seeds;
                           baselines (grey), fusions/routers (blue), oracle (red).

Palettes validated colorblind-safe with the dataviz skill's validator.

Usage:  python scripts/make_figures.py
"""
from __future__ import annotations

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

# ---- shared style -------------------------------------------------------- #
INK = "#0b0b0b"          # text-primary
INK2 = "#52514e"         # text-secondary (axes, labels)
GRID = "#e7e6e2"         # recessive grid
SURFACE = "white"
ENCODER_RAMP = ["#9ec5f4", "#3987e5", "#256abf", "#0d366b"]  # weak→strong, --ordinal ok
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
GREY = "#b3b1a8"

DATASET_LABEL = {
    "fiqa": "FiQA", "scidocs": "SciDocs", "scifact": "SciFact",
    "nfcorpus": "NFCorpus", "trec-covid": "TREC-COVID",
    "webis-touche2020": "Touché", "arguana": "ArguAna",
}
DATASET_ORDER = ["fiqa", "scidocs", "scifact", "nfcorpus", "trec-covid", "webis-touche2020"]
ENCODER_FILES = [
    ("TF-IDF", "controlled_tfidf.json"),
    ("MiniLM", "controlled_dense_minilm.json"),
    ("E5", "controlled_dense_e5.json"),
    ("BGE", "controlled_dense_bge.json"),
]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "semibold",
        "axes.labelsize": 8, "axes.labelcolor": INK2,
        "axes.edgecolor": INK2, "axes.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _load(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def _grid(ax, axis="y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, stem) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"),
                    bbox_inches="tight", facecolor=SURFACE, pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf/.png")


# -------------------------------------------------------------------------- #
def fig1_controlled_deltas() -> None:
    encoders = [(l, _load(f)) for l, f in ENCODER_FILES]
    encoders = [(l, r) for l, r in encoders if r]
    extra = _load("controlled_dense_e5_longdocs.json")
    arg = {l: _load(f"controlled_arguana_{k}.json")
           for l, k in [("TF-IDF", "tfidf"), ("E5", "e5"), ("BGE", "bge")]}

    def cells(res):
        out = {}
        for ds, r in res["results"].items():
            a = r["paragraph_by_agg"]["max"]
            s = a["significance_rr10"]
            out[ds] = (a["delta_rr10"], s["ci_low"], s["ci_high"], s["p_value"])
        return out

    data = {l: cells(r) for l, r in encoders}
    if extra and "E5" in data:
        data["E5"].update(cells(extra))
    for l, r in arg.items():
        if r and l in data:
            data[l].update(cells(r))

    order = DATASET_ORDER + ["arguana"]
    datasets = [d for d in order if any(d in v for v in data.values())]
    labels = [l for l, _ in ENCODER_FILES if l in data]
    colors = dict(zip(labels, ENCODER_RAMP))
    n = len(labels)
    width = 0.82 / n

    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    _grid(ax)
    for i, label in enumerate(labels):
        xs, ys, lo, hi, stars = [], [], [], [], []
        for j, ds in enumerate(datasets):
            if ds not in data[label]:
                continue
            d, cl, ch, p = data[label][ds]
            xs.append(j + (i - (n - 1) / 2) * width)
            ys.append(d); lo.append(d - cl); hi.append(ch - d); stars.append(p < 0.05)
        ax.bar(xs, ys, width=width * 0.92, color=colors[label], label=label,
               yerr=[lo, hi], capsize=1.6,
               error_kw={"lw": 0.7, "ecolor": INK2, "alpha": 0.9}, zorder=3)
        for x, y, s in zip(xs, ys, stars):
            if s:
                ax.annotate("$\\ast$", (x, y - 0.004 if y < 0 else y + 0.004),
                            ha="center", va="top" if y < 0 else "bottom",
                            fontsize=7, color=INK)
    ax.axhline(0, color=INK, lw=0.8, zorder=4)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets])
    ax.set_ylabel("$\\Delta$RR@10  (paragraph $-$ document)")
    ax.tick_params(length=0)
    ax.legend(title="encoder (weak$\\rightarrow$strong)", ncols=n,
              loc="lower right", handlelength=1.0, columnspacing=1.0,
              title_fontsize=7, borderpad=0.2)
    ax.margins(x=0.02)
    _save(fig, "fig1_controlled_deltas")


# -------------------------------------------------------------------------- #
def _binned(rows, xkey, ykey="delta_max", nb=4):
    rows = [r for r in rows
            if r.get("n_segments_min", r.get("n_segments", 2)) >= 2 and xkey in r]
    dem = demean_by_dataset(rows, ykey)
    return binned_means([r[xkey] for r in rows], dem, n_bins=nb)


def _bin_panel(ax, bins, color, title, xlabel, ylim):
    xs = np.arange(len(bins))
    ys = [b["mean_y"] for b in bins]
    ax.bar(xs, ys, width=0.66, color=color, zorder=3)
    ax.axhline(0, color=INK, lw=0.8, zorder=4)
    _grid(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(bins))])
    ax.set_ylim(*ylim)
    ax.set_title(title, color=INK)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:+.03f}", (x, y), ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=6.2, color=INK2,
                    xytext=(0, 2 if y >= 0 else -2), textcoords="offset points")


def fig2_locality_binned() -> None:
    rel = _load("locality_e5.json")
    dis = _load("distractor_locality_e5.json")
    if not rel or not dis:
        print("  skip fig2 (locality/distractor json missing)"); return
    b_rel = _binned(rel["rows"], "max_minus_mean")
    b_dis = _binned(dis["rows"], "competitive_gap_shift")
    allv = [b["mean_y"] for b in b_rel + b_dis]
    ylim = (min(allv) - 0.025, max(allv) + 0.025)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    _bin_panel(axes[0], b_rel, BLUE, "Relevant-side (whole-document)",
               "max$-$mean quartile   ($\\rho{=}{+}0.25$)", ylim)
    axes[0].set_ylabel("mean dataset-demeaned $\\Delta$RR@10")
    _bin_panel(axes[1], b_dis, ORANGE, "Distractor-side (competitors)",
               "competitive gap-shift quartile   ($\\rho{=}{-}0.44$)", ylim)
    fig.suptitle("Evidence locality predicts the per-query penalty "
                 "(E5, pooled, dataset-demeaned)",
                 fontsize=8.5, fontweight="semibold", color=INK, y=1.03)
    _save(fig, "fig2_locality_binned")


# -------------------------------------------------------------------------- #
def fig3_adaptive_systems() -> None:
    files = sorted(glob(os.path.join(RESULTS, "adaptive_e5_seed*.json")))
    if not files:
        print("  skip fig3 (no adaptive seed files)"); return
    runs = [json.load(open(f, encoding="utf-8")) for f in files]
    order = ["document", "paragraph", "rrf", "global_alpha",
             "router_hard", "router_soft", "oracle"]
    label = {"document": "document", "paragraph": "paragraph", "rrf": "RRF",
             "global_alpha": "global-$\\alpha$", "router_hard": "router\n(hard)",
             "router_soft": "router\n(soft)", "oracle": "oracle"}
    label = {**label, "router_hard": "router (hard)", "router_soft": "router (soft)"}
    systems = [s for s in order if s in runs[0]["pooled"]["mean_ndcg10"]]
    means = {s: [r["pooled"]["mean_ndcg10"][s] for r in runs] for s in systems}
    xs = {s: float(np.mean(means[s])) for s in systems}
    errs = {s: float(np.std(means[s])) for s in systems}

    def color(s):
        return GREY if s in ("document", "paragraph") else (RED if s == "oracle" else BLUE)

    # horizontal bars, top = best; single-column friendly
    sys_top = list(reversed(systems))
    ys = np.arange(len(sys_top))
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    _grid(ax, axis="x")
    ax.barh(ys, [xs[s] for s in sys_top], xerr=[errs[s] for s in sys_top],
            height=0.66, color=[color(s) for s in sys_top],
            capsize=2, error_kw={"lw": 0.7, "ecolor": INK2}, zorder=3)
    lo = min(xs.values()) - 0.03
    ax.set_xlim(max(0.0, lo), max(xs.values()) + max(errs.values()) + 0.02)
    ax.set_yticks(ys)
    ax.set_yticklabels([label[s] for s in sys_top])
    ax.set_xlabel("pooled test nDCG@10")
    ax.tick_params(length=0)
    for y, s in zip(ys, sys_top):
        ax.annotate(f"{xs[s]:.3f}", (xs[s] + errs[s], y), ha="left", va="center",
                    fontsize=6.4, color=INK2, xytext=(3, 0), textcoords="offset points")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="fixed granularity"),
                       Patch(color=BLUE, label="fusion / router"),
                       Patch(color=RED, label="oracle")],
              loc="upper right", ncols=1, handlelength=1.0, borderpad=0.3,
              labelspacing=0.3)
    _save(fig, "fig3_adaptive_systems")


if __name__ == "__main__":
    _style()
    print("generating figures ...")
    fig1_controlled_deltas()
    fig2_locality_binned()
    fig3_adaptive_systems()
