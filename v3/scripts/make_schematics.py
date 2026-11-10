"""make_schematics.py — friendly-but-clean vector redraws of the two concept
figures (motivation, failure), matching the palette/fonts of make_figures.py.

Writes figures/motivation.pdf/.png and figures/failure.pdf/.png.

Usage:  python scripts/make_schematics.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(V3, "figures")

INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
BLUE_FILL = "#dcebfb"
AQUA = "#1baf7a"
AQUA_FILL = "#d6f2e7"
ORANGE = "#eb6834"
RED = "#e34948"
RED_FILL = "#fbe0df"
GREY = "#b3b1a8"
PILL = "#f0efec"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "text.color": INK,
    })


def _ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, z=2, r=2.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, mutation_aspect=1))


def label(ax, x, y, s, size=8, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=6)


def textlines(ax, x, y, w, n=3, dy=2.4, color=GREY):
    """faint horizontal lines suggesting body text inside a document box."""
    for i in range(n):
        wi = w * (0.9 if i < n - 1 else 0.55)
        ax.add_line(plt.Line2D([x, x + wi], [y - i * dy, y - i * dy],
                    color=color, lw=0.8, zorder=3, alpha=0.7))


def pill(ax, cx, y, text, w=54, h=6.5):
    rbox(ax, cx - w / 2, y - h / 2, w, h, PILL, "#dcdbd6", lw=1.0, r=h / 2)
    label(ax, cx, y, text, size=8, color=INK2)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
        color=color, lw=lw, zorder=4, shrinkA=1, shrinkB=1))


def scorebar(ax, x, y, w, frac, color):
    ax.add_patch(Rectangle((x, y), w, 2.4, facecolor="#ecebe7", edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x, y), w * frac, 2.4, facecolor=color, edgecolor="none", zorder=4))


def star(ax, x, y, color=AQUA, size=80):
    ax.plot([x], [y], marker="*", markersize=size ** 0.5 * 1.2,
            color=color, markeredgecolor="none", zorder=7)


def check(ax, x, y, ok=True, r=2.7):
    c = AQUA if ok else RED
    ax.add_patch(plt.Circle((x, y), r, facecolor=c, edgecolor="none", zorder=6))
    if ok:  # draw a tick
        ax.add_line(plt.Line2D([x - 1.3, x - 0.3, x + 1.5], [y, y - 1.1, y + 1.3],
                    color="white", lw=1.6, zorder=7, solid_capstyle="round"))
    else:   # draw a cross
        for dx in (-1, 1):
            ax.add_line(plt.Line2D([x - 1.2 * dx, x + 1.2 * dx], [y - 1.2, y + 1.2],
                        color="white", lw=1.6, zorder=7, solid_capstyle="round"))


def _save(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{ext}"),
                    bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf/.png")


# --------------------------------------------------------------------------- #
def motivation() -> None:
    fig, ax = _ax(7.2, 3.5)
    H = 100 * 3.5 / 7.2  # ~48.6
    pill(ax, 50, H - 3.5, "Query:  dietary treatment of Crohn's disease")

    ax.add_line(plt.Line2D([50, 50], [4, H - 9], color="#e2e1dc", lw=1.0, zorder=1))
    label(ax, 25, H - 8.5, "Document-level", size=9, weight="semibold", color=INK)
    label(ax, 75, H - 8.5, "Paragraph-level", size=9, weight="semibold", color=INK)

    # ---- left: whole doc -> one vector ----
    rbox(ax, 8, 18, 20, 18, BLUE_FILL, BLUE, lw=1.4)
    textlines(ax, 11, 31.5, 14, n=6, dy=2.1)
    arrow(ax, 29.5, 27, 35.5, 27)
    for i in range(6):
        rbox(ax, 36.5 + i * 2.3, 25.5, 2.0, 3, BLUE, BLUE, lw=0, r=0.6)
    label(ax, 42.5, 21.5, "one vector", size=7.5, color=INK2)
    label(ax, 24, 12, "keeps full context, but the\nrelevant signal is diluted",
          size=7.2, color=INK2)

    # ---- right: split into segments, each scored ----
    segs = [("intro", 0.28), ("body", 0.34), ("relevant", 0.92),
            ("body", 0.30), ("conclusion", 0.24)]
    y0 = H - 13
    for i, (name, frac) in enumerate(segs):
        y = y0 - i * 5.7
        hl = name == "relevant"
        rbox(ax, 55, y - 4.4, 19, 4.6, AQUA_FILL if hl else "#eef4fc",
             AQUA if hl else BLUE, lw=1.5 if hl else 1.0)
        label(ax, 58, y - 2.1, name, size=6.8, ha="left",
              color=AQUA if hl else INK2, weight="semibold" if hl else "normal")
        if hl:
            star(ax, 71.5, y - 2.1, AQUA, size=70)
        scorebar(ax, 76.5, y - 3.3, 15, frac, AQUA if hl else "#9ec5f4")
    label(ax, 74, 5.5, "score every segment; $\\max$ keeps the strongest",
          size=7.2, color=INK2)
    _save(fig, "motivation")


# --------------------------------------------------------------------------- #
def failure() -> None:
    # single-column portrait: two stacked panels
    fig, ax = _ax(3.4, 3.7)
    W, H = 100, 100 * 3.7 / 3.4  # ~108.8
    pill(ax, 50, H - 4, "Query:  does fenugreek boost muscle strength?", w=92, h=7)

    # ---- top panel: document-level judges the whole text -> picks doc A ----
    label(ax, 50, H - 10.5, "Document-level  —  correct", size=8.5,
          weight="semibold", color=AQUA)

    def doc(x, title, fc, ec, frac, win):
        yb = H - 30
        rbox(ax, x, yb, 34, 13, fc, ec, lw=1.3)
        label(ax, x + 17, yb + 15.5, title, size=7, color=ec, weight="semibold")
        textlines(ax, x + 4, yb + 10, 24, n=4, dy=2.1)
        scorebar(ax, x + 2, yb - 4, 30, frac, ec if ec != GREY else GREY)
        if win:
            check(ax, x + 32, yb + 11, ok=True)
    doc(9, "doc A (on-topic)", AQUA_FILL, AQUA, 0.80, True)
    doc(57, "doc B (off-topic)", "#f3f2ee", GREY, 0.30, False)
    label(ax, 50, H - 36, "whole-document match  ·  ranks doc A first",
          size=7, color=INK2)

    # divider
    ax.add_line(plt.Line2D([6, 94], [H - 42, H - 42], color="#e2e1dc", lw=1.0, zorder=1))

    # ---- bottom panel: paragraph max promotes doc B's one look-alike segment ----
    label(ax, 50, H - 48, "Paragraph-level  —  misled", size=8.5,
          weight="semibold", color=RED)

    def segcol(x, name, fracs, star_i, ec, win):
        top = H - 55
        label(ax, x + 17, top + 3.5, name, size=7, color=ec, weight="semibold")
        for i, f in enumerate(fracs):
            y = top - i * 6.0
            hl = i == star_i
            rbox(ax, x, y - 4.4, 34, 4.8, RED_FILL if hl else "#f4f4f1",
                 RED if hl else "#d7d6d1", lw=1.4 if hl else 0.9)
            scorebar(ax, x + 2, y - 3.0, 30, f, RED if hl else "#c9c8c3")
            if hl:
                star(ax, x + 17, y + 1.2, RED, size=60)
        if win:
            check(ax, x + 32, top - 6.0, ok=False)
    segcol(9, "doc A", [0.42, 0.38, 0.45], None, AQUA, False)
    segcol(57, "doc B", [0.20, 0.95, 0.18], 1, RED, True)
    label(ax, 50, H - 82, "one look-alike fragment lifts the wrong doc B",
          size=7, color=INK2)
    ax.set_ylim(H - 88, H + 1)  # crop the empty lower area
    _save(fig, "failure")


if __name__ == "__main__":
    _style()
    print("generating schematics ...")
    motivation()
    failure()
