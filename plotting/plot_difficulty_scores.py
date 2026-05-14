#!/usr/bin/env python3
"""Visualise the FNO mean-cosine-distance difficulty scores used to rank
target-private classes for the hardest-first curriculum.

Reads:
  feature_distance_4known_fno_mean/rankings_4known_fno_mean.json   (OSDA)

Produces three styles (one PNG each):
  figures/difficulty_heatmap.png   — class × source-prototype distance matrix
  figures/difficulty_lollipop.png  — 2x3 pair grid, ranked horizontal lollipops
  figures/difficulty_strip.png     — single panel, y=pair, x=score, dot/class
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "feature_distance_4known_fno_mean",
                        "rankings_4known_fno_mean.json")
OUT_DIR = os.path.join(ROOT, "figures")

PAIRS = [
    "RealWorld -> Pamap2", "RealWorld -> MHEALTH",
    "Pamap2 -> RealWorld",  "Pamap2 -> MHEALTH",
    "MHEALTH -> RealWorld", "MHEALTH -> Pamap2",
]

DATASET_COLOR = {
    "RealWorld": "#4c78a8",
    "Pamap2":    "#e45756",
    "MHEALTH":   "#54a24b",
}


def load():
    with open(JSON_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------- heatmap

def plot_heatmap(data, out_path):
    """Stacked sub-plot per pair: rows = candidate target-private classes
    (ranked hardest -> easiest), columns = 4 source-known prototypes,
    plus a final 'min' column = curriculum score. Color = cosine distance."""
    knowns_order = ["lying", "running", "sitting", "walking"]
    cols = knowns_order + ["min (score)"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()

    vmin = min(min(r["mean_cosine_dist"] for r in data[p]["rankings"])
               for p in PAIRS)
    vmax = max(max(r["per_class_mean"][k]
                   for r in data[p]["rankings"]
                   for k in knowns_order)
               for p in PAIRS)

    last_im = None
    for ax, pair in zip(axes, PAIRS):
        rankings = sorted(data[pair]["rankings"],
                          key=lambda r: r["mean_cosine_dist"], reverse=True)
        classes = [r["unknown"] for r in rankings]
        mat = np.array([
            [r["per_class_mean"][k] for k in knowns_order]
            + [r["mean_cosine_dist"]]
            for r in rankings
        ])
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        last_im = im

        # cell annotations
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] < (vmin + vmax) / 2
                        else "black")

        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_title(pair, fontsize=11)
        # bold separator between knowns and 'min' column
        ax.axvline(len(knowns_order) - 0.5, color="white", linewidth=2.0)

    fig.colorbar(last_im, ax=axes, shrink=0.65, pad=0.02,
                 label="cosine distance (higher = harder)")
    fig.suptitle("FNO mean cosine distance — candidate target-private class to "
                 "each source-known prototype (5-seed avg)", fontsize=13)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


# ---------------------------------------------------------------- lollipop

def plot_lollipop(data, out_path):
    """2x3 grid. Per pair: horizontal lollipops, classes ranked hardest first
    on y-axis, x = mean cosine distance. Color of dot = nearest source-known."""
    known_colors = {"lying":  "#4c78a8",
                    "running": "#e45756",
                    "sitting": "#54a24b",
                    "walking": "#f58518"}

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    axes = axes.flatten()

    x_max = max(r["mean_cosine_dist"]
                for p in PAIRS for r in data[p]["rankings"]) * 1.10
    x_min = min(r["mean_cosine_dist"]
                for p in PAIRS for r in data[p]["rankings"]) * 0.95

    for ax, pair in zip(axes, PAIRS):
        rankings = sorted(data[pair]["rankings"],
                          key=lambda r: r["mean_cosine_dist"])
        classes = [r["unknown"] for r in rankings]
        scores  = [r["mean_cosine_dist"] for r in rankings]
        nearest = [r["nearest_known"]   for r in rankings]
        y = np.arange(len(classes))

        for yi, s, nk in zip(y, scores, nearest):
            ax.hlines(yi, x_min, s, color="#bbbbbb", linewidth=1.2, zorder=1)
            ax.scatter(s, yi, s=70, color=known_colors[nk],
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.text(s + (x_max - x_min) * 0.012, yi, f"{s:.3f}",
                    va="center", fontsize=7, color="black")

        ax.set_yticks(y)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_title(pair, fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlabel("mean cosine distance")

    # legend across the figure
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="",
                      markerfacecolor=c, markeredgecolor="black",
                      markersize=8, label=k)
               for k, c in known_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10,
               title="nearest source-known prototype", title_fontsize=10)

    fig.suptitle("Curriculum difficulty — candidate target-private classes "
                 "ranked easiest -> hardest per pair", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


# ---------------------------------------------------------------- strip

def plot_strip(data, out_path):
    """Single panel: y = pair, x = score, one dot per candidate class.
    Dot color = source dataset; top-2 hardest per pair labelled inline."""
    fig, ax = plt.subplots(figsize=(13, 6.5))

    for yi, pair in enumerate(PAIRS):
        src = pair.split(" -> ")[0]
        rankings = sorted(data[pair]["rankings"],
                          key=lambda r: r["mean_cosine_dist"], reverse=True)
        xs = [r["mean_cosine_dist"] for r in rankings]
        # jitter on y for readability when scores cluster
        ys = yi + np.linspace(-0.18, 0.18, len(xs))
        ax.scatter(xs, ys, s=110, color=DATASET_COLOR[src],
                   edgecolor="black", linewidth=0.6, alpha=0.85, zorder=3)
        # label top-2 hardest per pair (the ones picked first in curriculum)
        for k in range(min(2, len(rankings))):
            ax.annotate(rankings[k]["unknown"],
                        xy=(xs[k], ys[k]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, va="center")

    ax.set_yticks(np.arange(len(PAIRS)))
    ax.set_yticklabels(PAIRS, fontsize=10)
    ax.set_xlabel("mean cosine distance (higher = harder)")
    ax.set_title("FNO difficulty score per candidate class, all pairs in one view "
                 "(top-2 hardest per pair labelled)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=d) for d, c in DATASET_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", title="source dataset",
              framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load()
    plot_heatmap (data, os.path.join(OUT_DIR, "difficulty_heatmap.png"))
    plot_lollipop(data, os.path.join(OUT_DIR, "difficulty_lollipop.png"))
    plot_strip   (data, os.path.join(OUT_DIR, "difficulty_strip.png"))


if __name__ == "__main__":
    main()
