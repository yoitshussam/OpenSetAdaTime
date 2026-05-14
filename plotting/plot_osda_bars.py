#!/usr/bin/env python3
"""Bar-chart variant of plot_osda_curves.py.

Same 2x3 pair grid, same data — but each n on the x-axis becomes a group of
9 side-by-side bars (one per method). Error bars are ±1σ over 5 seeds.
OSDA-native methods (OSBP, TSFA) are drawn with a hatched fill so the
two families remain visually separable even without colour.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_osda_curves import (
    ALL_METHODS,
    METRIC_FILES,
    OSDA_METHODS,
    PAIRS,
    UNIDA_METHODS,
    collect,
    make_palette,
    report_coverage,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figures")


def plot_bars(data, metric_key, out_path):
    colors = make_palette()
    metric_label = {
        "H":   "H-score",
        "OS":  "OS*",
        "UNK": "UNK",
        "F1":  "Target F1-score",
    }[metric_key]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    axes = axes.flatten()

    n_methods = len(ALL_METHODS)
    group_width = 0.88                  # fraction of one n-unit used by bars
    bar_w = group_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_w

    for ax, pair in zip(axes, PAIRS):
        all_ns = sorted({n for m in ALL_METHODS for n in data[pair][m].keys()})
        if not all_ns:
            ax.set_title(f"{pair[0]} → {pair[1]} (no data)", fontsize=11)
            continue
        x_base = np.array(all_ns, dtype=float)

        for i, m in enumerate(ALL_METHODS):
            means, stds, xs = [], [], []
            for n_idx, n in enumerate(all_ns):
                vals = data[pair][m].get(n, [])
                if not vals:
                    continue
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(x_base[n_idx] + offsets[i])
            if not xs:
                continue
            hatch = "//" if m in OSDA_METHODS else None
            edgecolor = "black" if m in OSDA_METHODS else "none"
            ax.bar(xs, means, width=bar_w, color=colors[m],
                   yerr=stds, label=m, hatch=hatch, edgecolor=edgecolor,
                   linewidth=0.4, error_kw=dict(elinewidth=0.6, capsize=1.5, alpha=0.6))

        ax.set_title(f"{pair[0]} → {pair[1]}", fontsize=11)
        ax.set_xlabel("n (target-private classes added)")
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(int(n)) for n in x_base])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(metric_label)
    axes[3].set_ylabel(metric_label)

    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: (labels[i] not in OSDA_METHODS, labels[i]))
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=n_methods,
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9)
    fig.suptitle(f"OSDA: {metric_label} vs. target-private classes "
                 f"(hardest-first; bars = mean over 5 seeds, error = ±1σ)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"saved: {out_path.replace('.png', '.pdf')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", default="H", choices=list(METRIC_FILES))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect(args.metric)
    out_path = args.out or os.path.join(OUT_DIR, f"osda_{args.metric.lower()}_bars.png")
    plot_bars(data, args.metric, out_path)
    report_coverage(data)


if __name__ == "__main__":
    main()
