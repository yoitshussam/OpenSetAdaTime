#!/usr/bin/env python3
"""Curve plots of a metric vs. n_unknown, per (pair, algorithm).

Reads analysis/runs.csv produced by export_mlruns_csv.py.

Usage:
  python plot_curves.py --experiment OSDA  --metric H
  python plot_curves.py --experiment PDA   --metric F1
  python plot_curves.py --experiment UniDA --metric H

Layout: 2x3 grid (one panel per pair), x = n_unknown, y = metric.
A line is drawn per algorithm with ±1σ bands across seeds.
Methods native to the scenario (OSBP/TSFA for OSDA, SPADA/PDAAN for PDA)
are drawn with solid lines; UniDA methods are dashed.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "analysis", "runs.csv")
OUT_DIR  = os.path.join(ROOT, "figures")

UNIDA_METHODS = ["UDA", "OVANet", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT"]
OSDA_METHODS  = ["OSBP", "TSFA"]
PDA_METHODS   = ["SPADA", "PDAAN"]
CLOSED_BASELINES = [
    "ACON", "AdvSKM", "CDAN", "CLUDA", "CoDATS", "CoTMix", "DAAN", "DANN",
    "DDC", "Deep_Coral", "DIRT", "DSAN", "HoMM", "MMDA", "SASA",
    "SSSS_TSA", "SWL_Adapt", "uDAR",
]

# Master algorithm order: scenario-native methods first (so they get the
# distinctive tab20 colours), then UniDA methods, then closed-set baselines.
# The position in this list determines the colour — same algorithm gets the
# same colour across every chart in the project.
ALL_ALGOS = (OSDA_METHODS + PDA_METHODS + UNIDA_METHODS
             + sorted(CLOSED_BASELINES))

def _build_algo_colors():
    cmap1 = plt.cm.tab20(np.linspace(0, 1, 20))
    cmap2 = plt.cm.tab20b(np.linspace(0, 1, 20))
    combined = np.vstack([cmap1, cmap2])
    return {a: combined[i % len(combined)] for i, a in enumerate(ALL_ALGOS)}

ALGO_COLORS = _build_algo_colors()

EXPERIMENT_METHODS = {
    "OSDA":  OSDA_METHODS + UNIDA_METHODS,
    "PDA":   PDA_METHODS  + UNIDA_METHODS,
    "UniDA": UNIDA_METHODS,
}
NATIVE_METHODS = {
    "OSDA":  set(OSDA_METHODS),
    "PDA":   set(PDA_METHODS),
    "UniDA": set(UNIDA_METHODS),
}

# Default plottable metrics per experiment. Keys are CLI metric tokens.
METRIC_COL = {
    "H":      "H_score",
    "OS":     "OS_star",
    "UNK":    "UNK",
    "F1":     "target_f1",
    "SRCACC": "source_acc",
}
METRIC_LABEL = {
    "H":      "H-score",
    "OS":     "OS*",
    "UNK":    "UNK",
    "F1":     "Target F1-score",
    "SRCACC": "Source Accuracy",
}

PAIRS = [
    ("Pamap2",    "RealWorld"),
    ("Pamap2",    "MHEALTH"),
    ("RealWorld", "Pamap2"),
    ("RealWorld", "MHEALTH"),
    ("MHEALTH",   "Pamap2"),
    ("MHEALTH",   "RealWorld"),
]


def make_palette(methods):
    """Return {method -> RGBA} using the project-wide ALGO_COLORS mapping.
    Falls back to tab20 for any algorithm not in ALL_ALGOS."""
    fallback = plt.cm.tab20(np.linspace(0, 1, 20))
    return {m: ALGO_COLORS.get(m, fallback[i % 20])
            for i, m in enumerate(methods)}


def plot(df, experiment, metric_key, out_path):
    col = METRIC_COL[metric_key]
    df = df[df["scenario"] == experiment].dropna(subset=[col, "n_unknown"])
    if df.empty:
        print(f"[warn] no rows for scenario={experiment} metric={metric_key}")
        return

    methods = EXPERIMENT_METHODS[experiment]
    native = NATIVE_METHODS[experiment]
    colors = make_palette(methods)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes = axes.flatten()

    for ax, (src, tgt) in zip(axes, PAIRS):
        pair = f"{src} -> {tgt}"
        sub = df[df["pair"] == pair]
        if sub.empty:
            ax.set_title(f"{src} → {tgt}  (no data)", fontsize=11)
            continue

        for m in methods:
            mdf = sub[sub["algorithm"] == m]
            if mdf.empty:
                continue
            agg = mdf.groupby("n_unknown")[col].agg(["mean", "std", "count"])
            agg = agg.sort_index()
            xs = agg.index.values.astype(float)
            mus = agg["mean"].values
            sds = agg["std"].fillna(0.0).values

            linestyle = "-" if m in native else "--"
            ax.plot(xs, mus, marker="o", markersize=4,
                    linestyle=linestyle, color=colors[m], label=m, linewidth=1.6)
            ax.fill_between(xs, mus - sds, mus + sds, color=colors[m], alpha=0.12)

        ax.set_title(f"{src} → {tgt}", fontsize=11)
        ax.set_xlabel("n (target-private classes added)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        all_n = sorted(sub["n_unknown"].dropna().unique().astype(int))
        if all_n:
            ax.set_xticks(all_n)

    axes[0].set_ylabel(METRIC_LABEL[metric_key])
    axes[3].set_ylabel(METRIC_LABEL[metric_key])

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    order = sorted(range(len(labels)),
                   key=lambda i: (labels[i] not in native, labels[i]))
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 9),
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9)

    fig.suptitle(f"{experiment}: {METRIC_LABEL[metric_key]} vs. target-private "
                 f"classes (mean over seeds ± 1σ)", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True,
                   choices=["OSDA", "PDA", "UniDA"],
                   help="Which scenario to plot. closed_set has no n axis — use plot_bars.py.")
    p.add_argument("--metric", required=True, choices=list(METRIC_COL))
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["n_unknown"] = pd.to_numeric(df["n_unknown"], errors="coerce")
    for c in METRIC_COL.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(
        OUT_DIR, f"{args.experiment.lower()}_{args.metric.lower()}_curves.png")
    plot(df, args.experiment, args.metric, out)


if __name__ == "__main__":
    main()
