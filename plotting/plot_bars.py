#!/usr/bin/env python3
"""Bar-chart variant of plot_curves.py.

Reads analysis/runs.csv produced by export_mlruns_csv.py.

For OSDA / PDA / UniDA: same 2x3 pair grid; each n on the x-axis becomes a
group of side-by-side bars (one per method). Error bars = ±1σ across seeds.
Scenario-native methods (OSBP/TSFA for OSDA, SPADA/PDAAN for PDA) get a
hatched fill so the two families remain separable.

For closed_set: no n axis. The x-axis is the algorithm; one bar per method
within each pair panel (still 2x3 grid).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_curves import (
    CSV_PATH, OUT_DIR, EXPERIMENT_METHODS, NATIVE_METHODS,
    METRIC_COL, METRIC_LABEL, PAIRS, UNIDA_METHODS,
    OSDA_METHODS, PDA_METHODS, CLOSED_BASELINES, make_palette,
)

# Closed-set: every UniDA/OSDA/PDA method + the baselines that ran closed_set.
CLOSED_METHODS = sorted(
    set(UNIDA_METHODS + OSDA_METHODS + PDA_METHODS + CLOSED_BASELINES))


def _bars_with_n(df, experiment, metric_key, out_path):
    col = METRIC_COL[metric_key]
    df = df[df["scenario"] == experiment].dropna(subset=[col, "n_unknown"])
    if df.empty:
        print(f"[warn] no rows for scenario={experiment} metric={metric_key}")
        return

    methods = EXPERIMENT_METHODS[experiment]
    native = NATIVE_METHODS[experiment]
    colors = make_palette(methods)
    n_methods = len(methods)
    group_width = 0.88
    bar_w = group_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_w

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    axes = axes.flatten()

    for ax, (src, tgt) in zip(axes, PAIRS):
        pair = f"{src} -> {tgt}"
        sub = df[df["pair"] == pair]
        if sub.empty:
            ax.set_title(f"{src} → {tgt}  (no data)", fontsize=11)
            continue

        all_ns = sorted(sub["n_unknown"].unique().astype(int))
        x_base = np.array(all_ns, dtype=float)

        for i, m in enumerate(methods):
            mdf = sub[sub["algorithm"] == m]
            if mdf.empty:
                continue
            agg = mdf.groupby("n_unknown")[col].agg(["mean", "std"])
            xs, means, stds = [], [], []
            for n_idx, n in enumerate(all_ns):
                if n not in agg.index:
                    continue
                means.append(agg.loc[n, "mean"])
                stds.append(agg.loc[n, "std"] if not np.isnan(agg.loc[n, "std"]) else 0.0)
                xs.append(x_base[n_idx] + offsets[i])
            if not xs:
                continue
            ax.bar(xs, means, width=bar_w, color=colors[m],
                   yerr=stds, label=m, edgecolor="none",
                   error_kw=dict(elinewidth=0.6, capsize=1.5, alpha=0.6))

        ax.set_title(f"{src} → {tgt}", fontsize=11)
        ax.set_xlabel("n (target-private classes added)")
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(int(n)) for n in x_base])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

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
                 f"classes (bars = mean over seeds, error = ±1σ)", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def _bars_closed_set(df, metric_key, out_path):
    col = METRIC_COL[metric_key]
    df = df[df["scenario"] == "closed_set"].dropna(subset=[col])
    if df.empty:
        print(f"[warn] no rows for closed_set metric={metric_key}")
        return

    # Four method families in display order:
    #   closed-set baselines | OSDA-native | PDA-native | UniDA.
    available = set(df["algorithm"].unique())
    closed_list = [m for m in sorted(CLOSED_BASELINES) if m in available]
    osda_list   = [m for m in OSDA_METHODS              if m in available]
    pda_list    = [m for m in PDA_METHODS               if m in available]
    unida_list  = [m for m in UNIDA_METHODS             if m in available]
    methods = closed_list + osda_list + pda_list + unida_list

    closed_color = "#4c78a8"   # blue
    osda_color   = "#f58518"   # orange
    pda_color    = "#54a24b"   # green
    unida_color  = "#e45756"   # red

    family_of = {}
    family_of.update({m: ("closed", closed_color) for m in closed_list})
    family_of.update({m: ("osda",   osda_color)   for m in osda_list})
    family_of.update({m: ("pda",    pda_color)    for m in pda_list})
    family_of.update({m: ("unida",  unida_color)  for m in unida_list})

    # Indices at which family changes (for separator lines).
    sep_after = []
    cumulative = 0
    for group in (closed_list, osda_list, pda_list):
        cumulative += len(group)
        if cumulative and group:
            sep_after.append(cumulative)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    axes = axes.flatten()

    for ax, (src, tgt) in zip(axes, PAIRS):
        pair = f"{src} -> {tgt}"
        sub = df[df["pair"] == pair]
        if sub.empty:
            ax.set_title(f"{src} → {tgt}  (no data)", fontsize=11)
            continue

        agg = sub.groupby("algorithm")[col].agg(["mean", "std"]).reindex(methods)
        x = np.arange(len(methods))
        means = agg["mean"].values
        stds = agg["std"].fillna(0.0).values
        bar_colors = [family_of[m][1] for m in methods]

        ax.bar(x, means, yerr=stds, color=bar_colors,
               error_kw=dict(elinewidth=0.6, capsize=1.5, alpha=0.6))
        for idx in sep_after:
            ax.axvline(idx - 0.5, color="black", linestyle=":",
                       linewidth=0.8, alpha=0.5)
        ax.set_title(f"{src} → {tgt}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(METRIC_LABEL[metric_key])
    axes[3].set_ylabel(METRIC_LABEL[metric_key])

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=closed_color, label="closed-set methods"),
        Patch(color=osda_color,   label="OSDA-native (OSBP, TSFA)"),
        Patch(color=pda_color,    label="PDA-native (SPADA, PDAAN)"),
        Patch(color=unida_color,  label="UniDA methods"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10)

    fig.suptitle(f"closed_set: {METRIC_LABEL[metric_key]} per algorithm "
                 f"(bars = mean over seeds, error = ±1σ)", fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True,
                   choices=["closed_set", "OSDA", "PDA", "UniDA"])
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
        OUT_DIR, f"{args.experiment.lower()}_{args.metric.lower()}_bars.png")

    if args.experiment == "closed_set":
        _bars_closed_set(df, args.metric, out)
    else:
        _bars_with_n(df, args.experiment, args.metric, out)


if __name__ == "__main__":
    main()
