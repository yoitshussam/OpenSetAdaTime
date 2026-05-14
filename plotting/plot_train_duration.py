#!/usr/bin/env python3
"""Plot mean training duration per (scenario, algorithm) from mlruns/.

Reuses collect() from mean_train_duration.py. Produces a 2x2 grid (one panel
per scenario: closed_set / OSDA / PDA / UniDA). Within each panel, horizontal
bars show the mean duration (minutes) per algorithm, averaged across all
pairs and all n_unknown values, with ±1σ error bars over runs.

A second figure breaks each scenario down per pair (grouped horizontal bars)
for the case-by-case view.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from mean_train_duration import collect
from plot_curves import ALGO_COLORS, UNIDA_METHODS, OSDA_METHODS, PDA_METHODS

# Methods that actually run in the open-set / partial-set / UniDA experiments.
# Closed-set-only baselines (ACON, DANN, ...) are excluded.
ALLOWED_METHODS = set(OSDA_METHODS + PDA_METHODS + UNIDA_METHODS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figures")
SCENARIOS = ("closed_set", "OSDA", "PDA", "UniDA")

# Canonical dataset names — collapses "mhealth" (closed_set) and "MHEALTH"
# (OSDA/PDA/UniDA) into the same pair label.
_CANON = {"mhealth": "MHEALTH", "MHEALTH": "MHEALTH",
          "Pamap2": "Pamap2", "RealWorld": "RealWorld"}


def canon_pair(pair):
    src, _, tgt = pair.partition(" -> ")
    return f"{_CANON.get(src, src)} -> {_CANON.get(tgt, tgt)}"


def normalise(durations):
    """Return a copy of durations keyed by canonical pair names."""
    out = {}
    for (scn, pair, algo), vals in durations.items():
        key = (scn, canon_pair(pair), algo)
        out.setdefault(key, []).extend(vals)
    return out


def all_pairs(durations):
    return sorted({pair for (_scn, pair, _algo) in durations})


def panel_pair_overall(ax, durations, pair):
    by_algo = {}
    for (_scn, p, algo), vals in durations.items():
        if p != pair:
            continue
        by_algo.setdefault(algo, []).extend(vals)
    if not by_algo:
        ax.set_title(f"{pair} (no data)", fontsize=11)
        ax.set_axis_off()
        return

    algos = sorted(by_algo, key=lambda a: np.mean(by_algo[a]))
    means = np.array([np.mean(by_algo[a]) / 60.0 for a in algos])
    y = np.arange(len(algos))

    fallback = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = [ALGO_COLORS.get(a, fallback[i % 20]) for i, a in enumerate(algos)]
    ax.barh(y, means, color=colors)

    x_max = float(means.max())
    pad = x_max * 0.012
    for yi, mu in zip(y, means):
        ax.text(mu + pad, yi, f"{mu:.2f}",
                va="center", ha="left", fontsize=7, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(algos, fontsize=8)
    ax.set_xlabel("mean training duration (minutes)")
    ax.set_xlim(0, x_max * 1.18)
    n_runs = sum(len(v) for v in by_algo.values())
    ax.set_title(f"{pair}  ({n_runs} runs)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def plot_pair_overall(durations, out_path, ncols=2):
    pairs = all_pairs(durations)
    n = len(pairs)
    nrows = (n + ncols - 1) // ncols
    if ncols == 1:
        figsize = (10, 4.0 * nrows)
    else:
        figsize = (7 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    for ax, p in zip(axes, pairs):
        panel_pair_overall(ax, durations, p)
    for ax in axes[len(pairs):]:
        ax.set_axis_off()
    fig.suptitle("Mean training duration per algorithm — one panel per pair "
                 "(averaged across scenarios, n_unknown, seeds)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def panel_pair_by_scenario(ax, durations, pair):
    scn_set, algo_set = set(), set()
    for (scn, p, algo) in durations:
        if p == pair:
            scn_set.add(scn); algo_set.add(algo)
    if not algo_set:
        ax.set_title(f"{pair} (no data)", fontsize=11)
        ax.set_axis_off()
        return

    scenarios = [s for s in SCENARIOS if s in scn_set]
    by_algo_overall = {a: [] for a in algo_set}
    for (_s, p, a), vals in durations.items():
        if p == pair:
            by_algo_overall[a].extend(vals)
    algos = sorted(algo_set, key=lambda a: np.mean(by_algo_overall[a]))

    n_scn = len(scenarios)
    bar_h = 0.8 / n_scn
    offsets = (np.arange(n_scn) - (n_scn - 1) / 2) * bar_h
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, n_scn))

    y = np.arange(len(algos))
    all_means = []
    for i, scn in enumerate(scenarios):
        means = []
        for a in algos:
            vals = durations.get((scn, pair, a), [])
            means.append(np.mean(vals) / 60.0 if vals else np.nan)
        means = np.array(means)
        all_means.append(means)
        ax.barh(y + offsets[i], np.nan_to_num(means, nan=0.0),
                height=bar_h, color=palette[i], label=scn, edgecolor="none")

    x_max = float(np.nanmax(np.stack(all_means)))
    ax.set_yticks(y)
    ax.set_yticklabels(algos, fontsize=8)
    ax.set_xlabel("mean training duration (minutes)")
    ax.set_xlim(0, x_max * 1.05)
    ax.set_title(pair, fontsize=11)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def plot_pair_by_scenario(durations, out_path, ncols=2):
    pairs = all_pairs(durations)
    n = len(pairs)
    nrows = (n + ncols - 1) // ncols
    if ncols == 1:
        figsize = (11, 4.5 * nrows)
    else:
        figsize = (7.5 * ncols, 5.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    for ax, p in zip(axes, pairs):
        panel_pair_by_scenario(ax, durations, p)
    for ax in axes[len(pairs):]:
        ax.set_axis_off()
    fig.suptitle("Mean training duration per (algorithm, scenario) — "
                 "one panel per pair", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    durations = normalise(collect())
    # Keep only the 11 methods that appear in OSDA / PDA / UniDA experiments.
    durations = {k: v for k, v in durations.items() if k[2] in ALLOWED_METHODS}
    if not durations:
        print("No runs found.")
        return
    plot_pair_overall(durations,
                      os.path.join(OUT_DIR, "train_duration_by_pair.png"),
                      ncols=2)
    plot_pair_by_scenario(durations,
                          os.path.join(OUT_DIR, "train_duration_by_pair_scenarios.png"),
                          ncols=2)
    # Single-column variants (1 col × 6 rows)
    plot_pair_overall(durations,
                      os.path.join(OUT_DIR, "train_duration_by_pair_1col.png"),
                      ncols=1)
    plot_pair_by_scenario(durations,
                          os.path.join(OUT_DIR, "train_duration_by_pair_scenarios_1col.png"),
                          ncols=1)


if __name__ == "__main__":
    main()
