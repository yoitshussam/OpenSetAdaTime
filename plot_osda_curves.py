#!/usr/bin/env python3
"""Plot OSDA performance vs. number of target-private classes.

2x3 grid (one panel per source→target pair). Within each panel:
  x = n (target-private classes added under hardest-first ranking)
  y = chosen metric (default H-score)
  one line per method, mean ±1σ over 5 seeds.
OSDA-native methods (OSBP, TSFA)    drawn with solid lines.
UniDA methods (UDA, OVANet, DANCE, PPOT, UniOT, UniJDOT, RAINCOAT) drawn with
dashed lines so the two families are visually separable.

Reads MLflow experiments straight from ./mlruns. Active runs only —
soft-deleted experiments/runs (lifecycle_stage: deleted) are skipped.
"""
import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(ROOT, "mlruns")
OUT_DIR = os.path.join(ROOT, "figures")

OSDA_METHODS  = ["OSBP", "TSFA"]
UNIDA_METHODS = ["UDA", "OVANet", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT"]
ALL_METHODS   = OSDA_METHODS + UNIDA_METHODS

# Display order: top row "to RealWorld" sinks; bottom row "to MHEALTH"/Pamap2 sinks.
PAIRS = [
    ("RealWorld", "Pamap2"),
    ("RealWorld", "MHEALTH"),
    ("Pamap2",    "MHEALTH"),
    ("Pamap2",    "RealWorld"),
    ("MHEALTH",   "Pamap2"),
    ("MHEALTH",   "RealWorld"),
]

EXP_RE = re.compile(r"^(.+?) to (.+?)_OSDA_hard_n(\d+)(?:_fno_mean)?$")
RUN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*?)_run_(\d+)$")

METRIC_FILES = {
    "H":   "H_score",
    "OS":  "OS_star",
    "UNK": "UNK",
    "F1":  "Target F1-score",
}


def read_meta_field(path, field):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(field + ":"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def read_final_metric(metric_path):
    """MLflow metric file format: '<ts> <value> <step>' per line. Take last."""
    try:
        with open(metric_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        return float(lines[-1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def collect(metric_key):
    """Return data[pair][method][n] = list of seed values."""
    metric_file = METRIC_FILES[metric_key]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for exp_id in os.listdir(MLRUNS_DIR):
        exp_dir = os.path.join(MLRUNS_DIR, exp_id)
        meta = os.path.join(exp_dir, "meta.yaml")
        if not os.path.isfile(meta):
            continue
        if read_meta_field(meta, "lifecycle_stage") == "deleted":
            continue
        name = read_meta_field(meta, "name")
        if not name:
            continue
        m = EXP_RE.match(name)
        if not m:
            continue
        src, trg, n = m.group(1), m.group(2), int(m.group(3))
        pair = (src, trg)

        # Dedupe by (method, seed_idx) — pick the most recent seed run.
        seen = defaultdict(dict)  # (algo, seed) -> (mtime, val)
        for run_id in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_id)
            if not os.path.isdir(run_dir):
                continue
            run_meta = os.path.join(run_dir, "meta.yaml")
            if os.path.isfile(run_meta) and read_meta_field(run_meta, "lifecycle_stage") == "deleted":
                continue
            rname_path = os.path.join(run_dir, "tags", "mlflow.runName")
            if not os.path.isfile(rname_path):
                continue
            with open(rname_path) as f:
                rname = f.read().strip()
            rm = RUN_RE.match(rname)
            if not rm:
                continue
            algo = rm.group(1)
            if algo not in ALL_METHODS:
                continue
            val = read_final_metric(os.path.join(run_dir, "metrics", metric_file))
            if val is None:
                continue
            seed_idx = int(rm.group(2))
            mtime = os.path.getmtime(run_dir)
            prev = seen[(algo, seed_idx)]
            if not prev or mtime > prev[0]:
                seen[(algo, seed_idx)] = (mtime, val)

        for (algo, _seed), (_mt, val) in seen.items():
            data[pair][algo][n].append(val)

    return data


def make_palette():
    """Distinct colours, families kept visually separable via linestyle."""
    osda_cm  = plt.get_cmap("Set1")
    unida_cm = plt.get_cmap("tab10")
    colors = {}
    for i, m in enumerate(OSDA_METHODS):
        colors[m] = osda_cm(i)
    for i, m in enumerate(UNIDA_METHODS):
        colors[m] = unida_cm(i)
    return colors


def plot(data, metric_key, out_path):
    colors = make_palette()
    metric_label = {
        "H":   "H-score",
        "OS":  "OS*",
        "UNK": "UNK",
        "F1":  "Target F1-score",
    }[metric_key]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    axes = axes.flatten()

    for ax, pair in zip(axes, PAIRS):
        for m in ALL_METHODS:
            ns = sorted(data[pair][m].keys())
            if not ns:
                continue
            means = np.array([np.mean(data[pair][m][n]) for n in ns])
            stds  = np.array([np.std (data[pair][m][n]) for n in ns])
            ls = "-" if m in OSDA_METHODS else "--"
            lw = 2.0 if m in OSDA_METHODS else 1.3
            c  = colors[m]
            ax.plot(ns, means, ls, color=c, label=m, marker="o", lw=lw, ms=4.5)
            ax.fill_between(ns, means - stds, means + stds, color=c, alpha=0.12, linewidth=0)
        ax.set_title(f"{pair[0]} → {pair[1]}", fontsize=11)
        ax.set_xlabel("n (target-private classes added)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.0, 1.0)
        # Integer ticks only.
        all_ns = sorted({n for m in ALL_METHODS for n in data[pair][m].keys()})
        if all_ns:
            ax.set_xticks(all_ns)

    axes[0].set_ylabel(metric_label)
    axes[3].set_ylabel(metric_label)

    handles, labels = axes[0].get_legend_handles_labels()
    # Reorder so OSDA-native methods appear first in the legend.
    order = sorted(range(len(labels)),
                   key=lambda i: (labels[i] not in OSDA_METHODS, labels[i]))
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=len(ALL_METHODS),
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle(f"OSDA: {metric_label} vs. target-private classes (hardest-first)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"saved: {out_path.replace('.png', '.pdf')}")


def report_coverage(data):
    """Print (pair, method) cells where seed counts deviate from 5."""
    print("\nCoverage check (cells where seed count ≠ 5):")
    any_issue = False
    for pair in PAIRS:
        for m in ALL_METHODS:
            ns = sorted(data[pair][m].keys())
            for n in ns:
                k = len(data[pair][m][n])
                if k != 5:
                    print(f"  {pair[0]}→{pair[1]:<10s} {m:<10s} n={n}  seeds={k}")
                    any_issue = True
            if not ns:
                print(f"  {pair[0]}→{pair[1]:<10s} {m:<10s} NO DATA")
                any_issue = True
    if not any_issue:
        print("  all (pair, method, n) cells have exactly 5 seeds.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", default="H", choices=list(METRIC_FILES),
                   help="Metric to plot (default: H-score)")
    p.add_argument("--out", default=None, help="Output PNG path")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect(args.metric)
    out_path = args.out or os.path.join(OUT_DIR, f"osda_{args.metric.lower()}_curves.png")
    plot(data, args.metric, out_path)
    report_coverage(data)


if __name__ == "__main__":
    main()
