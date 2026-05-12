"""Compare CNN-swept vs FNO-swept run results across the 6 cross-family pairs.

Reads from two MLflow file stores:
  - mlrunsarchive/  -> CNN-swept hparams (older runs, ~2026-04-28)
  - mlruns/         -> FNO-swept hparams (newer runs, ~2026-05-02)

For each (pair, method) it averages over the 5 seeds and plots a grouped bar
chart per pair: CNN bar vs FNO bar per method, with std error bars.

Usage:
  python compare_cnn_vs_fno_runs.py                       # default: F1-score, _closed_set
  python compare_cnn_vs_fno_runs.py --metric "Target Accuracy"
  python compare_cnn_vs_fno_runs.py --exp_suffix _closed_set --metric "Target F1-score"
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Methods to include — exactly main.py's SCENARIO_METHODS flattened.
ALL_METHODS = [
    "OSBP", "TSFA",                                          # OSDA
    "UDA", "OVANet", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT",  # UniDA
    "SPADA", "PDAAN",                                        # PDA
]

# Pair display order (rows × cols of the subplot grid).
PAIR_ORDER = [
    ("RealWorld", "Pamap2"),
    ("RealWorld", "mhealth"),
    ("Pamap2", "RealWorld"),
    ("Pamap2", "mhealth"),
    ("mhealth", "RealWorld"),
    ("mhealth", "Pamap2"),
]

RUN_NAME_RE = re.compile(r"^([A-Za-z]+)_run_(\d+)$")
EXP_NAME_RE = re.compile(r"^(.+?) to (.+?)_(.+)$")  # "<src> to <trg>_<exp_name>"


def read_experiment_name(exp_dir):
    """Return (name, experiment_id) from an experiment dir's meta.yaml,
    or (None, None) if missing."""
    meta = os.path.join(exp_dir, "meta.yaml")
    if not os.path.isfile(meta):
        return None, None
    with open(meta) as f:
        data = yaml.safe_load(f)
    return data.get("name"), data.get("experiment_id")


def read_run_name(run_dir):
    """Return mlflow.runName tag value, or None."""
    p = os.path.join(run_dir, "tags", "mlflow.runName")
    if not os.path.isfile(p):
        return None
    return open(p).read().strip()


def read_metric_final(run_dir, metric):
    """Return the last logged value for `metric`, or None if absent.
    Metric file format: '<timestamp_ms> <value> <step>' per line."""
    p = os.path.join(run_dir, "metrics", metric)
    if not os.path.isfile(p):
        return None
    last = None
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    last = float(parts[1])
                except ValueError:
                    pass
    return last


def collect(mlruns_root, exp_suffix, metric):
    """Walk an mlruns-style directory; return dict[(src,trg,method)] -> [vals]."""
    out = defaultdict(list)
    if not os.path.isdir(mlruns_root):
        print(f"WARNING: {mlruns_root} does not exist — skipping.", file=sys.stderr)
        return out

    for entry in os.listdir(mlruns_root):
        exp_dir = os.path.join(mlruns_root, entry)
        if not os.path.isdir(exp_dir) or entry == ".trash":
            continue
        name, _ = read_experiment_name(exp_dir)
        if not name or not name.endswith(exp_suffix):
            continue
        m = EXP_NAME_RE.match(name)
        if not m:
            continue
        src, trg, _ = m.groups()

        for run_id in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_id)
            if not os.path.isdir(run_dir) or run_id.startswith("."):
                continue
            run_name = read_run_name(run_dir)
            if not run_name:
                continue
            rm = RUN_NAME_RE.match(run_name)
            if not rm:
                continue
            method = rm.group(1)
            if method not in ALL_METHODS:
                continue
            val = read_metric_final(run_dir, metric)
            if val is None:
                continue
            out[(src, trg, method)].append(val)
    return out


def plot(cnn_data, fno_data, metric, exp_suffix, out_path):
    """One subplot per pair, grouped bars per method (CNN vs FNO), std error bars."""
    n_methods = len(ALL_METHODS)
    x = np.arange(n_methods)
    bar_w = 0.4

    fig, axes = plt.subplots(2, 3, figsize=(20, 9), sharey=True)
    axes = axes.flatten()

    for ax, (src, trg) in zip(axes, PAIR_ORDER):
        cnn_means, cnn_stds, fno_means, fno_stds = [], [], [], []
        for method in ALL_METHODS:
            cnn = cnn_data.get((src, trg, method), [])
            fno = fno_data.get((src, trg, method), [])
            cnn_means.append(np.mean(cnn) if cnn else 0.0)
            cnn_stds .append(np.std(cnn)  if cnn else 0.0)
            fno_means.append(np.mean(fno) if fno else 0.0)
            fno_stds .append(np.std(fno)  if fno else 0.0)

        ax.bar(x - bar_w / 2, cnn_means, bar_w, yerr=cnn_stds, capsize=3,
               label="CNN-sweep (archive)", color="#4C72B0", alpha=0.85)
        ax.bar(x + bar_w / 2, fno_means, bar_w, yerr=fno_stds, capsize=3,
               label="FNO-sweep (live)",   color="#DD8452", alpha=0.85)
        ax.set_title(f"{src} → {trg}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_METHODS, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel(metric)
    axes[3].set_ylabel(metric)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"CNN-swept vs FNO-swept hparams  ·  metric: {metric}  ·  "
                 f"experiments: *{exp_suffix}  ·  mean ± std over 5 seeds",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved -> {out_path}")


def print_table(cnn_data, fno_data, metric):
    """Plain-text summary so you can sanity-check before opening the PNG."""
    print(f"\n{'pair':<25} {'method':<10} {'cnn_mean':>10} {'cnn_n':>6} "
          f"{'fno_mean':>10} {'fno_n':>6} {'Δ (fno-cnn)':>12}")
    print("-" * 84)
    for (src, trg) in PAIR_ORDER:
        for method in ALL_METHODS:
            cnn = cnn_data.get((src, trg, method), [])
            fno = fno_data.get((src, trg, method), [])
            cm = np.mean(cnn) if cnn else float("nan")
            fm = np.mean(fno) if fno else float("nan")
            delta = (fm - cm) if (cnn and fno) else float("nan")
            print(f"{src+'->'+trg:<25} {method:<10} "
                  f"{cm:>10.4f} {len(cnn):>6} "
                  f"{fm:>10.4f} {len(fno):>6} "
                  f"{delta:>+12.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn_dir", default="mlrunsarchive",
                   help="MLflow file-store dir for CNN-swept runs")
    p.add_argument("--fno_dir", default="mlruns",
                   help="MLflow file-store dir for FNO-swept runs")
    p.add_argument("--exp_suffix", default="_closed_set",
                   help="Experiment-name suffix to filter on (e.g. _closed_set, _EXP_NEW)")
    p.add_argument("--metric", default="Target F1-score",
                   help='Metric file name (e.g. "Target F1-score", "Target Accuracy", "H_score")')
    p.add_argument("--out", default="cnn_vs_fno_comparison.png")
    args = p.parse_args()

    cnn = collect(args.cnn_dir, args.exp_suffix, args.metric)
    fno = collect(args.fno_dir, args.exp_suffix, args.metric)

    print(f"CNN keys: {len(cnn)}  · FNO keys: {len(fno)}")
    print_table(cnn, fno, args.metric)
    plot(cnn, fno, args.metric, args.exp_suffix, args.out)


if __name__ == "__main__":
    main()
