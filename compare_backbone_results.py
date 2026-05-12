"""
Compare CNN- vs FNO-backbone results for the RealWorld -> mhealth closed_set
experiment. Splits runs by start_time: anything newer than CUTOFF_MS is the
recent CNN-backbone re-run, anything older is the original FNO runs.

For each algorithm we average the 5 runs of each backbone and plot a grouped
bar chart of Target Accuracy and Target F1-score.

Usage:
    python compare_backbone_results.py
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml


MLRUNS_DIR = "/mnt/data/home/tp2474/OpenSet-AdaTime/mlruns"
EXPERIMENT_ID = "773515370876137118"  # "RealWorld to mhealth_closed_set"
# CNN re-run started ~OSBP_run_0 at 1777626176972; previous (FNO) batch ended
# ~PDAAN_run_4 at 1777557660341. ~19 hour gap, anything between is safe.
CUTOFF_MS = 1777600000000

METRICS = ["Target Accuracy", "Target F1-score"]
# 11 algorithms re-run on CNN — restrict comparison to these.
ALGOS = ["DANCE", "OSBP", "OVANet", "PDAAN", "PPOT", "RAINCOAT",
         "SPADA", "TSFA", "UDA", "UniJDOT", "UniOT"]


def read_run(run_dir):
    meta = yaml.safe_load(open(os.path.join(run_dir, "meta.yaml")))
    name = meta.get("run_name", "")
    if "_run_" not in name:
        return None
    algo, run_id = name.rsplit("_run_", 1)
    try:
        run_id = int(run_id)
    except ValueError:
        return None

    metric_values = {}
    for m in METRICS:
        path = os.path.join(run_dir, "metrics", m)
        if not os.path.exists(path):
            continue
        # Final logged value = last line
        with open(path) as f:
            last = f.readlines()[-1].strip()
        metric_values[m] = float(last.split()[1])

    return {
        "algo": algo,
        "run_id": run_id,
        "start_time": int(meta["start_time"]),
        "metrics": metric_values,
    }


def collect():
    """Return {(algo, backbone, run_id): {metric: value}} keeping the most
    recent run for each (algo, backbone, run_id) tuple — duplicates older
    than that are dropped."""
    exp_dir = os.path.join(MLRUNS_DIR, EXPERIMENT_ID)
    latest = {}  # (algo, backbone, run_id) -> (start_time, metrics)
    for entry in os.listdir(exp_dir):
        run_dir = os.path.join(exp_dir, entry)
        if not os.path.isdir(run_dir) or not os.path.exists(os.path.join(run_dir, "meta.yaml")):
            continue
        r = read_run(run_dir)
        if r is None or r["algo"] not in ALGOS:
            continue
        backbone = "CNN" if r["start_time"] > CUTOFF_MS else "FNO"
        key = (r["algo"], backbone, r["run_id"])
        if key not in latest or latest[key][0] < r["start_time"]:
            latest[key] = (r["start_time"], r["metrics"])
    return latest


def aggregate(latest):
    """{(algo, backbone): {metric: [values across runs]}}"""
    grouped = defaultdict(lambda: defaultdict(list))
    for (algo, backbone, _), (_, metrics) in latest.items():
        for m, v in metrics.items():
            grouped[(algo, backbone)][m].append(v)
    return grouped


def plot(grouped):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(ALGOS))
    width = 0.38

    for ax, metric in zip(axes, METRICS):
        cnn_means, cnn_errs, fno_means, fno_errs = [], [], [], []
        for algo in ALGOS:
            cnn_vals = grouped.get((algo, "CNN"), {}).get(metric, [])
            fno_vals = grouped.get((algo, "FNO"), {}).get(metric, [])
            cnn_means.append(np.mean(cnn_vals) if cnn_vals else np.nan)
            cnn_errs.append(np.std(cnn_vals) if cnn_vals else 0.0)
            fno_means.append(np.mean(fno_vals) if fno_vals else np.nan)
            fno_errs.append(np.std(fno_vals) if fno_vals else 0.0)

        ax.bar(x - width/2, fno_means, width, yerr=fno_errs, capsize=3,
               label="FNO (sweep mismatch — original)", color="#4C72B0")
        ax.bar(x + width/2, cnn_means, width, yerr=cnn_errs, capsize=3,
               label="CNN (sweep-matched re-run)", color="#DD8452")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Print delta on top of each pair
        for xi, (cm, fm) in enumerate(zip(cnn_means, fno_means)):
            if not (np.isnan(cm) or np.isnan(fm)):
                delta = cm - fm
                ax.text(xi, max(cm, fm) + 0.03, f"Δ={delta:+.2f}",
                        ha="center", fontsize=8,
                        color="green" if abs(delta) < 0.05 else "red")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(ALGOS, rotation=30, ha="right")
    axes[0].legend(loc="upper right")
    axes[0].set_title("RealWorld → mhealth (closed_set): CNN vs FNO backbone, 5-run mean ± std")

    plt.tight_layout()
    out = "/mnt/data/home/tp2474/OpenSet-AdaTime/backbone_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


def print_table(grouped):
    print(f"\n{'algo':<10} | {'metric':<18} | {'FNO (n)':<12} | {'CNN (n)':<12} | Δ")
    print("-" * 75)
    for algo in ALGOS:
        for metric in METRICS:
            fno = grouped.get((algo, "FNO"), {}).get(metric, [])
            cnn = grouped.get((algo, "CNN"), {}).get(metric, [])
            fno_m = f"{np.mean(fno):.3f} (n={len(fno)})" if fno else "—"
            cnn_m = f"{np.mean(cnn):.3f} (n={len(cnn)})" if cnn else "—"
            delta = (np.mean(cnn) - np.mean(fno)) if fno and cnn else float("nan")
            print(f"{algo:<10} | {metric:<18} | {fno_m:<12} | {cnn_m:<12} | {delta:+.3f}")


if __name__ == "__main__":
    latest = collect()
    grouped = aggregate(latest)
    print_table(grouped)
    plot(grouped)
