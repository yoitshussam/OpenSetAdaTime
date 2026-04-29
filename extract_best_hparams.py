"""Pull the best trial per method from an MLflow sweep and write a
JSON file shaped exactly like `alg_hparams` in configs/hparams.py — i.e.

    {
      "UDA":   { "batch_size": 64, "learning_rate": 1e-4, ... },
      "PDAAN": { "batch_size": 32, "alpha": 1.5,         ... },
      ...
    }

ready to paste into `self.alg_hparams = {...}`.

Per-method "best" is auto-chosen by scenario:
    PDA methods (SPADA, PDAAN ...) -> max avg_f1_score
    OSDA / UniDA methods           -> max avg_H_score

Run:
    python extract_best_hparams.py --exp_name sweep_complete

The script writes best_hparams.json next to itself and prints nothing
useful to stdout beyond a one-line confirmation.
"""
import argparse
import json
import math
import os
import re
import sys

import mlflow
from mlflow.tracking import MlflowClient

from algorithms.algorithms import get_algorithm_class

METHODS = [
    "UDA", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT",
    "OSBP", "OVANet", "TSFA",
    "SPADA", "PDAAN",
]

SCENARIO_METRIC = {
    "PDA":   "avg_f1_score",
    "OSDA":  "avg_H_score",
    "UniDA": "avg_H_score",
    "CLOSED": "avg_acc",
}

INT_KEYS = {"batch_size", "K", "MQ_size", "n_batch", "trg_mem_size",
            "num_epochs", "num_epochs_pr", "num_epochs_correct"}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "best_hparams.json")


def coerce(key, value):
    """MLflow stores params as strings — turn them back into int/float."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if key in INT_KEYS:
        return int(f)
    return f


def parse_method(run_name):
    if not run_name:
        return None
    m = re.match(r"^([A-Za-z]+)_trial_\d+$", run_name)
    return m.group(1) if m else None


def scenario_for(method):
    try:
        return getattr(get_algorithm_class(method), "SCENARIO", "UniDA")
    except NotImplementedError:
        return "UniDA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True,
                    help="The --exp_name that was passed to main_sweep.py")
    ap.add_argument("--source_dataset", default="RealWorld_male")
    ap.add_argument("--target_dataset", default="RealWorld_female")
    args = ap.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    client = MlflowClient()

    experiment_name = (f"sweep_{args.source_dataset}_to_"
                       f"{args.target_dataset}_{args.exp_name}")
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        sys.exit(f"Experiment not found: {experiment_name!r}")

    runs = client.search_runs(experiment_ids=[exp.experiment_id],
                              max_results=10000)

    # Group by method, pick best by scenario-appropriate metric, with a
    # tie-breaker on avg_acc to avoid the degenerate "reject everything"
    # trap (e.g. PPOT thresh=1.0 ⇒ H_score=0 but acc=0 — vs. a real trial
    # that also has H=0 but actually classifies known samples).
    # We also drop trials with avg_acc==0 since those clearly aren't learning.
    best = {}  # method -> (score, tiebreak, run)
    for r in runs:
        method = parse_method(r.data.tags.get("mlflow.runName"))
        if method not in METHODS:
            continue
        metric = SCENARIO_METRIC[scenario_for(method)]
        score = r.data.metrics.get(metric)
        if score is None or not math.isfinite(score):
            continue
        tiebreak = r.data.metrics.get("avg_acc", 0.0) or 0.0
        if not math.isfinite(tiebreak) or tiebreak <= 0.0:
            continue
        key = (score, tiebreak)
        if method not in best or key > best[method][0:2]:
            best[method] = (score, tiebreak, r)

    if not best:
        sys.exit(f"No completed runs found in {experiment_name!r}")

    # alg_hparams-shaped dict, ordered to match METHODS.
    payload = {
        method: {k: coerce(k, v) for k, v in best[method][2].data.params.items()}
        for method in METHODS if method in best
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=4, sort_keys=True)
    print(f"Wrote {len(payload)} methods -> {OUT_PATH}")


if __name__ == "__main__":
    main()
