#!/usr/bin/env python3
"""Compute mean training duration per (scenario, pair, algorithm) from mlruns/.

For each active MLflow run named "<ALGO>_run_<N>" inside an active experiment,
duration = (end_time - start_time) / 1000 seconds (both fields are Unix ms in
the run's meta.yaml).

Aggregation:
  - one row per (scenario, pair, algorithm): mean over all n_unknown values
    and all seeds.
  - one row per (scenario, algorithm): mean over all pairs and n_unknown.

Sweep runs (*_trial_*) and FAILED/KILLED runs (status != 3) are excluded.
"""
import csv
import os
import re
from collections import defaultdict
from statistics import mean, stdev

MLRUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
OUT_CSV    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "analysis", "mean_train_duration.csv")

# MLflow run status codes — see RunStatus.proto. We only count FINISHED runs.
STATUS_FINISHED = "3"

# Same patterns as audit_mlruns.py.
RUN_NAME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*?)_run_(\d+)$")
SCENARIO_PATTERNS = [
    (re.compile(r"_closed_set$"),                 "closed_set"),
    (re.compile(r"_OSDA_hard_n\d+(_fno_mean)?$"), "OSDA"),
    (re.compile(r"_PDA_hard_n\d+(_fno_mean)?$"),  "PDA"),
    (re.compile(r"_UniDA_hard_n\d+(_fno_mean)?$"),"UniDA"),
]


def read_meta(meta_path):
    """Return dict of {name, lifecycle_stage, start_time, end_time, status, run_name}
    parsed from a meta.yaml. Missing keys become None."""
    out = {"name": None, "lifecycle_stage": None,
           "start_time": None, "end_time": None,
           "status": None, "run_name": None}
    try:
        with open(meta_path) as f:
            for line in f:
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip().strip("'\"")
                if k in out:
                    out[k] = v
    except OSError:
        pass
    return out


def classify_experiment(exp_name):
    """Return (scenario, n_unknown, pair) or (None, None, None)."""
    for pat, scenario in SCENARIO_PATTERNS:
        if pat.search(exp_name):
            n_match = re.search(r"_n(\d+)", exp_name)
            n = int(n_match.group(1)) if n_match else None
            pair = None
            pm = re.match(r"^(.+?) to (.+?)_(?:closed_set|OSDA|PDA|UniDA)_?",
                          exp_name)
            if pm:
                pair = f"{pm.group(1)} -> {pm.group(2)}"
            elif scenario == "closed_set":
                pm2 = re.match(r"^(.+?) to (.+?)_closed_set$", exp_name)
                if pm2:
                    pair = f"{pm2.group(1)} -> {pm2.group(2)}"
            return scenario, n, pair
    return None, None, None


def collect():
    """Walk mlruns/ and return:
      durations[(scenario, pair, algo)] = list of seconds (one per run)
    """
    durations = defaultdict(list)
    if not os.path.isdir(MLRUNS_DIR):
        print(f"[error] {MLRUNS_DIR} not found")
        return durations

    for exp_entry in sorted(os.listdir(MLRUNS_DIR)):
        exp_dir = os.path.join(MLRUNS_DIR, exp_entry)
        exp_meta_path = os.path.join(exp_dir, "meta.yaml")
        if not os.path.isfile(exp_meta_path):
            continue
        exp_meta = read_meta(exp_meta_path)
        exp_name = exp_meta["name"]
        if not exp_name or exp_name == "Default":
            continue
        if exp_meta["lifecycle_stage"] == "deleted":
            continue

        scenario, n_unknown, pair = classify_experiment(exp_name)
        if scenario is None or pair is None:
            continue

        for run_entry in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_entry)
            if not os.path.isdir(run_dir):
                continue
            run_meta_path = os.path.join(run_dir, "meta.yaml")
            if not os.path.isfile(run_meta_path):
                continue
            rm = read_meta(run_meta_path)
            if rm["lifecycle_stage"] == "deleted":
                continue
            if rm["status"] != STATUS_FINISHED:
                continue
            run_name = rm["run_name"]
            if not run_name:
                tag = os.path.join(run_dir, "tags", "mlflow.runName")
                if os.path.isfile(tag):
                    with open(tag) as f:
                        run_name = f.read().strip()
            if not run_name:
                continue
            m = RUN_NAME_RE.match(run_name)
            if not m:
                continue
            algo = m.group(1)
            try:
                start_ms = int(rm["start_time"])
                end_ms   = int(rm["end_time"])
            except (TypeError, ValueError):
                continue
            if end_ms <= start_ms:
                continue
            durations[(scenario, pair, algo)].append((end_ms - start_ms) / 1000.0)

    return durations


def fmt_min(seconds):
    return f"{seconds / 60.0:7.2f}"


def summarise_pair(durations):
    """Print one table per scenario with rows = (pair, algo)."""
    by_scn = defaultdict(list)
    for (scn, pair, algo), vals in durations.items():
        by_scn[scn].append((pair, algo, vals))

    for scn in ("closed_set", "OSDA", "PDA", "UniDA"):
        rows = by_scn.get(scn, [])
        if not rows:
            continue
        print(f"\n{'='*84}")
        print(f"  {scn}  —  per (pair, algorithm)  —  duration in minutes")
        print(f"{'='*84}")
        print(f"  {'pair':<28s} {'algo':<14s} {'n_runs':>7s} "
              f"{'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
        for pair, algo, vals in sorted(rows):
            n = len(vals)
            mu = mean(vals)
            sd = stdev(vals) if n > 1 else 0.0
            lo, hi = min(vals), max(vals)
            print(f"  {pair:<28s} {algo:<14s} {n:>7d} "
                  f"{fmt_min(mu)} {fmt_min(sd)} {fmt_min(lo)} {fmt_min(hi)}")


def summarise_overall(durations):
    """Print one table per scenario with rows = algo (averaged across pairs)."""
    by_scn_algo = defaultdict(lambda: defaultdict(list))
    for (scn, pair, algo), vals in durations.items():
        by_scn_algo[scn][algo].extend(vals)

    for scn in ("closed_set", "OSDA", "PDA", "UniDA"):
        algo_map = by_scn_algo.get(scn, {})
        if not algo_map:
            continue
        print(f"\n{'='*68}")
        print(f"  {scn}  —  overall per algorithm  —  duration in minutes")
        print(f"{'='*68}")
        print(f"  {'algo':<14s} {'n_runs':>7s} "
              f"{'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
        for algo in sorted(algo_map):
            vals = algo_map[algo]
            n = len(vals)
            mu = mean(vals)
            sd = stdev(vals) if n > 1 else 0.0
            print(f"  {algo:<14s} {n:>7d} "
                  f"{fmt_min(mu)} {fmt_min(sd)} "
                  f"{fmt_min(min(vals))} {fmt_min(max(vals))}")


def write_csv(durations, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "pair", "algorithm", "n_runs",
                    "mean_seconds", "std_seconds",
                    "min_seconds", "max_seconds",
                    "mean_minutes", "std_minutes"])
        for (scn, pair, algo), vals in sorted(durations.items()):
            n = len(vals)
            mu = mean(vals)
            sd = stdev(vals) if n > 1 else 0.0
            w.writerow([scn, pair, algo, n,
                        f"{mu:.3f}", f"{sd:.3f}",
                        f"{min(vals):.3f}", f"{max(vals):.3f}",
                        f"{mu/60.0:.4f}", f"{sd/60.0:.4f}"])
    print(f"\nwrote: {out_path}  ({len(durations)} rows)")


def main():
    durations = collect()
    if not durations:
        print("No runs found.")
        return
    summarise_pair(durations)
    summarise_overall(durations)
    write_csv(durations, OUT_CSV)


if __name__ == "__main__":
    main()
