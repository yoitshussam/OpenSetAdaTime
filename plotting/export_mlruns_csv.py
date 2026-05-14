#!/usr/bin/env python3
"""Export mlruns/ into a long-format CSV (one row per FINISHED, active run).

Columns:
  scenario          closed_set | OSDA | PDA | UniDA
  pair              "<src> -> <trg>" with mhealth canonicalised to MHEALTH
  algorithm         e.g. UDA, OVANet, RAINCOAT, ...
  n_unknown         integer for OSDA/PDA/UniDA, blank for closed_set
  seed              integer suffix from the run name <ALGO>_run_<N>
  duration_sec      (end_time - start_time) / 1000 from the run meta.yaml
  H_score           final value of the H_score metric file (empty if absent)
  OS_star           final value of OS_star
  UNK               final value of UNK
  target_f1         final value of "Target F1-score"
  source_f1         final value of "Source F1-score" (closed_set only)
  source_acc        final value of "Source Accuracy"

Soft-deleted runs (lifecycle_stage=deleted) and runs with status != FINISHED
are skipped. Sweep runs (*_trial_<N>) are skipped.

Output: analysis/runs.csv  (path is overridable with --out)
"""
import argparse
import csv
import os
import re
from collections import defaultdict

MLRUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "analysis", "runs.csv")
STATUS_FINISHED = "3"

RUN_NAME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*?)_run_(\d+)$")
SCENARIO_PATTERNS = [
    (re.compile(r"_closed_set$"),                 "closed_set"),
    (re.compile(r"_OSDA_hard_n\d+(_fno_mean)?$"), "OSDA"),
    (re.compile(r"_PDA_hard_n\d+(_fno_mean)?$"),  "PDA"),
    (re.compile(r"_UniDA_hard_n\d+(_fno_mean)?$"),"UniDA"),
]

# Metric file name on disk -> CSV column name.
METRIC_FILES = {
    "H_score":          "H_score",
    "OS_star":          "OS_star",
    "UNK":              "UNK",
    "Target F1-score":  "target_f1",
    "Source F1-score":  "source_f1",
    "Source Accuracy":  "source_acc",
}

# Pair-name canonicalisation: closed_set used lowercase "mhealth", open-set
# scenarios used uppercase "MHEALTH". Collapse to MHEALTH everywhere.
_CANON = {"mhealth": "MHEALTH", "MHEALTH": "MHEALTH",
          "Pamap2": "Pamap2", "RealWorld": "RealWorld"}


def canon_pair(src, tgt):
    return f"{_CANON.get(src, src)} -> {_CANON.get(tgt, tgt)}"


def read_meta(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def classify_experiment(exp_name):
    """Return (scenario, n_unknown, pair) or (None, None, None)."""
    for pat, scenario in SCENARIO_PATTERNS:
        if pat.search(exp_name):
            n_match = re.search(r"_n(\d+)", exp_name)
            n = int(n_match.group(1)) if n_match else None
            pm = re.match(r"^(.+?) to (.+?)_(?:closed_set|OSDA|PDA|UniDA)",
                          exp_name)
            pair = canon_pair(pm.group(1), pm.group(2)) if pm else None
            return scenario, n, pair
    return None, None, None


def read_final_metric(metric_path):
    """Return the value column (2nd whitespace token) of the last non-empty
    line in an MLflow metric file. Return None if the file is missing/empty."""
    try:
        with open(metric_path) as f:
            last = None
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if not last:
            return None
        parts = last.split()
        if len(parts) < 2:
            return None
        return float(parts[1])
    except (OSError, ValueError):
        return None


def collect_rows():
    rows = []
    skipped = defaultdict(int)
    if not os.path.isdir(MLRUNS_DIR):
        print(f"[error] {MLRUNS_DIR} not found")
        return rows, skipped

    for exp_entry in sorted(os.listdir(MLRUNS_DIR)):
        exp_dir = os.path.join(MLRUNS_DIR, exp_entry)
        exp_meta_path = os.path.join(exp_dir, "meta.yaml")
        if not os.path.isfile(exp_meta_path):
            continue
        em = read_meta(exp_meta_path)
        exp_name = em.get("name")
        if not exp_name or exp_name == "Default":
            continue
        if em.get("lifecycle_stage") == "deleted":
            skipped["experiment_deleted"] += 1
            continue
        scenario, n_unknown, pair = classify_experiment(exp_name)
        if scenario is None or pair is None:
            skipped["unrecognised_experiment"] += 1
            continue

        for run_entry in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_entry)
            if not os.path.isdir(run_dir):
                continue
            rm_path = os.path.join(run_dir, "meta.yaml")
            if not os.path.isfile(rm_path):
                continue
            rm = read_meta(rm_path)
            if rm.get("lifecycle_stage") == "deleted":
                skipped["run_deleted"] += 1
                continue
            if rm.get("status") != STATUS_FINISHED:
                skipped["run_not_finished"] += 1
                continue
            run_name = rm.get("run_name")
            if not run_name:
                tag_path = os.path.join(run_dir, "tags", "mlflow.runName")
                if os.path.isfile(tag_path):
                    with open(tag_path) as f:
                        run_name = f.read().strip()
            if not run_name:
                skipped["no_run_name"] += 1
                continue
            m = RUN_NAME_RE.match(run_name)
            if not m:
                skipped["sweep_or_unknown_run_name"] += 1
                continue
            algo, seed = m.group(1), int(m.group(2))

            try:
                start_ms = int(rm["start_time"])
                end_ms   = int(rm["end_time"])
                duration_sec = (end_ms - start_ms) / 1000.0 if end_ms > start_ms else None
            except (KeyError, ValueError):
                duration_sec = None

            metrics_dir = os.path.join(run_dir, "metrics")
            mvals = {col: "" for col in METRIC_FILES.values()}
            if os.path.isdir(metrics_dir):
                for fname, col in METRIC_FILES.items():
                    v = read_final_metric(os.path.join(metrics_dir, fname))
                    if v is not None:
                        mvals[col] = f"{v:.6f}"

            rows.append({
                "scenario":     scenario,
                "pair":         pair,
                "algorithm":    algo,
                "n_unknown":    "" if n_unknown is None else n_unknown,
                "seed":         seed,
                "duration_sec": "" if duration_sec is None else f"{duration_sec:.3f}",
                **mvals,
            })
    return rows, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Output CSV path (default: {DEFAULT_OUT})")
    args = p.parse_args()

    rows, skipped = collect_rows()
    if not rows:
        print("No rows to write.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = (["scenario", "pair", "algorithm", "n_unknown", "seed",
             "duration_sec"] + list(METRIC_FILES.values()))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote: {args.out}  ({len(rows)} rows, {len(cols)} cols)")

    by_scn = defaultdict(int)
    for r in rows:
        by_scn[r["scenario"]] += 1
    print("\nRows per scenario:")
    for scn in ("closed_set", "OSDA", "PDA", "UniDA"):
        print(f"  {scn:<10s} {by_scn.get(scn, 0):>6d}")

    if skipped:
        print("\nSkipped:")
        for k, v in sorted(skipped.items()):
            print(f"  {k:<26s} {v}")


if __name__ == "__main__":
    main()
