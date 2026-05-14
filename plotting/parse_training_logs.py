#!/usr/bin/env python3
"""Parse experiments_logs/curriculum/.../logs_*.log into a long CSV.

Every log file contains per-epoch loss blocks of the form:

    [Pr Epoch : 12/20]
    Pr_Src_cls_loss : 0.0775
    -------------------------------------
    [Epoch : 1/20]
    Total_loss : 0.0519
    loss_cls   : 0.0165
    loss_align : 0.0354
    -------------------------------------

Each block becomes one row per metric:
    scenario, pair, algorithm, n_unknown, seed, phase, epoch, total_epochs,
    metric, value

Phases observed:
    "Pr Epoch"      -> "pretrain"
    "Epoch"         -> "main"
    "Align Epoch"   -> "align"     (RAINCOAT phase 1)
    "Correct Epoch" -> "correct"   (RAINCOAT phase 2)

Output: analysis/training_logs.csv
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT, "experiments_logs", "curriculum")
OUT_CSV  = os.path.join(ROOT, "analysis", "training_logs.csv")

# Canonical dataset names — collapse case mismatches.
_CANON = {"mhealth": "MHEALTH", "MHEALTH": "MHEALTH",
          "Pamap2": "Pamap2", "RealWorld": "RealWorld"}

# Experiment directory name: "<src> to <tgt>_<scenario>_hard_n<N>[_fno_mean]"
EXP_RE = re.compile(
    r"^(.+?) to (.+?)_(closed_set|OSDA|PDA|UniDA)(?:_hard_n(\d+))?(_fno_mean)?$")

# Method subdir name: "<METHOD>_<scenario>_hard_n<N>[_fno_mean]"
METHOD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*?)_")

# Run subdir name: "<src>_to_<tgt>_run_<N>"
RUN_RE = re.compile(r"_run_(\d+)$")

# Epoch headers.
PHASE_HEADER = re.compile(
    r"^\[(?P<phase>Pr Epoch|Epoch|Align Epoch|Correct Epoch)\s*:\s*"
    r"(?P<epoch>\d+)/(?P<total>\d+)\]")
PHASE_NAME = {
    "Pr Epoch":      "pretrain",
    "Epoch":         "main",
    "Align Epoch":   "align",
    "Correct Epoch": "correct",
}

KV_LINE  = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_ ]*[A-Za-z0-9])\s*:\s*"
                      r"(?P<val>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")
SEPARATOR = re.compile(r"^-{5,}$")


def canon_pair(src, tgt):
    return f"{_CANON.get(src, src)} -> {_CANON.get(tgt, tgt)}"


def parse_log(path, ctx):
    """Yield row dicts for each (phase, epoch, metric) in this log."""
    cur_phase = cur_epoch = cur_total = None
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            m = PHASE_HEADER.match(line)
            if m:
                cur_phase = PHASE_NAME[m.group("phase")]
                cur_epoch = int(m.group("epoch"))
                cur_total = int(m.group("total"))
                continue
            if SEPARATOR.match(line):
                cur_phase = cur_epoch = cur_total = None
                continue
            if cur_phase is None:
                continue
            m = KV_LINE.match(line)
            if not m:
                continue
            metric = m.group("key").strip()
            try:
                val = float(m.group("val"))
            except ValueError:
                continue
            yield {**ctx,
                   "phase": cur_phase,
                   "epoch": cur_epoch,
                   "total_epochs": cur_total,
                   "metric": metric,
                   "value": val}


def walk_logs():
    rows = []
    if not os.path.isdir(LOGS_DIR):
        print(f"[error] {LOGS_DIR} not found")
        return rows

    for exp_name in sorted(os.listdir(LOGS_DIR)):
        exp_path = os.path.join(LOGS_DIR, exp_name)
        if not os.path.isdir(exp_path):
            continue
        em = EXP_RE.match(exp_name)
        if not em:
            continue
        src, tgt, scenario, n_unknown, _fno = em.groups()
        pair = canon_pair(src, tgt)
        n_unknown = int(n_unknown) if n_unknown else ""

        for method_dir in sorted(os.listdir(exp_path)):
            mp = os.path.join(exp_path, method_dir)
            if not os.path.isdir(mp):
                continue
            mm = METHOD_RE.match(method_dir)
            if not mm:
                continue
            algorithm = mm.group(1)

            for run_dir in sorted(os.listdir(mp)):
                rp = os.path.join(mp, run_dir)
                if not os.path.isdir(rp):
                    continue
                rm = RUN_RE.search(run_dir)
                if not rm:
                    continue
                seed = int(rm.group(1))

                logs = [f for f in os.listdir(rp) if f.endswith(".log")]
                if not logs:
                    continue
                # If multiple log files, take the most recently modified
                # (likely the most-complete run for this seed).
                logs.sort(key=lambda f: os.path.getmtime(os.path.join(rp, f)),
                          reverse=True)
                ctx = {"scenario": scenario, "pair": pair,
                       "algorithm": algorithm, "n_unknown": n_unknown,
                       "seed": seed}
                rows.extend(parse_log(os.path.join(rp, logs[0]), ctx))

    return rows


def main():
    rows = walk_logs()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cols = ["scenario", "pair", "algorithm", "n_unknown", "seed",
            "phase", "epoch", "total_epochs", "metric", "value"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote: {OUT_CSV}  ({len(rows)} rows)")

    # Quick summary
    from collections import Counter
    by_alg = Counter(r["algorithm"] for r in rows)
    print("\nrows per algorithm:")
    for a, c in sorted(by_alg.items()):
        print(f"  {a:<10s} {c:>7d}")


if __name__ == "__main__":
    main()
