#!/usr/bin/env python3
"""Audit mlruns/ for 5-run coverage per (pair, scenario) experiment.

Walks every experiment under ./mlruns, classifies it by name into one of
{closed_set, OSDA, PDA, UniDA}, and checks whether the expected set of
algorithms has exactly EXPECTED_RUNS runs each.

Reports:
  MISSING   — expected algorithm has 0 runs in this experiment
  PARTIAL   — expected algorithm has 1..EXPECTED_RUNS-1 runs
  EXCESS    — algorithm has >EXPECTED_RUNS runs
  UNEXPECTED— algorithm with runs that's not in the expected set (info)

Run names of the form <ALGO>_trial_<N> are sweep/hparam-tuning artefacts
and are excluded from the count — only <ALGO>_run_<N> runs are counted.
"""
import json
import os
import re
from collections import defaultdict

MLRUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
EXPECTED_RUNS = 5

# Ranking JSONs — used to derive the expected n-range per (pair, scenario).
# OSDA: ranking gives target-private classes per pair; n_max = len(rankings).
# PDA:  ranking gives source-private classes per pair; n_max = len(rankings).
# UniDA: takes the n hardest of BOTH lists → n_max = min(OSDA, PDA).
OSDA_RANKING_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "feature_distance_4known_fno_mean", "rankings_4known_fno_mean.json")
PDA_RANKING_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "feature_distance_4known_fno_mean_pda", "rankings_4known_fno_mean_pda.json")

# Official 5-seed run name pattern. Excludes "_trial_<N>" sweep runs.
RUN_NAME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*?)_run_(\d+)$")

# Experiment name → scenario tag.
# Order matters: longer/more-specific suffixes first.
SCENARIO_PATTERNS = [
    (re.compile(r"_closed_set$"),                 "closed_set"),
    (re.compile(r"_OSDA_hard_n\d+(_fno_mean)?$"), "OSDA"),
    (re.compile(r"_PDA_hard_n\d+(_fno_mean)?$"),  "PDA"),
    (re.compile(r"_UniDA_hard_n\d+(_fno_mean)?$"),"UniDA"),
]

UNIDA_METHODS = {"UDA", "OVANet", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT"}
OSDA_METHODS  = {"OSBP", "TSFA"}
PDA_METHODS   = {"SPADA", "PDAAN"}

# Closed-set baseline + DA methods that are also benchmarked closed-set.
CLOSED_METHODS = {
    "ACON", "AdvSKM", "CDAN", "CLUDA", "CoDATS", "CoTMix", "DAAN", "DANN",
    "DDC", "Deep_Coral", "DIRT", "DSAN", "HoMM", "MMDA", "SASA", "SSSS_TSA",
    "SWL_Adapt", "uDAR",
} | UNIDA_METHODS | OSDA_METHODS | PDA_METHODS

EXPECTED = {
    "closed_set": CLOSED_METHODS,
    "OSDA":       OSDA_METHODS | UNIDA_METHODS,
    "PDA":        PDA_METHODS  | UNIDA_METHODS,
    "UniDA":      UNIDA_METHODS,
}


def read_meta(meta_path):
    """Return (name, lifecycle_stage) from a meta.yaml. Either may be None."""
    name, lifecycle = None, None
    try:
        with open(meta_path) as f:
            for line in f:
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("lifecycle_stage:"):
                    lifecycle = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return name, lifecycle


def classify_experiment(exp_name):
    """Return (scenario, n_unknown_or_None, pair_or_None).

    pair = "<src> -> <trg>" (ranking-JSON key format) when the name has the
    "<src> to <trg>_<scenario>_..." shape; None for closed_set or
    unrecognised names.
    """
    for pat, scenario in SCENARIO_PATTERNS:
        if pat.search(exp_name):
            n_match = re.search(r"_n(\d+)", exp_name)
            n = int(n_match.group(1)) if n_match else None
            pair = None
            pm = re.match(r"^(.+?) to (.+?)_(?:closed_set|OSDA|PDA|UniDA)_", exp_name)
            if pm:
                pair = f"{pm.group(1)} -> {pm.group(2)}"
            return scenario, n, pair
    return None, None, None


def load_n_max():
    """Build {(pair, scenario) -> n_max} from the ranking JSONs.

    Returns an empty dict (with a warning) if either JSON is missing — the
    n-coverage check will simply be skipped in that case.
    """
    n_max = {}
    try:
        with open(OSDA_RANKING_JSON) as f:
            osda = json.load(f)
        for pair, payload in osda.items():
            if isinstance(payload, dict) and "rankings" in payload:
                n_max[(pair, "OSDA")] = len(payload["rankings"])
    except OSError:
        print(f"[warn] OSDA ranking JSON not readable: {OSDA_RANKING_JSON}")
    try:
        with open(PDA_RANKING_JSON) as f:
            pda = json.load(f)
        for pair, payload in pda.items():
            if isinstance(payload, dict) and "rankings" in payload:
                n_max[(pair, "PDA")] = len(payload["rankings"])
    except OSError:
        print(f"[warn] PDA ranking JSON not readable: {PDA_RANKING_JSON}")
    # UniDA = min of both budgets at the same pair.
    pairs = {p for (p, s) in n_max}
    for pair in pairs:
        o = n_max.get((pair, "OSDA"))
        p = n_max.get((pair, "PDA"))
        if o is not None and p is not None:
            n_max[(pair, "UniDA")] = min(o, p)
    return n_max


def collect_runs(exp_dir):
    """Return {algo_name: run_count} for active *_run_<N>* runs in this experiment.
    Runs whose meta.yaml has lifecycle_stage=deleted are skipped — MLflow's
    "delete" leaves files on disk and only flips the lifecycle flag."""
    counts = defaultdict(int)
    seen_seeds = defaultdict(set)
    for entry in os.listdir(exp_dir):
        run_dir = os.path.join(exp_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        run_meta = os.path.join(run_dir, "meta.yaml")
        if os.path.isfile(run_meta):
            _, lifecycle = read_meta(run_meta)
            if lifecycle == "deleted":
                continue
        run_name_path = os.path.join(run_dir, "tags", "mlflow.runName")
        if not os.path.isfile(run_name_path):
            continue
        with open(run_name_path) as f:
            run_name = f.read().strip()
        m = RUN_NAME_RE.match(run_name)
        if not m:
            continue
        algo, seed_idx = m.group(1), int(m.group(2))
        # Dedupe by seed index — duplicate seed runs (re-runs of same seed) count once.
        if seed_idx in seen_seeds[algo]:
            continue
        seen_seeds[algo].add(seed_idx)
        counts[algo] += 1
    return counts


def audit(verbose=False):
    if not os.path.isdir(MLRUNS_DIR):
        print(f"[error] {MLRUNS_DIR} not found")
        return

    issues = []
    audited = 0
    skipped_unknown = []
    # {(pair, scenario) -> set(n)} seen in mlruns (active experiments only).
    seen_n = defaultdict(set)

    for entry in sorted(os.listdir(MLRUNS_DIR)):
        exp_dir = os.path.join(MLRUNS_DIR, entry)
        meta_path = os.path.join(exp_dir, "meta.yaml")
        if not os.path.isfile(meta_path):
            continue
        exp_name, exp_lifecycle = read_meta(meta_path)
        if not exp_name or exp_name == "Default":
            continue
        if exp_lifecycle == "deleted":
            continue

        scenario, n_unknown, pair = classify_experiment(exp_name)
        if scenario is None:
            skipped_unknown.append(exp_name)
            continue
        if pair is not None and n_unknown is not None:
            seen_n[(pair, scenario)].add(n_unknown)

        expected = EXPECTED[scenario]
        counts = collect_runs(exp_dir)
        audited += 1

        per_exp_issues = []
        for algo in sorted(expected):
            c = counts.get(algo, 0)
            if c == 0:
                per_exp_issues.append(("MISSING",    algo, c))
            elif c < EXPECTED_RUNS:
                per_exp_issues.append(("PARTIAL",    algo, c))
            elif c > EXPECTED_RUNS:
                per_exp_issues.append(("EXCESS",     algo, c))

        unexpected = sorted(set(counts) - expected)
        for algo in unexpected:
            per_exp_issues.append(("UNEXPECTED", algo, counts[algo]))

        if per_exp_issues:
            issues.append((exp_name, scenario, n_unknown, per_exp_issues))
        elif verbose:
            print(f"[OK] {exp_name}  ({scenario}, n={n_unknown})  "
                  f"all {len(expected)} algos × {EXPECTED_RUNS} runs")

    print(f"\nAudited {audited} experiments under {MLRUNS_DIR}")
    print(f"Expected runs per algorithm per experiment: {EXPECTED_RUNS}\n")

    # ---- n-coverage check per (pair, scenario) ----
    n_max = load_n_max()
    if n_max:
        print(f"\n{'='*78}\n  n-COVERAGE per (pair, scenario)\n{'='*78}")
        # Group by scenario for readability.
        by_scn = defaultdict(list)
        for (pair, scn), nmax in n_max.items():
            by_scn[scn].append((pair, nmax))
        for scn in ("OSDA", "PDA", "UniDA"):
            entries = sorted(by_scn.get(scn, []))
            if not entries:
                continue
            print(f"\n  {scn}:")
            for pair, nmax in entries:
                expected_ns = set(range(1, nmax + 1))
                got = seen_n.get((pair, scn), set())
                missing = sorted(expected_ns - got)
                extra   = sorted(got - expected_ns)
                got_sorted = sorted(got)
                status = "OK" if not missing and not extra else "GAPS"
                line = (f"    [{status:<4s}] {pair:<28s} "
                        f"expected n=1..{nmax}  got={got_sorted}")
                print(line)
                if missing:
                    print(f"           missing n: {missing}")
                if extra:
                    print(f"           unexpected n (beyond ranking length): {extra}")
        # Pairs that have runs but no ranking entry (sanity).
        unknown_pairs = sorted(p for (p, s) in seen_n if (p, s) not in n_max and s != "closed_set")
        if unknown_pairs:
            print("\n  Pairs/scenarios with runs but no ranking entry "
                  "(can't compute expected n_max):")
            for p in unknown_pairs:
                print(f"    - {p}")

    if skipped_unknown:
        print("Experiments with unrecognised name (skipped):")
        for n in skipped_unknown:
            print(f"  - {n}")
        print()

    if not issues:
        print("All experiments fully covered.")
        return

    by_scenario = defaultdict(list)
    for exp_name, scenario, n, lst in issues:
        by_scenario[scenario].append((exp_name, n, lst))

    for scenario in ("closed_set", "OSDA", "PDA", "UniDA"):
        if scenario not in by_scenario:
            continue
        print(f"\n{'='*78}\n  {scenario.upper()}  —  {len(by_scenario[scenario])} experiment(s) with issues\n{'='*78}")
        for exp_name, n, lst in sorted(by_scenario[scenario]):
            print(f"\n  Experiment: {exp_name}"
                  + (f"  [n={n}]" if n is not None else ""))
            for kind, algo, c in sorted(lst, key=lambda x: (x[0], x[1])):
                print(f"    {kind:<11s} {algo:<20s} runs={c}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Also print experiments that are fully covered.")
    args = p.parse_args()
    audit(verbose=args.verbose)
