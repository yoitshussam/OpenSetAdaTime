"""Single-shot curriculum experiment runner.

Picks N private classes ordered by feature-distance similarity-to-known and
runs one training experiment with the resulting activity mappings. One CLI
invocation = one Python process = one experiment, so bash can fan out parallel
workers exactly like the sweep .sh scripts.

Scenarios:
  OSDA  — target grows with N target-privates (OSDA ranking).
  PDA   — source grows with N source-privates (PDA ranking).
  UniDA — symmetric: source grows with N source-privates from the PDA ranking
          AND target grows with N target-privates from the OSDA ranking.
          One `--n_unknown` controls both sides.

Example:
    python run_curriculum.py \\
        --source_dataset RealWorld_male \\
        --target_dataset RealWorld_female \\
        --scenario UniDA \\
        --strategy hard \\
        --n_unknown 3 \\
        --da_method UniOT \\
        --backbone FNO \\
        --num_runs 5 \\
        --rank_variant fno_mean \\
        --ranking_key "RealWorld -> RealWorld"

Experiment name auto-set to f"{scenario}_{strategy}_n{n_unknown}"
(e.g. "UniDA_hard_n3"). The MLflow experiment becomes
f"{src} to {trg}_{exp_name}".

The ranking_key defaults to f"{src_family} -> {trg_family}" with family
suffixes (_male, _female, _PDA, _OSDA) stripped. Override with --ranking_key
when the JSON uses a different label.
"""
import argparse
import json
import os
import sys

# Ensure similarity_comparison/ is importable regardless of how this script is
# invoked (cwd, missing PYTHONPATH, etc.) — that's where compute_feature_distance_4known
# lives.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "similarity_comparison"))

import mlflow

from compute_feature_distance_4known import DATASETS, PRETTY_NAMES
from trainers.train import Trainer

mlflow.set_tracking_uri("http://127.0.0.1:5001")

RANKINGS_PATHS = {
    "min":  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "feature_distance_4known", "rankings_4known.json"),
    "mean": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "feature_distance_4known_mean", "rankings_4known_mean.json"),
    # FNO backbone + 5-seed averaged. Best-aligned with what the actual DA
    # methods see. See compute_feature_distance_4known_fno_mean.py.
    "fno_mean": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "feature_distance_4known_fno_mean",
                             "rankings_4known_fno_mean.json"),
}
# PDA rankings: source-privates ranked by similarity to source-known centroids
# in an S-trained FNO. Same JSON shape as OSDA, keyed by "S -> T", but the
# rankings depend only on S so duplicates appear across pairs sharing S.
PDA_RANKINGS_PATHS = {
    "fno_mean": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "feature_distance_4known_fno_mean_pda",
                             "rankings_4known_fno_mean_pda.json"),
}
SUFFIXES = ("_male_female_PDA", "_male_female_OSDA", "_male_female",
            "_male", "_female", "_PDA", "_OSDA")


def split_family(dataset_name):
    """Strip a known suffix to recover the family name (e.g. RealWorld)."""
    for suf in SUFFIXES:
        if dataset_name.endswith(suf):
            return dataset_name[: -len(suf)]
    return dataset_name


def pretty_to_raw(pretty_name, target_family):
    """Map a ranking entry's display name back to the raw class string for
    the target dataset family. PRETTY_NAMES has duplicates ('Standing' covers
    both 'standing' and 'Standing still'), so we filter by what's actually
    present in the target family's class list."""
    candidates = [raw for raw, pretty in PRETTY_NAMES.items() if pretty == pretty_name]
    available = set(DATASETS[target_family]["all_classes"])
    for raw in candidates:
        if raw in available:
            return raw
    raise KeyError(
        f"No raw class found for pretty name {pretty_name!r} in family "
        f"{target_family!r}. Candidates={candidates}, available={available}"
    )


def _select_privates(ranking_key, paths, rank_variant, strategy, n_unknown,
                     private_family, side_label):
    """Pick top-n private classes from a rankings JSON in hard/easy order.
    Returns the raw class names resolved against `private_family`."""
    if rank_variant not in paths:
        raise ValueError(f"rank_variant {rank_variant!r} not available for "
                         f"{side_label}. Have: {list(paths)}")
    rankings_path = paths[rank_variant]
    with open(rankings_path) as f:
        rankings = json.load(f)
    if ranking_key not in rankings:
        raise KeyError(
            f"Pair {ranking_key!r} not in {rankings_path} ({side_label}). "
            f"Available: {list(rankings)}. Run the matching feature-distance "
            f"script for this pair, or pass --ranking_key explicitly."
        )

    ordered = list(rankings[ranking_key]["rankings"])  # ascending: hardest first
    if strategy == "easy":
        ordered.reverse()
    elif strategy != "hard":
        raise ValueError(f"strategy must be 'hard' or 'easy', got {strategy!r}")

    if n_unknown < 1 or n_unknown > len(ordered):
        raise ValueError(f"n_unknown={n_unknown} out of range [1, {len(ordered)}] "
                         f"for {side_label} ({rankings_path})")

    selected_pretty = [r["unknown"] for r in ordered[:n_unknown]]
    return [pretty_to_raw(p, private_family) for p in selected_pretty]


def build_curriculum_mappings(source_dataset, target_dataset, scenario,
                              strategy, n_unknown, ranking_key=None,
                              rank_variant="min"):
    """Return (src_activity_mapping, trg_activity_mapping,
              src_privates_added, trg_privates_added).

    OSDA:  target grows with n target-private classes (source stays at 4 knowns).
    PDA:   source grows with n source-private classes (target stays at 4 knowns).
    UniDA: symmetric — source grows with n source-privates from the PDA ranking
           AND target grows with n target-privates from the OSDA ranking.
           Same `rank_variant` applies to both sides; the only variant that
           currently exists for PDA is `fno_mean`, so UniDA effectively requires
           it.
    """
    src_family = split_family(source_dataset)
    trg_family = split_family(target_dataset)

    if src_family not in DATASETS or trg_family not in DATASETS:
        raise KeyError(f"Family lookup failed: src={src_family!r} "
                       f"trg={trg_family!r}. Known families: {list(DATASETS)}")

    if ranking_key is None:
        ranking_key = f"{src_family} -> {trg_family}"

    src_mapping = dict(DATASETS[src_family]["known_mapping"])
    trg_mapping = dict(DATASETS[trg_family]["known_mapping"])
    src_priv_raw, trg_priv_raw = [], []

    if scenario == "OSDA":
        trg_priv_raw = _select_privates(ranking_key, RANKINGS_PATHS, rank_variant,
                                         strategy, n_unknown, trg_family,
                                         "OSDA target-privates")
    elif scenario == "PDA":
        src_priv_raw = _select_privates(ranking_key, PDA_RANKINGS_PATHS, rank_variant,
                                         strategy, n_unknown, src_family,
                                         "PDA source-privates")
    elif scenario == "UniDA":
        src_priv_raw = _select_privates(ranking_key, PDA_RANKINGS_PATHS, rank_variant,
                                         strategy, n_unknown, src_family,
                                         "UniDA source-privates (PDA ranking)")
        trg_priv_raw = _select_privates(ranking_key, RANKINGS_PATHS, rank_variant,
                                         strategy, n_unknown, trg_family,
                                         "UniDA target-privates (OSDA ranking)")
    else:
        raise ValueError(f"Unknown scenario {scenario!r}")

    # Map each private class to itself so the trainer treats it as private on
    # whichever side it was added to.
    for raw in src_priv_raw:
        src_mapping[raw] = raw
    for raw in trg_priv_raw:
        trg_mapping[raw] = raw

    return src_mapping, trg_mapping, src_priv_raw, trg_priv_raw


class CurriculumTrainer(Trainer):
    """Trainer that injects custom activity mappings via get_configs()."""

    def __init__(self, args, source_mapping, target_mapping, curriculum_tags):
        # Stash before super().__init__ so get_configs() (called from
        # AbstractTrainer.__init__) can read them on the first pass — avoids
        # an eager load_data() on the wrong mapping.
        self._curriculum_src_mapping = source_mapping
        self._curriculum_trg_mapping = target_mapping
        self._curriculum_tags = curriculum_tags
        super().__init__(args)

    def get_configs(self):
        dataset_configs, hparams_class = super().get_configs()
        dataset_configs.source_activity_mapping = self._curriculum_src_mapping
        dataset_configs.target_activity_mapping = self._curriculum_trg_mapping
        # Classifier output dim = number of unique source-known canonical classes.
        dataset_configs.num_classes = len(set(self._curriculum_src_mapping.values()))
        return dataset_configs, hparams_class

    def _mlflow_context_params(self):
        # Attach curriculum metadata to every run's params. Lets us slice
        # MLflow by strategy/n_unknown without grepping run names.
        params = super()._mlflow_context_params()
        params.update(self._curriculum_tags)
        return params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dataset", required=True)
    p.add_argument("--target_dataset", required=True)
    p.add_argument("--scenario", required=True, choices=["OSDA", "PDA", "UniDA"])
    p.add_argument("--strategy", required=True, choices=["hard", "easy"])
    p.add_argument("--n_unknown", type=int, required=True,
                   help="Number of private classes per side. OSDA: target-"
                        "privates added to target. PDA: source-privates added "
                        "to source. UniDA: symmetric — n added to each side.")
    p.add_argument("--da_method", required=True)
    p.add_argument("--backbone", default="FNO")
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--data_path", default="../dataset")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_dir", default="experiments_logs/curriculum")
    p.add_argument("--ranking_key", default=None,
                   help='Override pair key, e.g. "RealWorld -> Pamap2"')
    p.add_argument("--rank_variant", default="min",
                   choices=["min", "mean", "fno_mean"],
                   help="Which feature-distance ranking to use: 'min' (v1, "
                        "min cosine dist on CNN, 1 seed), 'mean' (v2, mean "
                        "cosine dist on CNN, 1 seed), or 'fno_mean' (FNO "
                        "backbone, mean cosine dist, 5-seed averaged).")
    args = p.parse_args()

    src_map, trg_map, src_unknowns, trg_unknowns = build_curriculum_mappings(
        args.source_dataset, args.target_dataset, args.scenario,
        args.strategy, args.n_unknown, args.ranking_key,
        rank_variant=args.rank_variant,
    )

    if args.scenario == "UniDA":
        unknowns_repr = (f"src=[{','.join(src_unknowns)}] | "
                         f"trg=[{','.join(trg_unknowns)}]")
    else:
        # OSDA grows target only, PDA grows source only — exactly one is non-empty.
        unknowns_repr = ",".join(src_unknowns or trg_unknowns)

    print(f"[curriculum] scenario={args.scenario}  strategy={args.strategy}  "
          f"n_unknown={args.n_unknown}  rank_variant={args.rank_variant}",
          flush=True)
    print(f"[curriculum] source mapping ({len(src_map)} classes): {src_map}", flush=True)
    print(f"[curriculum] target mapping ({len(trg_map)} classes): {trg_map}", flush=True)
    print(f"[curriculum] unknowns added (in {args.strategy} order): {unknowns_repr}", flush=True)

    # Experiment name: <scenario>_<strategy>_n<n_unknown>, with a "_mean"
    # suffix for v2 rankings so v1 and v2 don't share an MLflow experiment.
    args.exp_name = f"{args.scenario}_{args.strategy}_n{args.n_unknown}"
    if args.rank_variant != "min":
        args.exp_name += f"_{args.rank_variant}"

    curriculum_tags = {
        "curriculum_scenario": args.scenario,
        "curriculum_strategy": args.strategy,
        "curriculum_n_unknown": str(args.n_unknown),
        "curriculum_unknowns": unknowns_repr,
        "curriculum_unknowns_src": ",".join(src_unknowns),
        "curriculum_unknowns_trg": ",".join(trg_unknowns),
        "curriculum_rank_variant": args.rank_variant,
        "curriculum_ranking_key": args.ranking_key
            or f"{split_family(args.source_dataset)} -> "
               f"{split_family(args.target_dataset)}",
    }

    trainer = CurriculumTrainer(args, src_map, trg_map, curriculum_tags)
    trainer.fit()


if __name__ == "__main__":
    main()
