"""
Cross-user hyperparameter sweep on MHEALTH.

Uses MHEALTH (12 activities, 10 subjects) as a held-out dataset for
Bayesian hyperparameter search. Leave-one-subject-out: for each trial,
all 10 rotations are evaluated and the average H-score is the objective.

The found hyperparameters are then used for the final cross-dataset
experiments on PAMAP2 / RealWorld — no leakage.

After sweeping, best hparams per algorithm are saved to
  experiments_logs/crossuser_sweep/best_hparams.json
which can be pasted directly into hparams.py alg_hparams.

Usage:
    python main_sweep_crossuser.py --da_method UDA --num_sweeps 50
    python main_sweep_crossuser.py --da_method ALL --num_sweeps 30
"""
import os
import json
import argparse
import collections
import numpy as np
import pandas as pd
import optuna
import mlflow

from trainers.sweep import Trainer as SweepTrainer, sample_hparam
from configs.sweep_params import sweep_alg_hparams, sweep_train_hparams
from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class

mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

# ─── MHEALTH cross-user scenarios ───────────────────────────────────────
# 10 subjects. Leave-one-subject-out: train on 9, test on 1.
MHEALTH_USERS = [f'subject{i}' for i in range(1, 11)]

CROSS_USER_SCENARIOS = []
for i, user in enumerate(MHEALTH_USERS):
    train_users = [u for u in MHEALTH_USERS if u != user]
    test_users = [user]
    CROSS_USER_SCENARIOS.append((train_users, test_users))

# Activity mappings for open-set on MHEALTH.
# 9 shared classes, 3 target-private unknowns (Cycling, Jogging, Jump front & back).
MHEALTH_SOURCE_MAPPING = {
    'Standing still': 'Standing still',
    'Sitting and relaxing': 'Sitting and relaxing',
    'Lying down': 'Lying down',
    'Walking': 'Walking',
    'Climbing stairs': 'Climbing stairs',
    'Waist bends forward': 'Waist bends forward',
    'Frontal elevation of arms': 'Frontal elevation of arms',
    'Knees bending (crouching)': 'Knees bending (crouching)',
    'Running': 'Running',
}

MHEALTH_TARGET_MAPPING = {
    **MHEALTH_SOURCE_MAPPING,
    # Target-private (unknown) classes:
    'Cycling': 'Cycling',
    'Jogging': 'Jogging',
    'Jump front & back': 'Jump front & back',
}


class CrossUserSweepTrainer(SweepTrainer):
    """Extends SweepTrainer to support cross-user on a single dataset."""

    def __init__(self, args, source_users, target_users):
        # Set user lists before super().__init__ calls load_data()
        self.source_users = source_users
        self.target_users = target_users
        # Store MHEALTH mappings — will be applied in load_data override
        self._src_mapping = getattr(args, '_mhealth_source_mapping', None)
        self._trg_mapping = getattr(args, '_mhealth_target_mapping', None)
        super().__init__(args)

    def load_data(self):
        """Override to inject MHEALTH activity mappings before loading."""
        if self._src_mapping is not None:
            self.dataset_configs.source_activity_mapping = self._src_mapping
            self.dataset_configs.target_activity_mapping = self._trg_mapping
        super().load_data()

    def sweep(self):
        """Not used — the outer loop drives the sweep."""
        raise NotImplementedError("Use run_crossuser_sweep() instead")


def run_crossuser_sweep(args):
    """Run Bayesian HP search across MHEALTH leave-one-out scenarios."""

    if args.hp_search_strategy == 'bayes':
        sampler = optuna.samplers.TPESampler(seed=42)
    elif args.hp_search_strategy == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)

    study_name = f"{args.da_method}_mhealth_crossuser_sweep"
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # maximize H-score
        sampler=sampler,
    )

    def objective(trial):
        # Sample hyperparameters
        hparams = {}
        for name, spec in sweep_train_hparams.items():
            hparams[name] = sample_hparam(trial, name, spec)
        alg_space = sweep_alg_hparams.get(args.da_method, {})
        for name, spec in alg_space.items():
            hparams[name] = sample_hparam(trial, name, spec)

        h_scores = []

        for scenario_idx, (src_users, trg_users) in enumerate(CROSS_USER_SCENARIOS):
            fix_randomness(scenario_idx)

            # Build trainer for this scenario
            trainer = CrossUserSweepTrainer(args, src_users, trg_users)
            trainer.hparams = {**trainer.hparams, **hparams}

            trainer.logger, trainer.scenario_log_dir = starting_logs(
                trainer.experiment_description, trainer.da_method, trainer.exp_log_dir,
                trainer.source_dataset, trainer.target_dataset, scenario_idx)

            trainer.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
            trainer.initialize_algorithm()

            try:
                trainer.last_model, trainer.best_model = trainer.algorithm.update(
                    trainer.src_train_dl, trainer.trg_train_dl, trainer.loss_avg_meters,
                    trainer.src_val_dl, trainer.logger)

                metrics = trainer.calculate_metrics()
                # metrics for uniDA: (acc, f1, H_score, OS_star, UNK, acc_mix, src_acc)
                h_score = metrics[2]
            except Exception as e:
                print(f"  Scenario {scenario_idx} failed: {e}")
                h_score = 0.0

            h_scores.append(h_score)
            print(f"  Scenario {scenario_idx} (test={trg_users[0]}): H={h_score:.4f}")

        avg_h = np.mean(h_scores)
        print(f"Trial {trial.number}: avg H-score = {avg_h:.4f} | params = {hparams}")
        return avg_h

    experiment_name = f"mhealth_crossuser_sweep_{args.da_method}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{args.da_method}_sweep"):
        study.optimize(objective, n_trials=args.num_sweeps)

        best = study.best_trial
        print(f"\n===== Best Trial =====")
        print(f"  H-score: {best.value:.4f}")
        print(f"  Params: {best.params}")

        mlflow.log_params(best.params)
        mlflow.log_metric("best_avg_H_score", best.value)

    return study


def save_best_hparams(all_results, save_dir):
    """Save best hparams per algorithm as JSON pasteable into hparams.py."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "best_hparams.json")

    # Convert numpy types to native Python for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = {}
    for method, info in all_results.items():
        clean[method] = {
            'best_H_score': to_native(info['best_H_score']),
            'hparams': {k: to_native(v) for k, v in info['hparams'].items()},
        }

    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=4)

    # Also print in a format ready to paste into hparams.py
    print(f"\nBest hparams saved to: {out_path}")
    print("\n# ---- Paste into hparams.py alg_hparams ----")
    for method, info in clean.items():
        print(f"'{method}': {json.dumps(info['hparams'], indent=4)},")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-user HP sweep on MHEALTH')

    parser.add_argument('--da_method',           default='ALL',    type=str)
    parser.add_argument('--data_path',           default=r'../dataset', type=str)
    parser.add_argument('--backbone',            default='FNO',    type=str)
    parser.add_argument('--num_runs',            default=1,        type=int)
    parser.add_argument('--device',              default="cuda",   type=str)
    parser.add_argument('--exp_name',            default='mhealth_sweep', type=str)
    parser.add_argument('--num_sweeps',          default=30,       type=int)
    parser.add_argument('--hp_search_strategy',  default="bayes",  type=str)
    parser.add_argument('--metric_to_minimize',  default="H_score", type=str)
    parser.add_argument('--save_dir',            default='experiments_logs/crossuser_sweep', type=str)

    # Fixed: both source and target are MHEALTH
    parser.add_argument('--source_dataset',      default='mhealth', type=str)
    parser.add_argument('--target_dataset',      default='mhealth', type=str)

    # Scenario groups (same as main.py)
    SCENARIO_METHODS = {
        "CLOSED": ["NO_ADAPT", "TARGET_ONLY"],
        "OSDA":   ["OSBP", "OVANet"],
        "UniDA":  ["UDA", "DANCE", "PPOT", "UniOT", "UniJDOT", "RAINCOAT", "TSFA"],
        "PDA":    ["SPADA", "PDAAN"],
    }
    ALL_METHODS = [m for ms in SCENARIO_METHODS.values() for m in ms]

    args = parser.parse_args()

    # Override activity mappings for MHEALTH
    args._mhealth_source_mapping = MHEALTH_SOURCE_MAPPING
    args._mhealth_target_mapping = MHEALTH_TARGET_MAPPING

    if args.da_method == 'ALL':
        methods = ALL_METHODS
    else:
        methods = [args.da_method]

    all_results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Sweeping: {method}")
        print(f"{'='*60}")
        args.da_method = method
        study = run_crossuser_sweep(args)

        best = study.best_trial
        all_results[method] = {
            'best_H_score': best.value,
            'hparams': dict(best.params),
        }

    # Save all best hparams to JSON
    save_best_hparams(all_results, args.save_dir)
