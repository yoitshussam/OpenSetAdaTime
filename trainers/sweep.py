import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
import os
import optuna
import mlflow
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import math

from configs.sweep_params import sweep_alg_hparams, sweep_train_hparams
from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def _safe_metrics(d):
    """Drop NaN/Inf values — MLflow's GraphQL layer cannot serialize them."""
    return {k: float(v) for k, v in d.items()
            if v is not None and math.isfinite(float(v))}


def sample_hparam(trial, name, spec):
    """Sample a hyperparameter from an Optuna trial given a sweep_params spec."""
    kind = spec[0]
    if kind == 'categorical':
        return trial.suggest_categorical(name, spec[1])
    elif kind == 'float':
        step = spec[3] if len(spec) > 3 else None
        return trial.suggest_float(name, spec[1], spec[2], step=step)
    elif kind == 'float_log':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif kind == 'int':
        return trial.suggest_int(name, spec[1], spec[2])
    else:
        raise ValueError(f"Unknown hparam spec type: {kind}")


class Trainer(AbstractTrainer):
    """
    Sweep trainer using Optuna for hyperparameter search and MLflow for logging.
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.num_sweeps = args.num_sweeps
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        if self.uniDA:
            self.results_columns = ["scenario", "run", "acc", "f1_score", "H_score", "OS_star", "UNK", "acc_mix", "src_acc"]
        else:
            self.results_columns = ["scenario", "run", "acc", "f1_score", "auroc", "H_score", "OS_star", "src_acc"]

        self.risks_columns = ["scenario", "run", "src_risk", "trg_risk"]

        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def sweep(self):
        """Run Optuna hyperparameter sweep with MLflow logging."""
        # Choose Optuna sampler
        if self.hp_search_strategy == 'bayes':
            sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=3)
        elif self.hp_search_strategy == 'random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif self.hp_search_strategy == 'grid':
            # Grid sampler needs the full search space upfront
            search_space = self._build_grid_search_space()
            sampler = optuna.samplers.GridSampler(search_space)
        else:
            sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=3)

        study_name = f"{self.da_method}_{self.backbone}_{self.source_dataset}_to_{self.target_dataset}"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=sampler,
        )
        study.optimize(self._objective, n_trials=self.num_sweeps)

        # Log best trial
        best = study.best_trial
        print(f"\n===== Best Trial =====")
        print(f"  Value ({self.metric_to_minimize}): {best.value:.4f}")
        print(f"  Params: {best.params}")

        return study

    def _build_grid_search_space(self):
        """Build grid search space dict for Optuna GridSampler."""
        space = {}
        merged = {**sweep_train_hparams, **sweep_alg_hparams.get(self.da_method, {})}
        for name, spec in merged.items():
            kind = spec[0]
            if kind == 'categorical':
                space[name] = spec[1]
            elif kind in ('float', 'float_log'):
                if len(spec) > 3 and spec[3] is not None:
                    # Use the step to generate the grid
                    space[name] = np.arange(spec[1], spec[2] + spec[3], spec[3]).tolist()
                else:
                    space[name] = np.linspace(spec[1], spec[2], 5).tolist()
            elif kind == 'int':
                space[name] = list(range(spec[1], spec[2] + 1))
        return space

    def _sample_hparams(self, trial):
        """Sample all hyperparameters for a trial."""
        hparams = {}
        # Train hparams (shared across algorithms)
        for name, spec in sweep_train_hparams.items():
            hparams[name] = sample_hparam(trial, name, spec)
        # Algorithm-specific hparams (override train hparams if same key)
        alg_space = sweep_alg_hparams.get(self.da_method, {})
        for name, spec in alg_space.items():
            hparams[name] = sample_hparam(trial, name, spec)
        return hparams

    def _objective(self, trial):
        """Optuna objective: train with sampled hparams, return metric to minimize."""
        import time, gc
        t_start = time.time()
        print(f">>> [trial {trial.number}] {self.da_method} starting", flush=True)

        # Free CUDA cache + previous-trial algorithm to avoid GPU drift across
        # many trials of heavy methods (TSFA, RAINCOAT...).
        if hasattr(self, 'algorithm'):
            del self.algorithm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Fix global RNGs per trial so hparam sampling and any pre-training
        # randomness are reproducible at the trial level. The inner per-run
        # fix_randomness(run_id) below reseeds for each training run.
        fix_randomness(trial.number)

        # Sample hyperparameters
        sampled = self._sample_hparams(trial)

        # Merge with base hparams (sampled values override defaults)
        prev_bs = self.hparams.get("batch_size")
        self.hparams = {**self.hparams, **sampled}

        # Rebuild dataloaders ONLY if batch_size changed. Without this guard
        # every trial re-reads ~440 MB of pkls silently (verbose=0), looking
        # exactly like a freeze between trials. Queue-based methods (UniOT)
        # still get a fresh loader whenever they need one.
        if self.hparams.get("batch_size") != prev_bs:
            print(f">>> [trial {trial.number}] reloading data (batch_size {prev_bs} -> {self.hparams.get('batch_size')})", flush=True)
            self.load_data()

        experiment_name = f"sweep_{self.source_dataset}_to_{self.target_dataset}_{self.exp_name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{self.da_method}_trial_{trial.number}"):
            mlflow.log_params(sampled)

            table_results = pd.DataFrame(columns=self.results_columns)
            table_risks = pd.DataFrame(columns=self.risks_columns)

            for run_id in range(self.num_runs):
                fix_randomness(run_id)

                self.logger, self.scenario_log_dir = starting_logs(
                    self.experiment_description, self.da_method, self.exp_log_dir,
                    self.source_dataset, self.target_dataset, run_id)

                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                self.initialize_algorithm()

                self.last_model, self.best_model = self.algorithm.update(
                    self.src_train_dl, self.trg_train_dl, self.loss_avg_meters,
                    self.src_val_dl, self.logger)

                self.save_checkpoint(self.home_path, self.scenario_log_dir,
                                     self.last_model, self.best_model)

                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                scenario = f"{self.source_dataset}_to_{self.target_dataset}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

                # Log per-run metrics
                if self.uniDA:
                    acc, f1_score, H_score, OS_star, UNK, acc_mix, src_acc = metrics
                    mlflow.log_metrics(_safe_metrics({
                        f"run_{run_id}/acc": acc,
                        f"run_{run_id}/f1_score": f1_score,
                        f"run_{run_id}/H_score": H_score,
                        f"run_{run_id}/OS_star": OS_star,
                        f"run_{run_id}/UNK": UNK,
                        f"run_{run_id}/src_acc": src_acc,
                    }))
                else:
                    acc, f1_score, auroc, _, _, src_acc = metrics
                    mlflow.log_metrics(_safe_metrics({
                        f"run_{run_id}/acc": acc,
                        f"run_{run_id}/f1_score": f1_score,
                        f"run_{run_id}/auroc": auroc,
                        f"run_{run_id}/src_acc": src_acc,
                    }))

                src_risk, trg_risk = risks
                mlflow.log_metrics(_safe_metrics({
                    f"run_{run_id}/src_risk": src_risk,
                    f"run_{run_id}/trg_risk": trg_risk,
                }))

            # Compute and log averages
            avg_src_risk = table_risks["src_risk"].mean()
            avg_trg_risk = table_risks["trg_risk"].mean()
            summary = {"avg_src_risk": avg_src_risk, "avg_trg_risk": avg_trg_risk}

            # `acc` and `f1_score` are in both column sets; the rest depend on scenario.
            summary["avg_acc"] = table_results["acc"].mean()
            summary["avg_f1_score"] = table_results["f1_score"].mean()
            if self.uniDA:
                for col in ["H_score", "OS_star", "UNK", "acc_mix", "src_acc"]:
                    summary[f"avg_{col}"] = table_results[col].mean()
            else:
                for col in ["auroc", "src_acc"]:
                    summary[f"avg_{col}"] = table_results[col].mean()

            mlflow.log_metrics(_safe_metrics(summary))

            # Save tables
            table_results = self.add_mean_std_table(table_results, self.results_columns)
            table_risks = self.add_mean_std_table(table_risks, self.risks_columns)
            self.save_tables_to_file(table_results, f'sweep_trial_{trial.number}_results')
            self.save_tables_to_file(table_risks, f'sweep_trial_{trial.number}_risks')

        print(f">>> [trial {trial.number}] {self.da_method} done in {time.time() - t_start:.1f}s", flush=True)

        # Return the metric to minimize. Optuna minimizes; for "max-better"
        # metrics (H_score, f1_score, acc) we negate.
        metric = self.metric_to_minimize

        # H_score is undefined for closed-set / PDA — auto-fall-back to F1.
        if metric == 'H_score' and not self.uniDA:
            metric = 'f1_score'

        if metric == 'trg_risk':
            return avg_trg_risk
        if metric == 'src_risk':
            return avg_src_risk
        if metric == 'H_score':
            return -summary.get('avg_H_score', 0.0)
        if metric == 'f1_score':
            return -summary.get('avg_f1_score', 0.0)
        if metric == 'acc':
            return -summary.get('avg_acc', 0.0)
        return avg_trg_risk
