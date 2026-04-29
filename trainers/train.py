import sys

import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import mlflow


class Trainer(AbstractTrainer):
    """
    This class contains the main training functions for OpenSet-AdaTime.
    """

    def __init__(self, args):
        super().__init__(args)

        if self.uniDA:
            self.results_columns = ["scenario", "run", "acc", "f1_score", "H_score", "OS_star", "UNK", "acc_mix", "src_acc"]
        else:
            self.results_columns = ["scenario", "run", "acc", "f1_score", "auroc", "H_score", "OS_star", "src_acc"]

        self.risks_columns = ["scenario", "run", "src_risk", "trg_risk"]

    def fit(self):
        # table with metrics
        table_results = pd.DataFrame(columns=self.results_columns)
        table_risks = pd.DataFrame(columns=self.risks_columns)

        # Trainer
        mlflow.set_experiment(self.experiment_description)

        for run_id in range(self.num_runs):
            with mlflow.start_run(run_name=f"{self.da_method}_run_{run_id}") as run:

                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(
                    self.experiment_description, self.da_method, self.exp_log_dir,
                    self.source_dataset, self.target_dataset, run_id)

                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Initiate the domain adaptation algorithm
                self.initialize_algorithm()
                mlflow.log_params(self.hparams)

                # Train the domain adaptation algorithm
                self.last_model, self.best_model = self.algorithm.update(
                    self.src_train_dl, self.trg_train_dl, self.loss_avg_meters,
                    self.src_val_dl, self.logger)

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, self.last_model, self.best_model)

                # Calculate metrics
                metrics = self.calculate_metrics()
                # metrics is: (acc, H_score, OS_star, UNK, acc_mix, src_acc)

                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{self.source_dataset}_to_{self.target_dataset}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

                # Log to MLflow
                if self.uniDA:
                    acc, f1, H_score, OS_star, UNK, acc_mix, src_acc = metrics
                    metrics_to_log = {
                        "Target Accuracy": acc,
                        "Target F1-score": f1,
                        "H_score": H_score,
                        "OS_star": OS_star,
                        "UNK": UNK,
                        "acc_mix": acc_mix,
                        "Source Accuracy": src_acc,
                    }
                else:
                    acc, f1, auroc, H_score, OS_star, src_acc = metrics
                    metrics_to_log = {
                        "Target Accuracy": acc,
                        "Target F1-score": f1,
                        "Target AUROC": auroc,
                        "Source Accuracy": src_acc,
                    }
                mlflow.log_metrics(metrics_to_log)

                print(f"Uploading logs from: {self.scenario_log_dir}")
                mlflow.log_artifacts(self.scenario_log_dir, artifact_path="logs_and_checkpoints")

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, self.results_columns)
        table_risks = self.add_mean_std_table(table_risks, self.risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')
