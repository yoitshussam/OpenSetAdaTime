import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import copy
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
from dataloader.dataloader import data_generator, few_shot_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class AbstractTrainer(object):
    """
    This class contains the main training functions for OpenSet-AdaTime.
    """

    def __init__(self, args):
        self.da_method = args.da_method
        self.source_dataset = args.source_dataset
        self.target_dataset = args.target_dataset

        self.backbone = args.backbone
        self.device = torch.device(args.device)

        # Derive scenario + uniDA flag from the algorithm's SCENARIO attribute.
        # Methods declare their scenario; the trainer follows. No CLI flag needed.
        self.scenario = getattr(get_algorithm_class(self.da_method), 'SCENARIO', 'UniDA')
        self.uniDA = self.scenario in ('OSDA', 'UniDA')
        print(f"[scenario] {self.da_method} -> {self.scenario} (uniDA={self.uniDA})")

        # Exp Description
        self.exp_name = args.exp_name
        self.experiment_description = f"{args.source_dataset} to {args.target_dataset}_{args.exp_name}"
        self.run_description = f"{args.da_method}_{args.exp_name}"

        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.source_data_path = os.path.join(args.data_path, self.source_dataset)
        self.target_data_path = os.path.join(args.data_path, self.target_dataset)

        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, self.run_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()
        # FNO backbone: set flag so classifiers/discriminators use the wider feature dim
        if args.backbone == "FNO":
            self.dataset_configs.isFNO = True

        # Compute effective feature dimension (used by classifiers, discriminators, etc.)
        # FNO output = CNN features + 2 * fourier_modes (amplitude + phase)
        if self.dataset_configs.isFNO:
            self.dataset_configs.feat_dim = (self.dataset_configs.features_len * self.dataset_configs.final_out_channels
                                             + 2 * self.dataset_configs.fourier_modes)
        else:
            self.dataset_configs.feat_dim = self.dataset_configs.features_len * self.dataset_configs.final_out_channels

        # Set da_method on configs so dataloader can check for return_index
        self.dataset_configs.da_method = self.da_method

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method], **self.hparams_class.train_params}

        # Load data (with open-set label encoding)
        self.load_data()

        # Initialize metrics
        self.init_metrics()

    def set_hparams(self, da_method):
        self.hparams = {**self.hparams_class.alg_hparams[da_method], **self.hparams_class.train_params}

    def init_metrics(self):
        self.num_classes = self.dataset_configs.num_classes
        if self.uniDA:
            # +1 for the unknown class in metrics computation
            metric_num_classes = self.num_classes + 1
        else:
            metric_num_classes = self.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=metric_num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=metric_num_classes, average="weighted")
        self.AUROC = AUROC(task="multiclass", num_classes=metric_num_classes)

    def initialize_algorithm(self):
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def train_model(self):
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)
        self.last_model, self.best_model = self.algorithm.update(
            self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.src_val_dl, self.logger)
        return self.last_model, self.best_model

    def evaluate(self, test_loader, src=False):
        """Delegate evaluation to the algorithm (handles per-algorithm unknown detection)."""
        self.loss, self.full_preds, self.full_labels = self.algorithm.evaluate(
            test_loader, self.trg_private_class, src=src)

    def get_trg_private(self, src_loader, trg_loader):
        """Compute target-private classes (classes in target not in source)."""
        trg_y = copy.deepcopy(trg_loader.dataset.y_data)
        src_y = src_loader.dataset.y_data
        pri_c = torch.Tensor(np.setdiff1d(trg_y.numpy(), src_y.numpy()))
        print("Private classes: ", pri_c)
        return pri_c

    def H_score(self, trg_pred, trg_y):
        """
        Compute H-score (harmonic mean of known class accuracy and unknown detection rate).

        Args:
            trg_pred: raw predictions tensor (before decision_function)
            trg_y: labels tensor with -1 for unknown classes
        Returns:
            H_score, acc_c (OS*), acc_p (UNK), acc_mix
        """
        class_c = np.where(trg_y != -1)  # known class indices
        class_p = np.where(trg_y == -1)  # unknown class indices

        trg_pred = self.algorithm.decision_function(trg_pred)

        label_c, pred_c = trg_y[class_c], trg_pred[class_c]
        label_p, pred_p = trg_y[class_p], trg_pred[class_p]

        acc_c = (pred_c == label_c).sum() / len(pred_c) if len(pred_c) != 0 else torch.Tensor([0])
        acc_p = (pred_p == label_p).sum() / len(pred_p) if len(pred_p) != 0 else torch.Tensor([0])

        acc_mix = (trg_y != -1).sum() / len(trg_y) * acc_c + (trg_y == -1).sum() / len(trg_y) * acc_p

        if acc_c + acc_p == 0:
            H = torch.Tensor([0])
        else:
            H = 2 * acc_c * acc_p / (acc_c + acc_p)

        return H, acc_c, acc_p, acc_mix

    def load_data(self):
        """Load source and target data."""
        # Cross-user mode: source_users / target_users set by caller (e.g. sweep)
        src_users = getattr(self, 'source_users', None)
        trg_users = getattr(self, 'target_users', None)

        # Shared label_list ensures source and target use the same integer
        # encoding for the same canonical activity. Without this, each side
        # runs np.unique on its own mapping values, producing divergent label
        # permutations and catastrophic cross-dataset evaluation.
        # Ordering: source-known classes first (0..K-1), then target-private
        # classes (K..N-1). This keeps the source classifier's K-way output
        # aligned with label indices 0..K-1, while target-private samples get
        # indices >= K that the evaluator masks to -1 (unknown).
        src_known = sorted(set(self.dataset_configs.source_activity_mapping.values()))
        trg_all = set(self.dataset_configs.target_activity_mapping.values())
        trg_private = sorted(trg_all - set(src_known))
        shared_label_list = np.array(src_known + trg_private)
        self.shared_label_list = shared_label_list

        # Load source data
        src_loaders = data_generator(self.source_data_path, self.dataset_configs, self.hparams, 'source',
                                     activity_mapping=self.dataset_configs.source_activity_mapping,
                                     source_users=src_users, target_users=trg_users,
                                     label_list=shared_label_list)
        self.src_train_dl = src_loaders[0]
        self.src_test_dl = src_loaders[1]
        self.src_val_dl = src_loaders[2]

        # Load target data
        trg_loaders = data_generator(self.target_data_path, self.dataset_configs, self.hparams, 'target',
                                     activity_mapping=self.dataset_configs.target_activity_mapping,
                                     source_users=src_users, target_users=trg_users,
                                     label_list=shared_label_list)
        self.trg_train_dl = trg_loaders[0]
        self.trg_test_dl = trg_loaders[1]

        print("\n--- Dataloader Size Check ---")
        print(f"Source Train batches:   {len(self.src_train_dl)}")
        print(f"Target Train batches:   {len(self.trg_train_dl)}")
        print("-------------------------------\n")

        # Compute target private classes
        if self.uniDA:
            self.trg_private_class = self.get_trg_private(self.src_train_dl, self.trg_train_dl)
        else:
            self.trg_private_class = torch.Tensor([])

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics(self):
        """Calculate metrics including H-score for open-set DA."""
        self.evaluate(self.trg_test_dl)

        if self.uniDA:
            # Mask target-private labels to -1
            mask = np.isin(self.full_labels.cpu().numpy(),
                           self.trg_private_class.numpy(), invert=True)
            self.full_labels[~torch.tensor(mask)] = -1

            # Compute H-score
            H_score, acc_c, acc_p, acc_mix = self.H_score(
                self.full_preds.cpu(), self.full_labels.cpu())
            H_score = H_score.item() if isinstance(H_score, torch.Tensor) else H_score
            acc_c = acc_c.item() if isinstance(acc_c, torch.Tensor) else acc_c
            acc_p = acc_p.item() if isinstance(acc_p, torch.Tensor) else acc_p
            acc_mix = acc_mix.item() if isinstance(acc_mix, torch.Tensor) else acc_mix

            # Also compute standard accuracy using decision_function
            preds_decided = self.algorithm.decision_function(self.full_preds.cpu())
            labels_cpu = self.full_labels.cpu()
            acc = (preds_decided == labels_cpu).float().mean().item()

            # Weighted F1 over K known + 1 unknown class. -1 (unknown) maps to
            # index = num_classes so the multiclass F1Score (configured with
            # num_classes+1) can score it like any other class.
            preds_for_f1 = preds_decided.clone() if isinstance(preds_decided, torch.Tensor) \
                else torch.tensor(preds_decided)
            labels_for_f1 = labels_cpu.clone()
            preds_for_f1[preds_for_f1 == -1] = self.num_classes
            labels_for_f1[labels_for_f1 == -1] = self.num_classes
            trg_f1 = self.F1(preds_for_f1.long(), labels_for_f1.long()).item()

            print(f"H_score: {H_score:.4f}, OS* (acc_c): {acc_c:.4f}, UNK (acc_p): {acc_p:.4f}, acc_mix: {acc_mix:.4f}, F1: {trg_f1:.4f}")

            # Source metrics
            self.evaluate(self.src_test_dl, src=True)
            src_preds_decided = self.algorithm.decision_function(self.full_preds.cpu())
            src_acc = (src_preds_decided == self.full_labels.cpu()).float().mean().item()

            return acc, trg_f1, H_score, acc_c, acc_p, acc_mix, src_acc
        else:
            # Standard closed-set metrics
            trg_acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
            trg_f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
            trg_auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

            self.evaluate(self.src_test_dl, src=True)
            src_acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()

            return trg_acc, trg_f1, trg_auroc, 0.0, 0.0, src_acc

    def calculate_risks(self):
        self.evaluate(self.src_test_dl, src=True)
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()
        return src_risk, trg_risk

    def save_tables_to_file(self, table_results, name):
        table_results.to_csv(os.path.join(self.exp_log_dir, f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def get_configs(self):
        # Pass scenario so the same source_dataset name (e.g. RealWorld_male)
        # can resolve to either the UniDA or PDA class split.
        dataset_class = get_dataset_class(self.source_dataset, scenario=self.scenario)
        hparams_class = get_hparams_class(self.source_dataset)
        return dataset_class(), hparams_class()

    def append_results_to_tables(self, table, scenario, run_id, metrics):
        results_row = [scenario, run_id, *metrics]
        results_df = pd.DataFrame([results_row], columns=table.columns)
        table = pd.concat([table, results_df], ignore_index=True)
        return table

    def add_mean_std_table(self, table, columns):
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x
        table = table.map(format_func)

        return table
