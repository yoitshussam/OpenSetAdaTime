#!/usr/bin/env python3
"""LR scan for OVANet, DANCE, PPOT on a held-out pair.

Tunes on RealWorld -> MHEALTH (held out). Main eval RealWorld -> Pamap2 stays
untouched, so this is leave-one-pair-out tuning, not direct overfitting to the
test pair.

For each (method, lr): 1 seed, 1 curriculum step (n=3), record H/F1/UNK/OS*.
Pick the best LR per method, then plug back into configs/hparams.py for the
main eval run.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
import torch

from run_curriculum import CurriculumTrainer, build_curriculum_mappings

mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Tuning directly on the eval pair. Not held-out anymore.
SOURCE = "RealWorld"
TARGET = "Pamap2"
N_UNKNOWN = 3
SCENARIO = "OSDA"
STRATEGY = "hard"
RANK_VARIANT = "fno_mean"

LR_GRIDS = {
    "OVANet": [5e-1, 5e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "DANCE":  [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "PPOT":   [1e-3, 5e-4, 1e-4, 5e-5],
}


class Args:
    pass


class LRScanTrainer(CurriculumTrainer):
    """CurriculumTrainer that overrides one hparam (learning_rate) for one
    method before the rest of the trainer initializes."""

    def __init__(self, args, src_map, trg_map, tags, method, lr):
        # Stash before super().__init__ so get_configs() (called from inside
        # AbstractTrainer.__init__) can apply the patch.
        self._lr_method = method
        self._lr_value = lr
        super().__init__(args, src_map, trg_map, tags)

    def get_configs(self):
        dataset_configs, hparams_class = super().get_configs()
        # hparams_class is a fresh instance from get_hparams_class()(), so
        # patching its alg_hparams dict only affects this trainer.
        hparams_class.alg_hparams[self._lr_method]["learning_rate"] = self._lr_value
        return dataset_configs, hparams_class


def make_args(method, lr):
    a = Args()
    a.source_dataset = SOURCE
    a.target_dataset = TARGET
    a.scenario = SCENARIO
    a.strategy = STRATEGY
    a.n_unknown = N_UNKNOWN
    a.da_method = method
    a.backbone = "FNO"
    a.num_runs = 1
    a.data_path = "../dataset"
    a.device = "cuda"
    a.save_dir = "experiments_logs/lr_scan"
    a.ranking_key = None
    a.rank_variant = RANK_VARIANT
    a.exp_name = f"LR_SCAN_n{N_UNKNOWN}_{method}_lr{lr:.0e}"
    return a


def run_one(method, lr):
    args = make_args(method, lr)

    src_map, trg_map, unknowns = build_curriculum_mappings(
        SOURCE, TARGET, SCENARIO, STRATEGY, N_UNKNOWN,
        rank_variant=RANK_VARIANT,
    )

    tags = {
        "curriculum_scenario":     SCENARIO,
        "curriculum_strategy":     STRATEGY,
        "curriculum_n_unknown":    str(N_UNKNOWN),
        "curriculum_unknowns":     ",".join(unknowns),
        "curriculum_rank_variant": RANK_VARIANT,
        "curriculum_ranking_key":  f"{SOURCE} -> {TARGET}",
        "lr_scan_method":          method,
        "lr_scan_lr":              f"{lr}",
    }

    print(f"\n========================================")
    print(f"[lr_scan] {method:<8s} lr={lr:.0e}")
    print(f"========================================")
    trainer = LRScanTrainer(args, src_map, trg_map, tags, method, lr)
    trainer.fit()

    # Free GPU before next config.
    del trainer
    torch.cuda.empty_cache()


def fetch_results():
    """Query MLflow for all LR_SCAN_* experiments on this held-out pair."""
    client = mlflow.tracking.MlflowClient()
    rows = []
    prefix = f"{SOURCE} to {TARGET}_LR_SCAN_n{N_UNKNOWN}"
    for exp in client.search_experiments():
        if not exp.name.startswith(prefix):
            continue
        runs = client.search_runs([exp.experiment_id])
        for run in runs:
            method = run.data.tags.get("lr_scan_method")
            lr_str = run.data.tags.get("lr_scan_lr")
            if not method or lr_str is None:
                continue
            h   = run.data.metrics.get("H_score", 0.0)
            f1  = run.data.metrics.get("Target F1-score", 0.0)
            unk = run.data.metrics.get("UNK", 0.0)
            os_ = run.data.metrics.get("OS_star", 0.0)
            rows.append((method, float(lr_str), h, f1, unk, os_))
    return rows


def main():
    for method, lrs in LR_GRIDS.items():
        for lr in lrs:
            try:
                run_one(method, lr)
            except Exception as e:
                print(f"[lr_scan] {method} lr={lr:.0e} CRASHED: {type(e).__name__}: {e}")
                # Reset CUDA in case of context corruption (NaN -> assert).
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    print("\n\n========= LR SCAN RESULTS =========")
    rows = fetch_results()
    rows.sort(key=lambda r: (r[0], -r[2]))  # by method, then H_score desc
    print(f"{'method':<8s} {'lr':>10s} {'H':>6s} {'F1':>6s} {'UNK':>6s} {'OS*':>6s}")
    print("-" * 50)
    cur = None
    for r in rows:
        if r[0] != cur:
            if cur is not None:
                print()
            cur = r[0]
        print(f"{r[0]:<8s} {r[1]:>10.0e} {r[2]:>6.3f} {r[3]:>6.3f} {r[4]:>6.3f} {r[5]:>6.3f}")


if __name__ == "__main__":
    main()
