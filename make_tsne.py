#!/usr/bin/env python3
"""Post-hoc t-SNE visualisation of trained feature extractors.

No retraining: loads the `last` weights from a saved checkpoint, pushes the
source + target test sets through the feature extractor, and produces two
PNGs per run — one coloured by domain (src vs trg), one coloured by class
(knowns + privates). Strictly visualisation — does not touch MLflow.

Examples:
    # OSDA curriculum run
    python make_tsne.py \\
        --da_method OSBP --source_dataset RealWorld --target_dataset Pamap2 \\
        --scenario OSDA --strategy hard --n_unknown 3 --rank_variant fno_mean \\
        --run 0

    # closed-set baseline (no curriculum args needed)
    python make_tsne.py \\
        --da_method OVANet --source_dataset RealWorld --target_dataset Pamap2 \\
        --scenario closed_set --run 0

    # batch over a method × n grid for one (src,trg)
    for m in OSBP OVANet UniOT; do
      for n in 1 3 5; do
        python make_tsne.py --da_method $m --source_dataset RealWorld \\
          --target_dataset Pamap2 --scenario OSDA --strategy hard \\
          --n_unknown $n --rank_variant fno_mean --run 0
      done
    done
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

# Make similarity_comparison/ importable for the curriculum-mapping helper.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "similarity_comparison"))

from algorithms.algorithms import get_algorithm_class
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from dataloader.dataloader import data_generator
from models.models import get_backbone_class

# Curriculum-mapping helper (same logic the trainer uses).
from run_curriculum import build_curriculum_mappings, split_family
from compute_feature_distance_4known import DATASETS


def build_configs(source_dataset, backbone, da_method, scenario):
    """Mirror the trainer's config setup so the algorithm builds with the
    correct feat_dim/num_classes."""
    dataset_configs = get_dataset_class(source_dataset, scenario=scenario)()
    hparams_class = get_hparams_class(source_dataset, backbone=backbone)()
    if backbone == "FNO":
        dataset_configs.isFNO = True
    if dataset_configs.isFNO:
        dataset_configs.feat_dim = (dataset_configs.features_len * dataset_configs.final_out_channels
                                    + 2 * dataset_configs.fourier_modes)
    else:
        dataset_configs.feat_dim = dataset_configs.features_len * dataset_configs.final_out_channels
    dataset_configs.da_method = da_method
    hparams = {**hparams_class.alg_hparams[da_method], **hparams_class.train_params}
    return dataset_configs, hparams


def resolve_mappings(args):
    """Return (src_mapping, trg_mapping, scenario_label_for_path).
    For closed_set the scenario label is 'closed_set'; for curriculum runs it
    matches the run_curriculum exp_name (e.g. 'OSDA_hard_n3_fno_mean')."""
    if args.scenario == "closed_set":
        src_family = split_family(args.source_dataset)
        trg_family = split_family(args.target_dataset)
        src_mapping = dict(DATASETS[src_family]["known_mapping"])
        trg_mapping = dict(DATASETS[trg_family]["known_mapping"])
        return src_mapping, trg_mapping, "closed_set"

    res = build_curriculum_mappings(
        args.source_dataset, args.target_dataset, args.scenario,
        args.strategy, args.n_unknown, args.ranking_key,
        rank_variant=args.rank_variant,
    )
    src_mapping, trg_mapping = res[0], res[1]
    exp_label = f"{args.scenario}_{args.strategy}_n{args.n_unknown}"
    if args.rank_variant != "min":
        exp_label += f"_{args.rank_variant}"
    return src_mapping, trg_mapping, exp_label


def find_checkpoint(args, exp_label):
    """Locate the trained checkpoint. Tries the conventional curriculum path
    (under experiments_logs/curriculum/...) and the closed-set path (under
    experiments_logs/...)."""
    src, trg = args.source_dataset, args.target_dataset
    run_folder = f"{src}_to_{trg}_run_{args.run}"
    candidates = []
    if exp_label == "closed_set":
        candidates.append(os.path.join(ROOT, "experiments_logs",
                                       f"{src} to {trg}_closed_set",
                                       f"{args.da_method}_closed_set",
                                       run_folder, "checkpoint.pt"))
    else:
        candidates.append(os.path.join(ROOT, "experiments_logs", "curriculum",
                                       f"{src} to {trg}_{exp_label}",
                                       f"{args.da_method}_{exp_label}",
                                       run_folder, "checkpoint.pt"))
    if args.checkpoint:
        candidates.insert(0, args.checkpoint)
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n  " + "\n  ".join(candidates) +
        "\nPass --checkpoint to override.")


def load_algorithm(da_method, backbone, dataset_configs, hparams, device,
                   ckpt_path, prefer="last"):
    """Build the algorithm and load weights. The trainer saves either
    self.network.state_dict() (most algos: keys '0.*' / '1.*') or
    self.state_dict() (e.g. UniOT: keys 'feature_extractor.*' etc.). We try
    both, ignoring missing/extra keys via strict=False."""
    algo_cls = get_algorithm_class(da_method)
    backbone_cls = get_backbone_class(backbone)
    algo = algo_cls(backbone_cls, dataset_configs, hparams, device).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get(prefer) or ckpt.get("best") or ckpt
    keys = list(sd.keys())
    looks_like_network = any(k.startswith(("0.", "1.")) for k in keys[:5])

    if looks_like_network and hasattr(algo, "network"):
        info = algo.network.load_state_dict(sd, strict=False)
    else:
        info = algo.load_state_dict(sd, strict=False)
    if info.missing_keys:
        print(f"[load] {len(info.missing_keys)} missing keys (e.g. {info.missing_keys[:3]})")
    if info.unexpected_keys:
        print(f"[load] {len(info.unexpected_keys)} unexpected keys (e.g. {info.unexpected_keys[:3]})")
    algo.eval()
    return algo


def make_loaders(args, dataset_configs, hparams, src_mapping, trg_mapping):
    """Build src + trg test loaders with the same shared label encoding the
    trainer uses (knowns 0..K-1, then target-privates K..N-1)."""
    src_known = sorted(set(src_mapping.values()))
    all_classes = set(src_mapping.values()) | set(trg_mapping.values())
    privates = sorted(all_classes - set(src_known))
    label_list = np.array(src_known + privates)

    src_loaders = data_generator(os.path.join(args.data_path, args.source_dataset),
                                  dataset_configs, hparams, "source",
                                  activity_mapping=src_mapping,
                                  label_list=label_list)
    trg_loaders = data_generator(os.path.join(args.data_path, args.target_dataset),
                                  dataset_configs, hparams, "target",
                                  activity_mapping=trg_mapping,
                                  label_list=label_list)
    return src_loaders[1], trg_loaders[1], label_list, len(src_known)


@torch.no_grad()
def extract(algo, loader, device):
    feats, labels = [], []
    fe = algo.feature_extractor
    fe.eval()
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.float().to(device)
        f = fe(x)
        if isinstance(f, tuple):
            f = f[0]
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, 0), np.concatenate(labels, 0)


def subsample(feats, labels, max_per_class, rng):
    """Keep at most max_per_class points per label so t-SNE stays fast and
    rare classes aren't drowned out."""
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return feats[keep], labels[keep]


def plot_tsne(coords, colors, legend_handles, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.7,
               linewidth=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", fontsize=8,
                  framealpha=0.9, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--da_method", required=True)
    p.add_argument("--source_dataset", required=True)
    p.add_argument("--target_dataset", required=True)
    p.add_argument("--scenario", required=True,
                   choices=["closed_set", "OSDA", "PDA", "UniDA"])
    p.add_argument("--strategy", default="hard", choices=["hard", "easy"])
    p.add_argument("--n_unknown", type=int, default=None)
    p.add_argument("--rank_variant", default="fno_mean",
                   choices=["min", "mean", "fno_mean"])
    p.add_argument("--ranking_key", default=None)
    p.add_argument("--backbone", default="FNO")
    p.add_argument("--run", type=int, default=0,
                   help="Which seed-index run (0..4) to load.")
    p.add_argument("--data_path", default="../dataset")
    p.add_argument("--checkpoint", default=None,
                   help="Override path to checkpoint.pt.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_dir", default="tsne_plots")
    p.add_argument("--max_per_class", type=int, default=400,
                   help="Cap on points per class per domain for t-SNE.")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_seed", type=int, default=0)
    args = p.parse_args()

    if args.scenario != "closed_set" and args.n_unknown is None:
        p.error("--n_unknown is required when --scenario is OSDA/PDA/UniDA")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    src_mapping, trg_mapping, exp_label = resolve_mappings(args)

    print(f"[tsne] {args.da_method} | {args.source_dataset} -> {args.target_dataset} "
          f"| scenario={args.scenario} | exp={exp_label} | run={args.run}")
    print(f"[tsne] src classes: {sorted(set(src_mapping.values()))}")
    print(f"[tsne] trg classes: {sorted(set(trg_mapping.values()))}")

    dataset_configs, hparams = build_configs(args.source_dataset, args.backbone,
                                              args.da_method, args.scenario)
    # num_classes: classifier output dim = unique source-known canonical classes.
    dataset_configs.num_classes = len(set(src_mapping.values()))
    dataset_configs.source_activity_mapping = src_mapping
    dataset_configs.target_activity_mapping = trg_mapping

    ckpt_path = find_checkpoint(args, exp_label)
    print(f"[tsne] checkpoint: {ckpt_path}")
    algo = load_algorithm(args.da_method, args.backbone, dataset_configs,
                          hparams, device, ckpt_path)

    src_test_dl, trg_test_dl, label_list, n_known = make_loaders(
        args, dataset_configs, hparams, src_mapping, trg_mapping)

    print("[tsne] extracting features...")
    src_feats, src_labels = extract(algo, src_test_dl, device)
    trg_feats, trg_labels = extract(algo, trg_test_dl, device)

    rng = np.random.default_rng(args.tsne_seed)
    src_feats, src_labels = subsample(src_feats, src_labels, args.max_per_class, rng)
    trg_feats, trg_labels = subsample(trg_feats, trg_labels, args.max_per_class, rng)

    feats = np.concatenate([src_feats, trg_feats], 0)
    domain = np.concatenate([np.zeros(len(src_feats), dtype=int),
                             np.ones(len(trg_feats), dtype=int)])
    labels = np.concatenate([src_labels, trg_labels])

    print(f"[tsne] running t-SNE on {feats.shape[0]} points "
          f"({feats.shape[1]}-D) perplexity={args.perplexity}...")
    perp = min(args.perplexity, max(5.0, (feats.shape[0] - 1) / 3))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=args.tsne_seed, learning_rate="auto")
    coords = tsne.fit_transform(feats)

    out_dir = os.path.join(ROOT, args.save_dir,
                           f"{args.source_dataset}_to_{args.target_dataset}_{exp_label}",
                           args.da_method)
    os.makedirs(out_dir, exist_ok=True)
    base = f"run{args.run}"

    # 1) Colour by domain.
    colors_dom = np.where(domain == 0, "#1f77b4", "#ff7f0e")
    handles_dom = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                   label="Source", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
                   label="Target", markersize=8),
    ]
    plot_tsne(coords, colors_dom, handles_dom,
              f"{args.da_method}  ·  {args.source_dataset}→{args.target_dataset}  ·  {exp_label}\nColoured by domain",
              os.path.join(out_dir, f"{base}_domain.png"))

    # 2) Colour by class. Knowns get qualitative colours; privates get a single
    # contrasting hue so they're easy to spot regardless of count.
    from matplotlib.colors import to_hex
    cmap = plt.get_cmap("tab10")
    colors_cls = ["#000000"] * len(labels)
    handles_cls = []
    for lab_int in np.unique(labels):
        is_private = lab_int >= n_known
        if is_private:
            color = "#e74c3c"
            name = f"{label_list[lab_int]} (private)" if lab_int < len(label_list) else f"private-{lab_int}"
        else:
            color = to_hex(cmap(lab_int % 10))
            name = str(label_list[lab_int]) if lab_int < len(label_list) else f"class-{lab_int}"
        for i in np.where(labels == lab_int)[0]:
            colors_cls[i] = color
        handles_cls.append(plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=color, label=name,
                                       markersize=8))
    plot_tsne(coords, colors_cls, handles_cls,
              f"{args.da_method}  ·  {args.source_dataset}→{args.target_dataset}  ·  {exp_label}\nColoured by class (red = private)",
              os.path.join(out_dir, f"{base}_class.png"))

    print(f"[tsne] done · plots in {out_dir}")


if __name__ == "__main__":
    main()
