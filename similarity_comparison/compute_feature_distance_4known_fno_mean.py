#!/usr/bin/env python3
"""
Feature-space centroid distance with 4 known classes — FNO backbone, multi-seed.

Differences from compute_feature_distance_4known_mean.py:
  - **FNO backbone** (matches what the actual DA methods use), not CNN.
  - **Multi-seed averaging** over 5 seeds: per-pair distances are averaged
    across seeds before taking the mean-over-knowns, which dampens single-init
    noise in the NO_ADAPT centroids.

The score per unknown is still: mean over the 4 source-known centroids of the
seed-averaged cosine distance from this unknown's centroid.

Outputs:
  feature_distance_4known_fno_mean/rankings_4known_fno_mean.json
  feature_distance_4known_fno_mean/feat4_*_<src>_to_<trg>.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from compute_feature_distance_4known import (
    DATASETS, PRETTY_NAMES,
    build_target_activity_mapping, build_source_activity_mapping,
    extract_features, compute_centroids, cosine_distance,
)
from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator
from models.models import get_backbone_class, classifier
from utils import fix_randomness

NO_ADAPT_HPARAMS = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 30,
    'num_epochs_pr': 20,
}

SEEDS = [42, 43, 44, 45, 46]


def train_no_adapt_fno(src_train_dl, dataset_configs, hparams, device, num_epochs=30):
    """NO_ADAPT training with FNO backbone. Mirrors v1's CNN trainer but swaps
    the backbone class."""
    backbone_class = get_backbone_class('FNO')
    feature_extractor = backbone_class(dataset_configs).to(device)
    cls = classifier(dataset_configs).to(device)

    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(cls.parameters()),
        lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'],
    )
    ce = nn.CrossEntropyLoss()

    feature_extractor.train(); cls.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0; n_batches = 0
        for src_x, src_y in src_train_dl:
            src_x, src_y = src_x.to(device), src_y.to(device)
            feat = feature_extractor(src_x)
            loss = ce(cls(feat), src_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); n_batches += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch}/{num_epochs} — loss: {epoch_loss / n_batches:.4f}")

    feature_extractor.eval(); cls.eval()
    return feature_extractor, cls


def setup_fno_configs():
    """Configure dataset_configs for FNO backbone (mirrors abstract_trainer.py:63-72)."""
    dataset_configs = get_dataset_class("ALL")()
    dataset_configs.isFNO = True
    dataset_configs.feat_dim = (
        dataset_configs.features_len * dataset_configs.final_out_channels
        + 2 * dataset_configs.fourier_modes
    )
    dataset_configs.da_method = ''
    return dataset_configs


def run_seed(source_name, target_name, seed, data_path, device):
    """Train NO_ADAPT with one seed and return raw per-pair distances dict.

    Returns: dict[(unk_pretty, src_known)] -> cos_dist  (both as raw float).
    """
    fix_randomness(seed)
    dataset_configs = setup_fno_configs()

    src_mapping = build_source_activity_mapping(source_name)
    trg_mapping = build_target_activity_mapping(target_name)
    dataset_configs.source_activity_mapping = src_mapping
    dataset_configs.target_activity_mapping = trg_mapping

    src_known = sorted(set(src_mapping.values()))
    trg_all = set(trg_mapping.values())
    trg_private = sorted(trg_all - set(src_known))
    shared_label_list = np.array(src_known + trg_private)
    name_to_label = {name: i for i, name in enumerate(shared_label_list)}
    dataset_configs.num_classes = len(src_known)

    hparams = dict(NO_ADAPT_HPARAMS)

    src_dp = os.path.join(data_path, source_name)
    trg_dp = os.path.join(data_path, target_name)
    src_loaders = data_generator(src_dp, dataset_configs, hparams, 'source',
                                  activity_mapping=src_mapping, label_list=shared_label_list)
    trg_loaders = data_generator(trg_dp, dataset_configs, hparams, 'target',
                                  activity_mapping=trg_mapping, label_list=shared_label_list)
    src_train_dl = src_loaders[0]
    trg_test_dl = trg_loaders[1]

    fe, _ = train_no_adapt_fno(src_train_dl, dataset_configs, hparams, device,
                                num_epochs=hparams['num_epochs'])

    src_feats, src_labels = extract_features(fe, src_train_dl, device)
    src_centroids = compute_centroids(src_feats, src_labels, name_to_label, src_known)
    trg_feats, trg_labels = extract_features(fe, trg_test_dl, device)
    trg_unk_centroids = compute_centroids(trg_feats, trg_labels, name_to_label, trg_private)

    # Build per-pair distances: keys are (pretty_unknown, src_known_canonical).
    pair_dists = {}
    for unk in trg_private:
        if unk not in trg_unk_centroids:
            continue
        pretty = PRETTY_NAMES.get(unk, unk)
        for k in src_known:
            if k not in src_centroids:
                continue
            pair_dists[(pretty, k)] = cosine_distance(trg_unk_centroids[unk], src_centroids[k])

    # Free GPU memory before next seed.
    del fe
    torch.cuda.empty_cache()

    return src_known, trg_private, pair_dists


def aggregate_seeds(per_seed_pair_dists):
    """Average per-pair distances across seeds; return (mean_dict, std_dict).

    Uses nanmean/nanstd so a single NaN seed (occasional Adam blow-up at
    epoch ~30 on RealWorld) doesn't poison the entire 5-seed average.
    """
    keys = per_seed_pair_dists[0].keys()
    means = {}
    stds = {}
    for key in keys:
        vals = [seed_d[key] for seed_d in per_seed_pair_dists]
        means[key] = float(np.nanmean(vals))
        stds[key]  = float(np.nanstd(vals))
    return means, stds


def plot_heatmap(pair_means, pair_stds, ranking_score, source_name, target_name,
                 src_known, save_dir):
    unknowns = sorted({u for (u, _) in pair_means.keys()},
                      key=lambda u: ranking_score[u])
    data = np.array([[pair_means.get((u, k), np.nan) for k in src_known] for u in unknowns])

    fig, ax = plt.subplots(figsize=(9, max(4, len(unknowns) * 0.7 + 1.5)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=[k.capitalize() for k in src_known],
                yticklabels=unknowns,
                cbar_kws={'label': 'Cosine distance (lower = more similar)'},
                ax=ax)
    ax.set_xlabel('Known Class (Source centroid, FNO + 5-seed avg)')
    ax.set_ylabel('Unknown Class (Target centroid)')
    ax.set_title(f'FNO Feature-Space Cosine Distance (5-seed avg, MEAN-ranked): '
                 f'{source_name} → {target_name}')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_cosine_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_ranking(ranking_score, nearest, source_name, target_name, src_known, save_dir):
    sorted_unknowns = sorted(ranking_score.keys(), key=lambda u: ranking_score[u])

    fig, ax = plt.subplots(figsize=(12, max(3, len(sorted_unknowns) * 0.55 + 1.5)))
    colors = {'lying': '#e74c3c', 'sitting': '#3498db',
              'walking': '#f39c12', 'running': '#9b59b6'}

    ax.barh(range(len(sorted_unknowns)),
            [ranking_score[u] for u in sorted_unknowns],
            color=[colors.get(nearest[u], '#95a5a6') for u in sorted_unknowns])

    ax.set_yticks(range(len(sorted_unknowns)))
    ax.set_yticklabels(sorted_unknowns)
    ax.invert_yaxis()
    for i, u in enumerate(sorted_unknowns):
        ax.text(ranking_score[u] + max(ranking_score.values()) * 0.01, i,
                f'{ranking_score[u]:.3f} (nearest: {nearest[u]})',
                va='center', fontsize=9)
    ax.set_xlabel('Mean Cosine Distance to All Known Centroids (FNO, 5-seed avg)')
    ax.set_title(f'Unknown Difficulty (FNO, MEAN, 5-seed avg): {source_name} → {target_name}\n'
                 f'(smaller = closer to knowns overall = harder to reject)')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k.capitalize()) for k in src_known]
    ax.legend(handles=legend_elements, title='Nearest Known (color only)', loc='lower right')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_ranking_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_pair(source_name, target_name, data_path, device, save_dir):
    print(f"\n{'='*70}")
    print(f"  {source_name} → {target_name}  (FNO, MEAN, multi-seed)")
    print(f"{'='*70}")

    per_seed_pair_dists = []
    src_known_ref, trg_private_ref = None, None
    for s in SEEDS:
        print(f"  -- seed {s} --")
        src_known, trg_private, pair_dists = run_seed(
            source_name, target_name, s, data_path, device)
        if src_known_ref is None:
            src_known_ref, trg_private_ref = src_known, trg_private
        per_seed_pair_dists.append(pair_dists)

    pair_means, pair_stds = aggregate_seeds(per_seed_pair_dists)

    # Mean-over-knowns aggregation (after seed averaging).
    pretty_unknowns = sorted({u for (u, _) in pair_means.keys()})
    ranking_score = {}
    nearest = {}
    for u in pretty_unknowns:
        per_known = {k: pair_means[(u, k)] for k in src_known_ref if (u, k) in pair_means}
        if not per_known:
            continue
        ranking_score[u] = float(np.mean(list(per_known.values())))
        nearest[u] = min(per_known, key=per_known.get)

    # Print summary table.
    print(f"\n  ── COSINE distances (FNO, 5-seed avg, mean-ranked) ──")
    print(f"  {'Unknown class':<28s} ", end='')
    for k in src_known_ref:
        print(f"{k:>10s}", end='')
    print(f"  {'Mean':>8s}  {'Nearest':>14s}")
    print("  " + "-" * (28 + 10*len(src_known_ref) + 8 + 14 + 4))
    for u in sorted(ranking_score.keys(), key=lambda u: ranking_score[u]):
        print(f"  {u:<28s} ", end='')
        for k in src_known_ref:
            print(f"{pair_means.get((u, k), float('nan')):>10.3f}", end='')
        print(f"  {ranking_score[u]:>8.3f}  {nearest[u]:>14s}")

    plot_heatmap(pair_means, pair_stds, ranking_score, source_name, target_name,
                 src_known_ref, save_dir)
    plot_ranking(ranking_score, nearest, source_name, target_name, src_known_ref, save_dir)

    return {
        'knowns': src_known_ref,
        'seeds': SEEDS,
        'rankings': [
            {'unknown': u,
             'mean_cosine_dist': float(ranking_score[u]),
             'nearest_known': nearest[u],
             'per_class_mean': {k: float(pair_means[(u, k)]) for k in src_known_ref
                                if (u, k) in pair_means},
             'per_class_std':  {k: float(pair_stds[(u, k)])  for k in src_known_ref
                                if (u, k) in pair_stds}}
            for u in sorted(ranking_score.keys(), key=lambda u: ranking_score[u])
        ],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  ·  seeds: {SEEDS}")

    # Dataset lives at <parent_of_project_root>/dataset (matches run_curriculum
    # default --data_path "../dataset"). Going up two dirnames gets there.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, '..', 'dataset')
    save_dir = os.path.join(project_root, 'feature_distance_4known_fno_mean')
    os.makedirs(save_dir, exist_ok=True)

    pairs = [
        ('RealWorld', 'Pamap2'),
        ('RealWorld', 'MHEALTH'),
        ('Pamap2', 'RealWorld'),
        ('Pamap2', 'MHEALTH'),
        ('MHEALTH', 'RealWorld'),
        ('MHEALTH', 'Pamap2'),
    ]

    all_results = {}
    for src, trg in pairs:
        all_results[f"{src} -> {trg}"] = run_pair(src, trg, data_path, device, save_dir)

    json_path = os.path.join(save_dir, 'rankings_4known_fno_mean.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved rankings JSON: {json_path}")
    print(f"All plots saved in: {save_dir}")


if __name__ == '__main__':
    main()
