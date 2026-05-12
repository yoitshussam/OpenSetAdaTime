#!/usr/bin/env python3
"""
Feature-space centroid distance with 4 known classes — MEAN variant.

Same pipeline as compute_feature_distance_4known.py except the ranking metric
is the **mean cosine distance** between each target-unknown centroid and ALL
source-known centroids, instead of the min over knowns.

Interpretation:
  - Low mean = the unknown lives in the general feature neighborhood of the
    known set (hard, from a "looks like knowns overall" perspective).
  - High mean = the unknown is geometrically isolated from the known set as
    a whole (easy: low overall overlap).

Outputs go to feature_distance_4known_mean/ to avoid clobbering v1.
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

# Reuse the dataset/PRETTY_NAMES tables and helpers from v1 — they're the
# same regardless of which distance summary we pick.
from compute_feature_distance_4known import (
    DATASETS, PRETTY_NAMES, KNOWN_CLASSES,
    get_unknown_classes, build_target_activity_mapping, build_source_activity_mapping,
    train_no_adapt, extract_features, compute_centroids, cosine_distance,
)
from configs.data_model_configs import get_dataset_class
from dataloader.dataloader import data_generator
from utils import fix_randomness

# Minimal hparams for source-only feature-extractor training. Avoids depending
# on the algorithm hparams classes (which were refactored to drop NO_ADAPT).
NO_ADAPT_HPARAMS = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 30,
    'num_epochs_pr': 20,
}


def compute_distance_matrix(src_centroids, trg_centroids, distance_fn,
                            source_known, target_unknowns):
    """Return per-class distances + mean-over-knowns and (still) nearest known.

    `nearest` is kept only as a color hint for plotting; it's no longer the
    ranking criterion.
    """
    distances, mean_dist, nearest = {}, {}, {}
    for unk in target_unknowns:
        if unk not in trg_centroids:
            continue
        pretty = PRETTY_NAMES.get(unk, unk)
        row = {}
        for k in source_known:
            if k not in src_centroids:
                continue
            row[k] = distance_fn(trg_centroids[unk], src_centroids[k])
        if not row:
            continue
        distances[pretty] = row
        mean_dist[pretty] = float(np.mean(list(row.values())))
        nearest[pretty] = min(row, key=row.get)
    return distances, mean_dist, nearest


def plot_heatmap(distances, mean_dist, source_name, target_name, source_known, save_dir):
    unknowns = sorted(distances.keys(), key=lambda u: mean_dist[u])
    data = np.array([[distances[u].get(k, np.nan) for k in source_known] for u in unknowns])

    fig, ax = plt.subplots(figsize=(9, max(4, len(unknowns) * 0.7 + 1.5)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=[k.capitalize() for k in source_known],
                yticklabels=unknowns,
                cbar_kws={'label': 'Cosine distance (lower = more similar)'},
                ax=ax)
    # Outline the per-row mean as an extra column on the right via title text;
    # heatmap stays the per-class view, mean is reported in the ranking plot.
    ax.set_xlabel('Known Class (Source centroid, 4-known)')
    ax.set_ylabel('Unknown Class (Target centroid)')
    ax.set_title(f'Feature-Space Cosine Distance (4 knowns, MEAN-ranked): '
                 f'{source_name} → {target_name}')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_cosine_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_ranking(mean_dist, nearest, source_name, target_name, source_known, save_dir):
    sorted_unknowns = sorted(mean_dist.keys(), key=lambda u: mean_dist[u])

    fig, ax = plt.subplots(figsize=(12, max(3, len(sorted_unknowns) * 0.55 + 1.5)))
    colors = {'lying': '#e74c3c', 'sitting': '#3498db',
              'walking': '#f39c12', 'running': '#9b59b6'}

    ax.barh(range(len(sorted_unknowns)),
            [mean_dist[u] for u in sorted_unknowns],
            color=[colors.get(nearest[u], '#95a5a6') for u in sorted_unknowns])

    ax.set_yticks(range(len(sorted_unknowns)))
    ax.set_yticklabels(sorted_unknowns)
    ax.invert_yaxis()
    for i, u in enumerate(sorted_unknowns):
        ax.text(mean_dist[u] + max(mean_dist.values()) * 0.01, i,
                f'{mean_dist[u]:.3f} (nearest: {nearest[u]})',
                va='center', fontsize=9)
    ax.set_xlabel('Mean Cosine Distance to All Known Centroids')
    ax.set_title(f'Unknown Difficulty (4 knowns, MEAN): {source_name} → {target_name}\n'
                 f'(smaller = closer to knowns overall = harder to reject)')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k.capitalize()) for k in source_known]
    ax.legend(handles=legend_elements, title='Nearest Known (color only)', loc='lower right')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_ranking_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_pair(source_name, target_name, data_path, device, save_dir, num_epochs=30, seed=42):
    print(f"\n{'='*70}")
    print(f"  {source_name} → {target_name}  (MEAN-ranked)")
    print(f"{'='*70}")

    fix_randomness(seed)
    dataset_configs = get_dataset_class("ALL")()
    hparams = dict(NO_ADAPT_HPARAMS)

    dataset_configs.da_method = ''
    dataset_configs.feat_dim = dataset_configs.features_len * dataset_configs.final_out_channels

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

    print(f"  Knowns ({len(src_known)}): {src_known}")
    print(f"  Unknowns ({len(trg_private)}): {trg_private}")

    src_dp = os.path.join(data_path, source_name)
    trg_dp = os.path.join(data_path, target_name)
    print("  Loading data...")
    src_loaders = data_generator(src_dp, dataset_configs, hparams, 'source',
                                  activity_mapping=src_mapping, label_list=shared_label_list)
    trg_loaders = data_generator(trg_dp, dataset_configs, hparams, 'target',
                                  activity_mapping=trg_mapping, label_list=shared_label_list)
    src_train_dl = src_loaders[0]
    trg_test_dl = trg_loaders[1]

    print(f"  Training NO_ADAPT ({num_epochs} epochs)...")
    fe, cls = train_no_adapt(src_train_dl, dataset_configs, hparams, device, num_epochs)

    print("  Extracting features...")
    src_feats, src_labels = extract_features(fe, src_train_dl, device)
    src_centroids = compute_centroids(src_feats, src_labels, name_to_label, src_known)
    trg_feats, trg_labels = extract_features(fe, trg_test_dl, device)
    trg_unk_centroids = compute_centroids(trg_feats, trg_labels, name_to_label, trg_private)

    distances, mean_dist, nearest = compute_distance_matrix(
        src_centroids, trg_unk_centroids, cosine_distance, src_known, trg_private)

    print(f"\n  ── COSINE distances (mean-ranked) ──")
    print(f"  {'Unknown class':<28s} ", end='')
    for k in src_known:
        print(f"{k:>10s}", end='')
    print(f"  {'Mean':>8s}  {'Nearest':>14s}")
    print("  " + "-" * (28 + 10*len(src_known) + 8 + 14 + 4))
    for u in sorted(mean_dist.keys(), key=lambda u: mean_dist[u]):
        print(f"  {u:<28s} ", end='')
        for k in src_known:
            print(f"{distances[u].get(k, float('nan')):>10.3f}", end='')
        print(f"  {mean_dist[u]:>8.3f}  {nearest[u]:>14s}")

    plot_heatmap(distances, mean_dist, source_name, target_name, src_known, save_dir)
    plot_ranking(mean_dist, nearest, source_name, target_name, src_known, save_dir)

    return {
        'knowns': src_known,
        'rankings': [
            {'unknown': u, 'mean_cosine_dist': float(mean_dist[u]),
             'nearest_known': nearest[u],
             'per_class': {k: float(distances[u].get(k, float('nan'))) for k in src_known}}
            for u in sorted(mean_dist.keys(), key=lambda u: mean_dist[u])
        ],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_distance_4known_mean')
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

    json_path = os.path.join(save_dir, 'rankings_4known_mean.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved rankings JSON: {json_path}")
    print(f"All plots saved in: {save_dir}")


if __name__ == '__main__':
    main()
