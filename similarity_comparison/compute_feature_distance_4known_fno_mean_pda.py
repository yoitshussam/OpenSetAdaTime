#!/usr/bin/env python3
"""
PDA variant: rank SOURCE-private classes by similarity to source-known
centroids, in a model trained on the source's 4 shared knowns.

Mirrors compute_feature_distance_4known_fno_mean.py (FNO + 5-seed avg) but the
unknowns being ranked come from the *source* dataset, not the target. The
embedding model is the same S-trained FNO that the OSDA variant uses for S as
source. Source-private set is a property of the source family alone (since the
4 shared knowns are fixed canonically), so we compute once per source family
and emit the same rankings for every (S, T) pair sharing that S — keeps the
JSON shape compatible with run_curriculum.py.

Outputs:
  feature_distance_4known_fno_mean_pda/rankings_4known_fno_mean_pda.json
  feature_distance_4known_fno_mean_pda/feat4_*_<src>_pda.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from compute_feature_distance_4known import (
    DATASETS, PRETTY_NAMES,
    get_unknown_classes, build_source_activity_mapping,
    extract_features, compute_centroids, cosine_distance,
)
from compute_feature_distance_4known_fno_mean import (
    NO_ADAPT_HPARAMS, SEEDS, train_no_adapt_fno, setup_fno_configs,
    aggregate_seeds,
)
from dataloader.dataloader import data_generator
from utils import fix_randomness


SOURCE_FAMILIES = ['RealWorld', 'Pamap2', 'MHEALTH']


def run_seed_pda(source_name, seed, data_path, device):
    """One seed: train FNO on src's 4 shared knowns, then extract centroids
    for src-known and src-private classes from the source dataset itself.
    Return (src_known, src_privates, pair_dists) where pair_dists is keyed by
    (private_pretty, src_known) -> cos_dist."""
    fix_randomness(seed)
    dataset_configs = setup_fno_configs()

    # Training mapping: 4 shared knowns only.
    train_mapping = build_source_activity_mapping(source_name)
    src_known = sorted(set(train_mapping.values()))

    # Centroid mapping: knowns + source-privates (mapped to themselves).
    src_privates = sorted(get_unknown_classes(source_name))
    full_mapping = dict(train_mapping)
    for p in src_privates:
        full_mapping[p] = p
    full_label_list = np.array(src_known + src_privates)
    name_to_label = {name: i for i, name in enumerate(full_label_list)}

    dataset_configs.source_activity_mapping = full_mapping
    dataset_configs.target_activity_mapping = full_mapping
    dataset_configs.num_classes = len(src_known)

    hparams = dict(NO_ADAPT_HPARAMS)
    src_dp = os.path.join(data_path, source_name)

    # Loader 1: 4-known mapping → for training (must match the OSDA variant's
    # training distribution exactly).
    train_loaders = data_generator(src_dp, dataset_configs, hparams, 'source',
                                    activity_mapping=train_mapping,
                                    label_list=full_label_list)
    src_train_dl = train_loaders[0]

    # Loader 2: full mapping → for extracting centroids of all classes through
    # the trained model.
    full_loaders = data_generator(src_dp, dataset_configs, hparams, 'source',
                                   activity_mapping=full_mapping,
                                   label_list=full_label_list)
    src_full_dl = full_loaders[0]

    fe, _ = train_no_adapt_fno(src_train_dl, dataset_configs, hparams, device,
                                num_epochs=hparams['num_epochs'])

    feats, labels = extract_features(fe, src_full_dl, device)
    src_known_centroids = compute_centroids(feats, labels, name_to_label, src_known)
    src_priv_centroids = compute_centroids(feats, labels, name_to_label, src_privates)

    pair_dists = {}
    for priv in src_privates:
        if priv not in src_priv_centroids:
            continue
        pretty = PRETTY_NAMES.get(priv, priv)
        for k in src_known:
            if k not in src_known_centroids:
                continue
            pair_dists[(pretty, k)] = cosine_distance(
                src_priv_centroids[priv], src_known_centroids[k]
            )

    del fe
    torch.cuda.empty_cache()
    return src_known, src_privates, pair_dists


def plot_heatmap(pair_means, ranking_score, source_name, src_known, save_dir):
    unknowns = sorted({u for (u, _) in pair_means.keys()},
                      key=lambda u: ranking_score[u])
    data = np.array([[pair_means.get((u, k), np.nan) for k in src_known] for u in unknowns])
    fig, ax = plt.subplots(figsize=(9, max(4, len(unknowns) * 0.7 + 1.5)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=[k.capitalize() for k in src_known],
                yticklabels=unknowns,
                cbar_kws={'label': 'Cosine distance (lower = more similar)'},
                ax=ax)
    ax.set_xlabel('Source Known centroid (FNO + 5-seed avg)')
    ax.set_ylabel('Source-Private class centroid')
    ax.set_title(f'PDA Source-Private vs Source-Known Cosine Distance: {source_name}')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_cosine_{source_name}_pda.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_ranking(ranking_score, nearest, source_name, src_known, save_dir):
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
    ax.set_xlabel('Mean Cosine Distance to Source-Known Centroids (FNO, 5-seed avg)')
    ax.set_title(f'PDA Source-Private Difficulty: {source_name}\n'
                 f'(smaller = closer to knowns = harder to suppress)')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k.capitalize()) for k in src_known]
    ax.legend(handles=legend_elements, title='Nearest Known (color)', loc='lower right')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_ranking_{source_name}_pda.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_family(source_name, data_path, device, save_dir):
    print(f"\n{'='*70}")
    print(f"  PDA: {source_name}  (FNO, 5-seed avg)")
    print(f"{'='*70}")

    per_seed_pair_dists = []
    src_known_ref, src_priv_ref = None, None
    for s in SEEDS:
        print(f"  -- seed {s} --")
        sk, sp, pd_dict = run_seed_pda(source_name, s, data_path, device)
        if src_known_ref is None:
            src_known_ref, src_priv_ref = sk, sp
        per_seed_pair_dists.append(pd_dict)

    pair_means, pair_stds = aggregate_seeds(per_seed_pair_dists)

    pretty_unknowns = sorted({u for (u, _) in pair_means.keys()})
    ranking_score = {}
    nearest = {}
    for u in pretty_unknowns:
        per_known = {k: pair_means[(u, k)] for k in src_known_ref if (u, k) in pair_means}
        if not per_known:
            continue
        ranking_score[u] = float(np.mean(list(per_known.values())))
        nearest[u] = min(per_known, key=per_known.get)

    print(f"\n  ── COSINE distances (FNO PDA, 5-seed avg) ──")
    print(f"  {'Source-private':<28s} ", end='')
    for k in src_known_ref:
        print(f"{k:>10s}", end='')
    print(f"  {'Mean':>8s}  {'Nearest':>14s}")
    print("  " + "-" * (28 + 10*len(src_known_ref) + 8 + 14 + 4))
    for u in sorted(ranking_score.keys(), key=lambda u: ranking_score[u]):
        print(f"  {u:<28s} ", end='')
        for k in src_known_ref:
            print(f"{pair_means.get((u, k), float('nan')):>10.3f}", end='')
        print(f"  {ranking_score[u]:>8.3f}  {nearest[u]:>14s}")

    plot_heatmap(pair_means, ranking_score, source_name, src_known_ref, save_dir)
    plot_ranking(ranking_score, nearest, source_name, src_known_ref, save_dir)

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

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, '..', 'dataset')
    # Save at project root so run_curriculum.py reads it via the same
    # `<project_root>/<dirname>/<json>` pattern as the OSDA file.
    save_dir = os.path.join(project_root, 'feature_distance_4known_fno_mean_pda')
    os.makedirs(save_dir, exist_ok=True)

    # Compute once per source family — rankings depend only on S.
    per_family = {}
    for src in SOURCE_FAMILIES:
        per_family[src] = run_family(src, data_path, device, save_dir)

    # Emit per (S, T) pair for run_curriculum.py compatibility.
    pairs = [
        ('RealWorld', 'Pamap2'), ('RealWorld', 'MHEALTH'),
        ('Pamap2', 'RealWorld'), ('Pamap2', 'MHEALTH'),
        ('MHEALTH', 'RealWorld'), ('MHEALTH', 'Pamap2'),
    ]
    all_results = {f"{s} -> {t}": per_family[s] for s, t in pairs}

    json_path = os.path.join(save_dir, 'rankings_4known_fno_mean_pda.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved rankings JSON: {json_path}")
    print(f"All plots saved in: {save_dir}")


if __name__ == '__main__':
    main()
