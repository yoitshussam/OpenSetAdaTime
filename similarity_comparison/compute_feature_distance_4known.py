#!/usr/bin/env python3
"""
Feature-space centroid distance with 4 known classes:
  sitting, lying, walking, running  (standing is now UNKNOWN).

Same pipeline as compute_feature_distance.py: train NO_ADAPT CNN on source
knowns, centroid each class in 128-d feature space, rank target unknowns
by min cosine distance to any known centroid (lower = harder).
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

from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from dataloader.dataloader import data_generator
from models.models import get_backbone_class, classifier
from utils import fix_randomness

KNOWN_CLASSES = ['lying', 'sitting', 'walking', 'running']  # 4 knowns, standing dropped

DATASETS = {
    'RealWorld': {
        'all_classes': ['walking', 'running', 'sitting', 'standing', 'lying',
                        'climbingup', 'climbingdown', 'jumping'],
        'known_mapping': {
            'lying': 'lying', 'sitting': 'sitting',
            'walking': 'walking', 'running': 'running',
        },
    },
    'Pamap2': {
        'all_classes': ['lying', 'sitting', 'standing', 'walking', 'running',
                        'cycling', 'Nordic walking', 'ascending stairs',
                        'descending stairs', 'vacuum cleaning', 'ironing',
                        'other (transient activities)', 'rope jumping'],
        'known_mapping': {
            'lying': 'lying', 'sitting': 'sitting',
            'walking': 'walking', 'running': 'running',
        },
    },
    'MHEALTH': {
        'all_classes': ['Standing still', 'Sitting and relaxing', 'Lying down',
                        'Walking', 'Climbing stairs', 'Waist bends forward',
                        'Frontal elevation of arms', 'Knees bending (crouching)',
                        'Cycling', 'Jogging', 'Running', 'Jump front & back'],
        'known_mapping': {
            'Lying down': 'lying', 'Sitting and relaxing': 'sitting',
            'Walking': 'walking', 'Running': 'running',
        },
    },
}

PRETTY_NAMES = {
    'standing': 'Standing',
    'Standing still': 'Standing',
    'climbingup': 'Climbing Up', 'climbingdown': 'Climbing Down', 'jumping': 'Jumping',
    'cycling': 'Cycling', 'Nordic walking': 'Nordic Walking',
    'ascending stairs': 'Ascending Stairs', 'descending stairs': 'Descending Stairs',
    'vacuum cleaning': 'Vacuum Cleaning', 'ironing': 'Ironing',
    'other (transient activities)': 'Other/Transient', 'rope jumping': 'Rope Jumping',
    'Climbing stairs': 'Climbing Stairs', 'Waist bends forward': 'Waist Bends',
    'Frontal elevation of arms': 'Arm Elevation', 'Knees bending (crouching)': 'Knees Bending',
    'Cycling': 'Cycling', 'Jogging': 'Jogging', 'Jump front & back': 'Jump F&B',
}


def get_unknown_classes(dataset_name):
    ds = DATASETS[dataset_name]
    known_raw = set(ds['known_mapping'].keys())
    return [c for c in ds['all_classes'] if c not in known_raw]


def build_target_activity_mapping(target_name):
    ds = DATASETS[target_name]
    mapping = dict(ds['known_mapping'])
    for unk in get_unknown_classes(target_name):
        mapping[unk] = unk
    return mapping


def build_source_activity_mapping(source_name):
    return dict(DATASETS[source_name]['known_mapping'])


def train_no_adapt(src_train_dl, dataset_configs, hparams, device, num_epochs=30):
    backbone_class = get_backbone_class('CNN')
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
            print(f"    Epoch {epoch}/{num_epochs} — loss: {epoch_loss / n_batches:.4f}")

    feature_extractor.eval(); cls.eval()
    return feature_extractor, cls


def extract_features(feature_extractor, data_loader, device):
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch[0], batch[1]
            data = data.float().to(device)
            feat = feature_extractor(data)
            all_feats.append(feat.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_feats), torch.cat(all_labels)


def compute_centroids(features, labels, name_to_label, class_names):
    centroids = {}
    for name in class_names:
        if name not in name_to_label:
            continue
        idx = name_to_label[name]
        mask = (labels == idx)
        if mask.sum() == 0:
            continue
        centroids[name] = features[mask].mean(dim=0)
    return centroids


def cosine_distance(a, b):
    return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_distance_matrix(src_centroids, trg_centroids, distance_fn,
                             source_known, target_unknowns):
    distances, min_dist, nearest = {}, {}, {}
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
        nearest_k = min(row, key=row.get)
        min_dist[pretty] = row[nearest_k]
        nearest[pretty] = nearest_k
    return distances, min_dist, nearest


def plot_heatmap(distances, min_dist, source_name, target_name, source_known, save_dir):
    unknowns = sorted(distances.keys(), key=lambda u: min_dist[u])
    data = np.array([[distances[u].get(k, np.nan) for k in source_known] for u in unknowns])

    fig, ax = plt.subplots(figsize=(9, max(4, len(unknowns) * 0.7 + 1.5)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=[k.capitalize() for k in source_known],
                yticklabels=unknowns,
                cbar_kws={'label': 'Cosine distance (lower = more similar)'},
                ax=ax)
    for i, u in enumerate(unknowns):
        best_k = min(distances[u], key=distances[u].get)
        j = source_known.index(best_k)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2.5))

    ax.set_xlabel('Known Class (Source centroid, 4-known)')
    ax.set_ylabel('Unknown Class (Target centroid)')
    ax.set_title(f'Feature-Space Cosine Distance (4 knowns): {source_name} → {target_name}')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_cosine_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_ranking(min_dist, nearest, source_name, target_name, source_known, save_dir):
    sorted_unknowns = sorted(min_dist.keys(), key=lambda u: min_dist[u])

    fig, ax = plt.subplots(figsize=(12, max(3, len(sorted_unknowns) * 0.55 + 1.5)))
    colors = {'lying': '#e74c3c', 'sitting': '#3498db',
              'walking': '#f39c12', 'running': '#9b59b6'}

    ax.barh(range(len(sorted_unknowns)),
            [min_dist[u] for u in sorted_unknowns],
            color=[colors.get(nearest[u], '#95a5a6') for u in sorted_unknowns])

    ax.set_yticks(range(len(sorted_unknowns)))
    ax.set_yticklabels(sorted_unknowns)
    ax.invert_yaxis()
    for i, u in enumerate(sorted_unknowns):
        ax.text(min_dist[u] + max(min_dist.values()) * 0.01, i,
                f'{min_dist[u]:.3f} → {nearest[u]}',
                va='center', fontsize=9)
    ax.set_xlabel('Min Cosine Distance to Nearest Known Centroid')
    ax.set_title(f'Unknown Difficulty (4 knowns): {source_name} → {target_name}\n'
                 f'(smaller = closer to a known = harder to reject)')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k.capitalize()) for k in source_known]
    ax.legend(handles=legend_elements, title='Nearest Known', loc='lower right')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat4_ranking_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_pair(source_name, target_name, data_path, device, save_dir, num_epochs=30, seed=42):
    print(f"\n{'='*70}")
    print(f"  {source_name} → {target_name}")
    print(f"{'='*70}")

    fix_randomness(seed)
    dataset_configs = get_dataset_class("ALL")()
    hparams_class = get_hparams_class("ALL")()
    hparams = {**hparams_class.alg_hparams['NO_ADAPT'], **hparams_class.train_params}

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

    distances, min_dist, nearest = compute_distance_matrix(
        src_centroids, trg_unk_centroids, cosine_distance, src_known, trg_private)

    print(f"\n  ── COSINE distances ──")
    print(f"  {'Unknown class':<28s} ", end='')
    for k in src_known:
        print(f"{k:>10s}", end='')
    print(f"  {'Min':>8s}  {'Nearest':>14s}")
    print("  " + "-" * (28 + 10*len(src_known) + 8 + 14 + 4))
    for u in sorted(min_dist.keys(), key=lambda u: min_dist[u]):
        print(f"  {u:<28s} ", end='')
        for k in src_known:
            print(f"{distances[u].get(k, float('nan')):>10.3f}", end='')
        print(f"  {min_dist[u]:>8.3f}  {nearest[u]:>14s}")

    plot_heatmap(distances, min_dist, source_name, target_name, src_known, save_dir)
    plot_ranking(min_dist, nearest, source_name, target_name, src_known, save_dir)

    return {
        'knowns': src_known,
        'rankings': [
            {'unknown': u, 'min_cosine_dist': float(min_dist[u]),
             'nearest_known': nearest[u],
             'per_class': {k: float(distances[u].get(k, float('nan'))) for k in src_known}}
            for u in sorted(min_dist.keys(), key=lambda u: min_dist[u])
        ],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_distance_4known')
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

    json_path = os.path.join(save_dir, 'rankings_4known.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved rankings JSON: {json_path}")
    print(f"All plots saved in: {save_dir}")


if __name__ == '__main__':
    main()
