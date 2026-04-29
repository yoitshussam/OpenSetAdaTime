#!/usr/bin/env python3
"""
Compute similarity scores between known and unknown HAR classes using
feature-space centroid distances from a NO_ADAPT (source-only) CNN model.

For each directed dataset pair:
  1. Train a source-only CNN on the 5 known classes
  2. Forward-pass target samples through the feature extractor
  3. Compute per-class centroids in the 128-d feature space
     (source centroids from source data; unknown centroids from target data)
  4. Compute cosine + L2 distances between each unknown centroid and each
     known centroid → min distance = how close unknown lies to a known cluster
  5. Rank unknowns by min distance (lower = harder to reject)

This bypasses softmax calibration issues that the confusion-based method
suffered from. Produces heatmap + ranking plots in the same format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from dataloader.dataloader import data_generator
from models.models import get_backbone_class, classifier
from utils import fix_randomness

# ── Dataset definitions (shared with compute_confusion_similarity.py) ──
KNOWN_CLASSES = ['lying', 'sitting', 'standing', 'walking', 'running']

DATASETS = {
    'RealWorld': {
        'all_classes': ['walking', 'running', 'sitting', 'standing', 'lying',
                        'climbingup', 'climbingdown', 'jumping'],
        'known_mapping': {
            'lying': 'lying', 'sitting': 'sitting', 'standing': 'standing',
            'walking': 'walking', 'running': 'running',
        },
    },
    'Pamap2': {
        'all_classes': ['lying', 'sitting', 'standing', 'walking', 'running',
                        'cycling', 'Nordic walking', 'ascending stairs',
                        'descending stairs', 'vacuum cleaning', 'ironing',
                        'other (transient activities)', 'rope jumping'],
        'known_mapping': {
            'lying': 'lying', 'sitting': 'sitting', 'standing': 'standing',
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
            'Standing still': 'standing', 'Walking': 'walking', 'Running': 'running',
        },
    },
}

PRETTY_NAMES = {
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
    cross_entropy = nn.CrossEntropyLoss()

    feature_extractor.train(); cls.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0; n_batches = 0
        for src_x, src_y in src_train_dl:
            src_x, src_y = src_x.to(device), src_y.to(device)
            feat = feature_extractor(src_x)
            loss = cross_entropy(cls(feat), src_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); n_batches += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{num_epochs} — loss: {epoch_loss / n_batches:.4f}")

    feature_extractor.eval(); cls.eval()
    return feature_extractor, cls


def extract_features(feature_extractor, data_loader, device):
    """Forward-pass all samples, return features + labels."""
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
    """Compute mean feature vector per class."""
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
    """1 - cosine similarity between two vectors."""
    return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def l2_distance(a, b):
    return torch.norm(a - b, p=2).item()


def compute_distance_matrix(src_centroids, trg_centroids, distance_fn,
                             source_known, target_unknowns):
    """
    For each unknown (target) class, compute distance to each known (source) class.

    Returns:
        distances: dict {unknown_pretty_name: {known_name: distance}}
        min_dist: dict {unknown_pretty_name: min distance to any known}
        nearest: dict {unknown_pretty_name: name of nearest known}
    """
    distances = {}
    min_dist = {}
    nearest = {}

    for unk_name in target_unknowns:
        if unk_name not in trg_centroids:
            continue
        pretty = PRETTY_NAMES.get(unk_name, unk_name)
        row = {}
        for k in source_known:
            if k not in src_centroids:
                continue
            row[k] = distance_fn(trg_centroids[unk_name], src_centroids[k])
        if not row:
            continue
        distances[pretty] = row
        nearest_k = min(row, key=row.get)
        min_dist[pretty] = row[nearest_k]
        nearest[pretty] = nearest_k

    return distances, min_dist, nearest


def plot_distance_heatmap(distances, min_dist, source_name, target_name,
                           source_known, save_dir, metric_name='cosine'):
    unknowns = list(distances.keys())
    # sort by min distance (hardest = smallest first)
    sort_idx = sorted(range(len(unknowns)), key=lambda i: min_dist[unknowns[i]])
    unknowns = [unknowns[i] for i in sort_idx]

    data = np.array([[distances[u].get(k, np.nan) for k in source_known] for u in unknowns])

    fig, ax = plt.subplots(figsize=(10, max(4, len(unknowns) * 0.7 + 1.5)))
    # Reverse colormap: green = low (similar), red = high (dissimilar)
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=[k.capitalize() for k in source_known],
                yticklabels=unknowns,
                cbar_kws={'label': f'{metric_name.capitalize()} distance (lower = more similar)'},
                ax=ax)

    # Mark the nearest known for each unknown with a bold box
    for i, u in enumerate(unknowns):
        row = distances[u]
        best_k = min(row, key=row.get)
        j = source_known.index(best_k)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2.5))

    ax.set_xlabel('Known Class (Source centroid)')
    ax.set_ylabel('Unknown Class (Target centroid)')
    ax.set_title(f'Feature-Space {metric_name.capitalize()} Distance: {source_name} → {target_name}\n'
                 f'(Target unknown centroids vs Source known centroids, 128-d CNN features)')
    plt.tight_layout()
    path = os.path.join(save_dir, f'feat_{metric_name}_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_distance_ranking(min_dist, nearest, source_name, target_name,
                           source_known, save_dir, metric_name='cosine'):
    sorted_unknowns = sorted(min_dist.keys(), key=lambda u: min_dist[u])

    fig, ax = plt.subplots(figsize=(12, max(3, len(sorted_unknowns) * 0.55 + 1.5)))

    colors = {'lying': '#e74c3c', 'sitting': '#3498db', 'standing': '#2ecc71',
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

    ax.set_xlabel(f'Min {metric_name.capitalize()} Distance to Nearest Known Centroid')
    ax.set_title(f'Unknown Class Difficulty ({metric_name.capitalize()}): {source_name} → {target_name}\n'
                 f'(smaller = closer to a known cluster = harder to reject)')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k.capitalize()) for k in source_known]
    ax.legend(handles=legend_elements, title='Nearest Known', loc='lower right')

    plt.tight_layout()
    path = os.path.join(save_dir, f'feat_ranking_{metric_name}_{source_name}_to_{target_name}.png')
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

    print(f"  Known: {src_known}")
    print(f"  Unknown: {trg_private}")

    source_data_path = os.path.join(data_path, source_name)
    target_data_path = os.path.join(data_path, target_name)

    print(f"\n  Loading data...")
    src_loaders = data_generator(source_data_path, dataset_configs, hparams, 'source',
                                  activity_mapping=src_mapping, label_list=shared_label_list)
    trg_loaders = data_generator(target_data_path, dataset_configs, hparams, 'target',
                                  activity_mapping=trg_mapping, label_list=shared_label_list)

    src_train_dl = src_loaders[0]
    trg_test_dl = trg_loaders[1]

    print(f"  Training ({num_epochs} epochs)...")
    feature_extractor, cls = train_no_adapt(src_train_dl, dataset_configs, hparams, device, num_epochs)

    # ── Extract features ──
    # Source centroids from the source TRAINING data (the 5 known clusters)
    print(f"  Extracting source features...")
    src_feats, src_labels = extract_features(feature_extractor, src_train_dl, device)
    src_centroids = compute_centroids(src_feats, src_labels, name_to_label, src_known)

    # Target centroids: knowns (for sanity) + unknowns from target data
    print(f"  Extracting target features...")
    trg_feats, trg_labels = extract_features(feature_extractor, trg_test_dl, device)
    trg_unk_centroids = compute_centroids(trg_feats, trg_labels, name_to_label, trg_private)

    # ── Compute distances ──
    for metric_name, dist_fn in [('cosine', cosine_distance), ('l2', l2_distance)]:
        distances, min_dist, nearest = compute_distance_matrix(
            src_centroids, trg_unk_centroids, dist_fn, src_known, trg_private)

        # Print
        print(f"\n  ── {metric_name.upper()} distances ──")
        print(f"  {'Unknown class':<28s} ", end='')
        for k in src_known:
            print(f"{k:>10s}", end='')
        print(f"  {'Min':>8s}  {'Nearest':>14s}")
        print("  " + "-" * (28 + 10*len(src_known) + 8 + 14 + 4))

        sorted_unknowns = sorted(min_dist.keys(), key=lambda u: min_dist[u])
        for u in sorted_unknowns:
            print(f"  {u:<28s} ", end='')
            for k in src_known:
                print(f"{distances[u].get(k, float('nan')):>10.3f}", end='')
            print(f"  {min_dist[u]:>8.3f}  {nearest[u]:>14s}")

        plot_distance_heatmap(distances, min_dist, source_name, target_name,
                               src_known, save_dir, metric_name)
        plot_distance_ranking(min_dist, nearest, source_name, target_name,
                               src_known, save_dir, metric_name)

    return src_centroids, trg_unk_centroids


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_distance_plots')
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 30
    seed = 42

    pairs = [
        ('RealWorld', 'Pamap2'),
        ('RealWorld', 'MHEALTH'),
        ('Pamap2', 'RealWorld'),
        ('Pamap2', 'MHEALTH'),
        ('MHEALTH', 'RealWorld'),
        ('MHEALTH', 'Pamap2'),
    ]

    for src, trg in pairs:
        run_pair(src, trg, data_path, device, save_dir, num_epochs, seed)

    print(f"\n\nAll plots saved in: {save_dir}")


if __name__ == '__main__':
    main()
