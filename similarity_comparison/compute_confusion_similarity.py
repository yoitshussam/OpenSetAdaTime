#!/usr/bin/env python3
"""
Compute similarity scores between known and unknown HAR classes using a
NO_ADAPT (source-only) CNN model's prediction confusion.

For each of the 3 target datasets, we:
  1. Train a source-only CNN on the 5 known classes (from a source dataset)
  2. Forward-pass ALL target samples (known + unknown) through the model
  3. For each unknown class, measure how the model distributes its predictions
     across the 5 known classes → this confusion pattern = similarity signal
  4. Rank unknowns by how confusable they are (higher max-confusion = harder)

Produces heatmap + ranking plots similar to the XHAR distance analysis.
"""

import sys
import os

# Make sure we can import from the OpenSet-AdaTime project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from dataloader.dataloader import data_generator
from models.models import get_backbone_class, classifier
from utils import fix_randomness, AverageMeter

# ── Dataset definitions ─────────────────────────────────────────────────
# The 5 known classes shared across all 3 datasets
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

# Pretty names for unknown classes (for plots)
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
    """Return unknown classes = all_classes minus the known mapping keys."""
    ds = DATASETS[dataset_name]
    known_raw = set(ds['known_mapping'].keys())
    return [c for c in ds['all_classes'] if c not in known_raw]


def build_target_activity_mapping(target_name):
    """Build activity mapping for target: known classes + all unknowns (each maps to itself)."""
    ds = DATASETS[target_name]
    mapping = dict(ds['known_mapping'])  # known classes
    for unk in get_unknown_classes(target_name):
        mapping[unk] = unk  # unknown classes map to themselves
    return mapping


def build_source_activity_mapping(source_name):
    """Build activity mapping for source: only known classes."""
    return dict(DATASETS[source_name]['known_mapping'])


def train_no_adapt(src_train_dl, dataset_configs, hparams, device, num_epochs=30):
    """Train a source-only CNN and return the trained model."""
    backbone_class = get_backbone_class('CNN')
    feature_extractor = backbone_class(dataset_configs).to(device)
    cls = classifier(dataset_configs).to(device)

    optimizer = torch.optim.Adam(
        list(feature_extractor.parameters()) + list(cls.parameters()),
        lr=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],
    )
    cross_entropy = nn.CrossEntropyLoss()

    feature_extractor.train()
    cls.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for src_x, src_y in src_train_dl:
            src_x, src_y = src_x.to(device), src_y.to(device)
            feat = feature_extractor(src_x)
            pred = cls(feat)
            loss = cross_entropy(pred, src_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{num_epochs} — loss: {epoch_loss / n_batches:.4f}")

    feature_extractor.eval()
    cls.eval()
    return feature_extractor, cls


def predict_all(feature_extractor, cls, data_loader, device):
    """Forward-pass all samples, return softmax probs and true labels."""
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch[0], batch[1]
            data = data.float().to(device)
            feat = feature_extractor(data)
            logits = cls(feat)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_probs), torch.cat(all_labels)


def compute_confusion_matrix(probs, labels, shared_label_list, source_known, target_unknowns):
    """
    For each unknown class, compute the distribution of model predictions
    across the known classes.

    Returns:
        confusion: dict {unknown_name: {known_name: fraction}}
        max_conf: dict {unknown_name: max confidence for any single known class}
        max_known: dict {unknown_name: which known class gets highest fraction}
        mean_conf: dict {unknown_name: mean softmax confidence}
    """
    label_to_name = {i: name for i, name in enumerate(shared_label_list)}
    name_to_label = {name: i for i, name in enumerate(shared_label_list)}

    # Known class indices (in the classifier output, these are 0..K-1)
    known_indices = [name_to_label[k] for k in source_known]

    confusion = {}
    max_conf = {}
    max_known = {}
    mean_confidence = {}

    for unk_name in target_unknowns:
        unk_idx = name_to_label[unk_name]

        # Mask for samples of this unknown class
        mask = (labels == unk_idx)
        if mask.sum() == 0:
            continue

        unk_probs = probs[mask]  # (N_unk, num_classes)

        # Distribution: for each known class, what fraction of predictions go there
        pred_classes = unk_probs[:, :len(known_indices)].argmax(dim=1)  # argmax over known dims only
        dist = {}
        for ki, kname in zip(known_indices, source_known):
            frac = (pred_classes == ki).float().mean().item()
            dist[kname] = frac

        # Max softmax confidence across all samples (how confident is the model?)
        max_softmax = unk_probs[:, :len(known_indices)].max(dim=1).values.mean().item()

        # Which known class gets the most predictions?
        best_known_idx = max(dist, key=dist.get)

        pretty = PRETTY_NAMES.get(unk_name, unk_name)
        confusion[pretty] = dist
        max_conf[pretty] = max(dist.values())
        max_known[pretty] = best_known_idx
        mean_confidence[pretty] = max_softmax

    return confusion, max_conf, max_known, mean_confidence


def plot_confusion_heatmap(confusion, source_name, target_name, known_classes, save_dir):
    """Plot heatmap of model confusion: unknown (rows) × known (cols)."""
    unknowns = list(confusion.keys())
    data = np.array([[confusion[u].get(k, 0) for k in known_classes] for u in unknowns])

    # Sort by max confusion (hardest first)
    sort_idx = np.argsort(-data.max(axis=1))
    data = data[sort_idx]
    unknowns = [unknowns[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, len(unknowns) * 0.7 + 1.5)))
    sns.heatmap(data * 100, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[k.capitalize() for k in known_classes],
                yticklabels=unknowns,
                cbar_kws={'label': '% of predictions'},
                ax=ax, vmin=0, vmax=100)
    ax.set_xlabel('Predicted Known Class')
    ax.set_ylabel('Unknown Class (Target)')
    ax.set_title(f'Model Confusion: {source_name} → {target_name}\n(% of unknown samples predicted as each known class)')
    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_ranking(confusion, max_conf, max_known, mean_confidence,
                           source_name, target_name, known_classes, save_dir):
    """Bar chart ranking unknowns by max confusion fraction (hardest = most confusable)."""
    # Sort by max confusion fraction (hardest first)
    sorted_unknowns = sorted(max_conf.keys(), key=lambda u: max_conf[u], reverse=True)

    fig, ax = plt.subplots(figsize=(12, max(3, len(sorted_unknowns) * 0.55 + 1.5)))

    colors = {'lying': '#e74c3c', 'sitting': '#3498db', 'standing': '#2ecc71',
              'walking': '#f39c12', 'running': '#9b59b6'}

    bars = ax.barh(range(len(sorted_unknowns)),
                   [max_conf[u] * 100 for u in sorted_unknowns],
                   color=[colors.get(max_known[u], '#95a5a6') for u in sorted_unknowns])

    ax.set_yticks(range(len(sorted_unknowns)))
    ax.set_yticklabels(sorted_unknowns)
    ax.invert_yaxis()

    for i, u in enumerate(sorted_unknowns):
        ax.text(max_conf[u] * 100 + 0.5, i,
                f'{max_conf[u]*100:.1f}% → {max_known[u]} (avg conf: {mean_confidence[u]:.2f})',
                va='center', fontsize=9)

    ax.set_xlabel('% Predicted as Most-Confused Known Class')
    ax.set_title(f'Unknown Class Difficulty (Model Confusion): {source_name} → {target_name}')
    ax.axvline(x=100 / len(known_classes), color='gray', linestyle='--', alpha=0.5, label='Random chance')

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k.capitalize()) for k in known_classes]
    ax.legend(handles=legend_elements, title='Most-confused as', loc='lower right')

    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_ranking_{source_name}_to_{target_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_pair(source_name, target_name, data_path, device, save_dir, num_epochs=30, seed=42):
    """Run the full pipeline for one source→target pair."""
    print(f"\n{'='*70}")
    print(f"  {source_name} → {target_name}")
    print(f"{'='*70}")

    fix_randomness(seed)

    # ── Configs ──
    dataset_configs = get_dataset_class("ALL")()
    hparams_class = get_hparams_class("ALL")()
    hparams = {**hparams_class.alg_hparams['NO_ADAPT'], **hparams_class.train_params}

    dataset_configs.da_method = ''  # no special return_index needed
    dataset_configs.feat_dim = dataset_configs.features_len * dataset_configs.final_out_channels  # 128

    # Activity mappings
    src_mapping = build_source_activity_mapping(source_name)
    trg_mapping = build_target_activity_mapping(target_name)

    dataset_configs.source_activity_mapping = src_mapping
    dataset_configs.target_activity_mapping = trg_mapping

    # Shared label list: known first, then unknown
    src_known = sorted(set(src_mapping.values()))
    trg_all = set(trg_mapping.values())
    trg_private = sorted(trg_all - set(src_known))
    shared_label_list = np.array(src_known + trg_private)

    dataset_configs.num_classes = len(src_known)  # classifier is K-way (known only)

    print(f"  Known classes: {src_known}")
    print(f"  Unknown classes: {trg_private}")
    print(f"  Shared label list: {list(shared_label_list)}")

    # ── Load data ──
    source_data_path = os.path.join(data_path, source_name)
    target_data_path = os.path.join(data_path, target_name)

    print(f"\n  Loading source data ({source_name})...")
    src_loaders = data_generator(source_data_path, dataset_configs, hparams, 'source',
                                  activity_mapping=src_mapping, label_list=shared_label_list)
    src_train_dl = src_loaders[0]

    print(f"  Loading target data ({target_name})...")
    trg_loaders = data_generator(target_data_path, dataset_configs, hparams, 'target',
                                  activity_mapping=trg_mapping, label_list=shared_label_list)
    trg_test_dl = trg_loaders[1]

    print(f"  Source train: {len(src_train_dl.dataset)} samples")
    print(f"  Target test:  {len(trg_test_dl.dataset)} samples")

    # ── Train ──
    print(f"\n  Training NO_ADAPT model ({num_epochs} epochs)...")
    feature_extractor, cls = train_no_adapt(src_train_dl, dataset_configs, hparams, device, num_epochs)

    # ── Predict on target ──
    print(f"  Running inference on target...")
    probs, labels = predict_all(feature_extractor, cls, trg_test_dl, device)

    # ── Source accuracy (sanity check) ──
    src_test_dl = src_loaders[1]
    src_probs, src_labels = predict_all(feature_extractor, cls, src_test_dl, device)
    src_preds = src_probs[:, :len(src_known)].argmax(dim=1)
    src_acc = (src_preds == src_labels).float().mean().item()
    print(f"  Source test accuracy: {src_acc:.4f}")

    # ── Compute confusion ──
    confusion, max_conf, max_known, mean_confidence = compute_confusion_matrix(
        probs, labels, shared_label_list, src_known, trg_private)

    # ── Print results ──
    print(f"\n  {'Unknown class':<28s} ", end='')
    for k in src_known:
        print(f"{k:>10s}", end='')
    print(f"  {'Max%':>8s}  {'Predicted as':>14s}  {'Avg Conf':>10s}")
    print("  " + "-" * (28 + 10*len(src_known) + 8 + 14 + 10 + 8))

    sorted_unknowns = sorted(max_conf.keys(), key=lambda u: max_conf[u], reverse=True)
    for u in sorted_unknowns:
        print(f"  {u:<28s} ", end='')
        for k in src_known:
            print(f"{confusion[u].get(k, 0)*100:>9.1f}%", end='')
        print(f"  {max_conf[u]*100:>7.1f}%  {max_known[u]:>14s}  {mean_confidence[u]:>10.3f}")

    # ── Plots ──
    plot_confusion_heatmap(confusion, source_name, target_name, src_known, save_dir)
    plot_confusion_ranking(confusion, max_conf, max_known, mean_confidence,
                           source_name, target_name, src_known, save_dir)

    return confusion, max_conf, max_known, mean_confidence


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_plots')
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 30
    seed = 42

    # All 6 directed pairs
    pairs = [
        ('RealWorld', 'Pamap2'),
        ('RealWorld', 'MHEALTH'),
        ('Pamap2', 'RealWorld'),
        ('Pamap2', 'MHEALTH'),
        ('MHEALTH', 'RealWorld'),
        ('MHEALTH', 'Pamap2'),
    ]

    all_results = {}
    for source_name, target_name in pairs:
        confusion, max_conf, max_known, mean_confidence = run_pair(
            source_name, target_name, data_path, device, save_dir, num_epochs, seed)
        all_results[(source_name, target_name)] = {
            'confusion': confusion, 'max_conf': max_conf,
            'max_known': max_known, 'mean_confidence': mean_confidence,
        }

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: Unknown classes ranked by model confusion")
    print(f"{'='*70}")
    for (src, trg), res in all_results.items():
        print(f"\n  {src} → {trg}:")
        sorted_unknowns = sorted(res['max_conf'].keys(),
                                  key=lambda u: res['max_conf'][u], reverse=True)
        for i, u in enumerate(sorted_unknowns, 1):
            print(f"    {i}. {u:<28s} ({res['max_conf'][u]*100:.1f}% → {res['max_known'][u]}, "
                  f"avg_conf={res['mean_confidence'][u]:.3f})")


if __name__ == '__main__':
    main()
