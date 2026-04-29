#!/usr/bin/env python3
"""
Side-by-side comparison of three unknown-class difficulty rankings:

  1. XHAR distance   — raw-signal DTW+EDR between window-averaged prototypes
  2. Confusion       — NO_ADAPT softmax: max % of unknown samples predicted as
                       the single most-confused known class
  3. Feature distance — cosine distance between 128-d NO_ADAPT centroids

For each of 6 directed pairs we train NO_ADAPT once and derive both the
confusion and feature-distance rankings from the same model. XHAR is a
model-free baseline.

Outputs:
  - similarity_comparison/rankings.json
  - similarity_comparison/ranking_table.csv
  - similarity_comparison/rank_correlation.csv
  - similarity_comparison/comparison_*.png (one combined plot per pair)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from dataloader.dataloader import data_generator
from models.models import get_backbone_class, classifier
from utils import fix_randomness

from scipy.stats import spearmanr

# ── Dataset definitions ───────────────────────────────────────────────────
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
    # RealWorld
    'climbingup': 'Climbing Up', 'climbingdown': 'Climbing Down', 'jumping': 'Jumping',
    # Pamap2
    'cycling': 'Cycling', 'Nordic walking': 'Nordic Walking',
    'ascending stairs': 'Ascending Stairs', 'descending stairs': 'Descending Stairs',
    'vacuum cleaning': 'Vacuum Cleaning', 'ironing': 'Ironing',
    'other (transient activities)': 'Other/Transient', 'rope jumping': 'Rope Jumping',
    # MHEALTH
    'Climbing stairs': 'Climbing Stairs', 'Waist bends forward': 'Waist Bends',
    'Frontal elevation of arms': 'Arm Elevation', 'Knees bending (crouching)': 'Knees Bending',
    'Cycling': 'Cycling', 'Jogging': 'Jogging', 'Jump front & back': 'Jump F&B',
}

PKL_PATHS = {
    'RealWorld': '/home/tp2474/dataset/RealWorld_processed.pkl',
    'Pamap2':    '/home/tp2474/dataset/Pamap2_processed.pkl',
    'MHEALTH':   '/home/tp2474/dataset/mhealth_processed.pkl',
}

WINDOW_SIZE = 150


# ══════════════════════════════════════════════════════════════════════════
# XHAR distance (DTW + EDR on window-averaged prototypes)
# ══════════════════════════════════════════════════════════════════════════
def dtw_distance(s, t):
    s = np.atleast_2d(s) if s.ndim == 1 else s
    t = np.atleast_2d(t) if t.ndim == 1 else t
    n, m = len(s), len(t)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(s[i-1] - t[j-1])
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    return cost[n, m] / (n + m)


def edr_distance(s, t, epsilon=None):
    s = np.atleast_2d(s) if s.ndim == 1 else s
    t = np.atleast_2d(t) if t.ndim == 1 else t
    n, m = len(s), len(t)
    if epsilon is None:
        combined = np.vstack([s, t])
        dists = np.linalg.norm(combined[::10] - np.roll(combined[::10], 1, axis=0), axis=1)
        epsilon = float(np.median(dists)) * 1.5
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(s[i-1] - t[j-1])
            match_cost = 0 if d < epsilon else 1
            dp[i, j] = min(dp[i-1, j-1] + match_cost, dp[i-1, j] + 1, dp[i, j-1] + 1)
    return dp[n, m] / max(n, m)


def xhar_distance(s, t, omega=1.0):
    return dtw_distance(s, t) + omega * edr_distance(s, t)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['user_split']


def compute_prototypes(user_split, class_labels, window_size=WINDOW_SIZE):
    class_data = defaultdict(list)
    for user, segments in user_split.items():
        for arr, lbls in segments:
            for cls in class_labels:
                mask = lbls == cls
                if mask.any():
                    class_data[cls].append(arr[mask])

    protos = {}
    for cls in class_labels:
        if cls not in class_data or not class_data[cls]:
            continue
        all_samples = np.concatenate(class_data[cls], axis=0)
        C = all_samples.shape[1]
        n_windows = len(all_samples) // window_size
        if n_windows == 0:
            padded = np.zeros((window_size, C))
            padded[:len(all_samples)] = all_samples
            padded[len(all_samples):] = all_samples[-1]
            protos[cls] = padded
        else:
            windowed = all_samples[:n_windows * window_size].reshape(n_windows, window_size, C)
            protos[cls] = windowed.mean(axis=0)
    return protos


def compute_xhar_ranking(source_name, target_name):
    """Returns {pretty_unknown: (min_dist, nearest_canonical)} ranked by dist."""
    src_data = load_pkl(PKL_PATHS[source_name])
    trg_data = load_pkl(PKL_PATHS[target_name])

    src_known_map = DATASETS[source_name]['known_mapping']
    src_known_labels = list(src_known_map.keys())

    trg_known_labels = list(DATASETS[target_name]['known_mapping'].keys())
    all_trg = set()
    for _, segs in trg_data.items():
        for arr, lbls in segs:
            all_trg.update(set(lbls))
    unknowns = sorted([u for u in (all_trg - set(trg_known_labels))
                       if u not in ('null', '0', 0)])

    src_protos = compute_prototypes(src_data, src_known_labels)
    trg_protos = compute_prototypes(trg_data, unknowns)

    out = {}
    for unk in unknowns:
        if unk not in trg_protos:
            continue
        best = None
        best_known = None
        for kn in src_known_labels:
            if kn not in src_protos:
                continue
            d = xhar_distance(src_protos[kn], trg_protos[unk])
            if best is None or d < best:
                best = d
                best_known = src_known_map[kn]
        pretty = PRETTY_NAMES.get(unk, unk)
        out[pretty] = (best, best_known)
    return out


# ══════════════════════════════════════════════════════════════════════════
# NO_ADAPT helpers (one train → both confusion & feature rankings)
# ══════════════════════════════════════════════════════════════════════════
def build_source_activity_mapping(name):
    return dict(DATASETS[name]['known_mapping'])


def build_target_activity_mapping(name):
    ds = DATASETS[name]
    mapping = dict(ds['known_mapping'])
    known_raw = set(ds['known_mapping'].keys())
    for c in ds['all_classes']:
        if c not in known_raw:
            mapping[c] = c
    return mapping


def get_target_unknowns(name):
    ds = DATASETS[name]
    known_raw = set(ds['known_mapping'].keys())
    return [c for c in ds['all_classes'] if c not in known_raw]


def train_no_adapt(src_train_dl, dataset_configs, hparams, device, num_epochs=20):
    backbone_class = get_backbone_class('CNN')
    fe = backbone_class(dataset_configs).to(device)
    cls = classifier(dataset_configs).to(device)
    opt = torch.optim.Adam(list(fe.parameters()) + list(cls.parameters()),
                           lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
    ce = nn.CrossEntropyLoss()
    fe.train(); cls.train()
    for epoch in range(1, num_epochs + 1):
        total = 0.0; n = 0
        for x, y in src_train_dl:
            x, y = x.to(device), y.to(device)
            loss = ce(cls(fe(x)), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"      Epoch {epoch}/{num_epochs} loss={total/n:.4f}")
    fe.eval(); cls.eval()
    return fe, cls


def forward_all(fe, cls, loader, device):
    feats, probs, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].float().to(device), batch[1]
            f = fe(x)
            p = F.softmax(cls(f), dim=1)
            feats.append(f.cpu()); probs.append(p.cpu()); labels.append(y.cpu())
    return torch.cat(feats), torch.cat(probs), torch.cat(labels)


def compute_confusion_ranking(probs, labels, shared_labels, src_known, target_unknowns):
    """Returns {pretty_unknown: (max_frac_predicted_as_known, nearest_canonical)}."""
    name_to_label = {n: i for i, n in enumerate(shared_labels)}
    known_indices = [name_to_label[k] for k in src_known]
    out = {}
    for unk in target_unknowns:
        idx = name_to_label[unk]
        mask = (labels == idx)
        if mask.sum() == 0:
            continue
        unk_probs = probs[mask]
        preds = unk_probs[:, :len(known_indices)].argmax(dim=1)
        fracs = {k: (preds == ki).float().mean().item()
                 for ki, k in zip(known_indices, src_known)}
        best = max(fracs, key=fracs.get)
        pretty = PRETTY_NAMES.get(unk, unk)
        out[pretty] = (fracs[best], best)
    return out


def compute_feature_ranking(feats, labels, shared_labels, src_known, target_unknowns,
                             src_feats_train, src_labels_train):
    """Cosine distance between target-unknown centroids and source-known centroids."""
    name_to_label = {n: i for i, n in enumerate(shared_labels)}
    src_centroids = {}
    for k in src_known:
        idx = name_to_label[k]
        m = (src_labels_train == idx)
        if m.sum() > 0:
            src_centroids[k] = src_feats_train[m].mean(dim=0)

    out = {}
    for unk in target_unknowns:
        idx = name_to_label[unk]
        m = (labels == idx)
        if m.sum() == 0:
            continue
        unk_centroid = feats[m].mean(dim=0)
        best = None; best_k = None
        for k, kc in src_centroids.items():
            d = 1.0 - F.cosine_similarity(unk_centroid.unsqueeze(0), kc.unsqueeze(0)).item()
            if best is None or d < best:
                best = d; best_k = k
        pretty = PRETTY_NAMES.get(unk, unk)
        out[pretty] = (best, best_k)
    return out


def run_no_adapt_pair(source_name, target_name, data_path, device, num_epochs=20, seed=42):
    print(f"    Training NO_ADAPT ({num_epochs} epochs)...")
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
    trg_private = sorted(set(trg_mapping.values()) - set(src_known))
    shared = np.array(src_known + trg_private)
    dataset_configs.num_classes = len(src_known)

    src_loaders = data_generator(os.path.join(data_path, source_name), dataset_configs,
                                  hparams, 'source', activity_mapping=src_mapping,
                                  label_list=shared)
    trg_loaders = data_generator(os.path.join(data_path, target_name), dataset_configs,
                                  hparams, 'target', activity_mapping=trg_mapping,
                                  label_list=shared)
    src_train_dl = src_loaders[0]
    trg_test_dl = trg_loaders[1]

    fe, cls = train_no_adapt(src_train_dl, dataset_configs, hparams, device, num_epochs)

    src_feats, _, src_labels = forward_all(fe, cls, src_train_dl, device)
    trg_feats, trg_probs, trg_labels = forward_all(fe, cls, trg_test_dl, device)

    conf = compute_confusion_ranking(trg_probs, trg_labels, shared, src_known, trg_private)
    feat = compute_feature_ranking(trg_feats, trg_labels, shared, src_known, trg_private,
                                    src_feats, src_labels)
    return conf, feat


# ══════════════════════════════════════════════════════════════════════════
# Combined comparison / plotting
# ══════════════════════════════════════════════════════════════════════════
def rank_order(scores_dict, reverse):
    """Return {name: rank} where rank 1 = hardest. reverse=True means higher=harder."""
    items = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=reverse)
    return {name: r + 1 for r, (name, _) in enumerate(items)}


def plot_comparison(pair_name, xhar, conf, feat, save_path):
    """Side-by-side bar chart of the three rankings for one pair."""
    # Align on intersection of names (XHAR might skip null classes)
    common = set(xhar) & set(conf) & set(feat)
    if not common:
        print(f"    WARN: no common unknowns for {pair_name}")
        return

    # Use feature-distance order as the reference (hardest first)
    order = sorted(common, key=lambda n: feat[n][0])

    xhar_vals = [xhar[n][0] for n in order]
    conf_vals = [conf[n][0] * 100 for n in order]
    feat_vals = [feat[n][0] for n in order]

    fig, axes = plt.subplots(1, 3, figsize=(17, max(3.5, 0.5 * len(order) + 2)),
                              sharey=True)

    colors = {'lying': '#e74c3c', 'sitting': '#3498db', 'standing': '#2ecc71',
              'walking': '#f39c12', 'running': '#9b59b6'}

    for ax, vals, title, xlabel, src in zip(
        axes,
        [xhar_vals, conf_vals, feat_vals],
        ['XHAR (DTW+EDR, raw signal)',
         'Confusion (NO_ADAPT softmax)',
         'Feature Distance (NO_ADAPT cosine)'],
        ['XHAR distance (lower = harder)',
         '% predicted as nearest known (higher = harder)',
         'Cosine distance (lower = harder)'],
        [xhar, conf, feat],
    ):
        bar_colors = [colors.get(src[n][1], '#95a5a6') for n in order]
        ax.barh(range(len(order)), vals, color=bar_colors)
        for i, (n, v) in enumerate(zip(order, vals)):
            ax.text(v + max(vals) * 0.01, i, f'{v:.2f} → {src[n][1]}',
                    va='center', fontsize=8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)

    fig.suptitle(f'{pair_name}  (rows ordered by feature-distance difficulty, hardest first)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'similarity_comparison')
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
        pair_key = f"{src} -> {trg}"
        print(f"\n{'='*70}\n  {pair_key}\n{'='*70}")

        print("  [1/3] XHAR distance (raw signal)...")
        xhar = compute_xhar_ranking(src, trg)

        print("  [2/3] Training NO_ADAPT + [3/3] confusion & feature-distance...")
        conf, feat = run_no_adapt_pair(src, trg, data_path, device, num_epochs=20)

        all_results[pair_key] = {
            'xhar':       {n: {'score': s, 'nearest': k} for n, (s, k) in xhar.items()},
            'confusion':  {n: {'score': s, 'nearest': k} for n, (s, k) in conf.items()},
            'feat_cosine':{n: {'score': s, 'nearest': k} for n, (s, k) in feat.items()},
        }

        plot_comparison(pair_key, xhar, conf, feat,
                        os.path.join(save_dir, f'comparison_{src}_to_{trg}.png'))

    # ── Save raw results ──
    json_path = os.path.join(save_dir, 'rankings.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved rankings → {json_path}")

    # ── Combined CSV (ranks, not raw scores) ──
    csv_path = os.path.join(save_dir, 'ranking_table.csv')
    with open(csv_path, 'w') as f:
        f.write("pair,unknown,xhar_rank,xhar_nearest,conf_rank,conf_nearest,feat_rank,feat_nearest\n")
        for pair_key, r in all_results.items():
            common = set(r['xhar']) & set(r['confusion']) & set(r['feat_cosine'])
            # For confusion, higher=harder → reverse=True; XHAR and feat lower=harder
            x_rank = rank_order({n: r['xhar'][n]['score'] for n in common}, reverse=False)
            c_rank = rank_order({n: r['confusion'][n]['score'] for n in common}, reverse=True)
            fr_rank = rank_order({n: r['feat_cosine'][n]['score'] for n in common}, reverse=False)
            for n in sorted(common, key=lambda x: fr_rank[x]):
                f.write(f"{pair_key},{n},"
                        f"{x_rank[n]},{r['xhar'][n]['nearest']},"
                        f"{c_rank[n]},{r['confusion'][n]['nearest']},"
                        f"{fr_rank[n]},{r['feat_cosine'][n]['nearest']}\n")
    print(f"Saved ranking table → {csv_path}")

    # ── Spearman rank correlation between methods (per pair + overall) ──
    corr_path = os.path.join(save_dir, 'rank_correlation.csv')
    with open(corr_path, 'w') as f:
        f.write("pair,xhar_vs_conf,xhar_vs_feat,conf_vs_feat,n_unknowns\n")
        agg = {'xc': [], 'xf': [], 'cf': []}
        for pair_key, r in all_results.items():
            common = list(set(r['xhar']) & set(r['confusion']) & set(r['feat_cosine']))
            if len(common) < 3:
                f.write(f"{pair_key},NA,NA,NA,{len(common)}\n")
                continue
            x = np.array([r['xhar'][n]['score'] for n in common])
            c = np.array([r['confusion'][n]['score'] for n in common])
            ft = np.array([r['feat_cosine'][n]['score'] for n in common])
            # higher=harder normalization: flip confusion sign
            # Use Spearman on scores directly (sign handled by negating conf)
            xc, _ = spearmanr(x, -c)   # both "low = hard" after flipping conf
            xf, _ = spearmanr(x, ft)
            cf, _ = spearmanr(-c, ft)
            f.write(f"{pair_key},{xc:.3f},{xf:.3f},{cf:.3f},{len(common)}\n")
            agg['xc'].append(xc); agg['xf'].append(xf); agg['cf'].append(cf)
        if agg['xc']:
            f.write(f"MEAN,{np.mean(agg['xc']):.3f},{np.mean(agg['xf']):.3f},"
                    f"{np.mean(agg['cf']):.3f},-\n")
    print(f"Saved correlations → {corr_path}")

    print(f"\nDone. All outputs in: {save_dir}")


if __name__ == '__main__':
    main()
