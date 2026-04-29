"""
Compute XHAR distance (DTW + EDR) between known and unknown classes
for cross-dataset domain adaptation pairs.

Based on the domain relevance calculator from:
    XHAR: Deep Domain Adaptation for Human Activity Recognition
    Zhou et al., IEEE SECON 2020

For each directed pair (Source -> Target):
  - Known classes: the 5 shared classes, prototypes from SOURCE
  - Unknown classes: remaining target classes, prototypes from TARGET
  - For each unknown, compute distance to every known class
  - Rank unknowns by min distance (nearest known) = similarity/difficulty

Output: full distance matrix + ranked unknown classes per pair.
"""

import pickle
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import warnings
warnings.filterwarnings("ignore")

PLOT_DIR = '/mnt/data/home/tp2474/SELF_HAR/xhar_plots'

# ============================================================
# Dataset paths and class mappings
# ============================================================

PKL_PATHS = {
    'RealWorld': '/mnt/data/home/tp2474/SELF_HAR/run/processed_datasets/RealWorld_processed.pkl',
    'Pamap2':    '/mnt/data/home/tp2474/SELF_HAR/run/processed_datasets/Pamap2_processed.pkl',
    'MHEALTH':   '/mnt/data/home/tp2474/SELF_HAR/run/processed_datasets/mhealth_processed.pkl',
}

# Map each dataset's raw label to a canonical name for the 5 known classes.
# Labels not listed here are "unknown" candidates when that dataset is the target.
KNOWN_CLASS_MAP = {
    'RealWorld': {
        'lying': 'lying', 'sitting': 'sitting', 'standing': 'standing',
        'walking': 'walking', 'running': 'running',
    },
    'Pamap2': {
        'lying': 'lying', 'sitting': 'sitting', 'standing': 'standing',
        'walking': 'walking', 'running': 'running',
    },
    'MHEALTH': {
        'Lying down': 'lying', 'Sitting and relaxing': 'sitting',
        'Standing still': 'standing', 'Walking': 'walking', 'Running': 'running',
    },
}

# Human-readable names for MHEALTH numeric-ish labels (already readable in our pkl)
# and for display purposes
DISPLAY_NAMES = {
    'RealWorld': {
        'climbingup': 'Climbing Up', 'climbingdown': 'Climbing Down', 'jumping': 'Jumping',
        'lying': 'Lying', 'sitting': 'Sitting', 'standing': 'Standing',
        'walking': 'Walking', 'running': 'Running',
    },
    'Pamap2': {
        'Nordic walking': 'Nordic Walking', 'ascending stairs': 'Ascending Stairs',
        'cycling': 'Cycling', 'descending stairs': 'Descending Stairs',
        'ironing': 'Ironing', 'vacuum cleaning': 'Vacuum Cleaning',
        'rope jumping': 'Rope Jumping', 'other (transient activities)': 'Other/Transient',
        'lying': 'Lying', 'sitting': 'Sitting', 'standing': 'Standing',
        'walking': 'Walking', 'running': 'Running',
    },
    'MHEALTH': {
        'null': 'Null', 'Standing still': 'Standing Still',
        'Sitting and relaxing': 'Sitting/Relaxing', 'Lying down': 'Lying Down',
        'Walking': 'Walking', 'Climbing stairs': 'Climbing Stairs',
        'Waist bends forward': 'Waist Bends', 'Frontal elevation of arms': 'Arm Elevation',
        'Knees bending (crouching)': 'Knees Bending', 'Cycling': 'Cycling',
        'Jogging': 'Jogging', 'Running': 'Running', 'Jump front & back': 'Jump F&B',
    },
}

WINDOW_SIZE = 150  # same as sequence_len in training config

# ============================================================
# DTW and EDR implementations
# ============================================================

def dtw_distance(s, t):
    """
    Dynamic Time Warping distance between two 1D or multivariate time series.
    s: (L1,) or (L1, C)
    t: (L2,) or (L2, C)
    Returns: normalized DTW distance (divided by path length).
    """
    s = np.atleast_2d(s) if s.ndim == 1 else s
    t = np.atleast_2d(t) if t.ndim == 1 else t
    # s: (L1, C), t: (L2, C)
    n, m = len(s), len(t)
    # Cost matrix: Euclidean distance at each pair
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(s[i-1] - t[j-1])
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    # Normalize by path length (n + m is an upper bound)
    return cost[n, m] / (n + m)


def edr_distance(s, t, epsilon=None):
    """
    Edit Distance on Real Sequences.
    Two points match if their Euclidean distance < epsilon.
    Returns: normalized EDR (divided by max length).
    """
    s = np.atleast_2d(s) if s.ndim == 1 else s
    t = np.atleast_2d(t) if t.ndim == 1 else t
    n, m = len(s), len(t)
    if epsilon is None:
        # Default epsilon: median pairwise distance in the combined set
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
            dp[i, j] = min(
                dp[i-1, j-1] + match_cost,
                dp[i-1, j] + 1,
                dp[i, j-1] + 1,
            )
    return dp[n, m] / max(n, m)


def xhar_distance(s, t, omega=1.0, epsilon=None):
    """
    XHAR distance = DTW_norm + omega * EDR_norm
    (Eq. 1 from Zhou et al., SECON 2020)
    """
    d_dtw = dtw_distance(s, t)
    d_edr = edr_distance(s, t, epsilon=epsilon)
    return d_dtw + omega * d_edr, d_dtw, d_edr


# ============================================================
# Data loading and prototype computation
# ============================================================

def load_dataset(path):
    """Load a processed pkl and return {label: list of (data_array, labels)} across all users."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['user_split']


def compute_class_prototypes(user_split, class_list, window_size=WINDOW_SIZE):
    """
    For each class in class_list, collect all windows of that class
    across all users, then average them into one prototype window.

    Returns: {class_label: prototype_array of shape (window_size, n_channels)}
    """
    # Collect all timestep-level data per class
    class_data = defaultdict(list)
    for user, segments in user_split.items():
        for arr, lbls in segments:
            for cls in class_list:
                mask = lbls == cls
                if mask.any():
                    class_data[cls].append(arr[mask])

    prototypes = {}
    for cls in class_list:
        if cls not in class_data or len(class_data[cls]) == 0:
            print(f"  WARNING: no data found for class '{cls}', skipping")
            continue
        all_samples = np.concatenate(class_data[cls], axis=0)  # (N_total, C)
        n_channels = all_samples.shape[1]

        # Cut into non-overlapping windows
        n_windows = len(all_samples) // window_size
        if n_windows == 0:
            # Not enough data for a full window, pad with last value
            padded = np.zeros((window_size, n_channels))
            padded[:len(all_samples)] = all_samples
            padded[len(all_samples):] = all_samples[-1]
            prototypes[cls] = padded
            continue

        windowed = all_samples[:n_windows * window_size].reshape(n_windows, window_size, n_channels)

        # Average across all windows to get prototype
        prototypes[cls] = windowed.mean(axis=0)  # (window_size, C)

    return prototypes


# ============================================================
# Main computation
# ============================================================

def compute_pair_distances(source_name, target_name, omega=1.0):
    """
    For a directed pair (source -> target):
    - Compute prototypes for 5 known classes from SOURCE data
    - Compute prototypes for unknown classes from TARGET data
    - Compute XHAR distance between each (known, unknown) pair
    """
    print(f"\n{'='*70}")
    print(f"  {source_name} -> {target_name}")
    print(f"{'='*70}")

    # Load data
    src_data = load_dataset(PKL_PATHS[source_name])
    trg_data = load_dataset(PKL_PATHS[target_name])

    # Known class labels in source dataset's naming
    src_known_map = KNOWN_CLASS_MAP[source_name]
    src_known_labels = list(src_known_map.keys())

    # Known class labels in target dataset's naming
    trg_known_map = KNOWN_CLASS_MAP[target_name]
    trg_known_labels = list(trg_known_map.keys())

    # All target classes
    all_trg_labels = set()
    for user, segments in trg_data.items():
        for arr, lbls in segments:
            all_trg_labels.update(set(lbls))

    # Unknown = target classes that are NOT in the known set
    unknown_labels = sorted(all_trg_labels - set(trg_known_labels))

    # Skip 'null' / '0' class if present (it's not a real activity)
    unknown_labels = [u for u in unknown_labels if u not in ('null', '0')]

    print(f"  Known classes (from {source_name}): {src_known_labels}")
    print(f"  Unknown classes (in {target_name}):  {unknown_labels}")
    print()

    # Compute prototypes
    print("  Computing source known prototypes...")
    src_prototypes = compute_class_prototypes(src_data, src_known_labels)

    print("  Computing target unknown prototypes...")
    trg_prototypes = compute_class_prototypes(trg_data, unknown_labels)

    # Compute distance matrix
    canonical_known = ['lying', 'sitting', 'standing', 'walking', 'running']
    trg_display = DISPLAY_NAMES.get(target_name, {})
    src_display = DISPLAY_NAMES.get(source_name, {})

    # Results storage
    results = {}
    print(f"\n  {'Unknown class':<25s}", end="")
    for k in src_known_labels:
        canonical = src_known_map[k]
        print(f"  {canonical:>10s}", end="")
    print(f"  {'MIN':>10s}  {'Nearest':>10s}")
    print("  " + "-" * (25 + 12 * (len(src_known_labels) + 2)))

    for unk in unknown_labels:
        if unk not in trg_prototypes:
            continue
        unk_proto = trg_prototypes[unk]
        row = {}
        for kn in src_known_labels:
            if kn not in src_prototypes:
                continue
            kn_proto = src_prototypes[kn]
            xhar_d, dtw_d, edr_d = xhar_distance(kn_proto, unk_proto, omega=omega)
            canonical = src_known_map[kn]
            row[canonical] = {'xhar': xhar_d, 'dtw': dtw_d, 'edr': edr_d}

        if not row:
            continue

        min_known = min(row, key=lambda k: row[k]['xhar'])
        min_dist = row[min_known]['xhar']

        unk_display = trg_display.get(unk, unk)
        print(f"  {unk_display:<25s}", end="")
        for kn in src_known_labels:
            canonical = src_known_map[kn]
            if canonical in row:
                print(f"  {row[canonical]['xhar']:10.4f}", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print(f"  {min_dist:10.4f}  {min_known:>10s}")

        results[unk] = {'distances': row, 'min_dist': min_dist, 'nearest_known': min_known}

    # Print ranking
    ranked = sorted(results.items(), key=lambda x: x[1]['min_dist'])
    print(f"\n  Difficulty ranking (hardest unknown first = smallest min distance):")
    print(f"  {'Rank':<6s} {'Unknown':<25s} {'Min XHAR Dist':>15s} {'Nearest Known':>15s}")
    print("  " + "-" * 65)
    for rank, (unk, info) in enumerate(ranked, 1):
        unk_display = trg_display.get(unk, unk)
        print(f"  {rank:<6d} {unk_display:<25s} {info['min_dist']:15.4f} {info['nearest_known']:>15s}")

    return results


def plot_pair_heatmap(source_name, target_name, results):
    """
    Heatmap: rows = unknown classes (sorted by min dist), cols = known classes.
    Annotated with distance values. The cell with the min distance per row is
    outlined.
    """
    trg_display = DISPLAY_NAMES.get(target_name, {})
    src_display = DISPLAY_NAMES.get(source_name, {})

    ranked = sorted(results.items(), key=lambda x: x[1]['min_dist'])
    canonical_known = ['lying', 'sitting', 'standing', 'walking', 'running']

    unk_labels = []
    matrix = []
    for unk, info in ranked:
        unk_labels.append(trg_display.get(unk, unk))
        row = []
        for k in canonical_known:
            row.append(info['distances'].get(k, {}).get('xhar', np.nan))
        matrix.append(row)

    matrix = np.array(matrix)
    n_unk, n_kn = matrix.shape

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.6 * n_unk + 1.5)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(n_kn))
    ax.set_xticklabels([k.capitalize() for k in canonical_known], fontsize=11)
    ax.set_yticks(range(n_unk))
    ax.set_yticklabels(unk_labels, fontsize=11)

    ax.set_xlabel('Known Class (Source)', fontsize=12, labelpad=8)
    ax.set_ylabel('Unknown Class (Target)', fontsize=12, labelpad=8)
    ax.set_title(f'XHAR Distance: {source_name} \u2192 {target_name}',
                 fontsize=14, fontweight='bold', pad=12)

    # Annotate cells
    for i in range(n_unk):
        min_j = int(np.nanargmin(matrix[i]))
        for j in range(n_kn):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = 'white' if val > (matrix.max() + matrix.min()) / 2 else 'black'
            weight = 'bold' if j == min_j else 'normal'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)
        # Highlight min cell with a box
        ax.add_patch(plt.Rectangle((min_j - 0.48, i - 0.48), 0.96, 0.96,
                                    fill=False, edgecolor='black', linewidth=2.5))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('XHAR Distance (lower = more similar)', fontsize=10)

    plt.tight_layout()
    fname = f'xhar_{source_name}_to_{target_name}.png'
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.join(PLOT_DIR, fname)}")


def plot_pair_barplot(source_name, target_name, results):
    """
    Horizontal bar chart: unknown classes sorted by min distance.
    Each bar is colored by the nearest known class. Shows the difficulty ranking
    at a glance.
    """
    trg_display = DISPLAY_NAMES.get(target_name, {})
    ranked = sorted(results.items(), key=lambda x: x[1]['min_dist'])

    known_colors = {
        'lying': '#e74c3c', 'sitting': '#3498db', 'standing': '#2ecc71',
        'walking': '#f39c12', 'running': '#9b59b6',
    }

    unk_labels = [trg_display.get(u, u) for u, _ in ranked]
    min_dists = [info['min_dist'] for _, info in ranked]
    nearest = [info['nearest_known'] for _, info in ranked]
    colors = [known_colors.get(n, '#95a5a6') for n in nearest]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(unk_labels) + 1.5)))

    y_pos = range(len(unk_labels))
    bars = ax.barh(y_pos, min_dists, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(unk_labels, fontsize=11)
    ax.invert_yaxis()  # hardest (smallest dist) at top
    ax.set_xlabel('Min XHAR Distance to Nearest Known Class', fontsize=12)
    ax.set_title(f'Unknown Class Difficulty: {source_name} \u2192 {target_name}',
                 fontsize=14, fontweight='bold', pad=12)

    # Annotate bars with distance and nearest class
    for i, (bar, dist, near) in enumerate(zip(bars, min_dists, nearest)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{dist:.1f} ({near})', va='center', fontsize=9, color='#333')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=k.capitalize())
                       for k, c in known_colors.items()]
    ax.legend(handles=legend_elements, title='Nearest Known', loc='lower right',
              fontsize=9, title_fontsize=10)

    # Add difficulty annotations
    ax.axvline(x=min(min_dists), color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=max(min_dists), color='gray', linestyle='--', alpha=0.3)
    ax.text(min(min_dists), len(unk_labels) + 0.1, 'Hardest', ha='center',
            fontsize=8, color='gray', style='italic')
    ax.text(max(min_dists), len(unk_labels) + 0.1, 'Easiest', ha='center',
            fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    fname = f'xhar_ranking_{source_name}_to_{target_name}.png'
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.join(PLOT_DIR, fname)}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    datasets = ['RealWorld', 'Pamap2', 'MHEALTH']
    omega = 1.0  # XHAR trade-off parameter (Eq. 1)

    all_results = {}
    for src in datasets:
        for trg in datasets:
            if src == trg:
                continue
            pair_key = f"{src} -> {trg}"
            all_results[pair_key] = compute_pair_distances(src, trg, omega=omega)

    # Generate plots
    print(f"\n\n{'='*70}")
    print("  Generating plots...")
    print(f"{'='*70}")
    for pair_key, results in all_results.items():
        src_name, trg_name = pair_key.split(' -> ')
        plot_pair_heatmap(src_name, trg_name, results)
        plot_pair_barplot(src_name, trg_name, results)

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY: Unknown classes ranked by difficulty per pair")
    print(f"{'='*70}")
    for pair_key, results in all_results.items():
        ranked = sorted(results.items(), key=lambda x: x[1]['min_dist'])
        src_name, trg_name = pair_key.split(' -> ')
        trg_display = DISPLAY_NAMES.get(trg_name, {})
        print(f"\n  {pair_key}:")
        for rank, (unk, info) in enumerate(ranked, 1):
            unk_display = trg_display.get(unk, unk)
            print(f"    {rank}. {unk_display:<25s} (dist={info['min_dist']:.4f}, nearest={info['nearest_known']})")


if __name__ == '__main__':
    main()
