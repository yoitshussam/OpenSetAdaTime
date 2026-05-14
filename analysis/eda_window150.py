import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Ensure project imports work when running from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_data import load_labelled_and_unlabelled

WINDOW = 150
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "eda_plots"))

DATASETS = {
    "RealWorld_full": "/home/tp2474/dataset/RealWorld_processed.pkl",
    "PAMAP2": "/home/tp2474/dataset/Pamap2_processed.pkl",
    "MHEALTH": "/home/tp2474/dataset/MHEALTH_processed.pkl",
}


def to_labels(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)
    return y.ravel()


def summarize_split(split, num_classes):
    _, y = split
    y = to_labels(y)
    total = len(y)
    counts = np.bincount(y, minlength=num_classes)
    return total, counts


def normalize_label_list(raw):
    label_list = raw.get("label_list")
    full_names = raw.get("label_list_full_name")
    # If label_list is numeric strings and full names exist, use full names.
    if full_names and all(isinstance(x, str) and x.isdigit() for x in (label_list or [])):
        label_list = full_names
    return label_list


def filter_null_class(labels):
    if labels is None:
        return labels
    return [l for l in labels if str(l).lower() != "null"]


def is_other_label(label):
    s = str(label).strip().lower()
    return ("other" in s) or s == "transient"


# Viridis-based palette for the three splits (train/val/test).
SPLIT_COLORS = {
    "train": plt.cm.viridis(0.20),
    "val":   plt.cm.viridis(0.55),
    "test":  plt.cm.viridis(0.85),
}


def plot_counts(title, labels, counts, out_path):
    plt.figure(figsize=(12, 5))
    x = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(labels), 1)))
    plt.bar(x, counts, color=colors, edgecolor="black", linewidth=0.5)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Window count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_split_stack(title, labels, split_counts, out_path):
    plt.figure(figsize=(12, 5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=int)

    for split_name, counts in split_counts.items():
        plt.bar(x, counts, bottom=bottom, label=split_name,
                color=SPLIT_COLORS.get(split_name), edgecolor="black",
                linewidth=0.4)
        bottom += counts

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Window count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_combined_overall(per_dataset, out_path, window):
    """One figure, 3 subplots stacked vertically (one per dataset). Each
    subplot shows the overall class distribution (sum of train+val+test)."""
    n = len(per_dataset)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.0 * n))
    if n == 1:
        axes = [axes]

    display_name = {"RealWorld_full": "RealWorld"}
    for ax, (name, info) in zip(axes, per_dataset.items()):
        labels = info["labels"]
        sc = info["split_counts"]
        totals = sc["train"] + sc["val"] + sc["test"]
        x = np.arange(len(labels))
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(labels), 1)))
        ax.bar(x, totals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{display_name.get(name, name)} class distribution "
                     f"(window={window})")
        ax.set_ylabel("Window count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_combined_split_stack(per_dataset, out_path, window):
    """Render all datasets' stacked train/val/test bars in one figure,
    one row per dataset. Each subplot keeps its own x-axis (class labels
    differ across datasets)."""
    n = len(per_dataset)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.2 * n))
    if n == 1:
        axes = [axes]

    split_order = ["train", "val", "test"]
    for ax, (name, info) in zip(axes, per_dataset.items()):
        labels = info["labels"]
        split_counts = info["split_counts"]
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels), dtype=int)
        for split_name in split_order:
            counts = split_counts[split_name]
            ax.bar(x, counts, bottom=bottom, label=split_name,
                   color=SPLIT_COLORS.get(split_name), edgecolor="black",
                   linewidth=0.4)
            bottom += counts
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{name} — train/val/test (window={window})")
        ax.set_ylabel("Window count")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_totals = {}
    per_dataset_splits = {}

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"Missing dataset: {path}")
            continue

        with open(path, "rb") as f:
            raw = pickle.load(f)

        label_list = normalize_label_list(raw)
        if label_list is None:
            print(f"No label_list for {name}; skipping")
            continue

        # Drop the null class (MHEALTH) from EDA plots
        label_list = filter_null_class(label_list)

        activity_mapping = {lbl: lbl for lbl in label_list}

        prepared, input_shape, output_shape = load_labelled_and_unlabelled(
            labelled_dataset_path=path,
            unlabelled_dataset_path=None,
            window_size=WINDOW,
            activity_mapping=activity_mapping,
            verbose=0,
            label_list=label_list,
        )

        splits = prepared["labelled"]
        split_counts = {}
        totals = {}

        for split_name in ["train", "val", "test"]:
            total, counts = summarize_split(splits[split_name], output_shape)
            totals[split_name] = total
            split_counts[split_name] = counts

        total_counts = split_counts["train"] + split_counts["val"] + split_counts["test"]

        # Drop classes with zero windows OR the catch-all PAMAP2 'Other' bucket.
        keep_mask = np.array([
            total_counts[i] > 0 and not is_other_label(label_list[i])
            for i in range(len(label_list))
        ])
        nonzero_idx = np.where(keep_mask)[0]
        dropped = [label_list[i] for i in range(len(label_list)) if not keep_mask[i]]
        if dropped:
            print(f"  dropping zero-window classes from plots: {dropped}")
            label_list  = [label_list[i] for i in nonzero_idx]
            split_counts = {s: c[nonzero_idx] for s, c in split_counts.items()}
            total_counts = total_counts[nonzero_idx]

        # Plot overall distribution
        plot_counts(
            title=f"{name} class distribution (window={WINDOW})",
            labels=label_list,
            counts=total_counts,
            out_path=os.path.join(OUTPUT_DIR, f"{name}_overall.png"),
        )

        # Plot stacked split distribution
        plot_split_stack(
            title=f"{name} train/val/test distribution (window={WINDOW})",
            labels=label_list,
            split_counts=split_counts,
            out_path=os.path.join(OUTPUT_DIR, f"{name}_splits.png"),
        )

        dataset_totals[name] = int(total_counts.sum())
        per_dataset_splits[name] = {"labels": label_list, "split_counts": split_counts}
        print(f"{name}: input_shape={input_shape}, classes={output_shape}, totals={totals}")

    # Combined vertically-stacked split plot (one subplot per dataset).
    if per_dataset_splits:
        plot_combined_split_stack(
            per_dataset=per_dataset_splits,
            out_path=os.path.join(OUTPUT_DIR, "all_splits_combined.png"),
            window=WINDOW,
        )
        # Combined overall class-distribution (no train/val/test split).
        plot_combined_overall(
            per_dataset=per_dataset_splits,
            out_path=os.path.join(OUTPUT_DIR, "all_overall_combined.png"),
            window=WINDOW,
        )

    # Plot dataset size comparison
    if dataset_totals:
        display_name = {"RealWorld_full": "RealWorld"}
        names = [display_name.get(n, n) for n in dataset_totals.keys()]
        sizes = list(dataset_totals.values())
        order = np.argsort(sizes)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
        bar_colors = [colors[r] for r in ranks]
        plt.figure(figsize=(8, 4))
        x = np.arange(len(names))
        bars = plt.bar(x, sizes, color=bar_colors, edgecolor="black",
                       linewidth=0.6)
        for b, s in zip(bars, sizes):
            plt.text(b.get_x() + b.get_width() / 2,
                     b.get_height() + max(sizes) * 0.015,
                     f"{s:,}", ha="center", va="bottom", fontsize=9)
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylim(0, max(sizes) * 1.15)
        plt.title(f"Dataset sizes (window={WINDOW})")
        plt.ylabel("Total windows")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "dataset_sizes.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
