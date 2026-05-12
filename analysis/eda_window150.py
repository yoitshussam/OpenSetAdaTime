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


def plot_counts(title, labels, counts, out_path):
    plt.figure(figsize=(12, 5))
    x = np.arange(len(labels))
    plt.bar(x, counts)
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
        plt.bar(x, counts, bottom=bottom, label=split_name)
        bottom += counts

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Window count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
            ax.bar(x, counts, bottom=bottom, label=split_name)
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

        # Drop classes with zero windows in this dataset (e.g. PAMAP2's
        # optional activities that no subject performed in the released subset).
        nonzero_idx = np.where(total_counts > 0)[0]
        dropped = [label_list[i] for i in range(len(label_list)) if i not in nonzero_idx]
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

    # Plot dataset size comparison
    if dataset_totals:
        names = list(dataset_totals.keys())
        sizes = [dataset_totals[n] for n in names]
        plt.figure(figsize=(8, 4))
        x = np.arange(len(names))
        plt.bar(x, sizes)
        plt.xticks(x, names, rotation=20, ha="right")
        plt.title(f"Dataset sizes (window={WINDOW})")
        plt.ylabel("Total windows")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "dataset_sizes.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
