"""Split an already-processed RealWorld pkl into male/female pkls.

Mirrors split_dataset_by_subject_groups_and_save from run_datasets.py but
inlined to avoid pulling the raw-data processing imports.
"""
import copy
import os
import pickle
import re

DATASET_DIR = "/mnt/data/home/tp2474/dataset"
SRC_PKL = os.path.join(DATASET_DIR, "RealWorld_processed.pkl")

MALE_IDS = {"proband2", "proband3", "proband4", "proband5",
            "proband7", "proband9", "proband10", "proband14"}
FEMALE_IDS = {"proband1", "proband6", "proband8", "proband11",
              "proband12", "proband13", "proband15"}


def _normalize_ids(ids):
    as_str = {str(x) for x in ids}
    as_int = set()
    for x in ids:
        try:
            as_int.add(int(x))
        except (TypeError, ValueError):
            pass
    return as_str, as_int


def split_user_datasets(user_datasets, group1_ids, group2_ids):
    g1_str, g1_int = _normalize_ids(group1_ids)
    g2_str, g2_int = _normalize_ids(group2_ids)
    g1, g2 = {}, {}
    for key, data in user_datasets.items():
        key_str = str(key)
        if key_str in g1_str:
            g1[key] = data
            continue
        if key_str in g2_str:
            g2[key] = data
            continue
        digits = re.sub(r"\D+", "", key_str)
        sid = int(digits) if digits else None
        if sid in g1_int:
            g1[key] = data
        elif sid in g2_int:
            g2[key] = data
        else:
            print(f"Warning: proband '{key}' not in either group; skipping.")
    return g1, g2


def save_group(content_template, group_name, group_users, out_dir):
    base = content_template.get("save_file_name", "RealWorld_processed.pkl")
    stem, ext = os.path.splitext(base)
    if stem.endswith("_processed"):
        stem = stem[:-len("_processed")]
    out_name = f"{stem}_{group_name}_processed{ext or '.pkl'}"

    content = copy.copy(content_template)
    content["name"] = f"{content_template.get('name', 'RealWorld HAR')}_{group_name}"
    content["save_file_name"] = out_name
    content["user_split"] = group_users

    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(content, f)
    print(f"Saved {group_name} ({len(group_users)} probands) -> {out_path}")
    return out_path


def main():
    with open(SRC_PKL, "rb") as f:
        content = pickle.load(f)

    user_datasets = content["user_split"]
    print(f"Loaded {len(user_datasets)} probands from {SRC_PKL}")
    print(f"Proband IDs: {sorted(user_datasets.keys())}")

    male_users, female_users = split_user_datasets(user_datasets, MALE_IDS, FEMALE_IDS)

    save_group(content, "male", male_users, DATASET_DIR)
    save_group(content, "female", female_users, DATASET_DIR)


if __name__ == "__main__":
    main()
