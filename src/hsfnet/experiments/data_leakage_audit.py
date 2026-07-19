from __future__ import print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import sys

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from data import CLASSES, paper_half_split_chime  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Audit CHIME train/test split for index, exact-image, and near-duplicate leakage.")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="reports/data_leakage_audit")
    parser.add_argument("--downsample", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--near-duplicate-cosine-threshold", type=float, default=0.999)
    return parser.parse_args()


def image_hashes(x):
    hashes = []
    for image in x:
        hashes.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
    return hashes


def downsample_mean(x, target):
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[..., 0]
    h, w = x.shape[1], x.shape[2]
    if h % target == 0 and w % target == 0:
        sh = h // target
        sw = w // target
        return x.reshape(x.shape[0], target, sh, target, sw).mean(axis=(2, 4))
    row_idx = np.linspace(0, h - 1, target).astype(np.int64)
    col_idx = np.linspace(0, w - 1, target).astype(np.int64)
    return x[:, row_idx][:, :, col_idx]


def nearest_cosine(train_x, test_x, train_y, test_y, target, top_k):
    train = downsample_mean(train_x, target).reshape(train_x.shape[0], -1).astype(np.float32)
    test = downsample_mean(test_x, target).reshape(test_x.shape[0], -1).astype(np.float32)
    train = train - train.mean(axis=1, keepdims=True)
    test = test - test.mean(axis=1, keepdims=True)
    train_norm = np.linalg.norm(train, axis=1, keepdims=True)
    test_norm = np.linalg.norm(test, axis=1, keepdims=True)
    train = train / np.maximum(train_norm, 1e-8)
    test = test / np.maximum(test_norm, 1e-8)

    sims = test.dot(train.T)
    best_train = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(sims.shape[0]), best_train]
    order = np.argsort(-best_sim)[:top_k]
    rows = []
    for test_local in order:
        train_local = int(best_train[test_local])
        rows.append(
            {
                "test_local_index": int(test_local),
                "test_class": CLASSES[int(test_y[test_local])],
                "nearest_train_local_index": train_local,
                "nearest_train_class": CLASSES[int(train_y[train_local])],
                "cosine_similarity_downsampled": float(best_sim[test_local]),
            }
        )
    return rows, best_sim


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, test_x, test_y, train_idx, test_idx = paper_half_split_chime(args.chime_data, args.chime_labels)

    index_overlap = sorted(set(train_idx.tolist()).intersection(set(test_idx.tolist())))
    train_hashes = image_hashes(train_x)
    test_hashes = image_hashes(test_x)
    train_hash_to_local = {h: i for i, h in enumerate(train_hashes)}
    exact_duplicates = []
    for test_local, h in enumerate(test_hashes):
        if h in train_hash_to_local:
            train_local = train_hash_to_local[h]
            exact_duplicates.append(
                {
                    "test_local_index": int(test_local),
                    "test_global_index": int(test_idx[test_local]),
                    "test_class": CLASSES[int(test_y[test_local])],
                    "train_local_index": int(train_local),
                    "train_global_index": int(train_idx[train_local]),
                    "train_class": CLASSES[int(train_y[train_local])],
                    "sha256": h,
                }
            )

    nearest_rows, best_sim = nearest_cosine(train_x, test_x, train_y, test_y, args.downsample, args.top_k)
    near_duplicates = [
        row for row in nearest_rows
        if row["cosine_similarity_downsampled"] >= args.near_duplicate_cosine_threshold
    ]

    class_counts = {}
    for idx, name in enumerate(CLASSES):
        class_counts[name] = {
            "train": int(np.sum(train_y == idx)),
            "test": int(np.sum(test_y == idx)),
        }

    summary = {
        "chime_data": args.chime_data,
        "chime_labels": args.chime_labels,
        "n_train": int(train_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "class_counts": class_counts,
        "index_overlap_count": int(len(index_overlap)),
        "index_overlap": [int(v) for v in index_overlap],
        "exact_duplicate_count": int(len(exact_duplicates)),
        "near_duplicate_threshold": float(args.near_duplicate_cosine_threshold),
        "near_duplicate_count_in_top_k": int(len(near_duplicates)),
        "max_downsampled_cosine_similarity": float(np.max(best_sim)),
        "mean_downsampled_nearest_cosine_similarity": float(np.mean(best_sim)),
        "downsample": int(args.downsample),
        "interpretation": (
            "No direct leakage detected."
            if len(index_overlap) == 0 and len(exact_duplicates) == 0
            else "Potential leakage found; inspect duplicates/overlaps."
        ),
    }

    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with (out_dir / "exact_duplicates.json").open("w", encoding="utf-8") as f:
        json.dump(exact_duplicates, f, indent=2, sort_keys=True)

    with (out_dir / "nearest_train_test_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["test_local_index", "test_class", "nearest_train_local_index", "nearest_train_class", "cosine_similarity_downsampled"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in nearest_rows:
            writer.writerow(row)

    with (out_dir / "audit_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Data Leakage Audit\n\n")
        f.write("- CHIME train samples: `%d`\n" % summary["n_train"])
        f.write("- CHIME test samples: `%d`\n" % summary["n_test"])
        f.write("- Train/test index overlap: `%d`\n" % summary["index_overlap_count"])
        f.write("- Exact duplicate waterfall hashes: `%d`\n" % summary["exact_duplicate_count"])
        f.write("- Max downsampled nearest-neighbor cosine similarity: `%.6f`\n" % summary["max_downsampled_cosine_similarity"])
        f.write("- Interpretation: **%s**\n\n" % summary["interpretation"])
        f.write("| Class | Train | Test |\n|---|---:|---:|\n")
        for name in CLASSES:
            f.write("| %s | %d | %d |\n" % (name, class_counts[name]["train"], class_counts[name]["test"]))

    print("Wrote data leakage audit to %s" % out_dir)


if __name__ == "__main__":
    main()

