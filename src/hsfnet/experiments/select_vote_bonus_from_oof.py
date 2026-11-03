"""Select the C1 vote bonus from development out-of-fold predictions only."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_FIELDS = (
    "sample_index",
    "fold",
    "true_label",
    "base_pred",
    "tri_prob_B",
    "tri_prob_C1",
    "tri_prob_C2",
    "bc1_prob_B",
    "bc1_prob_C1",
    "c1c2_prob_C1",
    "c1c2_prob_C2",
)
CLASSES = ("A", "B", "C", "C1", "C2")
ROUTE_SET = {1, 3, 4}


def parse_args():
    parser = argparse.ArgumentParser(description="Select beta using HSFNet-FRB development OOF predictions.")
    parser.add_argument("--oof-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_oof(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError("OOF fields must exactly match the preregistered schema: %s" % (EXPECTED_FIELDS,))
        rows = list(reader)
    sample_indices = [int(row["sample_index"]) for row in rows]
    folds = [int(row["fold"]) for row in rows]
    if len(rows) != 267 or len(set(sample_indices)) != 267:
        raise ValueError("OOF CSV must contain 267 unique development samples exactly once.")
    if set(folds) != set(range(5)):
        raise ValueError("OOF CSV must contain folds 0,1,2,3,4.")
    return rows


def fuse_row(row, beta):
    base_pred = int(row["base_pred"])
    if base_pred not in ROUTE_SET:
        return base_pred

    scores = {
        1: float(row["tri_prob_B"]),
        3: float(row["tri_prob_C1"]),
        4: float(row["tri_prob_C2"]),
    }
    if float(row["bc1_prob_C1"]) >= 0.5:
        scores[3] += beta
    else:
        scores[1] += beta
    if float(row["c1c2_prob_C2"]) >= 0.5:
        scores[4] += beta
    else:
        scores[3] += beta
    return max(scores.items(), key=lambda item: item[1])[0]


def evaluate_beta(rows, beta):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    y_true = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([fuse_row(row, beta) for row in rows], dtype=np.int64)
    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros((len(CLASSES),), dtype=np.float64),
        where=cm.sum(axis=1) != 0,
    )
    return {
        "beta": float(beta),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "c1_recall": float(recalls[3]),
        "c2_recall": float(recalls[4]),
        "n_routed": int(sum(int(row["base_pred"]) in ROUTE_SET for row in rows)),
    }


def main():
    args = parse_args()
    rows = load_oof(args.oof_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_values = [value / 100.0 for value in range(0, 51, 5)]
    sweep = [evaluate_beta(rows, beta) for beta in beta_values]
    selected = sorted(sweep, key=lambda row: (-row["macro_f1"], -row["accuracy"], row["beta"]))[0]

    sweep_path = output_dir / "validation_bonus_sweep.csv"
    fieldnames = ("beta", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "c2_recall", "n_routed")
    with sweep_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep)

    oof_bytes = Path(args.oof_csv).read_bytes()
    selection = {
        "selected_beta": float(selected["beta"]),
        "selection_metrics": selected,
        "selection_rule": [
            "maximize development OOF macro-F1",
            "break exact ties by higher accuracy",
            "break remaining exact ties by smaller beta",
        ],
        "beta_grid": beta_values,
        "n_oof_samples": 267,
        "oof_csv": args.oof_csv,
        "oof_csv_sha256": hashlib.sha256(oof_bytes).hexdigest(),
        "paper_test_metrics_access_before_selection": False,
        "validation_bonus_sweep_csv": str(sweep_path),
    }
    with (output_dir / "validation_bonus_selection.json").open("w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, sort_keys=True)
    print("Selected beta=%.2f from development OOF predictions." % selected["beta"])
    print("Saved: %s" % (output_dir / "validation_bonus_selection.json"))


if __name__ == "__main__":
    main()
