from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


B, C1, C2 = 1, 3, 4
ROUTE_SET = {B, C1, C2}


def parse_args():
    parser = argparse.ArgumentParser(description="Select the specialist vote bonus from leakage-free OOF predictions.")
    parser.add_argument("--oof-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=267)
    parser.add_argument("--betas", default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    return parser.parse_args()


def read_rows(path: Path):
    required = {
        "sample_index", "fold", "true_label", "base_pred",
        "tri_prob_B", "tri_prob_C1", "tri_prob_C2",
        "bc1_prob_B", "bc1_prob_C1", "c1c2_prob_C1", "c1c2_prob_C2",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing OOF columns: %s" % sorted(missing))
        rows = list(reader)

    sample_ids = [int(row["sample_index"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_index values in OOF predictions")
    return rows


def specialist_prediction(row, beta):
    base_pred = int(row["base_pred"])
    if base_pred not in ROUTE_SET:
        return base_pred

    scores = {
        B: float(row["tri_prob_B"]),
        C1: float(row["tri_prob_C1"]),
        C2: float(row["tri_prob_C2"]),
    }
    if float(row["bc1_prob_B"]) >= float(row["bc1_prob_C1"]):
        scores[B] += beta
    else:
        scores[C1] += beta
    if float(row["c1c2_prob_C1"]) >= float(row["c1c2_prob_C2"]):
        scores[C1] += beta
    else:
        scores[C2] += beta
    return max((B, C1, C2), key=lambda label: (scores[label], -label))


def metrics(y_true, y_pred):
    labels = range(5)
    n = len(y_true)
    accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / float(n)
    f1_values = []
    supports = []
    recalls = []
    for label in labels:
        tp = sum(int(a == label and b == label) for a, b in zip(y_true, y_pred))
        fp = sum(int(a != label and b == label) for a, b in zip(y_true, y_pred))
        fn = sum(int(a == label and b != label) for a, b in zip(y_true, y_pred))
        support = sum(int(a == label) for a in y_true)
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        supports.append(support)
        recalls.append(recall)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": sum(f1 * support for f1, support in zip(f1_values, supports)) / float(n),
        "c1_recall": recalls[C1],
        "c2_recall": recalls[C2],
    }


def main():
    args = parse_args()
    rows = read_rows(Path(args.oof_csv))
    if len(rows) != args.expected_samples:
        raise ValueError("Expected %d OOF samples, found %d" % (args.expected_samples, len(rows)))

    folds = sorted({int(row["fold"]) for row in rows})
    if len(folds) != 5:
        raise ValueError("Expected five folds, found %s" % folds)

    y_true = [int(row["true_label"]) for row in rows]
    betas = [float(value.strip()) for value in args.betas.split(",") if value.strip()]
    sweep = []
    for beta in betas:
        y_pred = [specialist_prediction(row, beta) for row in rows]
        result = metrics(y_true, y_pred)
        result["beta"] = beta
        sweep.append(result)

    selected = max(sweep, key=lambda row: (row["macro_f1"], row["accuracy"], -row["beta"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["beta", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "c2_recall"]
    with (output_dir / "validation_bonus_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep)

    payload = {
        "selection_data": "five-fold out-of-fold CHIME development predictions",
        "n_samples": len(rows),
        "folds": folds,
        "selection_rule": "maximize macro_f1; tie-break by accuracy; then choose smaller beta",
        "selected": selected,
    }
    (output_dir / "validation_bonus_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
