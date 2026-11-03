from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPERIMENTS = [
    ("baseline_cnn", "Baseline CNN reproduction"),
    ("resnet50_no_mask_noise", "ResNet50 without mask/noise"),
    ("resnet50_c1_repair", "ResNet50 C1 repair / no specialist"),
    ("hierarchical_resnet50_specialist", "Hierarchical ResNet50 + B/C1/C2 specialist"),
]


METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "weighted_precision",
    "weighted_recall",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize multiseed FRB experiment metrics as mean +/- std.")
    parser.add_argument("--base-dir", default="artifacts/runs/multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_path(base_dir, experiment, seed):
    return Path(base_dir) / ("%s_seed_%s" % (experiment, seed)) / "eval_paper_compare" / "metrics_summary.json"


def fmt_pct(values):
    values = np.asarray(values, dtype=np.float64) * 100.0
    return "%.2f ± %.2f" % (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else base_dir / "multiseed_summary"

    rows = []
    missing = []

    for experiment, label in EXPERIMENTS:
        records = []
        for seed in args.seeds:
            path = metric_path(base_dir, experiment, seed)
            if not path.exists():
                missing.append(str(path))
                continue
            payload = read_json(path)
            records.append((seed, payload))

        if not records:
            continue

        row = {
            "experiment": experiment,
            "label": label,
            "n_seeds": len(records),
            "seeds": " ".join(str(seed) for seed, _ in records),
        }
        for metric in METRICS:
            values = [payload[metric] for _, payload in records]
            row[metric] = fmt_pct(values)

        for class_name in ["A", "B", "C", "C1", "C2"]:
            values = [payload["per_class_recall"][class_name] for _, payload in records]
            row["recall_" + class_name] = fmt_pct(values)

        rows.append(row)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    fieldnames = ["experiment", "label", "n_seeds", "seeds"] + METRICS + ["recall_A", "recall_B", "recall_C", "recall_C1", "recall_C2"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Experiment | n | Accuracy | Macro-F1 | Weighted-F1 | C1 Recall |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| %s | %s | %s | %s | %s | %s |\n"
                % (
                    row["label"],
                    row["n_seeds"],
                    row["accuracy"],
                    row["macro_f1"],
                    row["weighted_f1"],
                    row["recall_C1"],
                )
            )

        if missing:
            f.write("\nMissing result files:\n\n")
            for path in missing:
                f.write("- `%s`\n" % path)

    print("Wrote %s" % csv_path)
    print("Wrote %s" % md_path)
    if missing:
        print("Missing %d result files. Completed seeds are still summarized." % len(missing))


if __name__ == "__main__":
    main()
