from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser(description="Summarize final binary-specialist fusion multiseed results.")
    parser.add_argument("--base-dir", default="artifacts/runs/multiseed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-prefix", default="artifacts/runs/multiseed/final_binary_fusion_multiseed_summary")
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_path(base_dir, seed):
    return Path(base_dir) / ("final_binary_specialist_fusion_seed_%s" % seed) / "eval_predicted_subset" / "metrics_summary.json"


def fmt_pct(values):
    values = np.asarray(values, dtype=np.float64) * 100.0
    return "%.2f ± %.2f" % (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def main():
    args = parse_args()
    records = []
    missing = []

    for seed in args.seeds:
        path = metric_path(args.base_dir, seed)
        if not path.exists():
            missing.append(str(path))
            continue
        records.append((seed, read_json(path), str(path)))

    if not records:
        raise SystemExit("No final binary fusion seed results found.")

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment": "final_binary_specialist_fusion",
        "label": "Final hierarchical ResNet50 + binary specialists",
        "n_seeds": len(records),
        "seeds": " ".join(str(seed) for seed, _, _ in records),
    }
    for metric in METRICS:
        row[metric] = fmt_pct([payload[metric] for _, payload, _ in records])
    for class_name in ["A", "B", "C", "C1", "C2"]:
        row["recall_" + class_name] = fmt_pct([payload["per_class_recall"][class_name] for _, payload, _ in records])

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    detail_path = output_prefix.with_name(output_prefix.name + "_per_seed.csv")

    fieldnames = ["experiment", "label", "n_seeds", "seeds"] + METRICS + ["recall_A", "recall_B", "recall_C", "recall_C1", "recall_C2"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames_detail = ["seed", "source"] + METRICS + ["recall_A", "recall_B", "recall_C", "recall_C1", "recall_C2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames_detail)
        writer.writeheader()
        for seed, payload, source in records:
            detail = {"seed": seed, "source": source}
            for metric in METRICS:
                detail[metric] = payload[metric]
            for class_name in ["A", "B", "C", "C1", "C2"]:
                detail["recall_" + class_name] = payload["per_class_recall"][class_name]
            writer.writerow(detail)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Experiment | n | Accuracy | Macro-F1 | Weighted-F1 | C1 Recall |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(
            "| %s | %s | %s | %s | %s | %s |\n"
            % (row["label"], row["n_seeds"], row["accuracy"], row["macro_f1"], row["weighted_f1"], row["recall_C1"])
        )
        if missing:
            f.write("\nMissing result files:\n\n")
            for path in missing:
                f.write("- `%s`\n" % path)

    print("Wrote %s" % csv_path)
    print("Wrote %s" % md_path)
    print("Wrote %s" % detail_path)
    if missing:
        print("Missing %d seed results. Completed seeds are still summarized." % len(missing))


if __name__ == "__main__":
    main()

