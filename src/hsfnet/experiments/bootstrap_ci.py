from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


CLASSES = ["A", "B", "C", "C1", "C2"]
METRICS = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "weighted_precision", "weighted_recall"]


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap 95% confidence intervals for FRB classification metrics.")
    parser.add_argument("--eval-dir", default=None, help="One evaluation directory containing y_true.txt and y_pred*.txt.")
    parser.add_argument("--base-dir", default=None, help="Run recursively over evaluation directories below this base directory.")
    parser.add_argument("--output", default=None, help="Output JSON/CSV path prefix. Defaults inside eval-dir/base-dir.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def find_prediction_file(eval_dir):
    eval_dir = Path(eval_dir)
    candidates = [
        "y_pred_binary_specialist_fusion.txt",
        "y_pred_siglip2_tf_specialist_fusion.txt",
        "y_pred_hierarchical.txt",
        "y_pred.txt",
    ]
    for name in candidates:
        path = eval_dir / name
        if path.exists():
            return path
    hierarchical = eval_dir / "y_pred_hierarchical.txt"
    plain = eval_dir / "y_pred.txt"
    if hierarchical.exists():
        return hierarchical
    if plain.exists():
        return plain
    raise FileNotFoundError("No y_pred_hierarchical.txt or y_pred.txt in %s" % eval_dir)


def read_labels(eval_dir):
    eval_dir = Path(eval_dir)
    y_true = np.loadtxt(str(eval_dir / "y_true.txt"), dtype=np.int64)
    y_pred = np.loadtxt(str(find_prediction_file(eval_dir)), dtype=np.int64)
    if y_true.ndim == 0:
        y_true = y_true.reshape(1)
    if y_pred.ndim == 0:
        y_pred = y_pred.reshape(1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Mismatched y_true/y_pred lengths in %s" % eval_dir)
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    for idx, class_name in enumerate(CLASSES):
        denom = np.sum(y_true == idx)
        out["recall_" + class_name] = float(np.sum((y_true == idx) & (y_pred == idx)) / denom) if denom else 0.0
    return out


def bootstrap_ci(y_true, y_pred, n_bootstrap, seed, alpha):
    rng = np.random.RandomState(seed)
    n = y_true.shape[0]
    original = compute_metrics(y_true, y_pred)
    samples = {key: [] for key in original.keys()}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        metrics = compute_metrics(y_true[idx], y_pred[idx])
        for key, value in metrics.items():
            samples[key].append(value)

    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    result = {}
    for key, value in original.items():
        arr = np.asarray(samples[key], dtype=np.float64)
        result[key] = {
            "value": float(value),
            "ci_low": float(np.percentile(arr, lo_q)),
            "ci_high": float(np.percentile(arr, hi_q)),
            "bootstrap_mean": float(np.mean(arr)),
            "bootstrap_std": float(np.std(arr, ddof=1)),
        }
    return result


def discover_eval_dirs(base_dir):
    base_dir = Path(base_dir)
    return sorted(path.parent for path in base_dir.rglob("y_true.txt") if find_prediction_file(path.parent).exists())


def infer_name(eval_dir):
    eval_dir = Path(eval_dir)
    parts = eval_dir.parts
    if len(parts) >= 2 and parts[-1] == "eval_paper_compare":
        return parts[-2]
    return eval_dir.name


def pct_interval(metric_payload):
    return "%.2f [%.2f, %.2f]" % (
        metric_payload["value"] * 100.0,
        metric_payload["ci_low"] * 100.0,
        metric_payload["ci_high"] * 100.0,
    )


def write_single_outputs(output_prefix, eval_dir, result):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump({"eval_dir": str(eval_dir), "metrics": result}, f, indent=2, sort_keys=True)

    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "ci_low", "ci_high", "bootstrap_mean", "bootstrap_std"])
        for key in sorted(result.keys()):
            payload = result[key]
            writer.writerow([key, payload["value"], payload["ci_low"], payload["ci_high"], payload["bootstrap_mean"], payload["bootstrap_std"]])


def write_batch_outputs(output_prefix, rows):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "eval_dir"] + METRICS + ["recall_A", "recall_B", "recall_C", "recall_C1", "recall_C2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with output_prefix.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write("| Evaluation | Accuracy 95% CI | Macro-F1 95% CI | Weighted-F1 95% CI | C1 Recall 95% CI |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| %s | %s | %s | %s | %s |\n"
                % (row["name"], row["accuracy"], row["macro_f1"], row["weighted_f1"], row["recall_C1"])
            )


def main():
    args = parse_args()
    if not args.eval_dir and not args.base_dir:
        raise SystemExit("Provide --eval-dir or --base-dir")

    if args.eval_dir:
        y_true, y_pred = read_labels(args.eval_dir)
        result = bootstrap_ci(y_true, y_pred, args.n_bootstrap, args.seed, args.alpha)
        output_prefix = args.output or str(Path(args.eval_dir) / "bootstrap_ci")
        write_single_outputs(output_prefix, args.eval_dir, result)
        print("Wrote %s.json/.csv" % output_prefix)

    if args.base_dir:
        rows = []
        for eval_dir in discover_eval_dirs(args.base_dir):
            y_true, y_pred = read_labels(eval_dir)
            result = bootstrap_ci(y_true, y_pred, args.n_bootstrap, args.seed, args.alpha)
            row = {"name": infer_name(eval_dir), "eval_dir": str(eval_dir)}
            for key, payload in result.items():
                row[key] = pct_interval(payload)
            rows.append(row)
        output_prefix = args.output or str(Path(args.base_dir) / "bootstrap_ci_summary")
        write_batch_outputs(output_prefix, rows)
        print("Wrote %s.json/.csv/.md" % output_prefix)


if __name__ == "__main__":
    main()
