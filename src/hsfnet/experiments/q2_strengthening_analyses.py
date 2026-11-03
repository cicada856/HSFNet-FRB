from __future__ import print_function

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CLASSES = ["A", "B", "C", "C1", "C2"]


ORIGINAL_PAPER_SINGLE_CM = [
    [43, 24, 7, 6, 1],
    [26, 84, 9, 12, 4],
    [2, 5, 18, 5, 1],
    [0, 6, 2, 1, 1],
    [0, 5, 0, 4, 1],
]

ORIGINAL_PAPER_OVO_CM = [
    [36, 111, 2, 9, 5],
    [13, 195, 22, 29, 11],
    [6, 10, 28, 18, 0],
    [1, 12, 3, 4, 0],
    [0, 7, 5, 6, 2],
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Q2/Q3 reviewer-strengthening analyses from retained HSFNet-FRB runs."
    )
    parser.add_argument("--output-dir", default="reports/q2_strengthening")
    parser.add_argument("--final-eval-dir", default="artifacts/runs/binary_specialist_fusion/eval_predicted_subset")
    parser.add_argument("--bonus-sweep-dir", default="artifacts/runs/q2_strengthening/c1_bonus_sweep")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--chime-msnr", default="data/chime/chime_interp_frbs_msnr.txt")
    parser.add_argument("--chime-file-order", default="data/chime/chime_file_order.txt")
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_ints(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return np.asarray([int(line.strip()) for line in f if line.strip()], dtype=np.int64)


def read_float_lines(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return np.asarray([float(line.strip()) for line in f if line.strip()], dtype=np.float64)


def read_lines(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def format_bonus(value):
    return ("%.2f" % float(value)).replace(".", "p")


def save_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_markdown_table(path, rows, columns):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(label for _, label in columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            values = []
            for key, _ in columns:
                value = row.get(key, "")
                if isinstance(value, float):
                    values.append("%.4f" % value)
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def metrics_from_confusion_matrix(cm):
    cm = np.asarray(cm, dtype=np.int64)
    n = int(cm.sum())
    correct = int(np.trace(cm))
    per_class = {}
    f1s = []
    weighted_f1 = 0.0
    recalls = []
    for idx, name in enumerate(CLASSES):
        tp = float(cm[idx, idx])
        support = float(cm[idx].sum())
        predicted = float(cm[:, idx].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
        f1s.append(f1)
        recalls.append(recall)
        weighted_f1 += f1 * support
    return {
        "n": n,
        "accuracy": float(correct) / float(n) if n else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_f1": weighted_f1 / float(n) if n else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
        "c1_recall": per_class["C1"]["recall"],
        "c2_recall": per_class["C2"]["recall"],
    }


def confusion_from_preds(y_true, y_pred):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for y, p in zip(y_true, y_pred):
        cm[int(y), int(p)] += 1
    return cm


def metrics_from_preds(y_true, y_pred):
    return metrics_from_confusion_matrix(confusion_from_preds(y_true, y_pred))


def metric_value(name, y_true, y_pred):
    metrics = metrics_from_preds(y_true, y_pred)
    return metrics[name]


def exact_mcnemar(y_true, pred_a, pred_b):
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum((~a_correct) & b_correct))
    b10 = int(np.sum(a_correct & (~b_correct)))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = 2.0 * sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return b01, b10, min(1.0, p)


def bootstrap_delta_ci(y_true, pred_ref, pred_final, metric, n_bootstrap, seed):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    observed = metric_value(metric, y_true, pred_final) - metric_value(metric, y_true, pred_ref)
    deltas = np.empty((n_bootstrap,), dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = metric_value(metric, y_true[idx], pred_final[idx]) - metric_value(metric, y_true[idx], pred_ref[idx])
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_mean": float(np.mean(deltas)),
        "bootstrap_std": float(np.std(deltas, ddof=1)),
    }


def find_prediction_file(eval_dir, candidates):
    eval_dir = Path(eval_dir)
    for name in candidates:
        path = eval_dir / name
        if path.exists():
            return path
    return None


def load_eval_predictions(eval_dir, candidates):
    eval_dir = Path(eval_dir)
    y_path = eval_dir / "y_true.txt"
    pred_path = find_prediction_file(eval_dir, candidates)
    if not y_path.exists() or pred_path is None:
        return None, None, None
    return read_ints(y_path), read_ints(pred_path), pred_path.name


def load_eval_metrics(eval_dir):
    eval_dir = Path(eval_dir)
    metrics_path = eval_dir / "metrics_summary.json"
    if metrics_path.exists():
        payload = read_json(metrics_path)
        per_class = payload.get("per_class_recall", {})
        return {
            "n": payload.get("n_samples", ""),
            "accuracy": float(payload.get("accuracy", 0.0)),
            "macro_f1": float(payload.get("macro_f1", 0.0)),
            "weighted_f1": float(payload.get("weighted_f1", 0.0)),
            "c1_recall": float(per_class.get("C1", 0.0)),
            "c2_recall": float(per_class.get("C2", 0.0)),
        }
    report_path = eval_dir / "classification_report.json"
    if report_path.exists():
        report = read_json(report_path)
        return {
            "n": int(report.get("macro avg", {}).get("support", 0)),
            "accuracy": float(report.get("accuracy", 0.0)),
            "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
            "c1_recall": float(report.get("C1", {}).get("recall", 0.0)),
            "c2_recall": float(report.get("C2", {}).get("recall", 0.0)),
        }
    return None


def percent_metrics_row(name, training_data, test_data, note, source, metrics):
    return {
        "model": name,
        "training_data": training_data,
        "test_data": test_data,
        "accuracy": 100.0 * metrics["accuracy"],
        "macro_f1": 100.0 * metrics["macro_f1"],
        "weighted_f1": 100.0 * metrics["weighted_f1"],
        "c1_recall": 100.0 * metrics["c1_recall"],
        "c2_recall": 100.0 * metrics["c2_recall"],
        "n": metrics.get("n", ""),
        "note": note,
        "source": source,
    }


def build_baseline_tables(out_dir):
    original_single = metrics_from_confusion_matrix(ORIGINAL_PAPER_SINGLE_CM)
    original_ovo = metrics_from_confusion_matrix(ORIGINAL_PAPER_OVO_CM)
    rows = [
        percent_metrics_row(
            "Original paper single multi-class classifier (Fig. 13, recomputed)",
            "sim + CHIME train half",
            "CHIME half-test",
            "from paper confusion matrix; same task, close split size",
            "2507.14854v2 Figure 13",
            original_single,
        ),
        percent_metrics_row(
            "Original paper One-vs-One binary framework (Fig. 12, recomputed)",
            "sim only",
            "all CHIME",
            "not paired with final split; context only",
            "2507.14854v2 Figure 12",
            original_ovo,
        ),
    ]

    eval_specs = [
        ("Local original notebook single CNN reproduction", "", "CHIME paper-test", "baseline reproduction", "artifacts/baselines/single_cnn_reproduction"),
        ("Local original One-vs-One reproduction", "", "CHIME paper-test", "baseline reproduction", "artifacts/baselines/one_vs_one_reproduction"),
        ("ResNet small + mask/noise", "sim + CHIME train", "CHIME paper-test", "initial stronger pipeline", "artifacts/runs/resnet_mask_noise_gpu/eval_paper_compare"),
        ("ResNet18 + mask/noise", "sim + CHIME train", "CHIME paper-test", "backbone comparison", "artifacts/runs/resnet18_aug_gpu0/eval_paper_compare"),
        ("ResNet50 + mask/noise", "sim + CHIME train", "CHIME paper-test", "backbone comparison", "artifacts/runs/resnet50_aug_gpu0/eval_paper_compare"),
        ("ResNet50 C1 repair base", "sim + CHIME train", "CHIME paper-test", "main base model", "artifacts/runs/resnet50_c1_repair_gpu0/eval_paper_compare"),
        ("Hierarchical ResNet50 + B/C1/C2 specialist", "sim + CHIME train", "CHIME paper-test", "tri-specialist fusion", "artifacts/runs/hierarchical_resnet50_b_c1_c2/eval_predicted_subset"),
        ("Final HSFNet-FRB binary specialist fusion", "sim + CHIME train", "CHIME paper-test", "current selected model", "artifacts/runs/binary_specialist_fusion/eval_predicted_subset"),
        ("Uncertain-subset routing", "sim + CHIME train", "CHIME paper-test", "routing ablation", "artifacts/runs/binary_specialist_fusion/eval_uncertain_subset_m018"),
        ("Noisebank v2 + binary specialists", "sim + CHIME train", "CHIME paper-test", "negative augmentation ablation", "artifacts/runs/noisebank_v2_binary_specialist_fusion/eval_predicted_subset"),
        ("Class-balanced focal + binary specialists", "sim + CHIME train", "CHIME paper-test", "negative loss ablation", "artifacts/runs/cb_focal_binary_specialist_fusion/eval_predicted_subset"),
        ("Sim-only final pipeline", "sim only", "CHIME paper-test", "domain-gap test", "artifacts/runs/sim_only_mainline/fusion_eval_seed_42_gpu0/eval_predicted_subset"),
    ]

    for name, training_data, test_data, note, source in eval_specs:
        metrics = load_eval_metrics(source)
        if metrics is None:
            continue
        rows.append(percent_metrics_row(name, training_data, test_data, note, source, metrics))

    fieldnames = ["model", "training_data", "test_data", "n", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "c2_recall", "note", "source"]
    save_csv(out_dir / "strong_baselines.csv", fieldnames, rows)
    save_markdown_table(
        out_dir / "strong_baselines.md",
        rows,
        [
            ("model", "model"),
            ("training_data", "training data"),
            ("test_data", "test data"),
            ("n", "n"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro-F1"),
            ("weighted_f1", "weighted-F1"),
            ("c1_recall", "C1 recall"),
            ("c2_recall", "C2 recall"),
            ("note", "note"),
        ],
    )

    ablation_rows = [
        row for row in rows
        if row["model"] not in {
            "Original paper single multi-class classifier (Fig. 13, recomputed)",
            "Original paper One-vs-One binary framework (Fig. 12, recomputed)",
            "Local original notebook single CNN reproduction",
            "Local original One-vs-One reproduction",
        }
    ]
    save_csv(out_dir / "clean_ablation_sequence.csv", fieldnames, ablation_rows)
    save_markdown_table(
        out_dir / "clean_ablation_sequence.md",
        ablation_rows,
        [
            ("model", "model"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro-F1"),
            ("weighted_f1", "weighted-F1"),
            ("c1_recall", "C1 recall"),
            ("c2_recall", "C2 recall"),
            ("note", "note"),
            ("source", "source"),
        ],
    )
    return rows


def build_paired_tests(out_dir, final_eval_dir, n_bootstrap, seed):
    final_eval_dir = Path(final_eval_dir)
    y_final = read_ints(final_eval_dir / "y_true.txt")
    pred_final = read_ints(final_eval_dir / "y_pred_binary_specialist_fusion.txt")

    comparison_specs = [
        ("Base prediction inside final fusion eval", final_eval_dir, ["y_pred_base.txt"]),
        ("ResNet50 C1 repair base", "artifacts/runs/resnet50_c1_repair_gpu0/eval_paper_compare", ["y_pred.txt"]),
        ("Tri-specialist fusion", "artifacts/runs/hierarchical_resnet50_b_c1_c2/eval_predicted_subset", ["y_pred_hierarchical.txt"]),
        ("ResNet small + mask/noise", "artifacts/runs/resnet_mask_noise_gpu/eval_paper_compare", ["y_pred.txt"]),
        ("ResNet18 + mask/noise", "artifacts/runs/resnet18_aug_gpu0/eval_paper_compare", ["y_pred.txt"]),
        ("ResNet50 + mask/noise", "artifacts/runs/resnet50_aug_gpu0/eval_paper_compare", ["y_pred.txt"]),
        ("Noisebank v2 + binary specialists", "artifacts/runs/noisebank_v2_binary_specialist_fusion/eval_predicted_subset", ["y_pred_binary_specialist_fusion.txt"]),
        ("Class-balanced focal + binary specialists", "artifacts/runs/cb_focal_binary_specialist_fusion/eval_predicted_subset", ["y_pred_binary_specialist_fusion.txt"]),
        ("Sim-only final pipeline", "artifacts/runs/sim_only_mainline/fusion_eval_seed_42_gpu0/eval_predicted_subset", ["y_pred_binary_specialist_fusion.txt"]),
    ]

    rows = []
    skipped = []
    for name, eval_dir, candidates in comparison_specs:
        y_ref, pred_ref, pred_file = load_eval_predictions(eval_dir, candidates)
        if y_ref is None:
            skipped.append({"comparison": name, "reason": "missing y_true/prediction file", "source": str(eval_dir)})
            continue
        if len(y_ref) != len(y_final) or not np.array_equal(y_ref, y_final):
            skipped.append({"comparison": name, "reason": "not the same ordered test set", "source": str(eval_dir)})
            continue

        b01, b10, p_value = exact_mcnemar(y_final, pred_ref, pred_final)
        for metric in ["accuracy", "macro_f1", "weighted_f1", "c1_recall", "c2_recall"]:
            stats = bootstrap_delta_ci(y_final, pred_ref, pred_final, metric, n_bootstrap, seed)
            rows.append(
                {
                    "comparison": "Final HSFNet-FRB vs " + name,
                    "metric": metric,
                    "final_value": metric_value(metric, y_final, pred_final),
                    "reference_value": metric_value(metric, y_final, pred_ref),
                    "delta": stats["delta"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "bootstrap_mean": stats["bootstrap_mean"],
                    "bootstrap_std": stats["bootstrap_std"],
                    "mcnemar_ref_wrong_final_correct": b01,
                    "mcnemar_ref_correct_final_wrong": b10,
                    "mcnemar_exact_p": p_value,
                    "reference_source": str(eval_dir),
                    "reference_prediction_file": pred_file,
                }
            )

    fieldnames = [
        "comparison",
        "metric",
        "final_value",
        "reference_value",
        "delta",
        "ci_low",
        "ci_high",
        "bootstrap_mean",
        "bootstrap_std",
        "mcnemar_ref_wrong_final_correct",
        "mcnemar_ref_correct_final_wrong",
        "mcnemar_exact_p",
        "reference_source",
        "reference_prediction_file",
    ]
    save_csv(out_dir / "paired_delta_tests.csv", fieldnames, rows)
    save_markdown_table(
        out_dir / "paired_delta_tests.md",
        rows,
        [
            ("comparison", "comparison"),
            ("metric", "metric"),
            ("final_value", "final"),
            ("reference_value", "reference"),
            ("delta", "delta"),
            ("ci_low", "CI low"),
            ("ci_high", "CI high"),
            ("mcnemar_exact_p", "McNemar p"),
        ],
    )
    write_json(out_dir / "paired_delta_tests_skipped.json", skipped)
    return rows, skipped


def build_bonus_sensitivity(out_dir, bonus_sweep_dir):
    rows = []
    seen = set()
    candidate_dirs = []
    bonus_sweep_dir = Path(bonus_sweep_dir)
    if bonus_sweep_dir.exists():
        candidate_dirs.extend(sorted([p for p in bonus_sweep_dir.iterdir() if p.is_dir()]))
    candidate_dirs.extend(
        [
            Path("artifacts/runs/binary_specialist_fusion/eval_predicted_subset"),
            Path("artifacts/runs/binary_specialist_fusion/eval_predicted_subset_bonus025"),
            Path("artifacts/runs/binary_specialist_fusion/eval_predicted_subset_bonus035"),
            Path("artifacts/runs/binary_specialist_fusion/eval_predicted_subset_bonus050"),
        ]
    )

    for eval_dir in candidate_dirs:
        metrics_path = eval_dir / "metrics_summary.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        bonus = float(metrics.get("c1_vote_bonus", np.nan))
        key = (round(bonus, 6), str(eval_dir))
        if key in seen:
            continue
        seen.add(key)
        per_class = metrics.get("per_class_recall", {})
        rows.append(
            {
                "c1_vote_bonus": bonus,
                "accuracy": 100.0 * float(metrics.get("accuracy", 0.0)),
                "macro_f1": 100.0 * float(metrics.get("macro_f1", 0.0)),
                "weighted_f1": 100.0 * float(metrics.get("weighted_f1", 0.0)),
                "c1_recall": 100.0 * float(per_class.get("C1", 0.0)),
                "c2_recall": 100.0 * float(per_class.get("C2", 0.0)),
                "n_routed": metrics.get("n_routed_to_specialists", ""),
                "source": str(eval_dir),
            }
        )

    rows.sort(key=lambda r: (r["c1_vote_bonus"], r["source"]))
    fieldnames = ["c1_vote_bonus", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "c2_recall", "n_routed", "source"]
    save_csv(out_dir / "c1_vote_bonus_sensitivity.csv", fieldnames, rows)
    save_markdown_table(
        out_dir / "c1_vote_bonus_sensitivity.md",
        rows,
        [
            ("c1_vote_bonus", "C1 bonus"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro-F1"),
            ("weighted_f1", "weighted-F1"),
            ("c1_recall", "C1 recall"),
            ("c2_recall", "C2 recall"),
            ("n_routed", "routed"),
            ("source", "source"),
        ],
    )
    return rows


def paper_half_indices(labels):
    y = np.asarray([CLASSES.index(label) for label in labels], dtype=np.int64)
    train_idx = []
    test_idx = []
    for class_idx in range(len(CLASSES)):
        idx = np.where(y == class_idx)[0]
        half = int(len(idx) / 2)
        train_idx.extend(idx[:half].tolist())
        test_idx.extend(idx[half:].tolist())
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def add_group_metrics(rows, group_type, group_name, indices, y_true, y_pred):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return
    metrics = metrics_from_preds(y_true[indices], y_pred[indices])
    rows.append(
        {
            "group_type": group_type,
            "group": group_name,
            "n": int(indices.size),
            "accuracy": 100.0 * metrics["accuracy"],
            "macro_f1": 100.0 * metrics["macro_f1"],
            "weighted_f1": 100.0 * metrics["weighted_f1"],
            "c1_recall": 100.0 * metrics["c1_recall"],
            "c2_recall": 100.0 * metrics["c2_recall"],
        }
    )


def load_first_npz_array(path):
    with np.load(str(path)) as z:
        return z[z.files[0]]


def build_stratified_performance(out_dir, final_eval_dir, args):
    final_eval_dir = Path(final_eval_dir)
    y_true = read_ints(final_eval_dir / "y_true.txt")
    y_pred = read_ints(final_eval_dir / "y_pred_binary_specialist_fusion.txt")
    labels = read_lines(args.chime_labels)
    _, test_idx = paper_half_indices(labels)
    rows = []

    for class_idx, name in enumerate(CLASSES):
        add_group_metrics(rows, "true_class", name, np.where(y_true == class_idx)[0], y_true, y_pred)

    if Path(args.chime_msnr).exists():
        all_msnr = read_float_lines(args.chime_msnr)
        test_msnr = all_msnr[test_idx]
        q1, q2 = np.percentile(test_msnr, [33.333, 66.667])
        add_group_metrics(rows, "msnr_tertile", "low <= %.3f" % q1, np.where(test_msnr <= q1)[0], y_true, y_pred)
        add_group_metrics(rows, "msnr_tertile", "%.3f < mid <= %.3f" % (q1, q2), np.where((test_msnr > q1) & (test_msnr <= q2))[0], y_true, y_pred)
        add_group_metrics(rows, "msnr_tertile", "high > %.3f" % q2, np.where(test_msnr > q2)[0], y_true, y_pred)

    non_positive_fraction = None
    if Path(args.chime_data).exists():
        arr = load_first_npz_array(args.chime_data)
        test_arr = arr[test_idx]
        non_positive_fraction = np.mean(test_arr <= 0, axis=tuple(range(1, test_arr.ndim)))
        q1, q2 = np.percentile(non_positive_fraction, [33.333, 66.667])
        add_group_metrics(rows, "non_positive_fraction_proxy", "low <= %.4f" % q1, np.where(non_positive_fraction <= q1)[0], y_true, y_pred)
        add_group_metrics(rows, "non_positive_fraction_proxy", "%.4f < mid <= %.4f" % (q1, q2), np.where((non_positive_fraction > q1) & (non_positive_fraction <= q2))[0], y_true, y_pred)
        add_group_metrics(rows, "non_positive_fraction_proxy", "high > %.4f" % q2, np.where(non_positive_fraction > q2)[0], y_true, y_pred)

    fieldnames = ["group_type", "group", "n", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "c2_recall"]
    save_csv(out_dir / "stratified_performance.csv", fieldnames, rows)
    save_markdown_table(
        out_dir / "stratified_performance.md",
        rows,
        [
            ("group_type", "group type"),
            ("group", "group"),
            ("n", "n"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro-F1"),
            ("weighted_f1", "weighted-F1"),
            ("c1_recall", "C1 recall"),
            ("c2_recall", "C2 recall"),
        ],
    )

    error_rows = []
    file_order = read_lines(args.chime_file_order) if Path(args.chime_file_order).exists() else [""] * len(labels)
    msnr = read_float_lines(args.chime_msnr) if Path(args.chime_msnr).exists() else np.full((len(labels),), np.nan)
    for local_idx, (y, p) in enumerate(zip(y_true, y_pred)):
        if int(y) == int(p):
            continue
        global_idx = int(test_idx[local_idx])
        error_rows.append(
            {
                "local_test_index": local_idx,
                "global_chime_index": global_idx,
                "file": file_order[global_idx] if global_idx < len(file_order) else "",
                "true_class": CLASSES[int(y)],
                "predicted_class": CLASSES[int(p)],
                "msnr": float(msnr[global_idx]) if global_idx < len(msnr) else "",
                "non_positive_fraction_proxy": float(non_positive_fraction[local_idx]) if non_positive_fraction is not None else "",
            }
        )
    save_csv(
        out_dir / "error_examples.csv",
        ["local_test_index", "global_chime_index", "file", "true_class", "predicted_class", "msnr", "non_positive_fraction_proxy"],
        error_rows,
    )
    return rows, error_rows


def build_mask_robustness_copy(out_dir):
    source = Path("reports/mask_robustness.csv")
    if not source.exists():
        return []
    rows = []
    with source.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    save_csv(out_dir / "fixed_mask_robustness.csv", reader.fieldnames, rows)
    save_markdown_table(
        out_dir / "fixed_mask_robustness.md",
        rows,
        [(name, name) for name in reader.fieldnames],
    )
    return rows


def write_readme(out_dir, args, summary):
    text = """# Q2/Q3 Strengthening Analyses

This directory contains reviewer-facing analyses generated from retained HSFNet-FRB runs.

Generated files:

- `paired_delta_tests.md`: paired bootstrap delta CIs and exact McNemar tests for final HSFNet-FRB versus available paired baselines.
- `paired_delta_tests_skipped.json`: comparisons skipped because paired predictions were unavailable or not on the same ordered test set.
- `c1_vote_bonus_sensitivity.md`: sensitivity of the final routing score to the C1 vote bonus.
- `strong_baselines.md`: original-paper, local reproduction, backbone, loss, augmentation, routing, and domain-gap baselines.
- `clean_ablation_sequence.md`: compact ablation table for the main technical story.
- `stratified_performance.md`: final-model performance stratified by true class, CHIME mSNR tertile, and non-positive intensity proxy tertile.
- `error_examples.csv`: misclassified CHIME paper-test examples with global CHIME index and metadata proxies.
- `fixed_mask_robustness.md`: retained fixed-mask robustness table copied into the Q2/Q3 package.

Important interpretation notes:

- C1 and C2 have only 10 samples each in the CHIME paper-test split, so minority-class recall should be described cautiously.
- McNemar tests only evaluate paired correctness changes on the same ordered test set.
- The non-positive intensity proxy is not a physical width or SNR measurement; it is a reproducible image-level proxy for blank/low-power regions in the retained CHIME array.
- No reliable burst-width metadata file was found in the cleaned project, so width-stratified claims should not be made from this package alone.

Run configuration:

- final eval dir: `{final_eval_dir}`
- bonus sweep dir: `{bonus_sweep_dir}`
- bootstrap resamples: `{n_bootstrap}`
- seed: `{seed}`
""".format(
        final_eval_dir=args.final_eval_dir,
        bonus_sweep_dir=args.bonus_sweep_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(text)
    write_json(out_dir / "summary.json", summary)


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    baseline_rows = build_baseline_tables(out_dir)
    paired_rows, skipped = build_paired_tests(out_dir, args.final_eval_dir, args.n_bootstrap, args.seed)
    bonus_rows = build_bonus_sensitivity(out_dir, args.bonus_sweep_dir)
    stratified_rows, error_rows = build_stratified_performance(out_dir, args.final_eval_dir, args)
    mask_rows = build_mask_robustness_copy(out_dir)

    summary = {
        "n_baseline_rows": len(baseline_rows),
        "n_paired_delta_rows": len(paired_rows),
        "n_skipped_paired_comparisons": len(skipped),
        "n_bonus_rows": len(bonus_rows),
        "n_stratified_rows": len(stratified_rows),
        "n_error_examples": len(error_rows),
        "n_mask_robustness_rows": len(mask_rows),
    }
    write_readme(out_dir, args, summary)
    print("Q2/Q3 strengthening analyses written to: %s" % out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
