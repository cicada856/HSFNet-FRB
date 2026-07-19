from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from data import CLASSES, PAPER_CLASS_NAMES, ensure_dir, load_chime_data, paper_half_split_chime, save_json


SPECIALIST_CLASSES = ["B", "C1", "C2"]
SPECIALIST_TO_GLOBAL = np.asarray([1, 3, 4], dtype=np.int64)
ROUTE_GLOBAL = set(SPECIALIST_TO_GLOBAL.tolist())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical FRB classifier: 5-way base + B/C1/C2 specialist.")
    parser.add_argument("--base-model", default="artifacts/runs/resnet50_c1_repair_gpu0/best_model")
    parser.add_argument("--specialist-model", default="artifacts/runs/specialist_b_c1_c2_gpu0/best_model")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="artifacts/runs/hierarchical_resnet50_b_c1_c2/eval_paper_compare")
    parser.add_argument("--split", choices=["paper_test", "paper_train", "all_chime"], default="paper_test")
    parser.add_argument("--route-mode", choices=["predicted_subset", "c1_sensitive"], default="predicted_subset")
    parser.add_argument("--c1-threshold", type=float, default=0.08, help="Used only with --route-mode c1_sensitive.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--gpu-id", default="0")
    return parser.parse_args()


def configure_device(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-frabjous")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def _prediction_dataset(tf, x, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(x)

    def preprocess(image):
        return tf.cast(image, tf.float32) / 255.0

    return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _select_split(args):
    if args.split == "all_chime":
        x, y, _ = load_chime_data(args.chime_data, args.chime_labels)
        return x, y, "chime_all_535"

    train_x, train_y, test_x, test_y, _, _ = paper_half_split_chime(args.chime_data, args.chime_labels)
    if args.split == "paper_train":
        return train_x, train_y, "chime_paper_train_half"
    return test_x, test_y, "chime_paper_test_half"


def _write_confidence_csv(path, base_probs, specialist_probs, final_pred, routed):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample", "routed", "final_pred"] + ["base_" + name for name in CLASSES] + ["specialist_" + name for name in SPECIALIST_CLASSES]
        writer.writerow(header)
        for idx in range(base_probs.shape[0]):
            spec = specialist_probs[idx] if routed[idx] else np.full((len(SPECIALIST_CLASSES),), np.nan)
            writer.writerow(
                [idx, int(routed[idx]), CLASSES[int(final_pred[idx])]]
                + ["%.8f" % float(v) for v in base_probs[idx]]
                + ["" if np.isnan(v) else "%.8f" % float(v) for v in spec]
            )


def _plot_confusion_matrices(out_dir, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums != 0) * 100.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "confusion_matrix_counts.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "confusion_matrix_percent.png"), dpi=200)
    plt.close()


def main():
    args = parse_args()
    configure_device(args)

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

    import losses  # noqa: F401 - registers custom serialized losses.

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    out_dir = ensure_dir(args.output_dir)
    x, y, dataset_name = _select_split(args)
    ds = _prediction_dataset(tf, x, args.batch_size)

    base_model = keras.models.load_model(args.base_model)
    specialist_model = keras.models.load_model(args.specialist_model)

    base_probs = base_model.predict(ds, verbose=1)
    base_pred = np.argmax(base_probs, axis=1).astype(np.int64)

    if args.route_mode == "predicted_subset":
        routed = np.asarray([int(pred) in ROUTE_GLOBAL for pred in base_pred], dtype=bool)
    else:
        routed = np.asarray([(int(pred) in ROUTE_GLOBAL) or (float(base_probs[i, 3]) >= args.c1_threshold) for i, pred in enumerate(base_pred)], dtype=bool)

    final_pred = base_pred.copy()
    specialist_probs_all = np.full((x.shape[0], len(SPECIALIST_CLASSES)), np.nan, dtype=np.float32)

    if np.any(routed):
        specialist_ds = _prediction_dataset(tf, x[routed], args.batch_size)
        specialist_probs = specialist_model.predict(specialist_ds, verbose=1)
        specialist_pred_local = np.argmax(specialist_probs, axis=1).astype(np.int64)
        specialist_pred_global = SPECIALIST_TO_GLOBAL[specialist_pred_local]
        final_pred[routed] = specialist_pred_global
        specialist_probs_all[routed] = specialist_probs

    report = classification_report(y, final_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, final_pred, labels=list(range(len(CLASSES))))

    np.savetxt(str(out_dir / "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(str(out_dir / "y_true.txt"), y, fmt="%d")
    np.savetxt(str(out_dir / "y_pred_base.txt"), base_pred, fmt="%d")
    np.savetxt(str(out_dir / "y_pred_hierarchical.txt"), final_pred, fmt="%d")
    save_json(out_dir / "classification_report.json", report)
    _write_confidence_csv(out_dir / "confidence_scores.csv", base_probs, specialist_probs_all, final_pred, routed)
    _plot_confusion_matrices(out_dir, cm)

    per_class_recall = {}
    for idx, name in enumerate(CLASSES):
        denom = cm[idx].sum()
        per_class_recall[name] = float(cm[idx, idx] / denom) if denom else 0.0

    summary = {
        "dataset": dataset_name,
        "route_mode": args.route_mode,
        "c1_threshold": float(args.c1_threshold),
        "n_samples": int(len(y)),
        "n_routed_to_specialist": int(np.sum(routed)),
        "class_names": CLASSES,
        "paper_class_names": PAPER_CLASS_NAMES,
        "accuracy": float(accuracy_score(y, final_pred)),
        "macro_precision": float(precision_score(y, final_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, final_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, final_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y, final_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y, final_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y, final_pred, average="weighted", zero_division=0)),
        "per_class_recall": per_class_recall,
    }
    save_json(out_dir / "metrics_summary.json", summary)
    print("Hierarchical evaluation saved to: %s" % out_dir)
    print("Accuracy: %.4f, macro-F1: %.4f, weighted-F1: %.4f, C1 recall: %.4f, routed: %d" % (
        summary["accuracy"],
        summary["macro_f1"],
        summary["weighted_f1"],
        summary["per_class_recall"]["C1"],
        summary["n_routed_to_specialist"],
    ))


if __name__ == "__main__":
    main()
