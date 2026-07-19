from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from data import (
    CLASSES,
    PAPER_CLASS_NAMES,
    ensure_dir,
    load_chime_data,
    load_simulation_dataset,
    paper_half_split_chime,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an FRB multiclass model with paper-compatible metrics.")
    parser.add_argument("--model", default="artifacts/runs/resnet_mask_noise/best_model", help="Saved Keras model path.")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="artifacts/runs/resnet_mask_noise/eval_chime")
    parser.add_argument("--split", choices=["paper_test", "paper_train", "all_chime"], default="paper_test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    parser.add_argument("--gpu-id", default="0", help="GPU id used when --device gpu, e.g. 0 or 1.")
    parser.add_argument("--disable-tf32", action="store_true", help="Disable TensorFloat-32 matmul on Ampere/Ada GPUs.")
    parser.add_argument("--eval-sim", action="store_true", help="Also evaluate a balanced simulated set.")
    parser.add_argument("--sim-dir", default="data/simulated/simfrb")
    parser.add_argument("--sim-samples-per-class", type=int, default=500)
    parser.add_argument("--snrs", default="15,25,35,50,100")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def configure_device(device):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-frabjous")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "gpu":
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def _load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prediction_dataset(tf, x, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _write_confidence_csv(path, probs):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample"] + CLASSES)
        for idx, row in enumerate(probs):
            writer.writerow([idx] + ["%.8f" % float(v) for v in row])


def _plot_confusion_matrices(out_dir, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums != 0) * 100.0

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_matrix_counts.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_matrix_percent.png"), dpi=200)
    plt.close()


def evaluate_dataset(tf, model, x, y, out_dir, batch_size, dataset_name):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

    out_dir = ensure_dir(out_dir)
    ds = _prediction_dataset(tf, x, y, batch_size)
    probs = model.predict(ds, verbose=1)
    pred = np.argmax(probs, axis=1)

    report = classification_report(y, pred, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES))))

    np.savetxt(str(out_dir / "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(str(out_dir / "y_true.txt"), y, fmt="%d")
    np.savetxt(str(out_dir / "y_pred.txt"), pred, fmt="%d")
    _write_confidence_csv(out_dir / "confidence_scores.csv", probs)
    save_json(out_dir / "classification_report.json", report)
    _plot_confusion_matrices(out_dir, cm)

    per_class_recall = {}
    for idx, name in enumerate(CLASSES):
        denom = cm[idx].sum()
        per_class_recall[name] = float(cm[idx, idx] / denom) if denom else 0.0

    summary = {
        "dataset": dataset_name,
        "n_samples": int(len(y)),
        "class_names": CLASSES,
        "paper_class_names": PAPER_CLASS_NAMES,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "per_class_recall": per_class_recall,
        "outputs": {
            "classification_report": "classification_report.json",
            "confusion_matrix_csv": "confusion_matrix.csv",
            "confusion_matrix_counts_png": "confusion_matrix_counts.png",
            "confusion_matrix_percent_png": "confusion_matrix_percent.png",
            "confidence_scores": "confidence_scores.csv",
        },
    }
    save_json(out_dir / "metrics_summary.json", summary)
    return summary


def main():
    args = parse_args()
    if args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    configure_device(args.device)

    import tensorflow as tf
    from tensorflow import keras

    import losses  # noqa: F401 - registers custom serialized losses.

    if args.disable_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    model = keras.models.load_model(args.model)
    out_dir = ensure_dir(args.output_dir)

    if args.split == "all_chime":
        x, y, _ = load_chime_data(args.chime_data, args.chime_labels)
        dataset_name = "chime_all_535"
    else:
        train_x, train_y, test_x, test_y, _, _ = paper_half_split_chime(args.chime_data, args.chime_labels)
        if args.split == "paper_train":
            x, y = train_x, train_y
            dataset_name = "chime_paper_train_half"
        else:
            x, y = test_x, test_y
            dataset_name = "chime_paper_test_half"

    chime_summary = evaluate_dataset(tf, model, x, y, out_dir, args.batch_size, dataset_name)

    comparison = {
        "paper_single_multiclass_chime_accuracy_approx": 0.55,
        "paper_single_multiclass_simulated_accuracy_approx": 0.73,
        "local_reproduced_single_multiclass_chime": _load_json_if_exists("artifacts/baselines/single_cnn_reproduction/classification_report.json"),
        "local_reproduced_one_vs_one_chime": _load_json_if_exists("artifacts/baselines/one_vs_one_reproduction/classification_report.json"),
        "new_model_chime_summary": chime_summary,
    }

    if args.eval_sim:
        snrs = [int(v.strip()) for v in args.snrs.split(",") if v.strip()]
        sim_x, sim_y, _ = load_simulation_dataset(
            args.sim_dir,
            samples_per_class=args.sim_samples_per_class,
            snrs=snrs,
            seed=args.seed,
            exclude_marked=True,
        )
        sim_summary = evaluate_dataset(tf, model, sim_x, sim_y, out_dir / "eval_simulated", args.batch_size, "simulated_balanced")
        comparison["new_model_simulated_summary"] = sim_summary

    save_json(out_dir / "paper_comparison.json", comparison)
    print("Evaluation saved to: %s" % out_dir)
    print("Accuracy: %.4f, macro-F1: %.4f, weighted-F1: %.4f" % (
        chime_summary["accuracy"],
        chime_summary["macro_f1"],
        chime_summary["weighted_f1"],
    ))


if __name__ == "__main__":
    main()
