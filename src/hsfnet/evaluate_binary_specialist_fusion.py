from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from data import CLASSES, PAPER_CLASS_NAMES, ensure_dir, load_chime_data, paper_half_split_chime, save_json


B_IDX = 1
C1_IDX = 3
C2_IDX = 4
ROUTE_SET = {B_IDX, C1_IDX, C2_IDX}
TRI_TO_GLOBAL = np.asarray([B_IDX, C1_IDX, C2_IDX], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical FRB classifier with B/C1/C2 and binary specialists.")
    parser.add_argument("--base-model", default="artifacts/runs/resnet50_c1_repair_gpu0/best_model")
    parser.add_argument("--tri-specialist-model", default="artifacts/runs/specialist_b_c1_c2_gpu0/best_model")
    parser.add_argument("--b-c1-model", required=True)
    parser.add_argument("--c1-c2-model", required=True)
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="artifacts/runs/binary_specialist_fusion/eval_paper_compare")
    parser.add_argument("--split", choices=["paper_test", "paper_train", "all_chime"], default="paper_test")
    parser.add_argument("--route-mode", choices=["predicted_subset", "uncertain_subset"], default="predicted_subset")
    parser.add_argument("--margin-threshold", type=float, default=0.18, help="Route extra samples when top1-top2 margin is below this value.")
    parser.add_argument("--c1-vote-bonus", type=float, default=0.15, help="Additional score for C1 when either binary expert votes C1.")
    parser.add_argument("--fixed-mask-ratio", type=float, default=0.0, help="Deterministically mask this fraction of frequency channels at evaluation time.")
    parser.add_argument("--fixed-mask-seed", type=int, default=123)
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


def apply_fixed_frequency_mask(x, ratio, seed):
    if ratio <= 0.0:
        return x
    rng = np.random.RandomState(seed)
    x = np.array(x, copy=True)
    n_freq = x.shape[1]
    n_mask = int(round(n_freq * ratio))
    if n_mask <= 0:
        return x
    rows = rng.choice(np.arange(n_freq), size=min(n_mask, n_freq), replace=False)
    x[:, rows, :, :] = 0
    return x


def route_samples(base_probs, base_pred, route_mode, margin_threshold):
    if route_mode == "predicted_subset":
        return np.asarray([int(pred) in ROUTE_SET for pred in base_pred], dtype=bool)

    sorted_probs = np.sort(base_probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.asarray([(int(pred) in ROUTE_SET) or (float(margins[i]) <= margin_threshold) for i, pred in enumerate(base_pred)], dtype=bool)


def fuse_predictions(base_pred, routed, tri_probs_all, bc1_probs_all, c1c2_probs_all, c1_vote_bonus):
    final_pred = base_pred.copy()
    local_indices = np.where(routed)[0]
    diagnostics = []

    for i in local_indices:
        scores = {
            B_IDX: float(tri_probs_all[i, 0]),
            C1_IDX: float(tri_probs_all[i, 1]),
            C2_IDX: float(tri_probs_all[i, 2]),
        }

        bc1_c1_prob = float(bc1_probs_all[i, 1])
        c1c2_c2_prob = float(c1c2_probs_all[i, 1])

        if bc1_c1_prob >= 0.5:
            scores[C1_IDX] += c1_vote_bonus
        else:
            scores[B_IDX] += c1_vote_bonus

        if c1c2_c2_prob >= 0.5:
            scores[C2_IDX] += c1_vote_bonus
        else:
            scores[C1_IDX] += c1_vote_bonus

        pred = max(scores.items(), key=lambda item: item[1])[0]
        final_pred[i] = pred
        diagnostics.append(
            {
                "sample": int(i),
                "base_pred": int(base_pred[i]),
                "final_pred": int(pred),
                "tri_pred": int(TRI_TO_GLOBAL[int(np.argmax(tri_probs_all[i]))]),
                "bc1_c1_prob": bc1_c1_prob,
                "c1c2_c2_prob": c1c2_c2_prob,
                "score_B": scores[B_IDX],
                "score_C1": scores[C1_IDX],
                "score_C2": scores[C2_IDX],
            }
        )

    return final_pred, diagnostics


def _write_confidence_csv(path, base_probs, tri_probs, bc1_probs, c1c2_probs, final_pred, routed):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample", "routed", "final_pred"]
            + ["base_" + name for name in CLASSES]
            + ["tri_B", "tri_C1", "tri_C2", "bc1_B", "bc1_C1", "c1c2_C1", "c1c2_C2"]
        )
        for idx in range(base_probs.shape[0]):
            if routed[idx]:
                extra = list(tri_probs[idx]) + list(bc1_probs[idx]) + list(c1c2_probs[idx])
            else:
                extra = [np.nan] * 7
            writer.writerow(
                [idx, int(routed[idx]), CLASSES[int(final_pred[idx])]]
                + ["%.8f" % float(v) for v in base_probs[idx]]
                + ["" if np.isnan(v) else "%.8f" % float(v) for v in extra]
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
    x = apply_fixed_frequency_mask(x, args.fixed_mask_ratio, args.fixed_mask_seed)
    ds = _prediction_dataset(tf, x, args.batch_size)

    base_model = keras.models.load_model(args.base_model)
    tri_model = keras.models.load_model(args.tri_specialist_model)
    bc1_model = keras.models.load_model(args.b_c1_model)
    c1c2_model = keras.models.load_model(args.c1_c2_model)

    base_probs = base_model.predict(ds, verbose=1)
    base_pred = np.argmax(base_probs, axis=1).astype(np.int64)
    routed = route_samples(base_probs, base_pred, args.route_mode, args.margin_threshold)

    tri_probs_all = np.full((x.shape[0], 3), np.nan, dtype=np.float32)
    bc1_probs_all = np.full((x.shape[0], 2), np.nan, dtype=np.float32)
    c1c2_probs_all = np.full((x.shape[0], 2), np.nan, dtype=np.float32)

    if np.any(routed):
        routed_ds = _prediction_dataset(tf, x[routed], args.batch_size)
        tri_probs_all[routed] = tri_model.predict(routed_ds, verbose=1)
        bc1_probs_all[routed] = bc1_model.predict(routed_ds, verbose=1)
        c1c2_probs_all[routed] = c1c2_model.predict(routed_ds, verbose=1)

    final_pred, diagnostics = fuse_predictions(
        base_pred,
        routed,
        tri_probs_all,
        bc1_probs_all,
        c1c2_probs_all,
        c1_vote_bonus=args.c1_vote_bonus,
    )

    report = classification_report(y, final_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, final_pred, labels=list(range(len(CLASSES))))

    np.savetxt(str(out_dir / "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(str(out_dir / "y_true.txt"), y, fmt="%d")
    np.savetxt(str(out_dir / "y_pred_base.txt"), base_pred, fmt="%d")
    np.savetxt(str(out_dir / "y_pred_binary_specialist_fusion.txt"), final_pred, fmt="%d")
    save_json(out_dir / "classification_report.json", report)
    save_json(out_dir / "routing_diagnostics.json", diagnostics)
    _write_confidence_csv(out_dir / "confidence_scores.csv", base_probs, tri_probs_all, bc1_probs_all, c1c2_probs_all, final_pred, routed)
    _plot_confusion_matrices(out_dir, cm)

    per_class_recall = {}
    for idx, name in enumerate(CLASSES):
        denom = cm[idx].sum()
        per_class_recall[name] = float(cm[idx, idx] / denom) if denom else 0.0

    summary = {
        "dataset": dataset_name,
        "route_mode": args.route_mode,
        "margin_threshold": float(args.margin_threshold),
        "c1_vote_bonus": float(args.c1_vote_bonus),
        "fixed_mask_ratio": float(args.fixed_mask_ratio),
        "n_samples": int(len(y)),
        "n_routed_to_specialists": int(np.sum(routed)),
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
    print("Binary specialist fusion evaluation saved to: %s" % out_dir)
    print("Accuracy: %.4f, macro-F1: %.4f, weighted-F1: %.4f, C1 recall: %.4f, routed: %d" % (
        summary["accuracy"],
        summary["macro_f1"],
        summary["weighted_f1"],
        summary["per_class_recall"]["C1"],
        summary["n_routed_to_specialists"],
    ))


if __name__ == "__main__":
    main()
