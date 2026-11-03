"""Evaluate final HSFNet-FRB specialists on complete true-label CHIME subsets."""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


HSFNET_DIR = Path(__file__).resolve().parents[1]
if str(HSFNET_DIR) not in sys.path:
    sys.path.insert(0, str(HSFNET_DIR))

from data import CLASS_TO_INDEX, ensure_dir, paper_half_split_chime, save_json  # noqa: E402


SPECIALISTS = (
    {
        "key": "tri_b_c1_c2",
        "title": "Tri-specialist: B / C1 / C2",
        "classes": ("B", "C1", "C2"),
        "model_arg": "tri_model",
        "expected_n": 155,
        "routed_probability_columns": ("tri_B", "tri_C1", "tri_C2"),
    },
    {
        "key": "binary_b_c1",
        "title": "Binary specialist: B / C1",
        "classes": ("B", "C1"),
        "model_arg": "b_c1_model",
        "expected_n": 145,
        "routed_probability_columns": ("bc1_B", "bc1_C1"),
    },
    {
        "key": "binary_c1_c2",
        "title": "Binary specialist: C1 / C2",
        "classes": ("C1", "C2"),
        "model_arg": "c1_c2_model",
        "expected_n": 20,
        "routed_probability_columns": ("c1c2_C1", "c1c2_C2"),
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate final specialists on all paper-test samples with applicable true labels."
    )
    parser.add_argument("--tri-model", default="artifacts/models/final/tri_specialist_b_c1_c2")
    parser.add_argument("--b-c1-model", default="artifacts/models/final/binary_specialist_b_vs_c1")
    parser.add_argument("--c1-c2-model", default="artifacts/models/final/binary_specialist_c1_vs_c2")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="reports/specialist_full_true_subsets")
    parser.add_argument(
        "--routed-confidence-csv",
        default="artifacts/runs/binary_specialist_fusion/eval_predicted_subset/confidence_scores.csv",
    )
    parser.add_argument(
        "--routed-y-true",
        default="artifacts/runs/binary_specialist_fusion/eval_predicted_subset/y_true.txt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu")
    parser.add_argument("--gpu-id", default="1")
    return parser.parse_args()


def configure_device(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hsfnet-specialists")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def prediction_dataset(tf, x, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(lambda image: tf.cast(image, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def calculate_metrics(y_true, y_pred, n_classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_recall_values": [float(value) for value in recalls],
    }, cm


def write_predictions(path, spec, test_positions, chime_indices, true_local, pred_local, probabilities):
    probability_headers = ["prob_" + name for name in spec["classes"]]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "paper_test_position",
                "chime_global_index",
                "true_local_label",
                "true_class",
                "pred_local_label",
                "pred_class",
            ]
            + probability_headers
        )
        for row_idx in range(len(true_local)):
            writer.writerow(
                [
                    int(test_positions[row_idx]),
                    int(chime_indices[row_idx]),
                    int(true_local[row_idx]),
                    spec["classes"][int(true_local[row_idx])],
                    int(pred_local[row_idx]),
                    spec["classes"][int(pred_local[row_idx])],
                ]
                + ["%.9f" % float(value) for value in probabilities[row_idx]]
            )


def read_routed_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def routed_subset_metrics(spec, rows, routed_y_global):
    class_global = np.asarray([CLASS_TO_INDEX[name] for name in spec["classes"]], dtype=np.int64)
    global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(class_global)}
    y_true = []
    y_pred = []

    for row in rows:
        position = int(row["sample"])
        true_global = int(routed_y_global[position])
        if int(row["routed"]) != 1 or true_global not in global_to_local:
            continue
        probability_values = [row[column] for column in spec["routed_probability_columns"]]
        if any(value == "" for value in probability_values):
            continue
        probabilities = np.asarray([float(value) for value in probability_values], dtype=np.float64)
        y_true.append(global_to_local[true_global])
        y_pred.append(int(np.argmax(probabilities)))

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    metrics, _ = calculate_metrics(y_true, y_pred, len(spec["classes"]))
    metrics["n_samples"] = int(len(y_true))
    metrics["selection"] = "base-prediction routed samples whose true label is in the expert task"
    metrics["per_class_recall"] = {
        name: value for name, value in zip(spec["classes"], metrics.pop("per_class_recall_values"))
    }
    return metrics


def setup_plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_confusion_matrices(results, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]})
    image = None

    for axis, result in zip(axes, results):
        cm = np.asarray(result["confusion_matrix"], dtype=np.int64)
        row_sums = cm.sum(axis=1, keepdims=True)
        percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0) * 100.0
        edges = np.arange(cm.shape[0] + 1)
        image = axis.pcolormesh(
            edges,
            edges,
            percent,
            vmin=0.0,
            vmax=100.0,
            cmap="Blues",
            shading="flat",
            edgecolors="white",
            linewidth=0.6,
        )
        axis.set_aspect("equal")
        axis.set_xlim(0, cm.shape[1])
        axis.set_ylim(cm.shape[0], 0)
        for row in range(cm.shape[0]):
            for column in range(cm.shape[1]):
                color = "white" if percent[row, column] >= 55.0 else "#202020"
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    "%d\n(%.1f%%)" % (cm[row, column], percent[row, column]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        classes = result["classes"]
        axis.set_xticks(np.arange(len(classes)) + 0.5)
        axis.set_yticks(np.arange(len(classes)) + 0.5)
        axis.set_xticklabels(classes)
        axis.set_yticklabels(classes)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        metrics = result["full_true_subset"]
        axis.set_title(
            "%s\n$n$=%d | Acc. %.1f%% | BAcc. %.1f%%\nMacro-F1 %.1f%%"
            % (
                result["title"],
                metrics["n_samples"],
                metrics["accuracy"] * 100.0,
                metrics["balanced_accuracy"] * 100.0,
                metrics["macro_f1"] * 100.0,
            ),
            pad=8,
            fontsize=9.2,
        )

    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.16, top=0.74, wspace=0.34)
    colorbar_axis = fig.add_axes([0.925, 0.20, 0.015, 0.55])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Row-normalized proportion (%)")

    stem = Path(output_dir) / "fig_specialist_full_true_subsets"
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    plt.close(fig)


def write_captions(path):
    content = """# Figure captions

## 中文

HSFNet-FRB 专家模型在独立 CHIME paper-test split 完整真实标签子集上的分类结果。左图为 B/C1/C2 三分类专家（n=155），中图为 B-vs-C1 二分类专家（n=145），右图为 C1-vs-C2 二分类专家（n=20）。每个混淆矩阵单元格给出样本计数及按真实类别归一化的比例。所有样本均依据真实标签纳入相应专家任务，未根据基础模型预测或路由状态筛选；隔离的 paper-test split 未参与训练、验证、模型选择或参数调整。

## English

Classification results of the HSFNet-FRB specialists on their complete true-label subsets from the independent CHIME paper-test split. The panels show the B/C1/C2 tri-specialist (n=155), the B-vs-C1 binary specialist (n=145), and the C1-vs-C2 binary specialist (n=20), respectively. Each cell reports the sample count and the row-normalized proportion. Samples were included solely according to their ground-truth labels, without filtering by base-model predictions or routing status. The isolated paper-test split was not used for training, validation, model selection, or parameter tuning.
"""
    Path(path).write_text(content, encoding="utf-8")


def write_comparison(path, results):
    lines = [
        "# Complete true-label subsets versus routed subsets",
        "",
        "The complete-subset evaluation measures each specialist on every applicable paper-test example. "
        "The routed comparison is conditional on the existing base-model predicted-subset routing and is therefore smaller and selection-dependent.",
        "",
        "| Specialist | Full n | Full accuracy | Full balanced accuracy | Full macro-F1 | Routed applicable n | Routed accuracy | Routed balanced accuracy | Routed macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        full = result["full_true_subset"]
        routed = result["existing_routed_subset"]
        lines.append(
            "| %s | %d | %.2f%% | %.2f%% | %.2f%% | %d | %.2f%% | %.2f%% | %.2f%% |"
            % (
                result["title"],
                full["n_samples"],
                full["accuracy"] * 100.0,
                full["balanced_accuracy"] * 100.0,
                full["macro_f1"] * 100.0,
                routed["n_samples"],
                routed["accuracy"] * 100.0,
                routed["balanced_accuracy"] * 100.0,
                routed["macro_f1"] * 100.0,
            )
        )
    lines.extend(
        [
            "",
            "Routed-subset values should not be interpreted as independent test-set estimates: routing depends on the base model's predictions. "
            "The complete true-label subsets are the direct standalone expert evaluation requested here.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    configure_device(args)

    import tensorflow as tf
    from tensorflow import keras

    import losses  # noqa: F401 - registers any serialized custom losses.

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    output_dir = ensure_dir(args.output_dir)
    _, _, test_x, test_y_global, _, test_indices = paper_half_split_chime(args.chime_data, args.chime_labels)
    model_paths = {
        "tri_model": args.tri_model,
        "b_c1_model": args.b_c1_model,
        "c1_c2_model": args.c1_c2_model,
    }
    routed_rows = read_routed_rows(args.routed_confidence_csv)
    routed_y_global = np.loadtxt(args.routed_y_true, dtype=np.int64)

    if len(routed_rows) != len(test_y_global) or len(routed_y_global) != len(test_y_global):
        raise ValueError("Existing routed outputs do not align with the %d-sample paper-test split." % len(test_y_global))

    results = []
    saved_indices = {}
    for spec in SPECIALISTS:
        class_global = np.asarray([CLASS_TO_INDEX[name] for name in spec["classes"]], dtype=np.int64)
        keep = np.isin(test_y_global, class_global)
        test_positions = np.where(keep)[0]
        subset_x = test_x[keep]
        subset_y_global = test_y_global[keep]
        subset_chime_indices = test_indices[keep]
        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(class_global)}
        true_local = np.asarray([global_to_local[int(value)] for value in subset_y_global], dtype=np.int64)

        if len(subset_x) != spec["expected_n"]:
            raise ValueError(
                "%s expected n=%d but paper-test filtering produced n=%d"
                % (spec["key"], spec["expected_n"], len(subset_x))
            )

        model = keras.models.load_model(model_paths[spec["model_arg"]])
        probabilities = model.predict(prediction_dataset(tf, subset_x, args.batch_size), verbose=0)
        if probabilities.shape != (len(subset_x), len(spec["classes"])):
            raise ValueError("Unexpected probability shape for %s: %s" % (spec["key"], probabilities.shape))
        pred_local = np.argmax(probabilities, axis=1).astype(np.int64)
        metrics, cm = calculate_metrics(true_local, pred_local, len(spec["classes"]))
        metrics["n_samples"] = int(len(true_local))
        metrics["selection"] = "complete paper-test subset selected only by ground-truth class"
        metrics["per_class_recall"] = {
            name: value for name, value in zip(spec["classes"], metrics.pop("per_class_recall_values"))
        }

        prediction_path = output_dir / ("specialist_predictions_%s.csv" % spec["key"])
        confusion_path = output_dir / ("specialist_confusion_%s.csv" % spec["key"])
        write_predictions(
            prediction_path,
            spec,
            test_positions,
            subset_chime_indices,
            true_local,
            pred_local,
            probabilities,
        )
        np.savetxt(str(confusion_path), cm, fmt="%d", delimiter=",")
        saved_indices[spec["key"]] = subset_chime_indices

        results.append(
            {
                "key": spec["key"],
                "title": spec["title"],
                "classes": list(spec["classes"]),
                "model_path": model_paths[spec["model_arg"]],
                "full_true_subset": metrics,
                "existing_routed_subset": routed_subset_metrics(spec, routed_rows, routed_y_global),
                "confusion_matrix": cm.tolist(),
                "prediction_csv": str(prediction_path),
                "confusion_matrix_csv": str(confusion_path),
            }
        )

        del model
        keras.backend.clear_session()

    np.savez_compressed(str(output_dir / "specialist_full_true_subset_indices.npz"), **saved_indices)
    payload = {
        "dataset": "independent CHIME paper-test half",
        "selection_policy": "complete true-label subset; no base-prediction or routing filter",
        "n_paper_test_total": int(len(test_y_global)),
        "chime_data": args.chime_data,
        "chime_labels": args.chime_labels,
        "existing_routed_source": args.routed_confidence_csv,
        "specialists": results,
    }
    save_json(output_dir / "specialist_full_true_subsets_metrics.json", payload)
    plot_confusion_matrices(results, output_dir)
    write_captions(output_dir / "fig_specialist_full_true_subsets_captions.md")
    write_comparison(output_dir / "specialist_full_vs_routed_comparison.md", results)

    for result in results:
        metrics = result["full_true_subset"]
        print(
            "%s: n=%d, accuracy=%.4f, balanced_accuracy=%.4f, macro_f1=%.4f"
            % (
                result["key"],
                metrics["n_samples"],
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
            )
        )
    print("Saved full-subset specialist evaluation to: %s" % output_dir)


if __name__ == "__main__":
    main()
