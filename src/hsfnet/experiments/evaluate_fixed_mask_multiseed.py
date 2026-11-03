"""Repeated fixed-mask inference for the final HSFNet-FRB model."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


HSFNET_DIR = Path(__file__).resolve().parents[1]
if str(HSFNET_DIR) not in sys.path:
    sys.path.insert(0, str(HSFNET_DIR))

from data import CLASSES, ensure_dir, paper_half_split_chime, save_json  # noqa: E402
from evaluate_binary_specialist_fusion import fuse_predictions, route_samples  # noqa: E402


METRICS = ("accuracy", "macro_f1", "weighted_f1", "c1_recall", "c2_recall")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated inference for fixed frequency-mask ratios using final HSFNet-FRB models."
    )
    parser.add_argument("--base-model", default="artifacts/models/final/base_resnet50")
    parser.add_argument("--tri-model", default="artifacts/models/final/tri_specialist_b_c1_c2")
    parser.add_argument("--b-c1-model", default="artifacts/models/final/binary_specialist_b_vs_c1")
    parser.add_argument("--c1-c2-model", default="artifacts/models/final/binary_specialist_c1_vs_c2")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="reports/fixed_mask_multiseed_final_hsfnet")
    parser.add_argument("--mask-ratios", default="0.0,0.1,0.2,0.3")
    parser.add_argument("--mask-seeds", default="123,124,125,126,127,128,129,130,131,132")
    parser.add_argument("--c1-vote-bonus", type=float, default=0.15)
    parser.add_argument("--expected-beta", type=float, default=None)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--require-empty-output", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu")
    parser.add_argument("--gpu-id", default="1")
    return parser.parse_args()


def configure_device(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fixed-mask-multiseed")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def validate_selection(selection_path, configured_beta):
    if selection_path is None:
        return None
    path = Path(selection_path)
    if not path.exists():
        raise FileNotFoundError("Validation-only beta selection JSON is missing: %s" % path)
    with path.open("r", encoding="utf-8") as f:
        selection = json.load(f)
    selected_beta = float(selection["selected_beta"])
    if abs(selected_beta - configured_beta) > 1e-12:
        raise ValueError(
            "Selection JSON beta %.8f does not match configured inference beta %.8f."
            % (selected_beta, configured_beta)
        )
    if selection.get("paper_test_metrics_access_before_selection") is not False:
        raise ValueError("Selection JSON does not certify validation-only beta selection.")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selected_beta": selected_beta,
    }


def prediction_dataset(tf, x, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(lambda image: tf.cast(image, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def apply_fixed_mask(x, ratio, seed):
    n_frequency_rows = x.shape[1]
    n_masked = int(round(n_frequency_rows * ratio))
    if n_masked <= 0:
        return np.array(x, copy=True), np.empty((0,), dtype=np.int64)

    rng = np.random.RandomState(seed)
    masked_rows = np.sort(
        rng.choice(np.arange(n_frequency_rows), size=min(n_masked, n_frequency_rows), replace=False)
    ).astype(np.int64)
    masked_x = np.array(x, copy=True)
    masked_x[:, masked_rows, :, :] = 0
    return masked_x, masked_rows


def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros((len(CLASSES),), dtype=np.float64),
        where=cm.sum(axis=1) != 0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "c1_recall": float(recalls[3]),
        "c2_recall": float(recalls[4]),
    }


def run_inference(tf, models, x, y, batch_size, c1_vote_bonus):
    base_model, tri_model, b_c1_model, c1_c2_model = models
    dataset = prediction_dataset(tf, x, batch_size)
    base_probabilities = base_model.predict(dataset, verbose=0)
    base_predictions = np.argmax(base_probabilities, axis=1).astype(np.int64)
    routed = route_samples(base_probabilities, base_predictions, "predicted_subset", 0.18)

    tri_probabilities = np.full((len(x), 3), np.nan, dtype=np.float32)
    b_c1_probabilities = np.full((len(x), 2), np.nan, dtype=np.float32)
    c1_c2_probabilities = np.full((len(x), 2), np.nan, dtype=np.float32)
    if np.any(routed):
        routed_dataset = prediction_dataset(tf, x[routed], batch_size)
        tri_probabilities[routed] = tri_model.predict(routed_dataset, verbose=0)
        b_c1_probabilities[routed] = b_c1_model.predict(routed_dataset, verbose=0)
        c1_c2_probabilities[routed] = c1_c2_model.predict(routed_dataset, verbose=0)

    final_predictions, _ = fuse_predictions(
        base_predictions,
        routed,
        tri_probabilities,
        b_c1_probabilities,
        c1_c2_probabilities,
        c1_vote_bonus=c1_vote_bonus,
    )
    metrics = calculate_metrics(y, final_predictions)
    metrics["n_routed"] = int(np.sum(routed))
    return {
        "metrics": metrics,
        "base_probabilities": base_probabilities,
        "base_predictions": base_predictions,
        "routed": routed,
        "tri_probabilities": tri_probabilities,
        "b_c1_probabilities": b_c1_probabilities,
        "c1_c2_probabilities": c1_c2_probabilities,
        "final_predictions": final_predictions,
    }


def ratio_tag(ratio):
    return ("%.2f" % ratio).replace(".", "p")


def write_run_artifacts(run_dir, run_result, test_y, test_indices, masked_rows, run_config):
    from sklearn.metrics import confusion_matrix

    run_dir = ensure_dir(run_dir)
    base_probabilities = run_result["base_probabilities"]
    base_predictions = run_result["base_predictions"]
    routed = run_result["routed"]
    tri_probabilities = run_result["tri_probabilities"]
    b_c1_probabilities = run_result["b_c1_probabilities"]
    c1_c2_probabilities = run_result["c1_c2_probabilities"]
    final_predictions = run_result["final_predictions"]
    probability_fields = (
        ["base_prob_" + name for name in CLASSES]
        + ["tri_prob_B", "tri_prob_C1", "tri_prob_C2"]
        + ["bc1_prob_B", "bc1_prob_C1"]
        + ["c1c2_prob_C1", "c1c2_prob_C2"]
    )
    fields = [
        "sample_position",
        "chime_global_index",
        "true_label",
        "true_class",
        "base_pred",
        "base_pred_class",
        "routed",
        "final_pred",
        "final_pred_class",
        *probability_fields,
    ]
    prediction_path = run_dir / "predictions_and_component_probabilities.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for position in range(len(test_y)):
            expert_values = np.concatenate(
                [
                    tri_probabilities[position],
                    b_c1_probabilities[position],
                    c1_c2_probabilities[position],
                ]
            )
            probability_values = list(base_probabilities[position]) + list(expert_values)
            row = {
                "sample_position": int(position),
                "chime_global_index": int(test_indices[position]),
                "true_label": int(test_y[position]),
                "true_class": CLASSES[int(test_y[position])],
                "base_pred": int(base_predictions[position]),
                "base_pred_class": CLASSES[int(base_predictions[position])],
                "routed": int(routed[position]),
                "final_pred": int(final_predictions[position]),
                "final_pred_class": CLASSES[int(final_predictions[position])],
            }
            for field, value in zip(probability_fields, probability_values):
                row[field] = "" if np.isnan(value) else "%.9f" % float(value)
            writer.writerow(row)

    cm = confusion_matrix(test_y, final_predictions, labels=list(range(len(CLASSES))))
    np.savetxt(str(run_dir / "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    save_json(run_dir / "metrics.json", {**run_config, **run_result["metrics"], "confusion_matrix": cm.tolist()})
    save_json(
        run_dir / "mask_channel_indices.json",
        {
            "mask_ratio": float(run_config["mask_ratio"]),
            "mask_seed": int(run_config["mask_seed"]),
            "n_masked_frequency_rows": int(len(masked_rows)),
            "masked_frequency_rows": [int(value) for value in masked_rows],
        },
    )
    return prediction_path


def validate_run_artifacts(prediction_path, expected_test_indices):
    with Path(prediction_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 268:
        raise ValueError("Prediction file must contain 268 rows: %s" % prediction_path)
    positions = [int(row["sample_position"]) for row in rows]
    indices = [int(row["chime_global_index"]) for row in rows]
    if positions != list(range(268)):
        raise ValueError("Test sample order changed in %s" % prediction_path)
    if indices != [int(value) for value in expected_test_indices]:
        raise ValueError("Global CHIME index order changed in %s" % prediction_path)

    max_base_error = 0.0
    max_expert_error = 0.0
    routed_count = 0
    for row in rows:
        base_sum = sum(float(row["base_prob_" + name]) for name in CLASSES)
        max_base_error = max(max_base_error, abs(base_sum - 1.0))
        if int(row["routed"]) == 1:
            routed_count += 1
            groups = (
                ("tri_prob_B", "tri_prob_C1", "tri_prob_C2"),
                ("bc1_prob_B", "bc1_prob_C1"),
                ("c1c2_prob_C1", "c1c2_prob_C2"),
            )
            for group in groups:
                if any(row[field] == "" for field in group):
                    raise ValueError("Missing routed expert probability in %s" % prediction_path)
                max_expert_error = max(
                    max_expert_error,
                    abs(sum(float(row[field]) for field in group) - 1.0),
                )
        else:
            expert_fields = (
                "tri_prob_B",
                "tri_prob_C1",
                "tri_prob_C2",
                "bc1_prob_B",
                "bc1_prob_C1",
                "c1c2_prob_C1",
                "c1c2_prob_C2",
            )
            if any(row[field] != "" for field in expert_fields):
                raise ValueError("Non-routed sample contains expert probabilities in %s" % prediction_path)
    return {
        "prediction_rows": 268,
        "n_routed": int(routed_count),
        "max_base_probability_sum_error": float(max_base_error),
        "max_expert_probability_sum_error": float(max_expert_error),
    }


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_results, mask_ratios, n_repeats):
    summary = []
    for ratio in mask_ratios:
        selected = [row for row in raw_results if row["mask_ratio"] == ratio]
        if len(selected) != n_repeats:
            raise ValueError("Mask ratio %.3f has %d runs; expected %d" % (ratio, len(selected), n_repeats))
        row = {
            "mask_ratio": float(ratio),
            "n_repeats": int(n_repeats),
            "std_definition": "sample standard deviation (ddof=1)",
        }
        for metric in METRICS:
            values = np.asarray([result[metric] for result in selected], dtype=np.float64)
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1))
        routed_values = np.asarray([result["n_routed"] for result in selected], dtype=np.float64)
        row["n_routed_mean"] = float(np.mean(routed_values))
        row["n_routed_std"] = float(np.std(routed_values, ddof=1))
        summary.append(row)
    return summary


def plot_results(raw_results, summary, output_dir, beta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    ratios = np.asarray([row["mask_ratio"] for row in summary], dtype=np.float64)
    seed_order = sorted(set(int(row["mask_seed"]) for row in raw_results))
    jitter_by_seed = {
        seed: offset for seed, offset in zip(seed_order, np.linspace(-0.008, 0.008, len(seed_order)))
    }
    colors = {
        "accuracy": "#2F5D8C",
        "macro_f1": "#C44E52",
        "weighted_f1": "#4C8C6A",
        "c1_recall": "#8A6BBE",
        "c2_recall": "#D28E45",
        "n_routed": "#555555",
    }
    labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro-F1",
        "weighted_f1": "Weighted-F1",
        "c1_recall": "C1 recall",
        "c2_recall": "C2 recall",
        "n_routed": "Routed samples",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    panels = (
        (axes[0], ("accuracy", "macro_f1", "weighted_f1"), "Overall performance", "Score"),
        (axes[1], ("c1_recall", "c2_recall"), "Minority-class recall", "Recall"),
        (axes[2], ("n_routed",), "Routing load", "Number of samples"),
    )
    for axis, metrics, title, ylabel in panels:
        for metric in metrics:
            for result in raw_results:
                x_value = result["mask_ratio"] + jitter_by_seed[int(result["mask_seed"])]
                axis.scatter(
                    x_value,
                    result[metric],
                    s=17,
                    color=colors[metric],
                    alpha=0.42,
                    edgecolors="none",
                    zorder=2,
                )
            mean_key = metric + "_mean"
            std_key = metric + "_std"
            means = np.asarray([row[mean_key] for row in summary], dtype=np.float64)
            stds = np.asarray([row[std_key] for row in summary], dtype=np.float64)
            axis.plot(
                ratios,
                means,
                marker="o",
                markersize=4.2,
                linewidth=1.8,
                color=colors[metric],
                label=labels[metric],
                zorder=3,
            )
            axis.fill_between(
                ratios,
                means - stds,
                means + stds,
                color=colors[metric],
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )
        axis.set_title(title)
        axis.set_xlabel("Fixed mask ratio")
        axis.set_ylabel(ylabel)
        axis.set_xticks(ratios)
        axis.set_xlim(-0.02, 0.32)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="best", title="Dots: runs; band: ±1 SD")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[2].set_ylim(0.0, 268.0)
    fig.suptitle("Final HSFNet-FRB fixed-mask robustness (validation-selected β=%.2f)" % beta, y=1.02)
    fig.tight_layout()

    beta_tag = "%03d" % int(round(beta * 100.0))
    figure_stem = "fig_fixed_mask_multiseed_beta%s" % beta_tag
    stem = Path(output_dir) / figure_stem
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [figure_stem + ".pdf", figure_stem + ".svg", figure_stem + ".png"]


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

    mask_ratios = parse_float_list(args.mask_ratios)
    mask_seeds = parse_int_list(args.mask_seeds)
    if mask_ratios != [0.0, 0.1, 0.2, 0.3]:
        raise ValueError("This experiment requires mask ratios 0.0, 0.1, 0.2, 0.3 exactly.")
    if len(mask_seeds) != 10 or len(set(mask_seeds)) != 10:
        raise ValueError("This experiment requires exactly 10 distinct fixed mask seeds.")
    if args.expected_beta is not None and abs(args.c1_vote_bonus - args.expected_beta) > 1e-12:
        raise ValueError(
            "Configured beta %.8f does not match required beta %.8f."
            % (args.c1_vote_bonus, args.expected_beta)
        )
    selection_record = validate_selection(args.selection_json, args.c1_vote_bonus)

    output_path = Path(args.output_dir)
    if args.require_empty_output and output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError("Output directory must be empty to avoid mixing old results: %s" % output_path)
    output_dir = ensure_dir(output_path)
    _, _, test_x, test_y, _, test_indices = paper_half_split_chime(args.chime_data, args.chime_labels)
    if len(test_y) != 268:
        raise ValueError("Expected 268 ordered CHIME paper-test samples, found %d." % len(test_y))

    model_paths = {
        "base": args.base_model,
        "tri_specialist": args.tri_model,
        "b_vs_c1_specialist": args.b_c1_model,
        "c1_vs_c2_specialist": args.c1_c2_model,
    }
    models = (
        keras.models.load_model(args.base_model),
        keras.models.load_model(args.tri_model),
        keras.models.load_model(args.b_c1_model),
        keras.models.load_model(args.c1_c2_model),
    )

    raw_results = []
    mask_rows = []
    completed_runs = []
    failed_runs = []
    run_index = 0
    for ratio in mask_ratios:
        for seed in mask_seeds:
            run_index += 1
            try:
                masked_x, rows = apply_fixed_mask(test_x, ratio, seed)
                inference = run_inference(
                    tf,
                    models,
                    masked_x,
                    test_y,
                    args.batch_size,
                    args.c1_vote_bonus,
                )
                metrics = inference["metrics"]
                result = {
                    "run_index": int(run_index),
                    "mask_ratio": float(ratio),
                    "mask_seed": int(seed),
                    "n_masked_frequency_rows": int(len(rows)),
                    **metrics,
                }
                run_dir = output_dir / "runs" / (
                    "mask_%s_seed_%d" % (ratio_tag(ratio), seed)
                )
                prediction_path = write_run_artifacts(
                    run_dir,
                    inference,
                    test_y,
                    test_indices,
                    rows,
                    {
                        "run_index": int(run_index),
                        "mask_ratio": float(ratio),
                        "mask_seed": int(seed),
                        "c1_vote_bonus": float(args.c1_vote_bonus),
                        "route_mode": "predicted_subset",
                        "n_test_samples": 268,
                    },
                )
                artifact_check = validate_run_artifacts(prediction_path, test_indices)
                if artifact_check["n_routed"] != metrics["n_routed"]:
                    raise ValueError("Routed count mismatch for run %d" % run_index)
                raw_results.append(result)
                mask_rows.append(
                    {
                        "run_index": int(run_index),
                        "mask_ratio": float(ratio),
                        "mask_seed": int(seed),
                        "masked_frequency_rows": [int(value) for value in rows],
                    }
                )
                completed_runs.append(
                    {
                        **result,
                        "run_dir": str(run_dir),
                        "predictions_file": str(prediction_path),
                        "artifact_check": artifact_check,
                    }
                )
                print(
                    "[%02d/40] mask=%.2f seed=%d accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f C1=%.3f C2=%.3f routed=%d"
                    % (
                        run_index,
                        ratio,
                        seed,
                        metrics["accuracy"],
                        metrics["macro_f1"],
                        metrics["weighted_f1"],
                        metrics["c1_recall"],
                        metrics["c2_recall"],
                        metrics["n_routed"],
                    )
                )
            except Exception as exc:
                failed_runs.append(
                    {
                        "run_index": int(run_index),
                        "mask_ratio": float(ratio),
                        "mask_seed": int(seed),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print("[FAILED %02d/40] mask=%.2f seed=%d: %s" % (run_index, ratio, seed, exc))

    raw_fields = [
        "run_index",
        "mask_ratio",
        "mask_seed",
        "n_masked_frequency_rows",
        "n_routed",
        *METRICS,
    ]
    write_csv(output_dir / "raw_results.csv", raw_fields, raw_results)
    with (output_dir / "fixed_mask_rows.json").open("w", encoding="utf-8") as f:
        json.dump(mask_rows, f, indent=2, sort_keys=True)
    save_json(output_dir / "failed_runs.json", {"failed_runs": failed_runs})

    if failed_runs or len(raw_results) != 40:
        integrity = {
            "status": "failed",
            "expected_runs": 40,
            "completed_runs": int(len(raw_results)),
            "failed_run_count": int(len(failed_runs)),
            "failed_runs": failed_runs,
        }
        save_json(output_dir / "integrity_report.json", integrity)
        raise RuntimeError("Fixed-mask experiment has failed or missing runs; see integrity_report.json")

    summary = summarize(raw_results, mask_ratios, len(mask_seeds))
    summary_fields = ["mask_ratio", "n_repeats", "std_definition"]
    for metric in METRICS:
        summary_fields.extend([metric + "_mean", metric + "_std"])
    summary_fields.extend(["n_routed_mean", "n_routed_std"])

    write_csv(output_dir / "summary.csv", summary_fields, summary)
    figure_files = plot_results(raw_results, summary, output_dir, args.c1_vote_bonus)

    run_keys = {(row["mask_ratio"], row["mask_seed"]) for row in raw_results}
    expected_keys = {(ratio, seed) for ratio in mask_ratios for seed in mask_seeds}
    integrity = {
        "status": "complete" if run_keys == expected_keys else "failed",
        "expected_runs": 40,
        "completed_runs": int(len(raw_results)),
        "failed_run_count": int(len(failed_runs)),
        "duplicate_run_count": int(len(raw_results) - len(run_keys)),
        "missing_runs": [
            {"mask_ratio": float(ratio), "mask_seed": int(seed)}
            for ratio, seed in sorted(expected_keys - run_keys)
        ],
        "test_order_sha256": hashlib.sha256(
            np.asarray(test_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "completed_run_artifact_checks": completed_runs,
    }
    save_json(output_dir / "integrity_report.json", integrity)
    if integrity["status"] != "complete":
        raise RuntimeError("Integrity audit failed; see integrity_report.json")

    payload = {
        "experiment": "final HSFNet-FRB repeated fixed-mask inference",
        "inference_only": True,
        "dataset": "CHIME paper-test half",
        "n_test_samples": int(len(test_y)),
        "test_sample_order": "unchanged paper_half_split_chime order",
        "test_indices_sha256": hashlib.sha256(np.asarray(test_indices, dtype=np.int64).tobytes()).hexdigest(),
        "model_paths": model_paths,
        "route_mode": "predicted_subset",
        "routing_logic": "unchanged final HSFNet-FRB routing",
        "beta_c1_vote_bonus": float(args.c1_vote_bonus),
        "validation_beta_selection": selection_record,
        "mask_ratios": mask_ratios,
        "mask_seeds": mask_seeds,
        "n_repeats_per_ratio": int(len(mask_seeds)),
        "standard_deviation": "sample standard deviation (ddof=1)",
        "raw_results": raw_results,
        "summary": summary,
        "mask_rows_file": "fixed_mask_rows.json",
        "predictions_and_probabilities": "runs/*/predictions_and_component_probabilities.csv",
        "figure_files": figure_files,
        "integrity_report": "integrity_report.json",
        "failed_runs_file": "failed_runs.json",
    }
    save_json(output_dir / "results.json", payload)
    print("Saved repeated fixed-mask results to: %s" % output_dir)


if __name__ == "__main__":
    main()
