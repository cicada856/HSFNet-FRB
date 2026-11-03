#!/usr/bin/env python
"""Leakage-controlled uncertainty-aware hard-routing ablation.

The script reuses the project's existing fusion functions and model interfaces.
Thresholds are selected on development-only OOF predictions. The locked
paper-test evaluation is run only after that selection and is repeated across
fixed inference seeds to document that routing has no training randomness.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HSFNET_SRC = PROJECT_ROOT / "src" / "hsfnet"
sys.path.insert(0, str(HSFNET_SRC))

from data import CLASS_TO_INDEX, CLASSES, load_npz_array, normalize_images_to_uint8  # noqa: E402
from evaluate_binary_specialist_fusion import fuse_predictions  # noqa: E402


LOCAL_NAMES = ("B", "C1", "C2")
LOCAL_INDICES = np.asarray([CLASS_TO_INDEX[name] for name in LOCAL_NAMES], dtype=np.int64)
LOCAL_SET = set(int(value) for value in LOCAL_INDICES)
STRATEGIES = ("top1_current", "top2_contains_local", "low_confidence", "small_margin")
INFERENCE_SEEDS = (123, 124, 125)
BETA = 0.20


def parse_args():
    parser = argparse.ArgumentParser(description="Run uncertainty-aware hard-routing ablation.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "routing_ablation"))
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inference-seeds", default="123,124,125")
    return parser.parse_args()


def configure_device(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-routing-ablation")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_dataset(tf, images, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(lambda image: tf.cast(image, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_chime_split():
    from data import paper_half_split_chime

    data_path = PROJECT_ROOT / "data/chime/chime_interp_frbs_data.npz"
    labels_path = PROJECT_ROOT / "data/chime/chime_labels.txt"
    train_x, train_y, test_x, test_y, train_idx, test_idx = paper_half_split_chime(data_path, labels_path)
    return train_x, train_y, test_x, test_y, train_idx, test_idx


def load_oof_rows(path):
    expected = (
        "sample_index", "fold", "true_label", "base_pred", "tri_prob_B", "tri_prob_C1",
        "tri_prob_C2", "bc1_prob_B", "bc1_prob_C1", "c1c2_prob_C1", "c1c2_prob_C2",
    )
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("Unexpected OOF schema in %s" % path)
        rows = list(reader)
    if len(rows) != 267 or len({int(row["sample_index"]) for row in rows}) != 267:
        raise ValueError("Expected 267 unique development OOF rows")
    return rows


def load_oof_base_probabilities(tf, keras, development_images_by_global, rows, batch_size):
    """Recover base probabilities from the existing fold models, never paper-test data."""
    base_probs = np.full((len(rows), len(CLASSES)), np.nan, dtype=np.float32)
    row_by_index = {int(row["sample_index"]): index for index, row in enumerate(rows)}
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        indices = np.asarray([int(row["sample_index"]) for row in fold_rows], dtype=np.int64)
        model_path = PROJECT_ROOT / "artifacts/runs/validation_beta_selection" / ("fold_%d/base/best_model" % fold)
        model = keras.models.load_model(str(model_path))
        fold_images = np.asarray([development_images_by_global[int(index)] for index in indices])
        probabilities = model.predict(prediction_dataset(tf, fold_images, batch_size), verbose=0)
        for local_index, sample_index in enumerate(indices):
            base_probs[row_by_index[int(sample_index)]] = probabilities[local_index]
        del model
    if not np.all(np.isfinite(base_probs)):
        raise RuntimeError("Failed to recover complete OOF base probabilities")
    return base_probs


def specialist_arrays_from_rows(rows):
    tri = np.asarray([[float(row["tri_prob_B"]), float(row["tri_prob_C1"]), float(row["tri_prob_C2"])] for row in rows], dtype=np.float32)
    bc1 = np.asarray([[float(row["bc1_prob_B"]), float(row["bc1_prob_C1"])] for row in rows], dtype=np.float32)
    c1c2 = np.asarray([[float(row["c1c2_prob_C1"]), float(row["c1c2_prob_C2"])] for row in rows], dtype=np.float32)
    return tri, bc1, c1c2


def route_mask(strategy, base_probs, threshold=None):
    base_pred = np.argmax(base_probs, axis=1)
    if strategy == "top1_current":
        return np.asarray([int(value) in LOCAL_SET for value in base_pred], dtype=bool)
    order = np.argsort(-base_probs, axis=1, kind="mergesort")
    if strategy == "top2_contains_local":
        return np.asarray([bool(set(order[index, :2].tolist()) & LOCAL_SET) for index in range(len(base_pred))], dtype=bool)
    max_probability = np.max(base_probs, axis=1)
    if strategy == "low_confidence":
        return max_probability < float(threshold)
    margin = base_probs[np.arange(len(base_probs)), order[:, 0]] - base_probs[np.arange(len(base_probs)), order[:, 1]]
    if strategy == "small_margin":
        return margin < float(threshold)
    raise ValueError("Unknown strategy: %s" % strategy)


def predictions_for_route(base_probs, tri, bc1, c1c2, routed):
    base_pred = np.argmax(base_probs, axis=1).astype(np.int64)
    final_pred, diagnostics = fuse_predictions(base_pred, routed, tri, bc1, c1c2, c1_vote_bonus=BETA)
    return base_pred, final_pred, diagnostics


def basic_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0)
    recalls = {
        name: float(cm[index, index] / cm[index].sum()) if cm[index].sum() else 0.0
        for index, name in enumerate(CLASSES)
    }
    local = np.isin(y_true, LOCAL_INDICES)
    local_report = classification_report(
        y_true[local], y_pred[local], labels=LOCAL_INDICES.tolist(), target_names=LOCAL_NAMES, output_dict=True, zero_division=0
    )
    local_accuracy = float(np.mean(y_pred[local] == y_true[local])) if np.any(local) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_recall": recalls,
        "local_subset_n": int(np.sum(local)),
        "local_subset_accuracy": local_accuracy,
        "local_subset_macro_f1": float(local_report["macro avg"]["f1-score"]),
        "local_subset_weighted_f1": float(local_report["weighted avg"]["f1-score"]),
        "local_subset_recall": {name: float(local_report[name]["recall"]) for name in LOCAL_NAMES},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def routing_diagnostics(y_true, base_pred, final_pred, routed, baseline_final_pred):
    true_local = np.isin(y_true, LOCAL_INDICES)
    true_ac = np.isin(y_true, [CLASS_TO_INDEX["A"], CLASS_TO_INDEX["C"]])
    sent_ac = routed & true_ac
    base_correct_ac = sent_ac & (base_pred == y_true)
    changed = final_pred != baseline_final_pred
    routed_base_ac = routed & np.isin(base_pred, [CLASS_TO_INDEX["A"], CLASS_TO_INDEX["C"]])
    return {
        "specialist_invocation_rate": float(np.mean(routed)),
        "n_routed": int(np.sum(routed)),
        "prediction_flips_vs_top1_current": int(np.sum(changed)),
        "prediction_flip_rate_vs_top1_current": float(np.mean(changed)),
        "a_c_true_samples_sent_to_specialist": int(np.sum(sent_ac)),
        "a_c_true_samples_sent_rate": float(np.mean(sent_ac[true_ac])) if np.any(true_ac) else 0.0,
        "a_c_base_correct_sent_to_specialist": int(np.sum(base_correct_ac)),
        "a_c_base_correct_became_wrong": int(np.sum(base_correct_ac & (final_pred != y_true))),
        "a_c_wrongly_sent_and_finally_wrong": int(np.sum(sent_ac & (final_pred != y_true))),
        "a_c_wrongly_sent_final_wrong_rate": float(np.mean((final_pred[sent_ac] != y_true[sent_ac]))) if np.any(sent_ac) else 0.0,
        "a_c_base_prediction_local_but_true_a_c": int(np.sum(routed_base_ac & true_ac)),
        "true_local_sample_count": int(np.sum(true_local)),
    }


def evaluate_strategy(y_true, base_probs, tri, bc1, c1c2, strategy, threshold, baseline_final_pred):
    routed = route_mask(strategy, base_probs, threshold)
    base_pred, final_pred, diagnostics = predictions_for_route(base_probs, tri, bc1, c1c2, routed)
    result = {
        "strategy": strategy,
        "threshold": None if threshold is None else float(threshold),
        "beta": BETA,
        "metrics": basic_metrics(y_true, final_pred),
        "routing": routing_diagnostics(y_true, base_pred, final_pred, routed, baseline_final_pred),
        "base_accuracy": float(np.mean(base_pred == y_true)),
        "n_samples": int(len(y_true)),
    }
    return result, routed, base_pred, final_pred, diagnostics


def choose_threshold(y_true, base_probs, tri, bc1, c1c2, strategy, baseline_final_pred):
    if strategy == "low_confidence":
        candidates = np.linspace(0.05, 0.95, 91)
    elif strategy == "small_margin":
        candidates = np.linspace(0.01, 0.95, 95)
    else:
        return None, []
    sweep = []
    for threshold in candidates:
        result, _, _, _, _ = evaluate_strategy(y_true, base_probs, tri, bc1, c1c2, strategy, threshold, baseline_final_pred)
        metrics = result["metrics"]
        sweep.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "specialist_invocation_rate": result["routing"]["specialist_invocation_rate"],
                "n_routed": result["routing"]["n_routed"],
            }
        )
    selected = sorted(sweep, key=lambda row: (-row["macro_f1"], -row["accuracy"], row["n_routed"], row["threshold"]))[0]
    return float(selected["threshold"]), sweep


def write_matrix(path, matrix):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + list(CLASSES))
        for name, row in zip(CLASSES, matrix):
            writer.writerow([name] + row)


def write_prediction_csv(path, names, y_true, base_pred, baseline_pred, strategy_data):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "name", "true_label", "base_pred", "top1_current_pred"] + ["%s_pred" % name for name in STRATEGIES] + ["%s_routed" % name for name in STRATEGIES])
        for index in range(len(names)):
            writer.writerow(
                [index, names[index], CLASSES[int(y_true[index])], CLASSES[int(base_pred[index])], CLASSES[int(baseline_pred[index])]]
                + [CLASSES[int(strategy_data[name]["final_pred"][index])] for name in STRATEGIES]
                + [int(strategy_data[name]["routed"][index]) for name in STRATEGIES]
            )


def main():
    args = parse_args()
    configure_device(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value.strip()) for value in args.inference_seeds.split(",") if value.strip())
    if len(seeds) < 3:
        raise ValueError("At least three inference seeds are required")

    import tensorflow as tf
    from tensorflow import keras

    import losses  # noqa: F401,E402

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    train_x, train_y, test_x, test_y, train_idx, test_idx = load_chime_split()
    oof_path = PROJECT_ROOT / "reports/validation_beta_selection/oof_component_predictions.csv"
    oof_rows = load_oof_rows(oof_path)
    # OOF sample_index is the original global CHIME index, so index by lookup.
    x_by_global_index = {int(index): train_x[position] for position, index in enumerate(train_idx)}
    oof_base_probs = load_oof_base_probabilities(tf, keras, x_by_global_index, oof_rows, args.batch_size)
    oof_tri, oof_bc1, oof_c1c2 = specialist_arrays_from_rows(oof_rows)
    oof_y = np.asarray([int(row["true_label"]) for row in oof_rows], dtype=np.int64)

    # Current beta=0.20 top-1 routing is the only baseline used for comparison.
    oof_base_pred = np.argmax(oof_base_probs, axis=1)
    oof_top1_routed = route_mask("top1_current", oof_base_probs)
    _, oof_top1_final, _ = predictions_for_route(oof_base_probs, oof_tri, oof_bc1, oof_c1c2, oof_top1_routed)
    thresholds = {"top1_current": None, "top2_contains_local": None}
    threshold_sweeps = {}
    for strategy in ("low_confidence", "small_margin"):
        thresholds[strategy], threshold_sweeps[strategy] = choose_threshold(
            oof_y, oof_base_probs, oof_tri, oof_bc1, oof_c1c2, strategy, oof_top1_final
        )

    config = {
        "experiment": "uncertainty-aware hard-routing ablation",
        "output_dir": str(output_dir),
        "project_root": str(PROJECT_ROOT),
        "strategies": list(STRATEGIES),
        "local_classes": list(LOCAL_NAMES),
        "beta": BETA,
        "beta_source": str(PROJECT_ROOT / "reports/validation_beta_selection/validation_bonus_selection.json"),
        "beta_source_sha256": sha256(PROJECT_ROOT / "reports/validation_beta_selection/validation_bonus_selection.json"),
        "threshold_selection": {
            "dataset": "267-sample CHIME development half only",
            "source_oof": str(oof_path),
            "source_oof_sha256": sha256(oof_path),
            "selection_rule": "maximize development OOF macro-F1; tie by accuracy, lower invocation count, then smaller threshold",
            "thresholds": thresholds,
            "top2_contains_local_has_no_threshold": True,
            "paper_test_access_before_selection": False,
        },
        "paper_test_protocol": "All strategies and thresholds fixed before a single locked final evaluation stage; repeated fixed-model inference seeds document deterministic routing.",
        "model_paths": {
            "base": str(PROJECT_ROOT / "artifacts/models/final/base_resnet50"),
            "tri_specialist": str(PROJECT_ROOT / "artifacts/models/final/tri_specialist_b_c1_c2"),
            "b_vs_c1": str(PROJECT_ROOT / "artifacts/models/final/binary_specialist_b_vs_c1"),
            "c1_vs_c2": str(PROJECT_ROOT / "artifacts/models/final/binary_specialist_c1_vs_c2"),
        },
        "inference_seeds": list(seeds),
    }
    save_json(output_dir / "config_locked.json", config)
    for strategy, sweep in threshold_sweeps.items():
        with (output_dir / ("development_%s_sweep.csv" % strategy)).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep[0].keys()))
            writer.writeheader()
            writer.writerows(sweep)

    development_results = {}
    for strategy in STRATEGIES:
        result, routed, base_pred, final_pred, diagnostics = evaluate_strategy(
            oof_y, oof_base_probs, oof_tri, oof_bc1, oof_c1c2, strategy, thresholds[strategy], oof_top1_final
        )
        development_results[strategy] = result
        save_json(output_dir / ("development_%s.json" % strategy), result)
    save_json(output_dir / "development_results.json", development_results)

    test_names_all = read_lines(PROJECT_ROOT / "data/chime/chime_file_order.txt")
    test_names = [Path(test_names_all[int(index)]).name.split("_waterfall", 1)[0] for index in test_idx]

    base_model = keras.models.load_model(str(PROJECT_ROOT / "artifacts/models/final/base_resnet50"))
    tri_model = keras.models.load_model(str(PROJECT_ROOT / "artifacts/models/final/tri_specialist_b_c1_c2"))
    bc1_model = keras.models.load_model(str(PROJECT_ROOT / "artifacts/models/final/binary_specialist_b_vs_c1"))
    c1c2_model = keras.models.load_model(str(PROJECT_ROOT / "artifacts/models/final/binary_specialist_c1_vs_c2"))
    test_dataset = prediction_dataset(tf, test_x, args.batch_size)
    test_base_probs = base_model.predict(test_dataset, verbose=1)
    test_tri = tri_model.predict(test_dataset, verbose=1)
    test_bc1 = bc1_model.predict(test_dataset, verbose=1)
    test_c1c2 = c1c2_model.predict(test_dataset, verbose=1)
    test_base_pred = np.argmax(test_base_probs, axis=1).astype(np.int64)
    test_top1_routed = route_mask("top1_current", test_base_probs)
    _, test_top1_final, _ = predictions_for_route(test_base_probs, test_tri, test_bc1, test_c1c2, test_top1_routed)

    all_seed_results = []
    canonical_strategy_data = None
    for seed in seeds:
        np.random.seed(seed)
        seed_dir = output_dir / "paper_test" / ("seed_%d" % seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_results = {}
        strategy_data = {}
        for strategy in STRATEGIES:
            result, routed, base_pred, final_pred, diagnostics = evaluate_strategy(
                test_y, test_base_probs, test_tri, test_bc1, test_c1c2, strategy, thresholds[strategy], test_top1_final
            )
            seed_results[strategy] = result
            strategy_data[strategy] = {"routed": routed, "base_pred": base_pred, "final_pred": final_pred, "diagnostics": diagnostics}
            write_matrix(seed_dir / ("%s_confusion_matrix.csv" % strategy), result["metrics"]["confusion_matrix"])
            save_json(seed_dir / ("%s_metrics.json" % strategy), result)
        write_prediction_csv(seed_dir / "predictions.csv", test_names, test_y, test_base_pred, test_top1_final, strategy_data)
        save_json(seed_dir / "seed_metadata.json", {"seed": seed, "training_randomness": False, "fixed_model_inference": True})
        all_seed_results.append({"seed": seed, "results": seed_results})
        if canonical_strategy_data is None:
            canonical_strategy_data = strategy_data

    # The three fixed-model seed runs must be identical; this is an explicit audit.
    deterministic = True
    for strategy in STRATEGIES:
        reference = all_seed_results[0]["results"][strategy]
        for run in all_seed_results[1:]:
            if run["results"][strategy]["metrics"] != reference["metrics"] or run["results"][strategy]["routing"] != reference["routing"]:
                deterministic = False
    final_payload = {
        "evaluation_type": "locked_paper_test_final",
        "n_samples": int(len(test_y)),
        "paper_test_access_stage": "after_threshold_selection_only",
        "inference_seeds": list(seeds),
        "fixed_model_results_identical_across_seeds": deterministic,
        "results_by_seed": all_seed_results,
        "canonical_seed": int(seeds[0]),
        "reference_paper_result": {
            "source": str(PROJECT_ROOT / "reports/fixed_mask_multiseed_final_hsfnet_beta020/results.json"),
            "accuracy": 0.6343283582089553,
            "macro_f1": 0.5334796516257037,
            "weighted_f1": 0.640265178642401,
            "c1_recall": 0.3,
            "c2_recall": 0.5,
        },
    }
    save_json(output_dir / "paper_test_results.json", final_payload)
    save_json(output_dir / "thresholds.json", {"beta": BETA, "thresholds": thresholds, "selection_source": str(oof_path)})

    current = all_seed_results[0]["results"]["top1_current"]
    lines = [
        "# Routing Ablation Summary",
        "",
        "## Protocol",
        "",
        "This experiment reuses the existing HSFNet-FRB base classifier, tri-specialist, binary specialists, and `fuse_predictions` interface. Beta is fixed at 0.20 from the existing development-only OOF selection. Low-confidence and small-margin thresholds are selected only on the 267-sample CHIME development half using existing validation OOF predictions. The paper-test split is evaluated only after all choices are locked.",
        "",
        "The three inference seeds are bookkeeping repeats with fixed saved models and no stochastic augmentation or training. Identical outputs across seeds are expected and are reported as a determinism audit.",
        "",
        "## Locked Thresholds",
        "",
        "| Strategy | Threshold | Development invocation rate | Development Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        dev = development_results[strategy]
        threshold = "n/a" if thresholds[strategy] is None else "%.2f" % thresholds[strategy]
        lines.append("| %s | %s | %.3f | %.4f |" % (strategy, threshold, dev["routing"]["specialist_invocation_rate"], dev["metrics"]["macro_f1"]))
    lines += [
        "",
        "## Paper-Test Results",
        "",
        "| Strategy | Accuracy | Balanced Acc. | Macro-F1 | Weighted-F1 | A recall | B recall | C recall | C1 recall | C2 recall | Invocation | Flips vs Top-1 | Local subset Acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        result = all_seed_results[0]["results"][strategy]
        metrics = result["metrics"]
        routing = result["routing"]
        lines.append(
            "| %s | %.4f | %.4f | %.4f | %.4f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %.4f |"
            % (
                strategy, metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"], metrics["weighted_f1"],
                metrics["per_class_recall"]["A"], metrics["per_class_recall"]["B"], metrics["per_class_recall"]["C"],
                metrics["per_class_recall"]["C1"], metrics["per_class_recall"]["C2"], routing["specialist_invocation_rate"],
                routing["prediction_flips_vs_top1_current"], metrics["local_subset_accuracy"],
            )
        )
    lines += [
        "",
        "## A/C Misrouting Diagnostics",
        "",
        "For every strategy, `a_c_true_samples_sent_to_specialist` counts true A/C examples routed into the local expert. `a_c_base_correct_became_wrong` is the strict harmful-override count: base classifier was correct, but specialist fusion changed it to an incorrect local label. `a_c_wrongly_sent_and_finally_wrong` counts all routed true A/C examples that remain incorrect after fusion.",
        "",
        "| Strategy | True A/C sent | Base-correct A/C became wrong | Routed A/C finally wrong |",
        "|---|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        routing = all_seed_results[0]["results"][strategy]["routing"]
        lines.append("| %s | %d | %d | %d |" % (strategy, routing["a_c_true_samples_sent_to_specialist"], routing["a_c_base_correct_became_wrong"], routing["a_c_wrongly_sent_and_finally_wrong"]))
    lines += [
        "",
        "## Comparison With Existing Paper Experiment",
        "",
        "The existing final beta=0.20 report gives Accuracy 0.6343, Macro-F1 0.5335, Weighted-F1 0.6403, C1 recall 0.300, and C2 recall 0.500 on the 268-sample paper-test split. The `top1_current` row above is the same locked model/fusion logic recomputed by this ablation script; small numerical differences would indicate an interface or preprocessing mismatch and should be investigated.",
        "",
        "Current Top-1 result in this run: Accuracy %.4f, Balanced Accuracy %.4f, Macro-F1 %.4f, Weighted-F1 %.4f, invocation %.3f." % (
            current["metrics"]["accuracy"], current["metrics"]["balanced_accuracy"], current["metrics"]["macro_f1"], current["metrics"]["weighted_f1"], current["routing"]["specialist_invocation_rate"]
        ),
        "",
        "## Recommendation",
        "",
    ]
    alternatives = [all_seed_results[0]["results"][strategy] for strategy in STRATEGIES if strategy != "top1_current"]
    best = max(alternatives, key=lambda result: (result["metrics"]["macro_f1"], result["metrics"]["balanced_accuracy"], result["metrics"]["accuracy"]))
    best_name = best["strategy"]
    if best["metrics"]["macro_f1"] > current["metrics"]["macro_f1"] and best["metrics"]["balanced_accuracy"] >= current["metrics"]["balanced_accuracy"]:
        lines.append("On the locked paper-test comparison, `%s` is the strongest candidate by Macro-F1 with no balanced-accuracy loss. It is a candidate replacement, but the small paper-test sample and local-label provenance mean the change should be treated as an ablation result rather than a definitive deployment decision." % best_name)
    else:
        lines.append("No uncertainty-aware alternative improves the locked paper-test result sufficiently over `top1_current` on the primary Macro-F1 and balanced-accuracy criteria. The current Top-1 routing should remain the final system; the alternatives are useful diagnostics but are not worth replacing it.")
    lines += [
        "",
        "All quantitative outputs, per-strategy confusion matrices, development threshold sweeps, locked configuration, and per-seed predictions are stored under this directory.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Routing ablation complete: %s" % output_dir)
    print("Locked thresholds: %s" % thresholds)
    for strategy in STRATEGIES:
        result = all_seed_results[0]["results"][strategy]
        print("%s accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f invocation=%.3f" % (strategy, result["metrics"]["accuracy"], result["metrics"]["balanced_accuracy"], result["metrics"]["macro_f1"], result["routing"]["specialist_invocation_rate"]))


if __name__ == "__main__":
    main()
