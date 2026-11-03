"""Run the single gated paper-test evaluation after validation-only beta selection."""

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

from data import CLASSES, PAPER_CLASS_NAMES, paper_half_split_chime, save_json  # noqa: E402
from evaluate_binary_specialist_fusion import (  # noqa: E402
    _plot_confusion_matrices,
    _prediction_dataset,
    _write_confidence_csv,
    fuse_predictions,
    route_samples,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the validation-selected beta on paper-test exactly once.")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--base-model", default="artifacts/models/final/base_resnet50")
    parser.add_argument("--tri-model", default="artifacts/models/final/tri_specialist_b_c1_c2")
    parser.add_argument("--b-c1-model", default="artifacts/models/final/binary_specialist_b_vs_c1")
    parser.add_argument("--c1-c2-model", default="artifacts/models/final/binary_specialist_c1_vs_c2")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="reports/validation_beta_selection/final_paper_test_evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu")
    parser.add_argument("--gpu-id", default="1")
    return parser.parse_args()


def configure_device(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-validation-beta")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def load_selection(path):
    selection_path = Path(path)
    if not selection_path.exists():
        raise FileNotFoundError(
            "Hard gate: validation_bonus_selection.json does not exist; paper-test evaluation is forbidden."
        )
    with selection_path.open("r", encoding="utf-8") as f:
        selection = json.load(f)
    if selection.get("paper_test_metrics_access_before_selection") is not False:
        raise ValueError("Selection JSON does not certify validation-only selection.")
    beta = float(selection["selected_beta"])
    valid_grid = [value / 100.0 for value in range(0, 51, 5)]
    if beta not in valid_grid:
        raise ValueError("Selected beta %.8f is outside the preregistered grid." % beta)
    return selection, beta


def write_state(path, status, selection_path, beta):
    save_json(
        path,
        {
            "status": status,
            "selection_json": str(selection_path),
            "selected_beta": float(beta),
            "paper_test_evaluation_policy": "single final evaluation; no paper-test beta sweep",
        },
    )


def main():
    args = parse_args()
    selection, selected_beta = load_selection(args.selection_json)
    configure_device(args)

    output_dir = Path(args.output_dir)
    state_path = output_dir / "evaluation_state.json"
    if output_dir.exists():
        raise FileExistsError(
            "Single-evaluation guard: output directory already exists and will not be evaluated again: %s"
            % output_dir
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_state(state_path, "started", args.selection_json, selected_beta)

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    import losses  # noqa: F401 - registers serialized custom losses.

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    _, _, test_x, test_y, _, test_indices = paper_half_split_chime(args.chime_data, args.chime_labels)
    if len(test_y) != 268:
        raise ValueError("Expected 268 paper-test samples, found %d." % len(test_y))
    dataset = _prediction_dataset(tf, test_x, args.batch_size)

    base_model = keras.models.load_model(args.base_model)
    tri_model = keras.models.load_model(args.tri_model)
    b_c1_model = keras.models.load_model(args.b_c1_model)
    c1_c2_model = keras.models.load_model(args.c1_c2_model)

    base_probabilities = base_model.predict(dataset, verbose=1)
    base_predictions = np.argmax(base_probabilities, axis=1).astype(np.int64)
    routed = route_samples(base_probabilities, base_predictions, "predicted_subset", 0.18)
    tri_probabilities = np.full((len(test_x), 3), np.nan, dtype=np.float32)
    b_c1_probabilities = np.full((len(test_x), 2), np.nan, dtype=np.float32)
    c1_c2_probabilities = np.full((len(test_x), 2), np.nan, dtype=np.float32)
    if np.any(routed):
        routed_dataset = _prediction_dataset(tf, test_x[routed], args.batch_size)
        tri_probabilities[routed] = tri_model.predict(routed_dataset, verbose=1)
        b_c1_probabilities[routed] = b_c1_model.predict(routed_dataset, verbose=1)
        c1_c2_probabilities[routed] = c1_c2_model.predict(routed_dataset, verbose=1)

    final_predictions, routing_diagnostics = fuse_predictions(
        base_predictions,
        routed,
        tri_probabilities,
        b_c1_probabilities,
        c1_c2_probabilities,
        c1_vote_bonus=selected_beta,
    )
    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(test_y, final_predictions, labels=labels)
    report = classification_report(
        test_y,
        final_predictions,
        labels=labels,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    per_class_recall = {
        name: float(cm[index, index] / cm[index].sum()) if cm[index].sum() else 0.0
        for index, name in enumerate(CLASSES)
    }
    metrics = {
        "dataset": "CHIME paper-test half",
        "n_samples": int(len(test_y)),
        "selected_beta": float(selected_beta),
        "selection_source": args.selection_json,
        "route_mode": "predicted_subset",
        "n_routed_to_specialists": int(np.sum(routed)),
        "accuracy": float(accuracy_score(test_y, final_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, final_predictions)),
        "macro_precision": float(precision_score(test_y, final_predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(test_y, final_predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(test_y, final_predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_y, final_predictions, average="weighted", zero_division=0)),
        "per_class_recall": per_class_recall,
        "class_names": CLASSES,
        "paper_class_names": PAPER_CLASS_NAMES,
    }

    np.savetxt(str(output_dir / "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(str(output_dir / "y_true.txt"), test_y, fmt="%d")
    np.savetxt(str(output_dir / "y_pred_base.txt"), base_predictions, fmt="%d")
    np.savetxt(str(output_dir / "y_pred_final.txt"), final_predictions, fmt="%d")
    np.savez_compressed(str(output_dir / "paper_test_indices.npz"), test_idx=test_indices)
    save_json(output_dir / "metrics_summary.json", metrics)
    save_json(output_dir / "classification_report.json", report)
    save_json(output_dir / "routing_diagnostics.json", routing_diagnostics)
    _write_confidence_csv(
        output_dir / "confidence_scores.csv",
        base_probabilities,
        tri_probabilities,
        b_c1_probabilities,
        c1_c2_probabilities,
        final_predictions,
        routed,
    )
    _plot_confusion_matrices(output_dir, cm)

    selection_bytes = Path(args.selection_json).read_bytes()
    save_json(
        output_dir / "run_config.json",
        {
            "inference_only": True,
            "single_final_paper_test_evaluation": True,
            "selection_json": args.selection_json,
            "selection_json_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "selected_beta": float(selected_beta),
            "selection_record": selection,
            "base_model": args.base_model,
            "tri_model": args.tri_model,
            "b_c1_model": args.b_c1_model,
            "c1_c2_model": args.c1_c2_model,
            "chime_data": args.chime_data,
            "chime_labels": args.chime_labels,
            "split": "paper_test",
            "route_mode": "predicted_subset",
            "batch_size": int(args.batch_size),
            "device": args.device,
            "gpu_id": args.gpu_id,
            "paper_test_beta_sweep": False,
        },
    )
    write_state(state_path, "complete", args.selection_json, selected_beta)
    print("Completed the single gated paper-test evaluation at beta=%.2f." % selected_beta)
    print("Saved to: %s" % output_dir)


if __name__ == "__main__":
    main()
