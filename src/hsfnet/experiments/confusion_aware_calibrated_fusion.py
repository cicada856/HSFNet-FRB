#!/usr/bin/env python
"""Phase 0/1 post-hoc CFSF pilot.

No model is trained here. Development OOF probabilities select temperatures,
log-score weights, and a selective gate. The paper-test arrays are loaded only
after selection_locked.json has been written.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HSFNET_SRC = PROJECT_ROOT / "src" / "hsfnet"
if str(HSFNET_SRC) not in sys.path:
    sys.path.insert(0, str(HSFNET_SRC))

from data import CLASSES, CLASS_TO_INDEX, load_npz_array, normalize_images_to_uint8, save_json
from evaluate_binary_specialist_fusion import fuse_predictions

B_IDX, C1_IDX, C2_IDX = 1, 3, 4
LOCAL_GLOBAL = np.asarray([B_IDX, C1_IDX, C2_IDX], dtype=np.int64)
LOCAL_SET = set(LOCAL_GLOBAL.tolist())
OOF_FIELDS = (
    "sample_index", "fold", "true_label", "base_pred", "tri_prob_B", "tri_prob_C1", "tri_prob_C2",
    "bc1_prob_B", "bc1_prob_C1", "c1c2_prob_C1", "c1c2_prob_C2",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 0/1 CFSF post-hoc pilot without retraining.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports/confusion_aware_calibrated_fusion"))
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--development-indices", default="artifacts/runs/resnet50_c1_repair_gpu0/paper_chime_split_indices.npz")
    parser.add_argument("--oof-csv", default="reports/validation_beta_selection/oof_component_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu")
    parser.add_argument("--gpu-id", default="0")
    return parser.parse_args()


def configure_device(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cfsf")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_dataset(tf, x, batch_size):
    return tf.data.Dataset.from_tensor_slices(x).map(
        lambda image: tf.cast(image, tf.float32) / 255.0,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def read_oof(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OOF_FIELDS:
            raise ValueError("Unexpected OOF schema: %s" % (reader.fieldnames,))
        rows = list(reader)
    if len(rows) != 267 or len({int(row["sample_index"]) for row in rows}) != 267:
        raise ValueError("OOF must contain 267 unique development samples")
    return rows


def load_development_only(args, rows):
    with np.load(str(PROJECT_ROOT / args.development_indices)) as split:
        dev_indices = np.asarray(split["train_idx"], dtype=np.int64)
    if set(dev_indices.tolist()) != {int(row["sample_index"]) for row in rows}:
        raise ValueError("OOF indices do not match the recorded development indices")
    raw = load_npz_array(PROJECT_ROOT / args.chime_data)
    dev_x = normalize_images_to_uint8(raw[dev_indices])[..., np.newaxis]
    return {int(index): dev_x[pos] for pos, index in enumerate(dev_indices)}


def recover_oof_base_probabilities(tf, keras, images_by_index, rows, batch_size):
    result = np.full((len(rows), 5), np.nan, dtype=np.float32)
    row_positions = {int(row["sample_index"]): pos for pos, row in enumerate(rows)}
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        indices = [int(row["sample_index"]) for row in fold_rows]
        model_path = PROJECT_ROOT / "artifacts/runs/validation_beta_selection" / ("fold_%d/base/best_model" % fold)
        model = keras.models.load_model(str(model_path))
        probs = model.predict(prediction_dataset(tf, np.asarray([images_by_index[i] for i in indices]), batch_size), verbose=0)
        for pos, index in enumerate(indices):
            result[row_positions[index]] = probs[pos]
        del model
        keras.backend.clear_session()
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Could not recover complete OOF base probabilities")
    return result


def component_probabilities(rows):
    tri = np.asarray([[float(row["tri_prob_B"]), float(row["tri_prob_C1"]), float(row["tri_prob_C2"])] for row in rows], dtype=np.float64)
    bc1 = np.asarray([[float(row["bc1_prob_B"]), float(row["bc1_prob_C1"])] for row in rows], dtype=np.float64)
    c1c2 = np.asarray([[float(row["c1c2_prob_C1"]), float(row["c1c2_prob_C2"])] for row in rows], dtype=np.float64)
    return tri, bc1, c1c2


def map_pair_labels(y, pair):
    if pair == "bc1":
        keep = np.isin(y, [B_IDX, C1_IDX])
        mapped = np.where(y[keep] == B_IDX, 0, 1)
    else:
        keep = np.isin(y, [C1_IDX, C2_IDX])
        mapped = np.where(y[keep] == C1_IDX, 0, 1)
    return keep, mapped


def nll(probs, labels):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1.0))))


def brier(probs, labels):
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ece(probs, labels, bins=10):
    confidence = np.max(probs, axis=1)
    prediction = np.argmax(probs, axis=1)
    total = 0.0
    for low, high in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1.0 else (confidence <= high))
        if np.any(mask):
            total += float(np.sum(mask)) / len(labels) * abs(float(np.mean(prediction[mask] == labels[mask])) - float(np.mean(confidence[mask])))
    return float(total)


def temperature_scale(probs, temperature):
    logits = np.log(np.clip(probs, 1e-8, 1.0))
    logits = logits / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def select_temperature(probs, labels):
    grid = np.linspace(0.5, 5.0, 91)
    rows = []
    for temperature in grid:
        scaled = temperature_scale(probs, temperature)
        rows.append((nll(scaled, labels), float(temperature), scaled))
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows[0][1], rows


def calibrated_components(base, tri, bc1, c1c2, y):
    base_t, base_sweep = select_temperature(base, y)
    tri_keep, tri_y = map_pair_labels(y, "tri") if False else (np.isin(y, LOCAL_GLOBAL), np.asarray([{B_IDX: 0, C1_IDX: 1, C2_IDX: 2}[int(v)] for v in y[np.isin(y, LOCAL_GLOBAL)]], dtype=np.int64))
    tri_t, tri_sweep = select_temperature(tri[tri_keep], tri_y)
    bc_keep, bc_y = map_pair_labels(y, "bc1")
    bc_t, bc_sweep = select_temperature(bc1[bc_keep], bc_y)
    cc_keep, cc_y = map_pair_labels(y, "c1c2")
    cc_t, cc_sweep = select_temperature(c1c2[cc_keep], cc_y)
    temps = {"base": base_t, "tri": tri_t, "bc1": bc_t, "c1c2": cc_t}
    scaled = {
        "base": temperature_scale(base, base_t),
        "tri": temperature_scale(tri, tri_t),
        "bc1": temperature_scale(bc1, bc_t),
        "c1c2": temperature_scale(c1c2, cc_t),
    }
    sweeps = {"base": base_sweep, "tri": tri_sweep, "bc1": bc_sweep, "c1c2": cc_sweep}
    return temps, scaled, sweeps


def calibration_record(before, after, labels, name):
    return {
        "model": name,
        "n": int(len(labels)),
        "temperature": None,
        "before": {"nll": nll(before, labels), "brier": brier(before, labels), "ece": ece(before, labels)},
        "after": {"nll": nll(after, labels), "brier": brier(after, labels), "ece": ece(after, labels)},
    }


def score_fusion(base, tri, bc1, c1c2, weights, eps=1e-8):
    # Binary experts contribute neutral evidence (1.0) to classes outside
    # their pair; the four weights remain independently interpretable.
    n = len(base)
    scores = np.zeros((n, 3), dtype=np.float64)
    local_base = base[:, [B_IDX, C1_IDX, C2_IDX]]
    local_tri = tri
    bc1_evidence = np.ones((n, 3), dtype=np.float64)
    bc1_evidence[:, 0] = bc1[:, 0]
    bc1_evidence[:, 1] = bc1[:, 1]
    c1c2_evidence = np.ones((n, 3), dtype=np.float64)
    c1c2_evidence[:, 1] = c1c2[:, 0]
    c1c2_evidence[:, 2] = c1c2[:, 1]
    for weight, probs in zip(weights, (local_base, local_tri, bc1_evidence, c1c2_evidence)):
        scores += float(weight) * np.log(np.clip(probs, eps, 1.0))
    return scores


def simplex_weights(step=0.1):
    values = np.arange(0.0, 1.0 + step / 2.0, step)
    return [np.asarray([a, b, c, 1.0 - a - b - c], dtype=np.float64) for a in values for b in values for c in values if a + b + c <= 1.0 + 1e-9]


def metrics(y, pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    cm = confusion_matrix(y, pred, labels=list(range(5)))
    rec = {name: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0 for i, name in enumerate(CLASSES)}
    local = np.isin(y, LOCAL_GLOBAL)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=list(range(5)), average="weighted", zero_division=0)),
        "per_class_recall": rec,
        "local_n": int(np.sum(local)),
        "local_balanced_accuracy": float(np.mean([rec[name] for name in ("B", "C1", "C2")])),
        "confusion_matrix": cm.tolist(),
    }


def corrected_stats(y, baseline, candidate, routed, gate_accept=None):
    changed = candidate != baseline
    if gate_accept is not None:
        changed = changed & gate_accept
    true_ac = np.isin(y, [0, 2])
    return {
        "candidate_changes_vs_fixed_beta": int(np.sum(changed)),
        "changed_to_correct": int(np.sum(changed & (candidate == y) & (baseline != y))),
        "changed_to_wrong": int(np.sum(changed & (candidate != y) & (baseline == y))),
        "routed_samples": int(np.sum(routed)),
        "accepted_corrections": int(np.sum(changed)),
        "a_c_routed": int(np.sum(routed & true_ac)),
        "a_c_base_correct_became_wrong": int(np.sum(routed & true_ac & (baseline == y) & (candidate != y))),
        "a_c_finally_wrong": int(np.sum(routed & true_ac & (candidate != y))),
    }


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    configure_device(args)
    output_dir = Path(args.output_dir).resolve()
    pilot_dir = output_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    if (pilot_dir / "selection_locked.json").exists():
        raise FileExistsError("Pilot already locked; refusing to overwrite %s" % pilot_dir)

    import tensorflow as tf
    from tensorflow import keras
    if args.device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU requested but TensorFlow cannot see a GPU")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    rows = read_oof(PROJECT_ROOT / args.oof_csv)
    images_by_index = load_development_only(args, rows)
    oof_base = recover_oof_base_probabilities(tf, keras, images_by_index, rows, args.batch_size)
    oof_tri, oof_bc1, oof_c1c2 = component_probabilities(rows)
    oof_y = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    raw_base_pred = np.argmax(oof_base, axis=1)
    top1_route = np.isin(raw_base_pred, LOCAL_GLOBAL)
    _, fixed_oof, _ = (lambda pred: (pred, fuse_predictions(raw_base_pred, top1_route, oof_tri, oof_bc1, oof_c1c2, 0.20)[0], None))(raw_base_pred)
    temps, scaled, temp_sweeps = calibrated_components(oof_base, oof_tri, oof_bc1, oof_c1c2, oof_y)

    cal_base_pred = np.argmax(scaled["base"], axis=1)
    route = np.isin(cal_base_pred, LOCAL_GLOBAL)
    raw_scores = score_fusion(scaled["base"], scaled["tri"], scaled["bc1"], scaled["c1c2"], [0.25, 0.35, 0.20, 0.20])
    weight_candidates = []
    for weights in simplex_weights(0.1):
        scores = score_fusion(scaled["base"], scaled["tri"], scaled["bc1"], scaled["c1c2"], weights)
        fused_local = np.argmax(scores, axis=1)
        fused_global = np.asarray([LOCAL_GLOBAL[i] for i in fused_local], dtype=np.int64)
        candidate = raw_base_pred.copy()
        candidate[route] = fused_global[route]
        m = metrics(oof_y, candidate)
        weight_candidates.append({"weights": [float(v) for v in weights], "metrics": m})
    weight_candidates.sort(key=lambda row: (-row["metrics"]["macro_f1"], -row["metrics"]["accuracy"], -row["metrics"]["local_balanced_accuracy"]))
    selected_weights = np.asarray(weight_candidates[0]["weights"], dtype=np.float64)
    selected_scores = score_fusion(scaled["base"], scaled["tri"], scaled["bc1"], scaled["c1c2"], selected_weights)
    fused_local = np.argmax(selected_scores, axis=1)
    fused_global = np.asarray([LOCAL_GLOBAL[i] for i in fused_local], dtype=np.int64)
    ungated = raw_base_pred.copy()
    ungated[route] = fused_global[route]
    margins = np.sort(selected_scores, axis=1)[:, -1] - np.sort(selected_scores, axis=1)[:, -2]
    gains = selected_scores[np.arange(len(oof_y)), fused_local] - selected_scores[np.arange(len(oof_y)), np.asarray([{B_IDX: 0, C1_IDX: 1, C2_IDX: 2}.get(int(v), 0) for v in raw_base_pred])]
    gate_candidates = []
    for threshold in np.linspace(0.0, 2.0, 81):
        accept = route & (margins >= threshold)
        pred = raw_base_pred.copy(); pred[accept] = fused_global[accept]
        gate_candidates.append({"gate": "fused_margin", "threshold": float(threshold), "metrics": metrics(oof_y, pred), "corrections": corrected_stats(oof_y, fixed_oof, pred, route, accept)})
    for threshold in np.linspace(-2.0, 2.0, 161):
        accept = route & (gains >= threshold)
        pred = raw_base_pred.copy(); pred[accept] = fused_global[accept]
        gate_candidates.append({"gate": "score_gain", "threshold": float(threshold), "metrics": metrics(oof_y, pred), "corrections": corrected_stats(oof_y, fixed_oof, pred, route, accept)})
    gate_candidates.sort(key=lambda row: (-row["metrics"]["macro_f1"], -row["metrics"]["accuracy"], -row["metrics"]["per_class_recall"]["C1"], -row["metrics"]["per_class_recall"]["C2"], row["corrections"]["changed_to_wrong"], row["threshold"]))
    selected_gate = gate_candidates[0]

    calibration = []
    calibration.append(calibration_record(oof_base, scaled["base"], oof_y, "base")); calibration[-1]["temperature"] = temps["base"]
    tri_keep, tri_y = map_pair_labels(oof_y, "tri") if False else (np.isin(oof_y, LOCAL_GLOBAL), np.asarray([{B_IDX: 0, C1_IDX: 1, C2_IDX: 2}[int(v)] for v in oof_y[np.isin(oof_y, LOCAL_GLOBAL)]], dtype=np.int64))
    calibration.append(calibration_record(oof_tri[tri_keep], scaled["tri"][tri_keep], tri_y, "tri")); calibration[-1]["temperature"] = temps["tri"]
    bc_keep, bc_y = map_pair_labels(oof_y, "bc1"); calibration.append(calibration_record(oof_bc1[bc_keep], scaled["bc1"][bc_keep], bc_y, "bc1")); calibration[-1]["temperature"] = temps["bc1"]
    cc_keep, cc_y = map_pair_labels(oof_y, "c1c2"); calibration.append(calibration_record(oof_c1c2[cc_keep], scaled["c1c2"][cc_keep], cc_y, "c1c2")); calibration[-1]["temperature"] = temps["c1c2"]
    save_json(pilot_dir / "calibration_metrics.json", {"development_oof_only": calibration, "temperature_search": {name: [{"nll": float(item[0]), "temperature": float(item[1])} for item in sweep] for name, sweep in temp_sweeps.items()}})

    baseline_oof = metrics(oof_y, fixed_oof)
    calibrated_oof = metrics(oof_y, ungated)
    gated_accept_oof = (route & ((margins >= selected_gate["threshold"]) if selected_gate["gate"] == "fused_margin" else (gains >= selected_gate["threshold"])))
    gated_oof = raw_base_pred.copy(); gated_oof[gated_accept_oof] = fused_global[gated_accept_oof]
    oof_rows_out = []
    for i, row in enumerate(rows):
        oof_rows_out.append({"sample_index": int(row["sample_index"]), "true_label": CLASSES[int(oof_y[i])], "fixed_beta_pred": CLASSES[int(fixed_oof[i])], "calibrated_fusion_pred": CLASSES[int(ungated[i])], "selective_gate_pred": CLASSES[int(gated_oof[i])], "routed": int(route[i]), "gate_accepted": int(gated_accept_oof[i]), "fused_margin": float(margins[i]), "score_gain": float(gains[i])})
    write_csv(pilot_dir / "oof_predictions.csv", oof_rows_out, list(oof_rows_out[0].keys()))
    write_csv(pilot_dir / "oof_metrics.csv", [{"method": "fixed_beta_0.20", **baseline_oof}, {"method": "calibrated_fusion", **calibrated_oof}, {"method": "calibrated_fusion_selective_gate", **metrics(oof_y, gated_oof)}], ["method", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "local_n", "local_balanced_accuracy", "per_class_recall", "confusion_matrix"])
    selected_record = {"phase": "Phase 1 pilot", "paper_test_loaded_before_lock": False, "oof_n": 267, "beta": 0.20, "temperatures": temps, "weights": [float(v) for v in selected_weights], "gate": {"type": selected_gate["gate"], "threshold": selected_gate["threshold"]}, "selection_rule": "development OOF macro-F1, then accuracy, C1/C2 recall and lower harmful correction count", "selected_weight_metrics": weight_candidates[0]["metrics"], "selected_gate_metrics": selected_gate["metrics"], "source_oof_sha256": sha256(PROJECT_ROOT / args.oof_csv)}
    save_json(pilot_dir / "selection_config.json", selected_record)
    save_json(pilot_dir / "selection_locked.json", {**selected_record, "config_sha256": hashlib.sha256(json.dumps(selected_record, sort_keys=True).encode("utf-8")).hexdigest()})
    save_json(output_dir / "phase0_audit.json", {"oof_schema_complete": True, "oof_n": 267, "unique_sample_indices": 267, "folds": [0, 1, 2, 3, 4], "classes": CLASSES, "local_indices": {"B": 1, "C1": 3, "C2": 4}, "base_probability_source": "existing validation_beta_selection fold base checkpoints; development OOF only", "paper_test_read_before_lock": False, "retraining": False})

    # Paper-test is loaded only after the lock file exists.
    from data import paper_half_split_chime
    _, _, test_x, test_y, _, test_indices = paper_half_split_chime(PROJECT_ROOT / args.chime_data, PROJECT_ROOT / args.chime_labels)
    model_paths = {"base": PROJECT_ROOT / "artifacts/models/final/base_resnet50", "tri": PROJECT_ROOT / "artifacts/models/final/tri_specialist_b_c1_c2", "bc1": PROJECT_ROOT / "artifacts/models/final/binary_specialist_b_vs_c1", "c1c2": PROJECT_ROOT / "artifacts/models/final/binary_specialist_c1_vs_c2"}
    models = {name: keras.models.load_model(str(path)) for name, path in model_paths.items()}
    test_ds = prediction_dataset(tf, test_x, args.batch_size)
    test_base = models["base"].predict(test_ds, verbose=0); test_tri = models["tri"].predict(test_ds, verbose=0); test_bc1 = models["bc1"].predict(test_ds, verbose=0); test_c1c2 = models["c1c2"].predict(test_ds, verbose=0)
    test_pred_base = np.argmax(test_base, axis=1); test_route = np.isin(test_pred_base, LOCAL_GLOBAL)
    test_fixed = fuse_predictions(test_pred_base, test_route, test_tri, test_bc1, test_c1c2, 0.20)[0]
    test_scaled = {"base": temperature_scale(test_base, temps["base"]), "tri": temperature_scale(test_tri, temps["tri"]), "bc1": temperature_scale(test_bc1, temps["bc1"]), "c1c2": temperature_scale(test_c1c2, temps["c1c2"])}
    test_scores = score_fusion(test_scaled["base"], test_scaled["tri"], test_scaled["bc1"], test_scaled["c1c2"], selected_weights); test_local = np.argmax(test_scores, axis=1); test_fused = np.asarray([LOCAL_GLOBAL[i] for i in test_local]); test_ungated = test_pred_base.copy(); test_ungated[test_route] = test_fused[test_route]
    test_margins = np.sort(test_scores, axis=1)[:, -1] - np.sort(test_scores, axis=1)[:, -2]
    base_local_idx = np.asarray([{B_IDX: 0, C1_IDX: 1, C2_IDX: 2}.get(int(v), 0) for v in test_pred_base])
    test_gains = test_scores[np.arange(len(test_y)), test_local] - test_scores[np.arange(len(test_y)), base_local_idx]
    test_accept = test_route & ((test_margins >= selected_gate["threshold"]) if selected_gate["gate"] == "fused_margin" else (test_gains >= selected_gate["threshold"]))
    test_gated = test_pred_base.copy(); test_gated[test_accept] = test_fused[test_accept]
    methods = {"fixed_beta_0.20": test_fixed, "calibrated_fusion": test_ungated, "calibrated_fusion_selective_gate": test_gated}
    test_metric_rows = []
    for name, pred in methods.items():
        item = metrics(test_y, pred); item["method"] = name; item["corrections"] = corrected_stats(test_y, test_fixed, pred, test_route, None if name == "fixed_beta_0.20" else (test_route if name == "calibrated_fusion" else test_accept)); test_metric_rows.append(item)
        cm = np.asarray(item["confusion_matrix"], dtype=np.int64); write_csv(pilot_dir / (name + "_confusion_matrix.csv"), [{"true_label": CLASSES[i], **{CLASSES[j]: int(cm[i, j]) for j in range(5)}} for i in range(5)], ["true_label"] + CLASSES)
    save_json(pilot_dir / "test_metrics.json", {item["method"]: item for item in test_metric_rows})
    write_csv(pilot_dir / "test_predictions.csv", [{"sample_index": int(test_indices[i]), "true_label": CLASSES[int(test_y[i])], "fixed_beta_pred": CLASSES[int(test_fixed[i])], "calibrated_fusion_pred": CLASSES[int(test_ungated[i])], "selective_gate_pred": CLASSES[int(test_gated[i])], "routed": int(test_route[i]), "gate_accepted": int(test_accept[i]), "fused_margin": float(test_margins[i]), "score_gain": float(test_gains[i])} for i in range(len(test_y))], ["sample_index", "true_label", "fixed_beta_pred", "calibrated_fusion_pred", "selective_gate_pred", "routed", "gate_accepted", "fused_margin", "score_gain"])
    write_csv(pilot_dir / "gate_diagnostics.csv", [{"method": item["method"], **item["corrections"]} for item in test_metric_rows], ["method"] + list(test_metric_rows[0]["corrections"].keys()))

    selected_test = test_metric_rows[2]
    selection_pass = {
        "accuracy_not_more_than_1pp_below": selected_test["accuracy"] >= test_metric_rows[0]["accuracy"] - 0.01,
        "macro_f1_positive_2pp_trend": selected_test["macro_f1"] >= test_metric_rows[0]["macro_f1"] + 0.02,
        "c1_not_decreased": selected_test["per_class_recall"]["C1"] >= test_metric_rows[0]["per_class_recall"]["C1"],
        "a_c_not_collapsed": selected_test["per_class_recall"]["A"] > 0.5 and selected_test["per_class_recall"]["C"] > 0.5,
        "correct_changes_exceed_wrong": selected_test["corrections"]["changed_to_correct"] > selected_test["corrections"]["changed_to_wrong"],
        "pilot_pass": False,
    }
    selection_pass["pilot_pass"] = all(value for key, value in selection_pass.items() if key != "pilot_pass")
    save_json(output_dir / "phase2_decision.json", {"pilot_pass": selection_pass["pilot_pass"], "criteria": selection_pass, "phase3_allowed": selection_pass["pilot_pass"], "reason": "Phase 3 is allowed only if every report criterion passes; this run itself performs Phase 0 and Phase 1 only."})
    failing = []
    for i in range(len(test_y)):
        if test_gated[i] != test_fixed[i] and test_gated[i] != test_y[i]:
            failing.append({"sample_index": int(test_indices[i]), "true_label": CLASSES[int(test_y[i])], "fixed_beta_pred": CLASSES[int(test_fixed[i])], "selective_gate_pred": CLASSES[int(test_gated[i])], "routed": int(test_route[i]), "gate_accepted": int(test_accept[i]), "fused_margin": float(test_margins[i]), "score_gain": float(test_gains[i]), "a_c_harmful": int(np.isin(test_y[i], [0, 2]) and test_fixed[i] == test_y[i])})
    write_csv(pilot_dir / "failure_examples.csv", failing[:50], ["sample_index", "true_label", "fixed_beta_pred", "selective_gate_pred", "routed", "gate_accepted", "fused_margin", "score_gain", "a_c_harmful"])
    write_csv(pilot_dir / "test_confusion_matrix.csv", [{"true_label": CLASSES[i], **{CLASSES[j]: int(np.asarray(selected_test["confusion_matrix"])[i, j]) for j in range(5)}} for i in range(5)], ["true_label"] + CLASSES)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = None
    save_json(output_dir / "run_metadata.json", {"python": platform.python_version(), "platform": platform.platform(), "git_commit": git_commit, "script_sha256": sha256(Path(__file__))})
    lines = ["# CFSF Phase 0/1 Pilot Summary", "", "## Phase 0", "", "- Existing OOF schema is complete: 267 unique development samples, five folds, tri-specialist and both binary specialists' continuous probabilities.", "- Base OOF probabilities were recovered from the existing five fold base checkpoints; no model was retrained.", "- Local class indices are B=1, C1=3, C2=4.", "- Paper-test was not read before `selection_locked.json`.", "", "## Selected Configuration", "", "- Temperatures: `%s`" % temps, "- Constrained log-score weights `[base, tri, bc1, c1c2]`: `%s`" % [float(v) for v in selected_weights], "- Selective gate: `%s >= %.4f`" % (selected_gate["gate"], selected_gate["threshold"]), "", "## OOF Metrics", "", "| Method | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Local balanced accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for label, item in (("Fixed beta=0.20", baseline_oof), ("Calibrated fusion", calibrated_oof), ("Calibrated fusion + gate", metrics(oof_y, gated_oof))):
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f |" % (label, item["accuracy"], item["balanced_accuracy"], item["macro_f1"], item["weighted_f1"], item["local_balanced_accuracy"]))
    lines += ["", "## Paper-Test Metrics", "", "| Method | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | C1 recall | Local balanced accuracy | Changes to correct | Changes to wrong |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in test_metric_rows:
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %d | %d |" % (item["method"], item["accuracy"], item["balanced_accuracy"], item["macro_f1"], item["weighted_f1"], item["per_class_recall"]["C1"], item["local_balanced_accuracy"], item["corrections"]["changed_to_correct"], item["corrections"]["changed_to_wrong"]))
    lines += ["", "## Decision", "", "- Pilot pass: `%s`" % selection_pass["pilot_pass"], "- Phase 3 allowed by the written criteria: `%s`" % selection_pass["pilot_pass"], "- The current fixed beta=0.20 system remains the main result unless a separately approved Phase 3 confirms a stable gain. This directory contains only post-hoc pilot outputs and does not modify any model, paper file, or existing report."]
    (pilot_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(output_dir / "run_complete.json", {"status": "complete", "phase": "Phase 0 and Phase 1", "retraining": False, "paper_test_after_lock": True, "phase3_allowed": False})
    print("CFSF Phase 0/1 pilot complete: %s" % pilot_dir, flush=True)


if __name__ == "__main__":
    main()
