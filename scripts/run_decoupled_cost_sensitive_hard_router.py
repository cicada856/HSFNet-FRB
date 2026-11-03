#!/usr/bin/env python
"""Decoupled cost-sensitive hard routing for HSFNet-FRB.

The published five-class base and all specialists remain frozen. A lightweight
binary router is trained from frozen base features to predict whether a sample
belongs to the local B/C1/C2 subspace. Router architecture, cost weighting, and
hard thresholds are selected with five-fold CHIME development-only OOF results.
Paper-test samples are indexed only after the locked configuration is written.
"""

from __future__ import print_function

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "hsfnet"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data import (  # noqa: E402
    CLASSES,
    CLASS_TO_INDEX,
    load_npz_array,
    load_simulation_dataset_with_snr,
    normalize_images_to_uint8,
    save_json,
)
from evaluate_binary_specialist_fusion import fuse_predictions  # noqa: E402
import run_routing_ablation as routing_legacy  # noqa: E402


LOCAL_NAMES = ("B", "C1", "C2")
LOCAL_INDICES = np.asarray([CLASS_TO_INDEX[name] for name in LOCAL_NAMES], dtype=np.int64)
LOCAL_SET = set(LOCAL_INDICES.tolist())
BETA = 0.20


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated decoupled cost-sensitive hard router experiment.")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports/decoupled_cost_sensitive_hard_router"))
    parser.add_argument("--artifact-dir", default=str(PROJECT_ROOT / "artifacts/runs/decoupled_cost_sensitive_hard_router"))
    parser.add_argument("--sim-dir", default="data/simulated/simfrb")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--split-indices", default="artifacts/runs/resnet50_c1_repair_gpu0/paper_chime_split_indices.npz")
    parser.add_argument("--oof-components", default="reports/validation_beta_selection/oof_component_predictions.csv")
    parser.add_argument("--samples-per-class", type=int, default=1200)
    parser.add_argument("--snrs", default="15,25,35,50,100")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--router-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def configure_device(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dlsr")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def parse_ints(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def write_status(report_dir, state, stage, **extra):
    payload = {"state": state, "stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    payload.update(extra)
    save_json(Path(report_dir) / "status.json", payload)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path, matrix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + list(CLASSES))
        for name, row in zip(CLASSES, matrix):
            writer.writerow([name] + list(row))


def git_commit_or_none():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_labels(path):
    labels = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray([CLASS_TO_INDEX[label] for label in labels], dtype=np.int64)


def load_indices(args):
    with np.load(str(PROJECT_ROOT / args.split_indices)) as split:
        development = np.asarray(split["train_idx"], dtype=np.int64)
        paper_test = np.asarray(split["test_idx"], dtype=np.int64)
    if len(development) != 267 or len(paper_test) != 268:
        raise ValueError("Expected immutable 267 development / 268 paper-test split")
    return development, paper_test


def load_oof_rows(args, development_indices):
    rows = routing_legacy.load_oof_rows(PROJECT_ROOT / args.oof_components)
    row_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if set(row_indices.tolist()) != set(development_indices.tolist()):
        raise ValueError("OOF rows do not match immutable development indices")
    return rows, row_indices


def load_development(args, row_indices):
    raw = load_npz_array(PROJECT_ROOT / args.chime_data)
    labels = read_labels(PROJECT_ROOT / args.chime_labels)
    return normalize_images_to_uint8(raw[row_indices])[..., np.newaxis], labels[row_indices]


def load_simulation(args):
    x, y, _, _ = load_simulation_dataset_with_snr(
        PROJECT_ROOT / args.sim_dir,
        samples_per_class=args.samples_per_class,
        snrs=parse_ints(args.snrs),
        seed=42,
        exclude_marked=True,
    )
    return x, y


def prediction_dataset(tf, images, batch_size):
    return tf.data.Dataset.from_tensor_slices(images).map(
        lambda image: tf.cast(image, tf.float32) / 255.0,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def router_inputs(features, probabilities):
    features = np.asarray(features, dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-7)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    order = np.sort(probabilities, axis=1)
    confidence = order[:, -1]
    margin = order[:, -1] - order[:, -2]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-7, 1.0)), axis=1) / np.log(5.0)
    local_mass = np.sum(probabilities[:, LOCAL_INDICES], axis=1)
    summary = np.stack([confidence, margin, entropy, local_mass], axis=1)
    return np.concatenate([features, probabilities, summary], axis=1).astype(np.float32)


def extract_base_bundle(tf, keras, images, model_path, batch_size, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(str(cache_path)) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if not (Path(model_path) / "saved_model.pb").exists():
        raise FileNotFoundError("Missing base model: %s" % model_path)
    import losses  # noqa: F401
    model = keras.models.load_model(str(model_path))
    extractor = keras.Model(
        model.input,
        [model.get_layer("global_avg_pool").output, model.output],
        name="frozen_base_with_router_features",
    )
    features, probabilities = extractor.predict(prediction_dataset(tf, images, batch_size), verbose=0)
    bundle = {
        "inputs": router_inputs(features, probabilities),
        "base_probs": np.asarray(probabilities, dtype=np.float32),
        "base_pred": np.argmax(probabilities, axis=1).astype(np.int64),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **bundle)
    del model, extractor, features, probabilities
    keras.backend.clear_session()
    gc.collect()
    return bundle


def fold_base_path(fold):
    return PROJECT_ROOT / "artifacts/runs/validation_beta_selection" / ("fold_%d/base/best_model" % fold)


def final_paths():
    root = PROJECT_ROOT / "artifacts/models/final"
    return {
        "base": root / "base_resnet50",
        "tri": root / "tri_specialist_b_c1_c2",
        "bc1": root / "binary_specialist_b_vs_c1",
        "c1c2": root / "binary_specialist_c1_vs_c2",
    }


def build_router(keras, input_dim, config):
    regularizer = keras.regularizers.l2(config["l2"])
    inputs = keras.layers.Input(shape=(input_dim,), name="frozen_base_features")
    x = keras.layers.Dense(
        config["hidden_dim"], activation="swish", kernel_regularizer=regularizer, name="router_hidden"
    )(inputs)
    x = keras.layers.Dropout(config["dropout"], name="router_dropout")(x)
    x = keras.layers.Dense(
        max(16, config["hidden_dim"] // 2), activation="swish", kernel_regularizer=regularizer,
        name="router_bottleneck",
    )(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="local_probability")(x)
    return keras.Model(inputs, output, name="decoupled_local_subspace_router")


def combine_training_data(sim_bundle, sim_y, dev_bundle, dev_y, train_mask, config):
    x = np.concatenate([sim_bundle["inputs"], dev_bundle["inputs"][train_mask]], axis=0)
    y = np.concatenate([np.isin(sim_y, LOCAL_INDICES), np.isin(dev_y[train_mask], LOCAL_INDICES)], axis=0).astype(np.float32)
    domain_weight = np.concatenate([
        np.ones(len(sim_y), dtype=np.float32),
        np.full(int(np.sum(train_mask)), config["development_weight"], dtype=np.float32),
    ])
    return x, y, domain_weight


def train_router(tf, keras, arrays, config, seed, output_dir, epochs):
    output_dir = Path(output_dir)
    weights_path = output_dir / "router.weights.h5"
    result_path = output_dir / "training.json"
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = build_router(keras, arrays[0].shape[1], config)
    if weights_path.exists() and result_path.exists():
        model.load_weights(str(weights_path))
        return model, json.loads(result_path.read_text(encoding="utf-8"))
    x, y, domain_weight = arrays
    dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y, domain_weight)).shuffle(
        min(len(x), 10000), seed=seed, reshuffle_each_iteration=True
    ).batch(config["batch_size"]).prefetch(tf.data.AUTOTUNE)
    optimizer = keras.optimizers.Adam(config["learning_rate"])

    @tf.function(reduce_retracing=True)
    def step(batch_x, batch_y, batch_domain):
        with tf.GradientTape() as tape:
            probability = tf.reshape(model(batch_x, training=True), [-1])
            probability = tf.clip_by_value(probability, 1e-6, 1.0 - 1e-6)
            bce = -(batch_y * tf.math.log(probability) + (1.0 - batch_y) * tf.math.log(1.0 - probability))
            pt = batch_y * probability + (1.0 - batch_y) * (1.0 - probability)
            focal = tf.pow(1.0 - pt, config["focal_gamma"])
            cost = batch_y * config["false_negative_cost"] + (1.0 - batch_y) * config["false_positive_cost"]
            weight = batch_domain * cost
            data_loss = tf.reduce_sum(bce * focal * weight) / tf.maximum(tf.reduce_sum(weight), 1.0)
            regularization = tf.add_n(model.losses) if model.losses else 0.0
            total = data_loss + regularization
        gradients = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total, data_loss

    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        values = []
        for batch_x, batch_y, batch_domain in dataset:
            total, data_loss = step(batch_x, batch_y, batch_domain)
            values.append((float(total.numpy()), float(data_loss.numpy())))
        means = np.mean(values, axis=0)
        history.append({"epoch": epoch + 1, "loss": float(means[0]), "data_loss": float(means[1])})
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(weights_path))
    result = {
        "seed": int(seed),
        "epochs": int(epochs),
        "seconds": float(time.perf_counter() - started),
        "parameters": int(model.count_params()),
        "config": config,
        "history": history,
        "paper_test_access": False,
    }
    save_json(result_path, result)
    return model, result


def candidate_configs(batch_size, smoke_only):
    epochs = 1 if smoke_only else None
    shared = {"batch_size": batch_size, "learning_rate": 1e-3, "dropout": 0.15, "l2": 1e-5}
    values = [
        ("balanced_h32", 32, 10.0, 1.0, 1.0, 0.0),
        ("global_guard_h32", 32, 12.0, 2.0, 1.0, 0.0),
        ("local_guard_h32", 32, 12.0, 1.0, 2.0, 0.0),
        ("focal_balanced_h32", 32, 12.0, 1.0, 1.0, 2.0),
        ("focal_global_h64", 64, 16.0, 2.0, 1.0, 2.0),
        ("focal_local_h64", 64, 16.0, 1.0, 2.0, 2.0),
    ]
    if smoke_only:
        values = values[:1]
    configs = []
    for name, hidden, development_weight, false_positive_cost, false_negative_cost, focal_gamma in values:
        config = dict(shared)
        config.update({
            "name": name,
            "hidden_dim": hidden,
            "development_weight": development_weight,
            "false_positive_cost": false_positive_cost,
            "false_negative_cost": false_negative_cost,
            "focal_gamma": focal_gamma,
        })
        configs.append(config)
    return configs, epochs


def predict_router(model, bundle, batch_size):
    return np.asarray(model.predict(bundle["inputs"], batch_size=batch_size, verbose=0)[:, 0], dtype=np.float32)


def specialist_arrays_from_rows(rows):
    return routing_legacy.specialist_arrays_from_rows(rows)


def base_margin(base_probs):
    order = np.sort(base_probs, axis=1)
    return order[:, -1] - order[:, -2]


def policy_grid():
    thresholds = tuple(float(round(value, 2)) for value in np.arange(0.20, 0.81, 0.05))
    for threshold in thresholds:
        yield {"name": "router_only", "threshold": threshold, "protect_base_c1": True}
    for threshold in thresholds:
        yield {"name": "dual_confirm", "confirm_threshold": threshold, "protect_base_c1": True}
    for confirm in thresholds:
        for rescue in (0.70, 0.80, 0.90, 0.95):
            for margin in (0.05, 0.10, 0.20):
                yield {
                    "name": "dual_confirm_rescue",
                    "confirm_threshold": confirm,
                    "rescue_threshold": rescue,
                    "rescue_base_margin": margin,
                    "protect_base_c1": True,
                }


def route_with_policy(router_probability, base_probs, policy):
    base_pred = np.argmax(base_probs, axis=1).astype(np.int64)
    c1 = base_pred == CLASS_TO_INDEX["C1"]
    if policy["name"] == "router_only":
        routed = router_probability >= policy["threshold"]
    else:
        ordinary_local = np.isin(base_pred, [CLASS_TO_INDEX["B"], CLASS_TO_INDEX["C2"]])
        routed = ordinary_local & (router_probability >= policy["confirm_threshold"])
        if policy["name"] == "dual_confirm_rescue":
            global_pred = np.isin(base_pred, [CLASS_TO_INDEX["A"], CLASS_TO_INDEX["C"]])
            rescue = global_pred & (router_probability >= policy["rescue_threshold"]) & (base_margin(base_probs) <= policy["rescue_base_margin"])
            routed |= rescue
    if policy.get("protect_base_c1", False):
        routed |= c1
    return routed.astype(bool)


def classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    recall = {name: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0 for i, name in enumerate(CLASSES)}
    local = np.isin(y_true, LOCAL_INDICES)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(5)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(5)), average="weighted", zero_division=0)),
        "per_class_recall": recall,
        "local_subset_accuracy": float(np.mean(y_pred[local] == y_true[local])),
        "local_subset_balanced_accuracy": float(np.mean([recall[name] for name in LOCAL_NAMES])),
        "local_subset_macro_f1": float(f1_score(y_true[local], y_pred[local], labels=LOCAL_INDICES.tolist(), average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def routing_metrics(y_true, routed):
    from sklearn.metrics import balanced_accuracy_score
    target = np.isin(y_true, LOCAL_INDICES)
    return {
        "accuracy": float(np.mean(target == routed)),
        "balanced_accuracy": float(balanced_accuracy_score(target.astype(np.int64), routed.astype(np.int64))),
        "miss_rate": float(np.mean(~routed[target])) if np.any(target) else 0.0,
        "false_positive_rate": float(np.mean(routed[~target])) if np.any(~target) else 0.0,
        "invocation_rate": float(np.mean(routed)),
        "n_invoked": int(np.sum(routed)),
    }


def correction_stats(y_true, base_pred, baseline_pred, candidate_pred, baseline_routed, candidate_routed):
    changed = candidate_pred != baseline_pred
    route_changed = candidate_routed != baseline_routed
    true_ac = np.isin(y_true, [CLASS_TO_INDEX["A"], CLASS_TO_INDEX["C"]])
    true_local = np.isin(y_true, LOCAL_INDICES)
    transitions = {}
    for source in range(5):
        for target in range(5):
            count = int(np.sum(changed & (baseline_pred == source) & (candidate_pred == target)))
            if count:
                transitions[CLASSES[source] + "->" + CLASSES[target]] = count
    return {
        "route_changes": int(np.sum(route_changed)),
        "newly_routed": int(np.sum((~baseline_routed) & candidate_routed)),
        "no_longer_routed": int(np.sum(baseline_routed & (~candidate_routed))),
        "prediction_flips": int(np.sum(changed)),
        "changed_to_correct": int(np.sum(changed & (candidate_pred == y_true) & (baseline_pred != y_true))),
        "changed_to_wrong": int(np.sum(changed & (candidate_pred != y_true) & (baseline_pred == y_true))),
        "changed_wrong_to_different_wrong": int(np.sum(changed & (candidate_pred != y_true) & (baseline_pred != y_true))),
        "local_changed_to_correct": int(np.sum(changed & true_local & (candidate_pred == y_true) & (baseline_pred != y_true))),
        "local_changed_to_wrong": int(np.sum(changed & true_local & (candidate_pred != y_true) & (baseline_pred == y_true))),
        "a_c_newly_routed": int(np.sum((~baseline_routed) & candidate_routed & true_ac)),
        "a_c_no_longer_routed": int(np.sum(baseline_routed & (~candidate_routed) & true_ac)),
        "a_c_correct_became_wrong": int(np.sum(changed & true_ac & (baseline_pred == y_true) & (candidate_pred != y_true))),
        "a_c_wrong_became_correct": int(np.sum(changed & true_ac & (baseline_pred != y_true) & (candidate_pred == y_true))),
        "base_correct_changed_to_wrong": int(np.sum(changed & (base_pred == y_true) & (candidate_pred != y_true))),
        "transitions": transitions,
    }


def evaluate_policy(y_true, base_probs, tri, bc1, c1c2, router_probability, policy, folds=None):
    base_pred = np.argmax(base_probs, axis=1).astype(np.int64)
    baseline_routed = np.isin(base_pred, LOCAL_INDICES)
    baseline_pred, _ = fuse_predictions(base_pred, baseline_routed, tri, bc1, c1c2, c1_vote_bonus=BETA)
    candidate_routed = route_with_policy(router_probability, base_probs, policy)
    candidate_pred, _ = fuse_predictions(base_pred, candidate_routed, tri, bc1, c1c2, c1_vote_bonus=BETA)
    item = classification_metrics(y_true, candidate_pred)
    route_item = routing_metrics(y_true, candidate_routed)
    corrections = correction_stats(y_true, base_pred, baseline_pred, candidate_pred, baseline_routed, candidate_routed)
    fold_rows = []
    if folds is not None:
        for fold in range(5):
            mask = folds == fold
            changed = candidate_pred[mask] != baseline_pred[mask]
            route_changed = candidate_routed[mask] != baseline_routed[mask]
            fold_rows.append({
                "fold": fold,
                "route_changes": int(np.sum(route_changed)),
                "prediction_flips": int(np.sum(changed)),
                "changed_to_correct": int(np.sum(changed & (candidate_pred[mask] == y_true[mask]) & (baseline_pred[mask] != y_true[mask]))),
                "changed_to_wrong": int(np.sum(changed & (candidate_pred[mask] != y_true[mask]) & (baseline_pred[mask] == y_true[mask]))),
            })
    return item, route_item, corrections, fold_rows, candidate_pred, candidate_routed, baseline_pred, baseline_routed


def selection_score(item, route_item):
    return float(
        0.35 * item["macro_f1"]
        + 0.25 * item["accuracy"]
        + 0.15 * item["local_subset_balanced_accuracy"]
        + 0.10 * item["per_class_recall"]["C1"]
        + 0.15 * route_item["balanced_accuracy"]
    )


def select_policy(y_true, base_probs, tri, bc1, c1c2, router_probability, folds):
    base_pred = np.argmax(base_probs, axis=1).astype(np.int64)
    baseline_routed = np.isin(base_pred, LOCAL_INDICES)
    baseline_pred, _ = fuse_predictions(base_pred, baseline_routed, tri, bc1, c1c2, c1_vote_bonus=BETA)
    baseline_metrics = classification_metrics(y_true, baseline_pred)
    baseline_route = routing_metrics(y_true, baseline_routed)
    rows = []
    payloads = []
    for policy in policy_grid():
        item, route_item, corrections, fold_rows, candidate_pred, candidate_routed, _, _ = evaluate_policy(
            y_true, base_probs, tri, bc1, c1c2, router_probability, policy, folds
        )
        active_folds = [row for row in fold_rows if row["route_changes"] > 0]
        prediction_folds = [row for row in fold_rows if row["prediction_flips"] > 0]
        harmful_folds = [row for row in prediction_folds if row["changed_to_wrong"] > row["changed_to_correct"]]
        qualified = bool(
            corrections["route_changes"] > 0
            and corrections["prediction_flips"] > 0
            and len(active_folds) >= 3
            and len(prediction_folds) >= 2
            and len(harmful_folds) <= 1
            and corrections["changed_to_correct"] >= corrections["changed_to_wrong"]
            and item["accuracy"] >= baseline_metrics["accuracy"] - 0.0075
            and item["macro_f1"] >= baseline_metrics["macro_f1"] - 0.005
            and item["per_class_recall"]["C1"] >= baseline_metrics["per_class_recall"]["C1"]
            and item["per_class_recall"]["A"] >= baseline_metrics["per_class_recall"]["A"] - (1.0 / 81.0)
            and item["per_class_recall"]["C"] >= baseline_metrics["per_class_recall"]["C"] - (1.0 / 31.0)
        )
        row = {
            "policy": policy["name"],
            "policy_json": json.dumps(policy, sort_keys=True),
            "qualified": qualified,
            "selection_score": selection_score(item, route_item),
            "accuracy": item["accuracy"],
            "balanced_accuracy": item["balanced_accuracy"],
            "macro_f1": item["macro_f1"],
            "weighted_f1": item["weighted_f1"],
            "c1_recall": item["per_class_recall"]["C1"],
            "local_balanced_accuracy": item["local_subset_balanced_accuracy"],
            "routing_balanced_accuracy": route_item["balanced_accuracy"],
            "invocation_rate": route_item["invocation_rate"],
            "route_changes": corrections["route_changes"],
            "prediction_flips": corrections["prediction_flips"],
            "changed_to_correct": corrections["changed_to_correct"],
            "changed_to_wrong": corrections["changed_to_wrong"],
            "active_folds": len(active_folds),
            "prediction_folds": len(prediction_folds),
            "harmful_folds": len(harmful_folds),
        }
        rows.append(row)
        payloads.append((row, policy, item, route_item, corrections, fold_rows))
    qualified = [value for value in payloads if value[0]["qualified"]]
    if not qualified:
        return None, rows, baseline_metrics, baseline_route
    selected = sorted(
        qualified,
        key=lambda value: (
            -value[0]["selection_score"],
            -(value[0]["changed_to_correct"] - value[0]["changed_to_wrong"]),
            value[0]["prediction_flips"],
            value[0]["route_changes"],
        ),
    )[0]
    return {
        "policy": selected[1],
        "metrics": selected[2],
        "routing": selected[3],
        "corrections": selected[4],
        "fold_diagnostics": selected[5],
        "selection_score": selected[0]["selection_score"],
    }, rows, baseline_metrics, baseline_route


def aggregate(rows, section):
    result = {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "local_subset_accuracy", "local_subset_balanced_accuracy", "local_subset_macro_f1"):
        values = [float(row[section][key]) for row in rows]
        result[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    for name in CLASSES:
        values = [float(row[section]["per_class_recall"][name]) for row in rows]
        result["recall_" + name] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    return result


def main():
    args = parse_args()
    configure_device(args)
    report_dir = Path(args.report_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_status(report_dir, "running", "initializing")

    import tensorflow as tf
    from tensorflow import keras
    if args.device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU requested but TensorFlow cannot see a GPU")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    seeds = parse_ints(args.seeds)
    if len(seeds) < 3 and not args.smoke_only:
        raise ValueError("At least three router training seeds are required")
    development_indices, test_indices = load_indices(args)
    oof_rows, row_indices = load_oof_rows(args, development_indices)
    dev_x, dev_y = load_development(args, row_indices)
    if not np.array_equal(dev_y, np.asarray([int(row["true_label"]) for row in oof_rows], dtype=np.int64)):
        raise ValueError("Development labels do not match OOF manifest")
    folds = np.asarray([int(row["fold"]) for row in oof_rows], dtype=np.int64)
    tri, bc1, c1c2 = specialist_arrays_from_rows(oof_rows)
    sim_x, sim_y = load_simulation(args)

    audit = {
        "experiment": "Decoupled Cost-Sensitive Hard Router (DLSR)",
        "frozen_components": ["five-class ResNet50 base", "tri-specialist", "B-vs-C1 specialist", "C1-vs-C2 specialist", "beta=0.20 fusion"],
        "new_component": "lightweight binary global/local router trained from frozen base features",
        "route_target": {"global": ["A", "C"], "local": ["B", "C1", "C2"]},
        "candidate_policies": ["router_only", "dual_confirm", "dual_confirm_rescue"],
        "cost_sensitive_training": True,
        "selection": "five-fold CHIME development-only OOF",
        "paper_test_samples_indexed_before_lock": False,
        "paper_latex_modified": False,
        "final_checkpoints_overwritten": False,
        "seeds": list(seeds),
    }
    save_json(report_dir / "static_audit.json", audit)

    configs, smoke_epochs = candidate_configs(args.router_batch_size, args.smoke_only)
    epochs = smoke_epochs if smoke_epochs is not None else args.epochs
    selection_rows = []
    for config_index, config in enumerate(configs):
        write_status(report_dir, "running", "development_oof", candidate=config["name"], completed=config_index, total=len(configs))
        pooled_probability = np.zeros(len(dev_y), dtype=np.float32)
        pooled_base_probs = np.zeros((len(dev_y), 5), dtype=np.float32)
        fold_training = []
        for fold in range(5):
            write_status(report_dir, "running", "development_oof_fold", candidate=config["name"], fold=fold)
            feature_dir = artifact_dir / "features" / ("fold_%d" % fold)
            sim_bundle = extract_base_bundle(tf, keras, sim_x, fold_base_path(fold), args.batch_size, feature_dir / "simulation.npz")
            dev_bundle = extract_base_bundle(tf, keras, dev_x, fold_base_path(fold), args.batch_size, feature_dir / "development.npz")
            train_mask = folds != fold
            validation_mask = folds == fold
            arrays = combine_training_data(sim_bundle, sim_y, dev_bundle, dev_y, train_mask, config)
            model, training = train_router(
                tf, keras, arrays, config, seeds[0] + fold,
                artifact_dir / "selection" / config["name"] / ("fold_%d" % fold), epochs,
            )
            pooled_probability[validation_mask] = predict_router(
                model, {key: value[validation_mask] for key, value in dev_bundle.items()}, args.router_batch_size
            )
            pooled_base_probs[validation_mask] = dev_bundle["base_probs"][validation_mask]
            fold_training.append({"fold": fold, "training": training, "n_validation": int(np.sum(validation_mask))})
            del model, arrays, sim_bundle, dev_bundle
            keras.backend.clear_session()
            gc.collect()
        selected_policy, sweep, baseline_metrics, baseline_route = select_policy(
            dev_y, pooled_base_probs, tri, bc1, c1c2, pooled_probability, folds
        )
        candidate = {
            "config": config,
            "selected_policy": selected_policy,
            "active_policy_found": selected_policy is not None,
            "baseline_metrics": baseline_metrics,
            "baseline_routing": baseline_route,
            "fold_training": fold_training,
        }
        selection_rows.append(candidate)
        candidate_dir = report_dir / "selection" / config["name"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        save_json(candidate_dir / "result.json", candidate)
        write_csv(candidate_dir / "policy_sweep.csv", sweep, list(sweep[0].keys()))
        np.savez_compressed(
            str(candidate_dir / "oof_router_predictions.npz"),
            router_probability=pooled_probability,
            base_probs=pooled_base_probs,
            y_true=dev_y,
            folds=folds,
        )

    active = [row for row in selection_rows if row["active_policy_found"]]
    if active:
        selected = sorted(
            active,
            key=lambda row: (
                -row["selected_policy"]["selection_score"],
                -(row["selected_policy"]["corrections"]["changed_to_correct"] - row["selected_policy"]["corrections"]["changed_to_wrong"]),
                row["config"]["hidden_dim"],
                row["config"]["name"],
            ),
        )[0]
    else:
        selected = selection_rows[0]
    active_selected = bool(selected["active_policy_found"])
    baseline_dev = selection_rows[0]["baseline_metrics"]
    selection_payload = {
        "baseline": baseline_dev,
        "candidates": selection_rows,
        "selected": selected,
        "selected_passed_development_guard": active_selected,
        "selection_scope": "CHIME development-only five-fold OOF",
        "paper_test_access": False,
    }
    save_json(report_dir / "selection_results.json", selection_payload)
    locked = {
        "method": "DLSR" if active_selected else "top1_current_fallback",
        "active": active_selected,
        "router_config": selected["config"],
        "policy": selected["selected_policy"]["policy"] if active_selected else None,
        "development_baseline": baseline_dev,
        "development_selected": selected["selected_policy"] if active_selected else None,
        "selection_scope": "CHIME development-only five-fold OOF",
        "paper_test_samples_indexed_before_lock": False,
        "beta": BETA,
        "seeds": list(seeds),
        "adoption_guard": {
            "max_accuracy_drop": 0.01,
            "max_macro_f1_drop": 0.005,
            "C1_recall_not_lower": True,
            "minimum_favorable_seeds": 2,
            "router_must_be_active": True,
        },
    }
    save_json(report_dir / "locked_config.json", locked)
    write_status(report_dir, "running", "configuration_locked", active=active_selected, candidate=selected["config"]["name"])

    if args.smoke_only:
        save_json(report_dir / "smoke_result.json", locked)
        write_status(report_dir, "complete", "smoke_complete")
        return

    final_model_paths = final_paths()
    final_feature_dir = artifact_dir / "features/final_system"
    sim_bundle = extract_base_bundle(tf, keras, sim_x, final_model_paths["base"], args.batch_size, final_feature_dir / "simulation.npz")
    dev_bundle = extract_base_bundle(tf, keras, dev_x, final_model_paths["base"], args.batch_size, final_feature_dir / "development.npz")

    # Reload the containing archive only after the development configuration is
    # locked, then index paper-test samples for the first time.
    raw = load_npz_array(PROJECT_ROOT / args.chime_data)
    all_labels = read_labels(PROJECT_ROOT / args.chime_labels)
    test_x = normalize_images_to_uint8(raw[test_indices])[..., np.newaxis]
    test_y = all_labels[test_indices]
    test_bundle = extract_base_bundle(tf, keras, test_x, final_model_paths["base"], args.batch_size, final_feature_dir / "paper_test_after_lock.npz")

    import losses  # noqa: F401
    specialist_cache = final_feature_dir / "paper_test_specialists_after_lock.npz"
    if specialist_cache.exists():
        with np.load(str(specialist_cache)) as data:
            test_tri = np.asarray(data["tri"])
            test_bc1 = np.asarray(data["bc1"])
            test_c1c2 = np.asarray(data["c1c2"])
    else:
        dataset = prediction_dataset(tf, test_x, args.batch_size)
        values = {}
        for name in ("tri", "bc1", "c1c2"):
            model = keras.models.load_model(str(final_model_paths[name]))
            values[name] = model.predict(dataset, verbose=0)
            del model
            keras.backend.clear_session()
            gc.collect()
        test_tri, test_bc1, test_c1c2 = values["tri"], values["bc1"], values["c1c2"]
        np.savez_compressed(str(specialist_cache), tri=test_tri, bc1=test_bc1, c1c2=test_c1c2)

    base_pred = test_bundle["base_pred"]
    baseline_routed = np.isin(base_pred, LOCAL_INDICES)
    baseline_pred, _ = fuse_predictions(base_pred, baseline_routed, test_tri, test_bc1, test_c1c2, c1_vote_bonus=BETA)
    baseline_metrics = classification_metrics(test_y, baseline_pred)
    baseline_route = routing_metrics(test_y, baseline_routed)
    write_matrix(report_dir / "current_top1_confusion_matrix.csv", baseline_metrics["confusion_matrix"])

    final_rows = []
    all_dev = np.ones(len(dev_y), dtype=bool)
    for seed_index, seed in enumerate(seeds):
        write_status(report_dir, "running", "final_seed_training", seed=seed, completed=seed_index, total=len(seeds))
        seed_report = report_dir / "final_seeds" / ("seed_%d" % seed)
        result_path = seed_report / "metrics.json"
        if result_path.exists():
            final_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        arrays = combine_training_data(sim_bundle, sim_y, dev_bundle, dev_y, all_dev, selected["config"])
        model, training = train_router(
            tf, keras, arrays, selected["config"], seed,
            artifact_dir / "final_seeds" / ("seed_%d" % seed), args.epochs,
        )
        started = time.perf_counter()
        for _ in range(20):
            router_probability = predict_router(model, test_bundle, args.router_batch_size)
        router_ms = (time.perf_counter() - started) * 1000.0 / (20 * len(test_y))
        if active_selected:
            policy = selected["selected_policy"]["policy"]
            item, route_item, corrections, _, candidate_pred, candidate_routed, _, _ = evaluate_policy(
                test_y, test_bundle["base_probs"], test_tri, test_bc1, test_c1c2, router_probability, policy
            )
        else:
            policy = None
            candidate_pred = baseline_pred.copy()
            candidate_routed = baseline_routed.copy()
            item = baseline_metrics
            route_item = baseline_route
            corrections = correction_stats(test_y, base_pred, baseline_pred, candidate_pred, baseline_routed, candidate_routed)
        payload = {
            "seed": seed,
            "policy": policy,
            "metrics": item,
            "routing": route_item,
            "baseline_metrics": baseline_metrics,
            "baseline_routing": baseline_route,
            "corrections": corrections,
            "router_parameters": int(model.count_params()),
            "router_ms_per_sample": float(router_ms),
            "training": training,
            "paper_test_access": "after locked_config.json",
        }
        seed_report.mkdir(parents=True, exist_ok=True)
        save_json(result_path, payload)
        write_matrix(seed_report / "confusion_matrix.csv", item["confusion_matrix"])
        rows = []
        for position, global_index in enumerate(test_indices):
            rows.append({
                "sample_index": int(global_index),
                "true_label": CLASSES[int(test_y[position])],
                "base_pred": CLASSES[int(base_pred[position])],
                "current_top1_pred": CLASSES[int(baseline_pred[position])],
                "dlsr_pred": CLASSES[int(candidate_pred[position])],
                "current_routed": int(baseline_routed[position]),
                "dlsr_routed": int(candidate_routed[position]),
                "router_local_probability": float(router_probability[position]),
            })
        write_csv(seed_report / "predictions.csv", rows, list(rows[0].keys()))
        final_rows.append(payload)
        print("DLSR seed=%d acc=%.4f macro=%.4f C1=%.3f route=%.3f flips=%d correct=%d wrong=%d" % (
            seed, item["accuracy"], item["macro_f1"], item["per_class_recall"]["C1"], route_item["invocation_rate"],
            corrections["prediction_flips"], corrections["changed_to_correct"], corrections["changed_to_wrong"],
        ), flush=True)
        del model, arrays, router_probability
        keras.backend.clear_session()
        gc.collect()

    aggregate_metrics = aggregate(final_rows, "metrics")
    favorable = sum(
        row["corrections"]["prediction_flips"] > 0
        and row["corrections"]["changed_to_correct"] >= row["corrections"]["changed_to_wrong"]
        for row in final_rows
    )
    active_seeds = sum(row["corrections"]["route_changes"] > 0 for row in final_rows)
    criteria = {
        "accuracy_within_1pp": bool(aggregate_metrics["accuracy"]["mean"] >= baseline_metrics["accuracy"] - 0.01),
        "macro_f1_within_0p5pp": bool(aggregate_metrics["macro_f1"]["mean"] >= baseline_metrics["macro_f1"] - 0.005),
        "c1_recall_not_lower": bool(aggregate_metrics["recall_C1"]["mean"] >= baseline_metrics["per_class_recall"]["C1"]),
        "a_recall_drop_at_most_one_sample": bool(aggregate_metrics["recall_A"]["mean"] >= baseline_metrics["per_class_recall"]["A"] - (1.0 / 82.0)),
        "c_recall_drop_at_most_one_sample": bool(aggregate_metrics["recall_C"]["mean"] >= baseline_metrics["per_class_recall"]["C"] - (1.0 / 31.0)),
        "at_least_two_favorable_seeds": bool(favorable >= 2),
        "router_active_all_seeds": bool(active_seeds == len(seeds)),
        "development_guard_passed": active_selected,
    }
    criteria["recommend_replacement"] = bool(all(criteria.values()))
    final = {
        "locked_config": locked,
        "current_top1": {"metrics": baseline_metrics, "routing": baseline_route},
        "aggregate": aggregate_metrics,
        "per_seed": final_rows,
        "criteria": criteria,
        "paper_test_used_for_selection": False,
    }
    save_json(report_dir / "final_results.json", final)
    metric_rows = []
    for row in final_rows:
        metric_rows.append({
            "seed": row["seed"],
            "accuracy": row["metrics"]["accuracy"],
            "balanced_accuracy": row["metrics"]["balanced_accuracy"],
            "macro_f1": row["metrics"]["macro_f1"],
            "weighted_f1": row["metrics"]["weighted_f1"],
            "c1_recall": row["metrics"]["per_class_recall"]["C1"],
            "local_balanced_accuracy": row["metrics"]["local_subset_balanced_accuracy"],
            "routing_balanced_accuracy": row["routing"]["balanced_accuracy"],
            "invocation_rate": row["routing"]["invocation_rate"],
            "route_changes": row["corrections"]["route_changes"],
            "prediction_flips": row["corrections"]["prediction_flips"],
            "changed_to_correct": row["corrections"]["changed_to_correct"],
            "changed_to_wrong": row["corrections"]["changed_to_wrong"],
            "router_parameters": row["router_parameters"],
            "router_ms_per_sample": row["router_ms_per_sample"],
        })
    write_csv(report_dir / "multiseed_metrics.csv", metric_rows, list(metric_rows[0].keys()))

    lines = [
        "# Decoupled Cost-Sensitive Hard Router", "", "## Protocol", "",
        "- The published five-class base and all specialists are frozen.",
        "- A separate binary router predicts global A/C versus local B/C1/C2 from frozen base features.",
        "- Router architecture, asymmetric costs, policy, and thresholds were selected with five-fold CHIME development-only OOF results.",
        "- Paper-test samples were first indexed after `locked_config.json` was written.",
        "- The existing beta=0.20 specialist fusion is unchanged.", "", "## Locked Configuration", "",
        "- Active: `%s`" % active_selected,
        "- Router: `%s`" % selected["config"],
        "- Policy: `%s`" % (selected["selected_policy"]["policy"] if active_selected else None), "", "## Paper-Test Aggregate", "",
        "| Model | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | C1 recall | Local balanced acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Current Top-1 | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            baseline_metrics["accuracy"], baseline_metrics["balanced_accuracy"], baseline_metrics["macro_f1"],
            baseline_metrics["weighted_f1"], baseline_metrics["per_class_recall"]["C1"], baseline_metrics["local_subset_balanced_accuracy"],
        ),
        "| DLSR mean +/- std | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f |" % (
            aggregate_metrics["accuracy"]["mean"], aggregate_metrics["accuracy"]["std"],
            aggregate_metrics["balanced_accuracy"]["mean"], aggregate_metrics["balanced_accuracy"]["std"],
            aggregate_metrics["macro_f1"]["mean"], aggregate_metrics["macro_f1"]["std"],
            aggregate_metrics["weighted_f1"]["mean"], aggregate_metrics["weighted_f1"]["std"],
            aggregate_metrics["recall_C1"]["mean"], aggregate_metrics["recall_C1"]["std"],
            aggregate_metrics["local_subset_balanced_accuracy"]["mean"], aggregate_metrics["local_subset_balanced_accuracy"]["std"],
        ), "", "## Decision", "",
        "- Criteria: `%s`" % criteria,
        "- Recommendation: %s" % ("DLSR is a candidate replacement pending independent review; current final checkpoints remain untouched." if criteria["recommend_replacement"] else "retain current Top-1 routing; keep DLSR as an ablation."),
        "", "The paper LaTeX, final checkpoints, and all prior reports were not modified.",
    ]
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(report_dir / "run_metadata.json", {"python": platform.python_version(), "platform": platform.platform(), "git_commit": git_commit_or_none()})
    save_json(report_dir / "run_complete.json", {"status": "complete", "active": active_selected, "n_seeds": len(seeds), "recommend_replacement": criteria["recommend_replacement"]})
    write_status(report_dir, "complete", "all_complete", recommendation=criteria["recommend_replacement"])


if __name__ == "__main__":
    main()
