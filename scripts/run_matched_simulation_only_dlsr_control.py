#!/usr/bin/env python
"""Matched simulation-only DLSR control and paired paper-test analysis."""

from __future__ import print_function

import argparse
import csv
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src/hsfnet", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data import CLASSES, CLASS_TO_INDEX, load_npz_array, load_simulation_dataset_with_snr, normalize_images_to_uint8, save_json  # noqa: E402
from evaluate_binary_specialist_fusion import fuse_predictions  # noqa: E402
import run_decoupled_cost_sensitive_hard_router as dlsr  # noqa: E402


LOCAL = np.asarray([CLASS_TO_INDEX[x] for x in ("B", "C1", "C2")], dtype=np.int64)
SIM_ROOT = ROOT / "artifacts/runs/sim_only_mainline"
ADAPTED_REPORT = ROOT / "reports/decoupled_cost_sensitive_hard_router"


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", default=str(ROOT / "reports/matched_simulation_only_dlsr_control"))
    p.add_argument("--artifact-dir", default=str(ROOT / "artifacts/runs/matched_simulation_only_dlsr_control"))
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--router-batch-size", type=int, default=256)
    p.add_argument("--gpu-id", default="1")
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    return p.parse_args()


def configure(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-matched-dlsr")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def read_labels(path):
    values = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return np.asarray([CLASS_TO_INDEX[x] for x in values], dtype=np.int64)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path, matrix):
    rows = []
    for name, values in zip(CLASSES, matrix):
        row = {"true\\pred": name}
        row.update({label: int(value) for label, value in zip(CLASSES, values)})
        rows.append(row)
    write_csv(path, rows, ["true\\pred"] + list(CLASSES))


def status(report, stage, state="running", **extra):
    payload = {"state": state, "stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    payload.update(extra)
    save_json(Path(report) / "status.json", payload)


def model_paths():
    return {
        "base": SIM_ROOT / "resnet50_base_seed_42_gpu0/best_model",
        "tri": SIM_ROOT / "specialist_b_c1_c2_seed_42_gpu0/best_model",
        "bc1": SIM_ROOT / "pair_B_vs_C1_seed_42_gpu0/best_model",
        "c1c2": SIM_ROOT / "pair_C1_vs_C2_seed_42_gpu0/best_model",
    }


def predict_specialists(tf, keras, images, paths, batch_size, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(str(cache_path)) as data:
            return tuple(np.asarray(data[k]) for k in ("tri", "bc1", "c1c2"))
    values = {}
    import losses  # noqa: F401
    for key in ("tri", "bc1", "c1c2"):
        model = keras.models.load_model(str(paths[key]))
        values[key] = np.asarray(model.predict(dlsr.prediction_dataset(tf, images, batch_size), verbose=0), dtype=np.float32)
        del model
        keras.backend.clear_session()
        gc.collect()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **values)
    return values["tri"], values["bc1"], values["c1c2"]


def cm_metrics(y, pred):
    cm = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros(5), where=support != 0)
    precision = np.divide(tp, predicted, out=np.zeros(5), where=predicted != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) != 0)
    local_mask = np.isin(y, LOCAL)
    local_cm = np.bincount(y[local_mask] * 5 + pred[local_mask], minlength=25).reshape(5, 5)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.sum(f1 * support) / np.sum(support)),
        "per_class_recall": {name: float(recall[i]) for i, name in enumerate(CLASSES)},
        "local_subset_accuracy": float(np.mean(y[local_mask] == pred[local_mask])),
        "local_subset_balanced_accuracy": float(np.mean(recall[LOCAL])),
        "confusion_matrix": cm.tolist(),
        "local_confusion_matrix": local_cm.tolist(),
    }


def exact_mcnemar(y, control, adapted):
    control_ok = control == y
    adapted_ok = adapted == y
    n01 = int(np.sum((~control_ok) & adapted_ok))
    n10 = int(np.sum(control_ok & (~adapted_ok)))
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"control_wrong_adapted_correct": n01, "control_correct_adapted_wrong": n10, "discordant": n, "exact_two_sided_p": p}


def metric_value(y, pred, key):
    return cm_metrics(y, pred)[key]


def stratified_indices(y, rng):
    parts = []
    for label in range(5):
        idx = np.where(y == label)[0]
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def paired_resampling(y, control, adapted, replicates, seed):
    rng = np.random.RandomState(seed)
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "local_subset_balanced_accuracy")
    observed = {k: metric_value(y, adapted, k) - metric_value(y, control, k) for k in keys}
    boot = {k: np.empty(replicates, dtype=np.float64) for k in keys}
    perm = {k: np.empty(replicates, dtype=np.float64) for k in keys}
    for r in range(replicates):
        idx = stratified_indices(y, rng)
        for k in keys:
            boot[k][r] = metric_value(y[idx], adapted[idx], k) - metric_value(y[idx], control[idx], k)
        swap = rng.rand(len(y)) < 0.5
        left = np.where(swap, adapted, control)
        right = np.where(swap, control, adapted)
        for k in keys:
            perm[k][r] = metric_value(y, right, k) - metric_value(y, left, k)
    result = {}
    for k in keys:
        result[k] = {
            "adapted_minus_control": float(observed[k]),
            "bootstrap_95_ci": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))],
            "paired_randomization_two_sided_p": float((1 + np.sum(np.abs(perm[k]) >= abs(observed[k]))) / (replicates + 1)),
            "replicates": int(replicates),
        }
    return result


def correction_counts(y, control, adapted, control_route, adapted_route):
    changed = control != adapted
    c_ok = control == y
    a_ok = adapted == y
    return {
        "prediction_flips": int(np.sum(changed)),
        "control_wrong_adapted_correct": int(np.sum(changed & (~c_ok) & a_ok)),
        "control_correct_adapted_wrong": int(np.sum(changed & c_ok & (~a_ok))),
        "wrong_to_different_wrong": int(np.sum(changed & (~c_ok) & (~a_ok))),
        "same_prediction": int(np.sum(~changed)),
        "route_changes": int(np.sum(control_route != adapted_route)),
        "control_only_routes": int(np.sum(control_route & (~adapted_route))),
        "adapted_only_routes": int(np.sum((~control_route) & adapted_route)),
    }


def aggregate(rows, section):
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "local_subset_accuracy", "local_subset_balanced_accuracy")
    out = {}
    for key in keys:
        values = np.asarray([r[section][key] for r in rows], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    for name in CLASSES:
        values = np.asarray([r[section]["per_class_recall"][name] for r in rows], dtype=np.float64)
        out["recall_" + name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return out


def load_adapted(seed, expected_indices, expected_y):
    path = ADAPTED_REPORT / "final_seeds" / ("seed_%d" % seed) / "predictions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    indices = np.asarray([int(r["sample_index"]) for r in rows], dtype=np.int64)
    y = np.asarray([CLASS_TO_INDEX[r["true_label"]] for r in rows], dtype=np.int64)
    if not np.array_equal(indices, expected_indices) or not np.array_equal(y, expected_y):
        raise ValueError("Adapted DLSR predictions are not aligned for seed %d" % seed)
    return {
        "pred": np.asarray([CLASS_TO_INDEX[r["dlsr_pred"]] for r in rows], dtype=np.int64),
        "route": np.asarray([bool(int(r["dlsr_routed"])) for r in rows]),
        "router_probability": np.asarray([float(r["router_local_probability"]) for r in rows], dtype=np.float32),
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    args = args_parser()
    configure(args)
    report = Path(args.report_dir).resolve()
    artifact = Path(args.artifact_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    status(report, "initializing")

    locked = json.loads((ADAPTED_REPORT / "locked_config.json").read_text(encoding="utf-8"))
    config = dict(locked["router_config"])
    policy = dict(locked["policy"])
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    paths = model_paths()
    for key, path in paths.items():
        if not (path / "saved_model.pb").exists():
            raise FileNotFoundError("Missing simulation-only %s model: %s" % (key, path))

    protocol = {
        "experiment": "matched simulation-only DLSR control",
        "control_training_samples": "simulation only",
        "chime_development_samples_used_by_control_optimizer": 0,
        "configuration_origin": "inherited unchanged from development-locked adapted DLSR",
        "router_config": config,
        "policy": policy,
        "beta": float(locked["beta"]),
        "seeds": list(seeds),
        "test_split": "same immutable 268-sample CHIME paper-test half",
        "selection_on_paper_test": False,
        "claim_boundary": "simulation-only training under a fixed DLSR configuration; architecture selection historically used CHIME development OOF",
        "model_paths": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }
    save_json(report / "locked_protocol.json", protocol)

    import tensorflow as tf
    from tensorflow import keras
    if args.device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU requested but TensorFlow cannot see a GPU")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    status(report, "loading_simulation")
    sim_x, sim_y, _, _ = load_simulation_dataset_with_snr(
        ROOT / "data/simulated/simfrb", samples_per_class=1200,
        snrs=(15, 25, 35, 50, 100), seed=42, exclude_marked=True,
    )
    feature_dir = artifact / "features"
    sim_bundle = dlsr.extract_base_bundle(tf, keras, sim_x, paths["base"], args.batch_size, feature_dir / "simulation.npz")

    split_path = ROOT / "artifacts/runs/resnet50_c1_repair_gpu0/paper_chime_split_indices.npz"
    with np.load(str(split_path)) as split:
        test_indices = np.asarray(split["test_idx"], dtype=np.int64)
    raw = load_npz_array(ROOT / "data/chime/chime_interp_frbs_data.npz")
    labels = read_labels(ROOT / "data/chime/chime_labels.txt")
    test_x = normalize_images_to_uint8(raw[test_indices])[..., np.newaxis]
    test_y = labels[test_indices]
    test_bundle = dlsr.extract_base_bundle(tf, keras, test_x, paths["base"], args.batch_size, feature_dir / "paper_test.npz")
    tri, bc1, c1c2 = predict_specialists(tf, keras, test_x, paths, args.batch_size, feature_dir / "paper_test_specialists.npz")

    base_pred = np.argmax(test_bundle["base_probs"], axis=1).astype(np.int64)
    baseline_route = np.isin(base_pred, LOCAL)
    baseline_pred, _ = fuse_predictions(base_pred, baseline_route, tri, bc1, c1c2, c1_vote_bonus=float(locked["beta"]))
    save_json(report / "data_manifest.json", {
        "n_simulation_router_train": int(len(sim_y)),
        "n_paper_test": int(len(test_y)),
        "paper_test_indices": test_indices.tolist(),
        "simulation_class_counts": {name: int(np.sum(sim_y == i)) for i, name in enumerate(CLASSES)},
        "paper_test_class_counts": {name: int(np.sum(test_y == i)) for i, name in enumerate(CLASSES)},
    })
    save_json(report / "checkpoint_manifest.json", {
        k: {"path": str(v.relative_to(ROOT)), "train_config": str((v.parent / "train_config.json").relative_to(ROOT))}
        for k, v in paths.items()
    })

    final_rows = []
    for seed in seeds:
        status(report, "training_router", seed=seed)
        arrays = (
            sim_bundle["inputs"],
            np.isin(sim_y, LOCAL).astype(np.float32),
            np.ones(len(sim_y), dtype=np.float32),
        )
        model, training = dlsr.train_router(tf, keras, arrays, config, seed, artifact / "final_seeds" / ("seed_%d" % seed), args.epochs)
        router_probability = dlsr.predict_router(model, test_bundle, args.router_batch_size)
        control_metrics, control_routing, _, _, control_pred, control_route, _, _ = dlsr.evaluate_policy(
            test_y, test_bundle["base_probs"], tri, bc1, c1c2, router_probability, policy
        )
        adapted = load_adapted(seed, test_indices, test_y)
        adapted_metrics = cm_metrics(test_y, adapted["pred"])
        corrections = correction_counts(test_y, control_pred, adapted["pred"], control_route, adapted["route"])
        mcnemar = exact_mcnemar(test_y, control_pred, adapted["pred"])
        paired = paired_resampling(test_y, control_pred, adapted["pred"], args.bootstrap_replicates, seed + 1000)

        seed_dir = report / "final_seeds" / ("seed_%d" % seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_matrix(seed_dir / "simulation_only_confusion_matrix.csv", control_metrics["confusion_matrix"])
        write_matrix(seed_dir / "adapted_confusion_matrix.csv", adapted_metrics["confusion_matrix"])
        rows = []
        for i, sample_index in enumerate(test_indices):
            row = {
                "sample_index": int(sample_index), "true_label": CLASSES[int(test_y[i])],
                "simulation_only_base_pred": CLASSES[int(base_pred[i])],
                "simulation_only_dlsr_pred": CLASSES[int(control_pred[i])],
                "adapted_dlsr_pred": CLASSES[int(adapted["pred"][i])],
                "simulation_only_correct": int(control_pred[i] == test_y[i]),
                "adapted_correct": int(adapted["pred"][i] == test_y[i]),
                "prediction_flip": int(control_pred[i] != adapted["pred"][i]),
                "simulation_only_routed": int(control_route[i]), "adapted_routed": int(adapted["route"][i]),
                "simulation_only_router_local_probability": float(router_probability[i]),
                "adapted_router_local_probability": float(adapted["router_probability"][i]),
            }
            for j, name in enumerate(CLASSES):
                row["simulation_only_base_prob_" + name] = float(test_bundle["base_probs"][i, j])
            for j, name in enumerate(("B", "C1", "C2")):
                row["simulation_only_tri_prob_" + name] = float(tri[i, j])
            row.update({
                "simulation_only_bc1_prob_B": float(bc1[i, 0]), "simulation_only_bc1_prob_C1": float(bc1[i, 1]),
                "simulation_only_c1c2_prob_C1": float(c1c2[i, 0]), "simulation_only_c1c2_prob_C2": float(c1c2[i, 1]),
            })
            rows.append(row)
        write_csv(seed_dir / "matched_predictions.csv", rows)
        payload = {
            "seed": seed, "simulation_only_metrics": control_metrics, "simulation_only_routing": control_routing,
            "adapted_metrics": adapted_metrics, "corrections_control_to_adapted": corrections,
            "mcnemar": mcnemar, "paired_resampling": paired, "training": training,
        }
        save_json(seed_dir / "metrics_and_paired_tests.json", payload)
        final_rows.append(payload)
        print("seed=%d control acc=%.4f macro=%.4f adapted acc=%.4f macro=%.4f flips=%d W->C=%d C->W=%d" % (
            seed, control_metrics["accuracy"], control_metrics["macro_f1"], adapted_metrics["accuracy"], adapted_metrics["macro_f1"],
            corrections["prediction_flips"], corrections["control_wrong_adapted_correct"], corrections["control_correct_adapted_wrong"]
        ), flush=True)
        del model
        keras.backend.clear_session()
        gc.collect()

    control_agg = aggregate(final_rows, "simulation_only_metrics")
    adapted_agg = aggregate(final_rows, "adapted_metrics")
    metric_rows = []
    for row in final_rows:
        c, a = row["simulation_only_metrics"], row["adapted_metrics"]
        metric_rows.append({
            "seed": row["seed"], "control_accuracy": c["accuracy"], "adapted_accuracy": a["accuracy"],
            "accuracy_difference_pp": 100 * (a["accuracy"] - c["accuracy"]),
            "control_macro_f1": c["macro_f1"], "adapted_macro_f1": a["macro_f1"],
            "macro_f1_difference_pp": 100 * (a["macro_f1"] - c["macro_f1"]),
            "control_balanced_accuracy": c["balanced_accuracy"], "adapted_balanced_accuracy": a["balanced_accuracy"],
            "control_c1_recall": c["per_class_recall"]["C1"], "adapted_c1_recall": a["per_class_recall"]["C1"],
            "prediction_flips": row["corrections_control_to_adapted"]["prediction_flips"],
            "wrong_to_correct": row["corrections_control_to_adapted"]["control_wrong_adapted_correct"],
            "correct_to_wrong": row["corrections_control_to_adapted"]["control_correct_adapted_wrong"],
            "mcnemar_p": row["mcnemar"]["exact_two_sided_p"],
        })
    write_csv(report / "multiseed_metrics.csv", metric_rows)

    final = {
        "protocol": protocol, "simulation_only_aggregate": control_agg, "adapted_aggregate": adapted_agg,
        "adapted_minus_control": {
            key: 100.0 * (adapted_agg[key]["mean"] - control_agg[key]["mean"])
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "local_subset_balanced_accuracy")
        },
        "per_seed": final_rows,
    }
    save_json(report / "final_results.json", final)

    acc_delta = final["adapted_minus_control"]["accuracy"]
    macro_delta = final["adapted_minus_control"]["macro_f1"]
    correction_mean = np.mean([r["corrections_control_to_adapted"]["control_wrong_adapted_correct"] for r in final_rows])
    harm_mean = np.mean([r["corrections_control_to_adapted"]["control_correct_adapted_wrong"] for r in final_rows])
    lines = [
        "# Matched Simulation-Only DLSR Control", "", "## Protocol", "",
        "- The DLSR architecture, focal costs, dual-confirm policy, threshold 0.40, C1 protection, beta=0.20, test samples, preprocessing, and label mapping are fixed to the adapted system.",
        "- Every trainable control checkpoint (base, tri-specialist, pair specialists, and router) receives simulation samples only.",
        "- No CHIME development sample enters the control optimizer; no paper-test sample is used for tuning or selection.",
        "- The configuration itself is inherited from the previously development-locked DLSR, so the precise claim is simulation-only training under a fixed DLSR configuration.",
        "", "## Paper-Test Results", "",
        "| System | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | C1 recall | Local balanced acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Simulation-only DLSR | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f |" % (
            control_agg["accuracy"]["mean"], control_agg["accuracy"]["std"], control_agg["balanced_accuracy"]["mean"], control_agg["balanced_accuracy"]["std"],
            control_agg["macro_f1"]["mean"], control_agg["macro_f1"]["std"], control_agg["weighted_f1"]["mean"], control_agg["weighted_f1"]["std"],
            control_agg["recall_C1"]["mean"], control_agg["recall_C1"]["std"], control_agg["local_subset_balanced_accuracy"]["mean"], control_agg["local_subset_balanced_accuracy"]["std"]),
        "| CHIME-adapted DLSR | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f | %.4f +/- %.4f |" % (
            adapted_agg["accuracy"]["mean"], adapted_agg["accuracy"]["std"], adapted_agg["balanced_accuracy"]["mean"], adapted_agg["balanced_accuracy"]["std"],
            adapted_agg["macro_f1"]["mean"], adapted_agg["macro_f1"]["std"], adapted_agg["weighted_f1"]["mean"], adapted_agg["weighted_f1"]["std"],
            adapted_agg["recall_C1"]["mean"], adapted_agg["recall_C1"]["std"], adapted_agg["local_subset_balanced_accuracy"]["mean"], adapted_agg["local_subset_balanced_accuracy"]["std"]),
        "", "## Matched Effect", "",
        "- Accuracy difference (adapted - control): %.2f percentage points." % acc_delta,
        "- Macro-F1 difference (adapted - control): %.2f percentage points." % macro_delta,
        "- Mean wrong-to-correct changes per seed: %.2f." % correction_mean,
        "- Mean correct-to-wrong changes per seed: %.2f." % harm_mean,
        "- Exact McNemar and 10,000-replicate paired stratified bootstrap/randomization results are stored per seed.",
        "", "This matched control supplies the missing per-sample evidence for the simulation-to-real adaptation claim.",
    ]
    (report / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(report / "run_metadata.json", {"python": platform.python_version(), "platform": platform.platform(), "git_commit": git_commit()})
    save_json(report / "run_complete.json", {"status": "complete", "n_seeds": len(seeds), "bootstrap_replicates": args.bootstrap_replicates})
    status(report, "all_complete", state="complete")


if __name__ == "__main__":
    main()
