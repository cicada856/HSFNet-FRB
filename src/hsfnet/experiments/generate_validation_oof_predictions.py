"""Train five outer-CV HSFNet-FRB component sets and emit development OOF predictions."""

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

from data import (  # noqa: E402
    CLASSES,
    CLASS_TO_INDEX,
    build_real_noise_bank,
    ensure_dir,
    load_npz_array,
    load_simulation_dataset_with_snr,
    normalize_images_to_uint8,
    save_json,
    stratified_train_val_split,
)


OOF_FIELDS = (
    "sample_index",
    "fold",
    "true_label",
    "base_pred",
    "tri_prob_B",
    "tri_prob_C1",
    "tri_prob_C2",
    "bc1_prob_B",
    "bc1_prob_C1",
    "c1c2_prob_C1",
    "c1c2_prob_C2",
)
TRI_CLASSES = ("B", "C1", "C2")
TRI_GLOBAL = np.asarray([CLASS_TO_INDEX[name] for name in TRI_CLASSES], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate leakage-controlled HSFNet-FRB OOF component predictions.")
    parser.add_argument("--sim-dir", default="data/simulated/simfrb")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument(
        "--development-indices",
        default="artifacts/runs/resnet50_c1_repair_gpu0/paper_chime_split_indices.npz",
        help="Existing immutable paper split file; only train_idx is accessed in this stage.",
    )
    parser.add_argument("--output-dir", default="reports/validation_beta_selection")
    parser.add_argument("--model-root", default="artifacts/runs/validation_beta_selection")
    parser.add_argument("--samples-per-class", type=int, default=1800)
    parser.add_argument("--snrs", default="15,25,35,50,100")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size-base", type=int, default=8)
    parser.add_argument("--batch-size-specialist", type=int, default=16)
    parser.add_argument("--batch-size-inference", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu")
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def configure_device(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def read_development_labels(label_path, development_indices):
    wanted = set(int(value) for value in development_indices)
    labels_by_index = {}
    with Path(label_path).open("r", encoding="utf-8") as f:
        for index, raw_line in enumerate(f):
            if index not in wanted:
                continue
            label = raw_line.strip()
            if label not in CLASS_TO_INDEX:
                raise ValueError("Unknown development label at CHIME index %d: %s" % (index, label))
            labels_by_index[index] = CLASS_TO_INDEX[label]
    if set(labels_by_index) != wanted:
        missing = sorted(wanted - set(labels_by_index))
        raise ValueError("Missing development labels for CHIME indices: %s" % missing[:10])
    return np.asarray([labels_by_index[int(index)] for index in development_indices], dtype=np.int64)


def load_development_only(data_path, label_path, split_path):
    with np.load(str(split_path)) as split:
        development_indices = np.asarray(split["train_idx"], dtype=np.int64)
    if len(development_indices) != 267 or len(np.unique(development_indices)) != 267:
        raise ValueError("Expected 267 unique development indices, found %d." % len(development_indices))

    all_images = load_npz_array(data_path)
    development_x = normalize_images_to_uint8(all_images[development_indices])[..., np.newaxis]
    del all_images
    development_y = read_development_labels(label_path, development_indices)
    return development_x, development_y, development_indices


def build_folds(development_y, seed):
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.full((len(development_y),), -1, dtype=np.int64)
    placeholder = np.zeros((len(development_y), 1), dtype=np.float32)
    for fold, (_, validation_positions) in enumerate(splitter.split(placeholder, development_y)):
        folds[validation_positions] = fold
    if np.any(folds < 0):
        raise RuntimeError("Fold assignment is incomplete.")
    return folds


def write_manifest(path, development_indices, development_y, folds):
    rows = sorted(
        (
            {
                "sample_index": int(development_indices[position]),
                "fold": int(folds[position]),
                "true_label": int(development_y[position]),
            }
            for position in range(len(development_y))
        ),
        key=lambda row: row["sample_index"],
    )
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("sample_index", "fold", "true_label"))
        writer.writeheader()
        writer.writerows(rows)


def prediction_dataset(tf, x, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(lambda image: tf.cast(image, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def remap_labels(y_global, classes):
    global_indices = np.asarray([CLASS_TO_INDEX[name] for name in classes], dtype=np.int64)
    global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(global_indices)}
    keep = np.isin(y_global, global_indices)
    y_local = np.asarray([global_to_local[int(value)] for value in y_global[keep]], dtype=np.int64)
    return keep, y_local


def write_history(path, history):
    save_json(path, {key: [float(value) for value in values] for key, values in history.history.items()})


def train_component(
    tf,
    keras,
    make_tf_dataset,
    build_model,
    component_dir,
    component_name,
    backbone,
    num_classes,
    train_pool_x,
    train_pool_y,
    noise_bank,
    args,
    batch_size,
    learning_rate,
    dropout,
    mask_prob,
    mask_max_fraction,
    noise_prob,
    noise_scale,
    class_aug_multipliers,
    class_weights,
    outer_train_indices,
    outer_validation_indices,
    train_pool_snr=None,
):
    component_dir = ensure_dir(component_dir)
    best_model_path = component_dir / "best_model"
    config_path = component_dir / "train_config.json"
    model_exists = (best_model_path / "saved_model.pb").exists() and config_path.exists()
    if model_exists and args.resume:
        print("[resume] %s" % component_dir)
        return best_model_path
    if model_exists:
        raise FileExistsError("Model already exists; use --resume to reuse it: %s" % best_model_path)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    train_x, train_y, val_x, val_y, internal_train_idx, internal_val_idx = stratified_train_val_split(
        train_pool_x,
        train_pool_y,
        val_fraction=args.inner_val_fraction,
        seed=args.seed,
    )
    if train_pool_snr is None:
        train_snr = None
        val_snr = None
    else:
        train_snr = train_pool_snr[internal_train_idx]
        val_snr = train_pool_snr[internal_val_idx]

    np.savez_compressed(
        str(component_dir / "internal_split_indices.npz"),
        train_idx=internal_train_idx,
        val_idx=internal_val_idx,
    )
    save_json(
        config_path,
        {
            "component": component_name,
            "outer_cv_seed": int(args.seed),
            "backbone": backbone,
            "num_classes": int(num_classes),
            "epochs": int(args.epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "dropout": float(dropout),
            "mask_prob": float(mask_prob),
            "mask_max_fraction": float(mask_max_fraction),
            "noise_prob": float(noise_prob),
            "noise_scale": float(noise_scale),
            "class_aug_multipliers": [float(value) for value in class_aug_multipliers],
            "class_weights": [float(value) for value in class_weights],
            "n_train_pool": int(len(train_pool_y)),
            "n_internal_train": int(len(train_y)),
            "n_internal_validation": int(len(val_y)),
            "outer_train_chime_indices": [int(value) for value in outer_train_indices],
            "outer_validation_chime_indices": [int(value) for value in outer_validation_indices],
            "noise_bank_chime_indices": [int(value) for value in outer_train_indices],
            "leakage_assertion": "outer validation is absent from train pool, internal validation, checkpoint selection, and noise bank",
        },
    )

    train_ds = make_tf_dataset(
        train_x,
        train_y,
        batch_size=batch_size,
        training=True,
        seed=args.seed,
        mask_prob=mask_prob,
        mask_max_fraction=mask_max_fraction,
        noise_bank=noise_bank,
        noise_prob=noise_prob,
        noise_scale=noise_scale,
        class_aug_multipliers=class_aug_multipliers,
        class_weights=class_weights,
        sample_snr=train_snr,
    )
    val_ds = make_tf_dataset(
        val_x,
        val_y,
        batch_size=batch_size,
        training=False,
        sample_snr=val_snr,
    )

    keras.backend.clear_session()
    model = build_model(backbone, num_classes=num_classes, dropout=dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(best_model_path), monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.early_stop_patience,
            restore_best_weights=False,
        ),
        keras.callbacks.CSVLogger(str(component_dir / "training_history.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    write_history(component_dir / "history.json", history)
    del model
    keras.backend.clear_session()
    return best_model_path


def predict_component(keras, tf, model_path, x, batch_size):
    model = keras.models.load_model(str(model_path))
    probabilities = model.predict(prediction_dataset(tf, x, batch_size), verbose=0)
    del model
    keras.backend.clear_session()
    return probabilities


def write_fold_oof(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OOF_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_fold_oof(path, expected_fold, expected_indices):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or tuple(rows[0].keys()) != OOF_FIELDS:
        raise ValueError("Unexpected OOF fields in %s" % path)
    indices = [int(row["sample_index"]) for row in rows]
    if set(indices) != set(int(value) for value in expected_indices) or len(indices) != len(set(indices)):
        raise ValueError("Fold OOF indices do not match fold manifest: %s" % path)
    if any(int(row["fold"]) != expected_fold for row in rows):
        raise ValueError("Fold id mismatch in %s" % path)
    return rows


def main():
    args = parse_args()
    configure_device(args)

    import tensorflow as tf
    from tensorflow import keras

    from augment import make_tf_dataset
    from models import build_model

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    output_dir = ensure_dir(args.output_dir)
    model_root = ensure_dir(args.model_root)
    development_x, development_y, development_indices = load_development_only(
        args.chime_data,
        args.chime_labels,
        args.development_indices,
    )
    folds = build_folds(development_y, args.seed)
    write_manifest(output_dir / "fold_manifest.csv", development_indices, development_y, folds)

    snrs = [int(value.strip()) for value in args.snrs.split(",") if value.strip()]
    print("Loading simulated samples once for all folds...")
    sim_x, sim_y, sim_snr, _ = load_simulation_dataset_with_snr(
        args.sim_dir,
        samples_per_class=args.samples_per_class,
        snrs=snrs,
        seed=args.seed,
        exclude_marked=True,
    )

    all_oof_rows = []
    for fold in range(5):
        fold_dir = ensure_dir(model_root / ("fold_%d" % fold))
        fold_oof_path = fold_dir / "fold_oof_predictions.csv"
        outer_validation_positions = np.where(folds == fold)[0]
        outer_train_positions = np.where(folds != fold)[0]
        outer_validation_indices = development_indices[outer_validation_positions]
        outer_train_indices = development_indices[outer_train_positions]

        if set(outer_validation_indices.tolist()) & set(outer_train_indices.tolist()):
            raise RuntimeError("Outer fold train/validation overlap detected in fold %d." % fold)

        required_fold_outputs = [
            fold_dir / "base" / "best_model" / "saved_model.pb",
            fold_dir / "base" / "train_config.json",
            fold_dir / "tri_specialist" / "best_model" / "saved_model.pb",
            fold_dir / "tri_specialist" / "train_config.json",
            fold_dir / "pair_b_c1" / "best_model" / "saved_model.pb",
            fold_dir / "pair_b_c1" / "train_config.json",
            fold_dir / "pair_c1_c2" / "best_model" / "saved_model.pb",
            fold_dir / "pair_c1_c2" / "train_config.json",
            fold_dir / "inference_log.txt",
            fold_oof_path,
        ]
        if args.resume and all(path.exists() for path in required_fold_outputs):
            print("[resume fold OOF] fold %d" % fold)
            all_oof_rows.extend(read_fold_oof(fold_oof_path, fold, outer_validation_indices))
            continue

        outer_train_x = development_x[outer_train_positions]
        outer_train_y = development_y[outer_train_positions]
        outer_validation_x = development_x[outer_validation_positions]
        outer_validation_y = development_y[outer_validation_positions]
        noise_bank = build_real_noise_bank(
            outer_train_x,
            bank_size=512,
            blank_fraction=0.2,
            seed=args.seed,
        )
        save_json(
            fold_dir / "fold_config.json",
            {
                "fold": int(fold),
                "outer_cv_seed": int(args.seed),
                "n_outer_train_chime": int(len(outer_train_positions)),
                "n_outer_validation_chime": int(len(outer_validation_positions)),
                "outer_train_chime_indices": [int(value) for value in outer_train_indices],
                "outer_validation_chime_indices": [int(value) for value in outer_validation_indices],
                "noise_bank_chime_indices": [int(value) for value in outer_train_indices],
                "paper_test_access": "none",
            },
        )

        base_pool_x = np.concatenate([sim_x, outer_train_x], axis=0)
        base_pool_y = np.concatenate([sim_y, outer_train_y], axis=0)
        base_pool_snr = np.concatenate(
            [sim_snr, np.zeros((len(outer_train_y),), dtype=np.float32)], axis=0
        )
        base_path = train_component(
            tf,
            keras,
            make_tf_dataset,
            build_model,
            fold_dir / "base",
            "base_resnet50",
            "resnet50",
            5,
            base_pool_x,
            base_pool_y,
            noise_bank,
            args,
            args.batch_size_base,
            1e-4,
            0.3,
            0.7,
            0.25,
            0.6,
            0.025,
            [0.5, 1.0, 1.0, 0.4, 1.0],
            [1.15, 0.75, 1.0, 3.0, 1.2],
            outer_train_indices,
            outer_validation_indices,
            train_pool_snr=base_pool_snr,
        )

        tri_sim_keep, tri_sim_y = remap_labels(sim_y, TRI_CLASSES)
        tri_chime_keep, tri_chime_y = remap_labels(outer_train_y, TRI_CLASSES)
        tri_pool_x = np.concatenate([sim_x[tri_sim_keep], outer_train_x[tri_chime_keep]], axis=0)
        tri_pool_y = np.concatenate([tri_sim_y, tri_chime_y], axis=0)
        tri_path = train_component(
            tf,
            keras,
            make_tf_dataset,
            build_model,
            fold_dir / "tri_specialist",
            "tri_specialist_b_c1_c2",
            "resnet18",
            3,
            tri_pool_x,
            tri_pool_y,
            noise_bank,
            args,
            args.batch_size_specialist,
            2e-4,
            0.3,
            0.65,
            0.22,
            0.55,
            0.02,
            [1.0, 0.35, 0.85],
            [0.7, 4.0, 1.3],
            outer_train_indices,
            outer_validation_indices,
        )

        pair_paths = {}
        for pair_key, pair_classes, aug, weights in (
            ("b_c1", ("B", "C1"), [1.0, 0.35], [0.7, 4.0]),
            ("c1_c2", ("C1", "C2"), [0.35, 0.85], [3.5, 1.3]),
        ):
            sim_keep, pair_sim_y = remap_labels(sim_y, pair_classes)
            chime_keep, pair_chime_y = remap_labels(outer_train_y, pair_classes)
            pair_pool_x = np.concatenate([sim_x[sim_keep], outer_train_x[chime_keep]], axis=0)
            pair_pool_y = np.concatenate([pair_sim_y, pair_chime_y], axis=0)
            pair_paths[pair_key] = train_component(
                tf,
                keras,
                make_tf_dataset,
                build_model,
                fold_dir / ("pair_" + pair_key),
                "pair_" + pair_key,
                "resnet18",
                2,
                pair_pool_x,
                pair_pool_y,
                noise_bank,
                args,
                args.batch_size_specialist,
                2e-4,
                0.3,
                0.6,
                0.2,
                0.5,
                0.02,
                aug,
                weights,
                outer_train_indices,
                outer_validation_indices,
            )

        base_probabilities = predict_component(
            keras, tf, base_path, outer_validation_x, args.batch_size_inference
        )
        tri_probabilities = predict_component(
            keras, tf, tri_path, outer_validation_x, args.batch_size_inference
        )
        b_c1_probabilities = predict_component(
            keras, tf, pair_paths["b_c1"], outer_validation_x, args.batch_size_inference
        )
        c1_c2_probabilities = predict_component(
            keras, tf, pair_paths["c1_c2"], outer_validation_x, args.batch_size_inference
        )

        fold_rows = []
        for local_position, development_position in enumerate(outer_validation_positions):
            fold_rows.append(
                {
                    "sample_index": int(development_indices[development_position]),
                    "fold": int(fold),
                    "true_label": int(outer_validation_y[local_position]),
                    "base_pred": int(np.argmax(base_probabilities[local_position])),
                    "tri_prob_B": "%.9f" % float(tri_probabilities[local_position, 0]),
                    "tri_prob_C1": "%.9f" % float(tri_probabilities[local_position, 1]),
                    "tri_prob_C2": "%.9f" % float(tri_probabilities[local_position, 2]),
                    "bc1_prob_B": "%.9f" % float(b_c1_probabilities[local_position, 0]),
                    "bc1_prob_C1": "%.9f" % float(b_c1_probabilities[local_position, 1]),
                    "c1c2_prob_C1": "%.9f" % float(c1_c2_probabilities[local_position, 0]),
                    "c1c2_prob_C2": "%.9f" % float(c1_c2_probabilities[local_position, 1]),
                }
            )
        fold_rows.sort(key=lambda row: row["sample_index"])
        write_fold_oof(fold_oof_path, fold_rows)
        with (fold_dir / "inference_log.txt").open("w", encoding="utf-8") as f:
            f.write("fold=%d\n" % fold)
            f.write("n_outer_validation=%d\n" % len(fold_rows))
            f.write("sample_indices=%s\n" % ",".join(str(row["sample_index"]) for row in fold_rows))
            f.write("base_model=%s\n" % base_path)
            f.write("tri_model=%s\n" % tri_path)
            f.write("b_c1_model=%s\n" % pair_paths["b_c1"])
            f.write("c1_c2_model=%s\n" % pair_paths["c1_c2"])
            f.write("expert_probabilities_saved_for_all_outer_validation_samples=true\n")
        all_oof_rows.extend(fold_rows)

    all_oof_rows.sort(key=lambda row: int(row["sample_index"]))
    sample_indices = [int(row["sample_index"]) for row in all_oof_rows]
    if len(all_oof_rows) != 267 or len(set(sample_indices)) != 267:
        raise RuntimeError("OOF coverage must contain 267 unique development samples exactly once.")
    if set(sample_indices) != set(int(value) for value in development_indices):
        raise RuntimeError("OOF sample indices do not exactly cover the development split.")
    write_fold_oof(output_dir / "oof_component_predictions.csv", all_oof_rows)
    save_json(
        output_dir / "oof_generation_config.json",
        {
            **vars(args),
            "n_development_samples": 267,
            "n_outer_folds": 5,
            "fold_numbering": "0-4",
            "oof_fields": list(OOF_FIELDS),
            "selection_stage_paper_test_metrics_access": "forbidden and not performed",
        },
    )
    print("Saved complete OOF predictions to: %s" % (output_dir / "oof_component_predictions.csv"))


if __name__ == "__main__":
    main()
