from __future__ import print_function

import argparse
import os
from pathlib import Path

import numpy as np

from data import (
    CLASSES,
    PAPER_CLASS_NAMES,
    build_bucketed_real_noise_bank,
    build_real_noise_bank,
    ensure_dir,
    load_simulation_dataset,
    load_simulation_dataset_with_snr,
    paper_half_split_chime,
    save_json,
    stratified_train_val_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an improved FRB multiclass classifier with masking and real-noise augmentation.")
    parser.add_argument("--sim-dir", default="data/simulated/simfrb", help="Paper-style simulation directory containing simdata/.")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz", help="Interpolated CHIME data .npz.")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt", help="CHIME label file.")
    parser.add_argument("--output-dir", default="artifacts/runs/resnet_mask_noise", help="Directory for model and logs.")
    parser.add_argument("--backbone", default="resnet_small", help="simple_cnn, resnet_small, resnet18, resnet50, efficientnetb0, or convnext_tiny.")
    parser.add_argument("--samples-per-class", type=int, default=1800, help="Number of simulated examples per morphology class.")
    parser.add_argument("--snrs", default="15,25,35,50,100", help="Comma-separated SNR values to use from simulated data.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu", help="Default cpu avoids TensorFlow GPU allocation failures on constrained devices.")
    parser.add_argument("--gpu-id", default="0", help="GPU id used when --device gpu, e.g. 0 or 1.")
    parser.add_argument("--disable-tf32", action="store_true", help="Disable TensorFloat-32 matmul on Ampere/Ada GPUs.")
    parser.add_argument("--no-chime-train", action="store_true", help="Do not add the paper-style first half of CHIME data to training.")
    parser.add_argument("--mask-prob", type=float, default=0.8)
    parser.add_argument("--mask-max-fraction", type=float, default=0.35)
    parser.add_argument("--noise-prob", type=float, default=0.7)
    parser.add_argument("--noise-scale", type=float, default=0.04)
    parser.add_argument("--noise-bank-size", type=int, default=512)
    parser.add_argument("--noise-bank-mode", choices=["plain", "bucketed"], default="plain")
    parser.add_argument("--band-rfi-prob", type=float, default=0.0)
    parser.add_argument("--band-rfi-max-fraction", type=float, default=0.08)
    parser.add_argument("--time-dropout-prob", type=float, default=0.0)
    parser.add_argument("--time-dropout-max-fraction", type=float, default=0.08)
    parser.add_argument("--patch-corrupt-prob", type=float, default=0.0)
    parser.add_argument("--patch-corrupt-max-fraction", type=float, default=0.12)
    parser.add_argument(
        "--class-aug-multipliers",
        default="A:1,B:1,C:1,C1:1,C2:1",
        help="Per-class multiplier for mask/noise strength, e.g. A:0.35,B:1,C:1,C1:1,C2:1.",
    )
    parser.add_argument(
        "--class-weights",
        default="A:1,B:1,C:1,C1:1,C2:1",
        help="Per-class loss weights, e.g. A:1.6,B:0.8,C:1,C1:1.2,C2:1.2.",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal", "class_balanced_focal"],
        default="cross_entropy",
        help="Training loss. class_balanced_focal uses effective-number class weights.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument(
        "--class-balance-source",
        choices=["train_pool", "chime_train"],
        default="chime_train",
        help="Labels used to compute class-balanced focal alpha.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)
    return parser.parse_args()


def configure_device(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "gpu":
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def parse_class_values(spec, default_value=1.0):
    values = {name: float(default_value) for name in CLASSES}
    if not spec:
        return [values[name] for name in CLASSES], values
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Expected CLASS:VALUE item in %s" % item)
        name, value = item.split(":", 1)
        name = name.strip()
        if name not in values:
            raise ValueError("Unknown class %s. Expected one of %s" % (name, CLASSES))
        values[name] = float(value)
    return [values[name] for name in CLASSES], values


class FinalEpochModelSaver:
    """Persist the requested final-epoch weights before early stopping restores best weights."""

    def __init__(self, output_path, total_epochs):
        self.output_path = str(output_path)
        self.total_epochs = total_epochs

    def as_callback(self, keras):
        output_path = self.output_path
        total_epochs = self.total_epochs

        class _Saver(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch + 1 == total_epochs:
                    self.model.save(output_path, save_format="tf")

        return _Saver()


def main():
    args = parse_args()
    if args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    configure_device(args.device)

    import tensorflow as tf
    from tensorflow import keras

    from augment import make_tf_dataset
    from losses import build_sparse_classification_loss
    from models import available_backbones, build_model

    if args.disable_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    snrs = [int(v.strip()) for v in args.snrs.split(",") if v.strip()]
    class_aug_multipliers, class_aug_multipliers_dict = parse_class_values(args.class_aug_multipliers)
    class_weights, class_weights_dict = parse_class_values(args.class_weights)

    if args.backbone not in available_backbones():
        raise ValueError("Unknown backbone %s. Available: %s" % (args.backbone, available_backbones()))

    print("Loading simulated training data...")
    sim_x, sim_y, sim_snr, sim_meta = load_simulation_dataset_with_snr(
        args.sim_dir,
        samples_per_class=args.samples_per_class,
        snrs=snrs,
        seed=args.seed,
        exclude_marked=True,
    )

    print("Loading CHIME paper split...")
    chime_train_x, chime_train_y, chime_test_x, chime_test_y, chime_train_idx, chime_test_idx = paper_half_split_chime(
        args.chime_data, args.chime_labels
    )

    if args.no_chime_train:
        train_pool_x = sim_x
        train_pool_y = sim_y
        train_pool_snr = sim_snr
    else:
        train_pool_x = np.concatenate([sim_x, chime_train_x], axis=0)
        train_pool_y = np.concatenate([sim_y, chime_train_y], axis=0)
        train_pool_snr = np.concatenate([sim_snr, np.zeros((chime_train_x.shape[0],), dtype=np.float32)], axis=0)

    train_x, train_y, val_x, val_y, train_idx, val_idx = stratified_train_val_split(
        train_pool_x, train_pool_y, val_fraction=args.val_fraction, seed=args.seed
    )
    train_snr = train_pool_snr[train_idx]
    val_snr = train_pool_snr[val_idx]

    print("Building CHIME-derived real-noise bank...")
    noise_bank = None
    bucketed_noise_bank = None
    noise_bank_summary = None
    if args.noise_prob <= 0.0 or args.noise_scale <= 0.0:
        print("Skipping real-noise bank because noise augmentation is disabled.")
    elif args.noise_bank_mode == "bucketed":
        bucketed_noise_bank, noise_bank_summary = build_bucketed_real_noise_bank(
            chime_train_x,
            bank_size=args.noise_bank_size,
            blank_fraction=0.2,
            seed=args.seed,
            n_buckets=3,
        )
    else:
        noise_bank = build_real_noise_bank(
            chime_train_x,
            bank_size=args.noise_bank_size,
            blank_fraction=0.2,
            seed=args.seed,
        )

    train_ds = make_tf_dataset(
        train_x,
        train_y,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
        mask_prob=args.mask_prob,
        mask_max_fraction=args.mask_max_fraction,
        noise_bank=noise_bank,
        bucketed_noise_bank=bucketed_noise_bank,
        noise_prob=args.noise_prob,
        noise_scale=args.noise_scale,
        class_aug_multipliers=class_aug_multipliers,
        class_weights=class_weights,
        sample_snr=train_snr,
        band_rfi_prob=args.band_rfi_prob,
        band_rfi_max_fraction=args.band_rfi_max_fraction,
        time_dropout_prob=args.time_dropout_prob,
        time_dropout_max_fraction=args.time_dropout_max_fraction,
        patch_corrupt_prob=args.patch_corrupt_prob,
        patch_corrupt_max_fraction=args.patch_corrupt_max_fraction,
    )
    val_ds = make_tf_dataset(val_x, val_y, batch_size=args.batch_size, training=False, sample_snr=val_snr)

    if args.class_balance_source == "chime_train":
        y_for_balance = chime_train_y
    else:
        y_for_balance = train_pool_y
    loss_fn, loss_metadata = build_sparse_classification_loss(
        args.loss,
        y_for_balance=y_for_balance,
        num_classes=len(CLASSES),
        gamma=args.focal_gamma,
        beta=args.class_balance_beta,
    )

    model = build_model(args.backbone, dropout=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(out_dir / "best_model"), monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=args.early_stop_patience, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(out_dir / "training_history.csv")),
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "tensorboard")),
        FinalEpochModelSaver(out_dir / "final_model", args.epochs).as_callback(keras),
    ]

    config = vars(args).copy()
    config.update({
        "classes": CLASSES,
        "paper_class_names": PAPER_CLASS_NAMES,
        "n_sim_loaded": int(sim_x.shape[0]),
        "n_chime_train": int(chime_train_x.shape[0]),
        "n_chime_test_reserved_for_evaluation": int(chime_test_x.shape[0]),
        "n_train_after_split": int(train_x.shape[0]),
        "n_val_after_split": int(val_x.shape[0]),
        "noise_bank_summary": noise_bank_summary,
        "class_aug_multipliers_parsed": class_aug_multipliers_dict,
        "class_weights_parsed": class_weights_dict,
        "loss_metadata": loss_metadata,
        "innovation": [
            "masked-channel aware training",
            "real CHIME blank-region noise augmentation",
            "optional SNR-matched bucketed CHIME noise augmentation",
            "optional banded RFI, time dropout, and patch corruption",
            "switchable stronger backbone",
            "optional morphology-aware augmentation and class-weighted loss",
            "optional focal and class-balanced focal loss",
        ],
    })
    save_json(out_dir / "train_config.json", config)
    np.savez_compressed(
        str(out_dir / "paper_chime_split_indices.npz"),
        train_idx=chime_train_idx,
        test_idx=chime_test_idx,
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    best_model = keras.models.load_model(str(out_dir / "best_model"))
    val_metrics = best_model.evaluate(val_ds, verbose=0)
    save_json(out_dir / "validation_metrics.json", dict(zip(best_model.metrics_names, [float(v) for v in val_metrics])))
    save_json(out_dir / "history.json", {k: [float(v) for v in values] for k, values in history.history.items()})
    print("Saved best model to: %s" % (out_dir / "best_model"))
    print("Saved final model to: %s" % (out_dir / "final_model"))


if __name__ == "__main__":
    main()
