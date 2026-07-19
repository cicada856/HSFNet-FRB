from __future__ import print_function

import argparse
import os

import numpy as np

from data import (
    CLASSES,
    CLASS_TO_INDEX,
    build_real_noise_bank,
    ensure_dir,
    load_simulation_dataset,
    paper_half_split_chime,
    save_json,
    stratified_train_val_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a pairwise FRB morphology specialist.")
    parser.add_argument("--class-a", required=True, choices=CLASSES, help="Class encoded as binary label 0.")
    parser.add_argument("--class-b", required=True, choices=CLASSES, help="Class encoded as binary label 1.")
    parser.add_argument("--sim-dir", default="data/simulated/simfrb")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="resnet18", help="simple_cnn, resnet_small, resnet18, resnet50, efficientnetb0, or convnext_tiny.")
    parser.add_argument("--samples-per-class", type=int, default=1800)
    parser.add_argument("--snrs", default="15,25,35,50,100")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--no-chime-train", action="store_true", help="Train and validate only on simulated data; reserve CHIME for final testing.")
    parser.add_argument("--mask-prob", type=float, default=0.6)
    parser.add_argument("--mask-max-fraction", type=float, default=0.2)
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--noise-bank-size", type=int, default=512)
    parser.add_argument("--class-a-aug-multiplier", type=float, default=1.0)
    parser.add_argument("--class-b-aug-multiplier", type=float, default=1.0)
    parser.add_argument("--class-a-weight", type=float, default=1.0)
    parser.add_argument("--class-b-weight", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    return parser.parse_args()


def configure_device(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def filter_pair_and_remap(x, y, class_a, class_b):
    a_idx = CLASS_TO_INDEX[class_a]
    b_idx = CLASS_TO_INDEX[class_b]
    keep = (y == a_idx) | (y == b_idx)
    x_keep = x[keep]
    y_keep = y[keep]
    y_binary = np.where(y_keep == b_idx, 1, 0).astype(np.int64)
    return x_keep, y_binary


def main():
    args = parse_args()
    configure_device(args)

    import tensorflow as tf
    from tensorflow import keras

    from augment import make_tf_dataset
    from models import available_backbones, build_model

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    if args.backbone not in available_backbones():
        raise ValueError("Unknown backbone %s. Available: %s" % (args.backbone, available_backbones()))

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)
    snrs = [int(v.strip()) for v in args.snrs.split(",") if v.strip()]

    print("Loading simulated data for %s vs %s..." % (args.class_a, args.class_b))
    sim_x_all, sim_y_all, _ = load_simulation_dataset(
        args.sim_dir,
        samples_per_class=args.samples_per_class,
        snrs=snrs,
        seed=args.seed,
        exclude_marked=True,
    )
    sim_x, sim_y = filter_pair_and_remap(sim_x_all, sim_y_all, args.class_a, args.class_b)

    print("Loading CHIME paper split for %s vs %s..." % (args.class_a, args.class_b))
    chime_train_x_all, chime_train_y_all, chime_test_x_all, chime_test_y_all, chime_train_idx, chime_test_idx = paper_half_split_chime(
        args.chime_data, args.chime_labels
    )
    chime_train_x, chime_train_y = filter_pair_and_remap(chime_train_x_all, chime_train_y_all, args.class_a, args.class_b)
    chime_test_x, chime_test_y = filter_pair_and_remap(chime_test_x_all, chime_test_y_all, args.class_a, args.class_b)

    if args.no_chime_train:
        train_pool_x = sim_x
        train_pool_y = sim_y
    else:
        train_pool_x = np.concatenate([sim_x, chime_train_x], axis=0)
        train_pool_y = np.concatenate([sim_y, chime_train_y], axis=0)
    train_x, train_y, val_x, val_y, _, _ = stratified_train_val_split(
        train_pool_x, train_pool_y, val_fraction=args.val_fraction, seed=args.seed
    )

    if args.noise_prob <= 0.0 or args.noise_scale <= 0.0:
        print("Skipping real-noise bank because noise augmentation is disabled.")
        noise_bank = None
    else:
        noise_bank = build_real_noise_bank(
            chime_train_x_all,
            bank_size=args.noise_bank_size,
            blank_fraction=0.2,
            seed=args.seed,
        )

    class_aug_multipliers = [args.class_a_aug_multiplier, args.class_b_aug_multiplier]
    class_weights = [args.class_a_weight, args.class_b_weight]

    train_ds = make_tf_dataset(
        train_x,
        train_y,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
        mask_prob=args.mask_prob,
        mask_max_fraction=args.mask_max_fraction,
        noise_bank=noise_bank,
        noise_prob=args.noise_prob,
        noise_scale=args.noise_scale,
        class_aug_multipliers=class_aug_multipliers,
        class_weights=class_weights,
    )
    val_ds = make_tf_dataset(val_x, val_y, batch_size=args.batch_size, training=False)

    model = build_model(args.backbone, num_classes=2, dropout=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(out_dir / "best_model"), monitor="val_accuracy", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=args.early_stop_patience, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(out_dir / "training_history.csv")),
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "tensorboard")),
    ]

    save_json(
        out_dir / "train_config.json",
        {
            **vars(args),
            "label_0": args.class_a,
            "label_1": args.class_b,
            "global_label_0": CLASS_TO_INDEX[args.class_a],
            "global_label_1": CLASS_TO_INDEX[args.class_b],
            "n_sim_pair": int(sim_x.shape[0]),
            "n_chime_train_pair": int(chime_train_x.shape[0]),
            "n_chime_test_pair_reserved": int(chime_test_x.shape[0]),
            "n_train_after_split": int(train_x.shape[0]),
            "n_val_after_split": int(val_x.shape[0]),
        },
    )
    np.savez_compressed(
        str(out_dir / "paper_chime_split_indices.npz"),
        train_idx=chime_train_idx,
        test_idx=chime_test_idx,
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(str(out_dir / "final_model"), save_format="tf")

    best_model = keras.models.load_model(str(out_dir / "best_model"))
    val_metrics = best_model.evaluate(val_ds, verbose=0)
    save_json(out_dir / "validation_metrics.json", dict(zip(best_model.metrics_names, [float(v) for v in val_metrics])))
    save_json(out_dir / "history.json", {k: [float(v) for v in values] for k, values in history.history.items()})
    print("Saved pair specialist best model to: %s" % (out_dir / "best_model"))


if __name__ == "__main__":
    main()
