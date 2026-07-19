from __future__ import print_function

import tensorflow as tf


def _apply_frequency_mask(image, max_fraction):
    freq = tf.shape(image)[0]
    mask_fraction = tf.random.uniform([], minval=0.0, maxval=max_fraction, dtype=tf.float32)
    keep = tf.cast(tf.random.uniform([freq, 1, 1], dtype=tf.float32) > mask_fraction, image.dtype)
    return image * keep


def _apply_real_noise(image, noise_bank, max_scale):
    n_noise = tf.shape(noise_bank)[0]
    idx = tf.random.uniform([], minval=0, maxval=n_noise, dtype=tf.int32)
    noise = tf.gather(noise_bank, idx)
    scale = tf.random.uniform([], minval=0.0, maxval=max_scale, dtype=tf.float32)
    return tf.clip_by_value(image + noise * scale, 0.0, 1.0)


def _snr_to_noise_bucket(snr, n_buckets):
    random_bucket = tf.random.uniform([], minval=0, maxval=n_buckets, dtype=tf.int32)

    def from_snr():
        return tf.case(
            [
                (snr <= 20.0, lambda: tf.minimum(n_buckets - 1, 2)),
                (snr <= 35.0, lambda: tf.minimum(n_buckets - 1, 1)),
            ],
            default=lambda: tf.constant(0, dtype=tf.int32),
            exclusive=False,
        )

    return tf.cond(snr > 0.0, from_snr, lambda: random_bucket)


def _apply_bucketed_real_noise(image, bucketed_noise_bank, snr, max_scale):
    n_buckets = tf.shape(bucketed_noise_bank)[0]
    n_per_bucket = tf.shape(bucketed_noise_bank)[1]
    bucket = _snr_to_noise_bucket(tf.cast(snr, tf.float32), n_buckets)
    idx = tf.random.uniform([], minval=0, maxval=n_per_bucket, dtype=tf.int32)
    noise = bucketed_noise_bank[bucket, idx]
    scale = tf.random.uniform([], minval=0.0, maxval=max_scale, dtype=tf.float32)
    return tf.clip_by_value(image + noise * scale, 0.0, 1.0)


def _apply_frequency_band_rfi(image, max_band_fraction):
    freq = tf.shape(image)[0]
    max_width = tf.maximum(1, tf.cast(tf.round(tf.cast(freq, tf.float32) * max_band_fraction), tf.int32))
    width = tf.random.uniform([], minval=1, maxval=max_width + 1, dtype=tf.int32)
    start = tf.random.uniform([], minval=0, maxval=tf.maximum(1, freq - width + 1), dtype=tf.int32)
    rows = tf.range(freq)
    in_band = (rows >= start) & (rows < start + width)
    keep = 1.0 - tf.cast(in_band, image.dtype)
    keep = tf.reshape(keep, [freq, 1, 1])
    return image * keep


def _apply_time_dropout(image, max_time_fraction):
    time = tf.shape(image)[1]
    max_width = tf.maximum(1, tf.cast(tf.round(tf.cast(time, tf.float32) * max_time_fraction), tf.int32))
    width = tf.random.uniform([], minval=1, maxval=max_width + 1, dtype=tf.int32)
    start = tf.random.uniform([], minval=0, maxval=tf.maximum(1, time - width + 1), dtype=tf.int32)
    cols = tf.range(time)
    in_band = (cols >= start) & (cols < start + width)
    keep = 1.0 - tf.cast(in_band, image.dtype)
    keep = tf.reshape(keep, [1, time, 1])
    return image * keep


def _apply_patch_corruption(image, max_patch_fraction):
    freq = tf.shape(image)[0]
    time = tf.shape(image)[1]
    max_h = tf.maximum(1, tf.cast(tf.round(tf.cast(freq, tf.float32) * max_patch_fraction), tf.int32))
    max_w = tf.maximum(1, tf.cast(tf.round(tf.cast(time, tf.float32) * max_patch_fraction), tf.int32))
    h = tf.random.uniform([], minval=1, maxval=max_h + 1, dtype=tf.int32)
    w = tf.random.uniform([], minval=1, maxval=max_w + 1, dtype=tf.int32)
    row0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, freq - h + 1), dtype=tf.int32)
    col0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, time - w + 1), dtype=tf.int32)
    rows = tf.range(freq)
    cols = tf.range(time)
    row_mask = (rows >= row0) & (rows < row0 + h)
    col_mask = (cols >= col0) & (cols < col0 + w)
    patch = tf.logical_and(tf.reshape(row_mask, [freq, 1, 1]), tf.reshape(col_mask, [1, time, 1]))
    keep = 1.0 - tf.cast(patch, image.dtype)
    fill = tf.random.uniform(tf.shape(image), minval=0.0, maxval=0.05, dtype=image.dtype)
    return image * keep + fill * (1.0 - keep)


def make_tf_dataset(
    x,
    y,
    batch_size,
    training,
    seed=42,
    mask_prob=0.8,
    mask_max_fraction=0.35,
    noise_bank=None,
    bucketed_noise_bank=None,
    noise_prob=0.7,
    noise_scale=0.04,
    class_aug_multipliers=None,
    class_weights=None,
    sample_snr=None,
    band_rfi_prob=0.0,
    band_rfi_max_fraction=0.08,
    time_dropout_prob=0.0,
    time_dropout_max_fraction=0.08,
    patch_corrupt_prob=0.0,
    patch_corrupt_max_fraction=0.12,
):
    if sample_snr is None:
        sample_snr = tf.zeros([len(x)], dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y, sample_snr))
    if training:
        ds = ds.shuffle(min(len(x), 10000), seed=seed, reshuffle_each_iteration=True)

    use_noise = noise_bank is not None and len(noise_bank) > 0
    if use_noise:
        noise_bank_tf = tf.constant(noise_bank, dtype=tf.float32)
    else:
        noise_bank_tf = None

    use_bucketed_noise = bucketed_noise_bank is not None and len(bucketed_noise_bank) > 0 and len(bucketed_noise_bank[0]) > 0
    if use_bucketed_noise:
        bucketed_noise_bank_tf = tf.constant(bucketed_noise_bank, dtype=tf.float32)
    else:
        bucketed_noise_bank_tf = None

    if class_aug_multipliers is not None:
        class_aug_multipliers_tf = tf.constant(class_aug_multipliers, dtype=tf.float32)
    else:
        class_aug_multipliers_tf = None

    if class_weights is not None:
        class_weights_tf = tf.constant(class_weights, dtype=tf.float32)
    else:
        class_weights_tf = None

    def preprocess(image, label, snr):
        image = tf.cast(image, tf.float32) / 255.0
        aug_multiplier = 1.0
        if class_aug_multipliers_tf is not None:
            aug_multiplier = tf.gather(class_aug_multipliers_tf, label)
        if training:
            if mask_max_fraction > 0.0 and mask_prob > 0.0:
                image = tf.cond(
                    tf.random.uniform([], dtype=tf.float32) < mask_prob,
                    lambda: _apply_frequency_mask(image, mask_max_fraction * aug_multiplier),
                    lambda: image,
                )
            if use_bucketed_noise and noise_scale > 0.0 and noise_prob > 0.0:
                image = tf.cond(
                    tf.random.uniform([], dtype=tf.float32) < noise_prob,
                    lambda: _apply_bucketed_real_noise(image, bucketed_noise_bank_tf, snr, noise_scale * aug_multiplier),
                    lambda: image,
                )
            elif use_noise and noise_scale > 0.0 and noise_prob > 0.0:
                image = tf.cond(
                    tf.random.uniform([], dtype=tf.float32) < noise_prob,
                    lambda: _apply_real_noise(image, noise_bank_tf, noise_scale * aug_multiplier),
                    lambda: image,
                )
            if band_rfi_prob > 0.0 and band_rfi_max_fraction > 0.0:
                image = tf.cond(
                    tf.random.uniform([], dtype=tf.float32) < band_rfi_prob,
                    lambda: _apply_frequency_band_rfi(image, band_rfi_max_fraction * aug_multiplier),
                    lambda: image,
                )
            if time_dropout_prob > 0.0 and time_dropout_max_fraction > 0.0:
                image = tf.cond(
                    tf.random.uniform([], dtype=tf.float32) < time_dropout_prob,
                    lambda: _apply_time_dropout(image, time_dropout_max_fraction * aug_multiplier),
                    lambda: image,
                )
            if patch_corrupt_prob > 0.0 and patch_corrupt_max_fraction > 0.0:
                image = tf.cond(
                    tf.random.uniform([], dtype=tf.float32) < patch_corrupt_prob,
                    lambda: _apply_patch_corruption(image, patch_corrupt_max_fraction * aug_multiplier),
                    lambda: image,
                )
        if class_weights_tf is not None:
            sample_weight = tf.gather(class_weights_tf, label)
            return image, label, sample_weight
        return image, label

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
