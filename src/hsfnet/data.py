from __future__ import print_function

import json
from pathlib import Path

import numpy as np


CLASSES = ["A", "B", "C", "C1", "C2"]
PAPER_CLASS_NAMES = ["I", "II", "III", "IV", "V"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}
INDEX_TO_CLASS = {idx: name for name, idx in CLASS_TO_INDEX.items()}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, payload):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def normalize_images_to_uint8(images):
    """Normalize each dynamic spectrum for paper-compatible preprocessing and store as uint8."""
    images = np.asarray(images)
    out = np.empty(images.shape, dtype=np.uint8)
    for i in range(images.shape[0]):
        img = np.nan_to_num(images[i].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        immax = float(np.max(img))
        if not np.isfinite(immax) or immax <= 0.0:
            out[i] = 0
        else:
            out[i] = np.clip(img / immax * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def read_label_file(label_file):
    with Path(label_file).open("r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    unknown = sorted(set(labels) - set(CLASSES))
    if unknown:
        raise ValueError("Unknown labels in %s: %s" % (label_file, unknown))
    return labels


def load_npz_array(path):
    path = Path(path)
    with np.load(str(path)) as z:
        return z[z.files[0]]


def _read_remove_indices(path, n_items):
    """Read author-style index files and tolerate 0-based or 1-based values."""
    path = Path(path)
    if not path.exists():
        return set()

    raw = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(int(float(line.split()[0])))
            except ValueError:
                continue

    if not raw:
        return set()

    values = np.asarray(raw, dtype=np.int64)
    if values.min() >= 1 and values.max() <= n_items:
        values = values - 1
    values = values[(values >= 0) & (values < n_items)]
    if len(set(values.tolist())) > int(0.8 * n_items):
        print("Ignoring %s as a remove list because it covers most of the file." % path)
        return set()
    return set(values.tolist())


def _remove_file_candidates(sim_dir, class_name, snr):
    sim_dir = Path(sim_dir)
    candidates = []
    if class_name == "C1":
        candidates.append(sim_dir / ("remove_type_C1_SNR_%s.txt" % snr))
    if class_name == "C2":
        candidates.append(sim_dir / ("type_C2_SNR_%s.txt" % snr))
    return candidates


def _valid_indices_for_sim_file(sim_dir, class_name, snr, n_items):
    remove = set()
    for path in _remove_file_candidates(sim_dir, class_name, snr):
        remove.update(_read_remove_indices(path, n_items))
    if not remove:
        return np.arange(n_items, dtype=np.int64)
    return np.asarray([idx for idx in range(n_items) if idx not in remove], dtype=np.int64)


def load_simulation_dataset_with_snr(sim_dir, samples_per_class=1800, snrs=None, seed=42, exclude_marked=True):
    """Load a balanced subset of simulated FRBs from the paper-style simdata tree."""
    sim_dir = Path(sim_dir)
    simdata_dir = sim_dir / "simdata"
    if snrs is None:
        snrs = [15, 25, 35, 50, 100]

    rng = np.random.RandomState(seed)
    xs = []
    ys = []
    snr_values = []
    metadata = []

    for class_name in CLASSES:
        selected_for_class = []
        selected_snrs_for_class = []
        selected_meta_for_class = []
        per_snr = int(np.ceil(float(samples_per_class) / float(len(snrs))))

        for snr in snrs:
            npz_path = simdata_dir / ("type_%s" % class_name) / ("SNR_%s" % snr) / ("simulatefrbs_type_%s.npz" % class_name)
            if not npz_path.exists():
                raise FileNotFoundError("Missing simulated data file: %s" % npz_path)

            arr = load_npz_array(npz_path)
            valid = np.arange(arr.shape[0], dtype=np.int64)
            if exclude_marked:
                valid = _valid_indices_for_sim_file(sim_dir, class_name, snr, arr.shape[0])
            if valid.size == 0:
                continue

            take = min(per_snr, valid.size)
            chosen = rng.choice(valid, size=take, replace=False)
            selected = normalize_images_to_uint8(arr[chosen])
            selected_for_class.append(selected)
            selected_snrs_for_class.append(np.full((selected.shape[0],), int(snr), dtype=np.float32))
            selected_meta_for_class.extend([{"source": "sim", "class": class_name, "snr": int(snr), "index": int(i)} for i in chosen])

        if not selected_for_class:
            raise ValueError("No simulated examples loaded for class %s" % class_name)

        class_x = np.concatenate(selected_for_class, axis=0)
        class_snr = np.concatenate(selected_snrs_for_class, axis=0)
        class_meta = selected_meta_for_class
        if class_x.shape[0] > samples_per_class:
            keep = rng.choice(np.arange(class_x.shape[0]), size=samples_per_class, replace=False)
            class_x = class_x[keep]
            class_snr = class_snr[keep]
            class_meta = [class_meta[int(i)] for i in keep]
        xs.append(class_x)
        ys.append(np.full((class_x.shape[0],), CLASS_TO_INDEX[class_name], dtype=np.int64))
        snr_values.append(class_snr)
        metadata.extend(class_meta)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    snr_values = np.concatenate(snr_values, axis=0)
    order = rng.permutation(np.arange(x.shape[0]))
    x = x[order][..., np.newaxis]
    y = y[order]
    snr_values = snr_values[order]
    metadata = [metadata[int(i)] for i in order]
    return x, y, snr_values, metadata


def load_simulation_dataset(sim_dir, samples_per_class=1800, snrs=None, seed=42, exclude_marked=True):
    x, y, _, metadata = load_simulation_dataset_with_snr(
        sim_dir,
        samples_per_class=samples_per_class,
        snrs=snrs,
        seed=seed,
        exclude_marked=exclude_marked,
    )
    return x, y, metadata


def load_chime_data(data_file, label_file):
    arr = load_npz_array(data_file)
    labels = read_label_file(label_file)
    if len(labels) != arr.shape[0]:
        raise ValueError("CHIME labels count %d does not match data count %d" % (len(labels), arr.shape[0]))
    x = normalize_images_to_uint8(arr)[..., np.newaxis]
    y = np.asarray([CLASS_TO_INDEX[label] for label in labels], dtype=np.int64)
    return x, y, labels


def paper_half_split_chime(data_file, label_file):
    """Reproduce the paper/notebook split: first half of each class for training, second half for test."""
    x, y, labels = load_chime_data(data_file, label_file)
    train_idx = []
    test_idx = []
    for class_idx in range(len(CLASSES)):
        idx = np.where(y == class_idx)[0]
        half = int(len(idx) / 2)
        train_idx.extend(idx[:half].tolist())
        test_idx.extend(idx[half:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx], train_idx, test_idx


def stratified_train_val_split(x, y, val_fraction=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    for class_idx in sorted(np.unique(y).tolist()):
        idx = np.where(y == class_idx)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], train_idx, val_idx


def build_real_noise_bank(chime_x, bank_size=512, blank_fraction=0.2, seed=42):
    """Build 256x256 real-noise templates from low-power CHIME time columns."""
    rng = np.random.RandomState(seed)
    chime_x = np.asarray(chime_x)
    if chime_x.ndim == 4:
        chime_x = chime_x[..., 0]
    if chime_x.shape[0] == 0:
        return np.empty((0, 256, 256, 1), dtype=np.float32)

    bank = []
    n_blank = max(8, int(round(chime_x.shape[2] * blank_fraction)))
    for _ in range(bank_size):
        img = chime_x[rng.randint(0, chime_x.shape[0])].astype(np.float32) / 255.0
        col_power = np.mean(img, axis=0)
        blank_cols = np.argsort(col_power)[:n_blank]
        sampled_cols = rng.choice(blank_cols, size=img.shape[1], replace=True)
        noise = img[:, sampled_cols]
        noise = noise - np.mean(noise)
        std = float(np.std(noise))
        if std > 1e-6:
            noise = noise / std
        bank.append(noise[..., np.newaxis].astype(np.float32))
    return np.asarray(bank, dtype=np.float32)


def build_bucketed_real_noise_bank(chime_x, bank_size=768, blank_fraction=0.2, seed=42, n_buckets=3):
    """Build CHIME noise templates bucketed by blank-region noise level.

    Buckets are ordered low -> high noise. The training pipeline maps high-SNR
    simulated bursts to lower-noise buckets and low-SNR bursts to higher-noise
    buckets, while real CHIME training examples sample buckets at random.
    """
    rng = np.random.RandomState(seed)
    chime_x = np.asarray(chime_x)
    if chime_x.ndim == 4:
        chime_x = chime_x[..., 0]
    if chime_x.shape[0] == 0:
        return np.empty((n_buckets, 0, 256, 256, 1), dtype=np.float32), []

    candidates = []
    n_blank = max(8, int(round(chime_x.shape[2] * blank_fraction)))
    n_candidates = max(bank_size * 2, n_buckets)

    for _ in range(n_candidates):
        img_idx = int(rng.randint(0, chime_x.shape[0]))
        img = chime_x[img_idx].astype(np.float32) / 255.0
        col_power = np.mean(img, axis=0)
        blank_cols = np.argsort(col_power)[:n_blank]
        sampled_cols = rng.choice(blank_cols, size=img.shape[1], replace=True)
        noise = img[:, sampled_cols]
        raw_std = float(np.std(noise))
        noise = noise - np.mean(noise)
        std = float(np.std(noise))
        if std > 1e-6:
            noise = noise / std
        candidates.append((raw_std, noise[..., np.newaxis].astype(np.float32), img_idx))

    candidates.sort(key=lambda item: item[0])
    per_bucket = max(1, int(np.ceil(float(bank_size) / float(n_buckets))))
    buckets = []
    summary = []

    for bucket_idx in range(n_buckets):
        lo = int(round(bucket_idx * len(candidates) / float(n_buckets)))
        hi = int(round((bucket_idx + 1) * len(candidates) / float(n_buckets)))
        bucket_candidates = candidates[lo:hi] or candidates
        chosen = rng.choice(np.arange(len(bucket_candidates)), size=per_bucket, replace=len(bucket_candidates) < per_bucket)
        bucket = [bucket_candidates[int(i)][1] for i in chosen]
        buckets.append(np.asarray(bucket, dtype=np.float32))
        raw_stds = [bucket_candidates[int(i)][0] for i in chosen]
        summary.append(
            {
                "bucket": int(bucket_idx),
                "noise_level": ["low", "medium", "high"][bucket_idx] if n_buckets == 3 else str(bucket_idx),
                "n_templates": int(len(bucket)),
                "raw_std_mean": float(np.mean(raw_stds)),
                "raw_std_min": float(np.min(raw_stds)),
                "raw_std_max": float(np.max(raw_stds)),
            }
        )

    return np.asarray(buckets, dtype=np.float32), summary
