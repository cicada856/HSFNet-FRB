from __future__ import print_function

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from data import paper_half_split_chime  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Measure parameter count and inference time for the current best pipeline.")
    parser.add_argument("--base-model", default="artifacts/runs/resnet50_c1_repair_gpu0/best_model")
    parser.add_argument("--tri-specialist-model", default="artifacts/runs/specialist_b_c1_c2_gpu0/best_model")
    parser.add_argument("--b-c1-model", default="artifacts/runs/pair_specialist_B_vs_C1_gpu0/best_model")
    parser.add_argument("--c1-c2-model", default="artifacts/runs/pair_specialist_C1_vs_C2_gpu0/best_model")
    parser.add_argument("--chime-data", default="data/chime/chime_interp_frbs_data.npz")
    parser.add_argument("--chime-labels", default="data/chime/chime_labels.txt")
    parser.add_argument("--output-dir", default="reports/inference_cost")
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="gpu")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def configure_device(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def make_ds(tf, x, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(x)

    def preprocess(image):
        return tf.cast(image, tf.float32) / 255.0

    return ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def time_predict(model, ds, n_samples, repeats):
    times = []
    model.predict(ds, verbose=0)
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(ds, verbose=0)
        times.append(time.perf_counter() - start)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_seconds": float(np.mean(arr)),
        "std_seconds": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "mean_ms_per_sample": float(np.mean(arr) * 1000.0 / n_samples),
    }


def main():
    args = parse_args()
    configure_device(args)

    import tensorflow as tf
    from tensorflow import keras

    import losses  # noqa: F401

    if args.device in ("gpu", "auto"):
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    physical_gpus = tf.config.list_physical_devices("GPU")

    _, _, x, _, _, _ = paper_half_split_chime(args.chime_data, args.chime_labels)
    ds = make_ds(tf, x, args.batch_size)

    models = {
        "base_resnet50": args.base_model,
        "tri_specialist_b_c1_c2": args.tri_specialist_model,
        "binary_specialist_b_vs_c1": args.b_c1_model,
        "binary_specialist_c1_vs_c2": args.c1_c2_model,
    }
    rows = {}
    for name, path in models.items():
        model = keras.models.load_model(path)
        rows[name] = {
            "path": path,
            "parameters": int(model.count_params()),
            "timing_on_268_chime_test": time_predict(model, ds, x.shape[0], args.repeats),
        }

    rows["pipeline_total"] = {
        "parameters_if_all_models_loaded": int(sum(v["parameters"] for v in rows.values() if "parameters" in v)),
        "note": "Timing rows are per-model full-test timings. The deployed routed pipeline only runs specialists for routed samples.",
        "device_requested": args.device,
        "gpu_id_requested": str(args.gpu_id),
        "physical_gpus_visible": [str(gpu) for gpu in physical_gpus],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "inference_cost.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    with (out_dir / "inference_cost.md").open("w", encoding="utf-8") as f:
        f.write("# Inference Cost\n\n")
        f.write("| Component | Parameters | Mean ms/sample on CHIME test |\n|---|---:|---:|\n")
        for name, row in rows.items():
            if "parameters" not in row:
                continue
            f.write(
                "| %s | %d | %.3f |\n"
                % (name, row["parameters"], row["timing_on_268_chime_test"]["mean_ms_per_sample"])
            )
        f.write("\nTotal loaded parameters: `%d`\n" % rows["pipeline_total"]["parameters_if_all_models_loaded"])
        f.write("\nDevice requested: `%s`\n" % rows["pipeline_total"]["device_requested"])
        f.write("\nVisible physical GPUs: `%s`\n" % (", ".join(rows["pipeline_total"]["physical_gpus_visible"]) or "none"))

    print("Wrote inference cost to %s" % out_dir)


if __name__ == "__main__":
    main()
