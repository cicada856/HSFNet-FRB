from __future__ import print_function

import csv
import json
import argparse
from pathlib import Path


OUT_DIR = Path("reports")


def parse_args():
    parser = argparse.ArgumentParser(description="Compile publication-ready result tables from local experiment outputs.")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value):
    if value is None:
        return ""
    return "%.2f" % (float(value) * 100.0)


def metrics_from_summary(path):
    payload = read_json(path)
    if payload is None:
        return None
    return {
        "accuracy": payload.get("accuracy"),
        "macro_f1": payload.get("macro_f1"),
        "weighted_f1": payload.get("weighted_f1"),
        "macro_precision": payload.get("macro_precision"),
        "macro_recall": payload.get("macro_recall"),
        "weighted_precision": payload.get("weighted_precision"),
        "weighted_recall": payload.get("weighted_recall"),
        "per_class_recall": payload.get("per_class_recall", {}),
        "path": str(path),
    }


def metrics_from_classification_report(path):
    payload = read_json(path)
    if payload is None:
        return None
    return {
        "accuracy": payload.get("accuracy"),
        "macro_f1": payload.get("macro avg", {}).get("f1-score"),
        "weighted_f1": payload.get("weighted avg", {}).get("f1-score"),
        "macro_precision": payload.get("macro avg", {}).get("precision"),
        "macro_recall": payload.get("macro avg", {}).get("recall"),
        "weighted_precision": payload.get("weighted avg", {}).get("precision"),
        "weighted_recall": payload.get("weighted avg", {}).get("recall"),
        "per_class_recall": {name: payload.get(name, {}).get("recall") for name in ["A", "B", "C", "C1", "C2"]},
        "path": str(path),
    }


def row(label, result, training_data="", test_data="CHIME paper-test", note=""):
    if result is None:
        return {
            "model": label,
            "training_data": training_data,
            "test_data": test_data,
            "accuracy": "",
            "macro_f1": "",
            "weighted_f1": "",
            "c1_recall": "",
            "source": "missing",
            "note": note,
        }
    return {
        "model": label,
        "training_data": training_data,
        "test_data": test_data,
        "accuracy": pct(result.get("accuracy")),
        "macro_f1": pct(result.get("macro_f1")),
        "weighted_f1": pct(result.get("weighted_f1")),
        "c1_recall": pct(result.get("per_class_recall", {}).get("C1")),
        "source": result.get("path", ""),
        "note": note,
    }


def write_table(name, rows, columns):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / ("%s.csv" % name)
    md_path = OUT_DIR / ("%s.md" % name)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for item in rows:
            writer.writerow({col: item.get(col, "") for col in columns})

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---" for _ in columns]) + "|\n")
        for item in rows:
            f.write("| " + " | ".join(str(item.get(col, "")) for col in columns) + " |\n")
    return csv_path, md_path


def load_all_results():
    return {
        "paper_repro_single_cnn": metrics_from_classification_report("artifacts/baselines/single_cnn_reproduction/classification_report.json"),
        "one_vs_one_repro": metrics_from_classification_report("artifacts/baselines/one_vs_one_reproduction/classification_report.json"),
        "resnet_mask_noise": metrics_from_summary("artifacts/runs/resnet_mask_noise_gpu/eval_paper_compare/metrics_summary.json"),
        "resnet18_aug": metrics_from_summary("artifacts/runs/resnet18_aug_gpu0/eval_paper_compare/metrics_summary.json"),
        "resnet50_aug": metrics_from_summary("artifacts/runs/resnet50_aug_gpu0/eval_paper_compare/metrics_summary.json"),
        "resnet50_c1_repair": metrics_from_summary("artifacts/runs/resnet50_c1_repair_gpu0/eval_paper_compare/metrics_summary.json"),
        "tri_specialist": metrics_from_summary("artifacts/runs/hierarchical_resnet50_b_c1_c2/eval_predicted_subset/metrics_summary.json"),
        "current_best": metrics_from_summary("artifacts/runs/binary_specialist_fusion/eval_predicted_subset/metrics_summary.json"),
        "current_best_uncertain": metrics_from_summary("artifacts/runs/binary_specialist_fusion/eval_uncertain_subset_m018/metrics_summary.json"),
        "sim_only_base": metrics_from_summary("artifacts/runs/sim_only_mainline/resnet50_base_seed_42_gpu0/eval_paper_compare/metrics_summary.json"),
        "sim_only_fusion": metrics_from_summary("artifacts/runs/sim_only_mainline/fusion_eval_seed_42_gpu0/eval_predicted_subset/metrics_summary.json"),
        "noisebank_v2_base": metrics_from_summary("artifacts/runs/resnet50_noisebank_v2_gpu0/eval_paper_compare/metrics_summary.json"),
        "noisebank_v2_fusion": metrics_from_summary("artifacts/runs/noisebank_v2_binary_specialist_fusion/eval_predicted_subset/metrics_summary.json"),
        "cb_focal_base": metrics_from_summary("artifacts/runs/resnet50_cb_focal_gpu0/eval_paper_compare/metrics_summary.json"),
        "cb_focal_fusion": metrics_from_summary("artifacts/runs/cb_focal_binary_specialist_fusion/eval_predicted_subset/metrics_summary.json"),
        "convnext": metrics_from_summary("artifacts/runs/convnext_tiny_aug_gpu0/eval_paper_compare/metrics_summary.json"),
    }


def make_main_comparison(results):
    rows = [
        row("Original notebook single CNN reproduction", results["paper_repro_single_cnn"], "paper-style", "CHIME paper-test"),
        row("ResNet50 C1 repair base", results["resnet50_c1_repair"], "sim + CHIME train", "CHIME paper-test"),
        row("Hierarchical ResNet50 + B/C1/C2 specialist", results["tri_specialist"], "sim + CHIME train", "CHIME paper-test"),
        row("Final hierarchical ResNet50 + binary specialists", results["current_best"], "sim + CHIME train", "CHIME paper-test", "current best"),
        row("Sim-only final pipeline", results["sim_only_fusion"], "sim only", "CHIME paper-test", "domain-gap test"),
    ]
    write_table("main_comparison", rows, ["model", "training_data", "test_data", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "note", "source"])


def make_ablation_table(results):
    rows = [
        row("Single CNN reproduction", results["paper_repro_single_cnn"], note="baseline reproduction"),
        row("ResNet small + mask/noise", results["resnet_mask_noise"], note="initial stronger pipeline"),
        row("ResNet18 + mask/noise", results["resnet18_aug"], note="backbone comparison"),
        row("ResNet50 + mask/noise", results["resnet50_aug"], note="backbone comparison"),
        row("ResNet50 C1 repair", results["resnet50_c1_repair"], note="main base model"),
        row("+ B/C1/C2 tri-specialist", results["tri_specialist"], note="hierarchical specialist"),
        row("+ B-vs-C1 and C1-vs-C2 binary specialists", results["current_best"], note="current best"),
        row("Uncertain-subset routing", results["current_best_uncertain"], note="routing ablation"),
        row("Noisebank v2 + binary specialists", results["noisebank_v2_fusion"], note="negative augmentation ablation"),
        row("Class-balanced focal + binary specialists", results["cb_focal_fusion"], note="negative loss ablation"),
        row("Sim-only final pipeline", results["sim_only_fusion"], training_data="sim only", note="simulation-to-real domain gap"),
    ]
    write_table("ablation_table", rows, ["model", "training_data", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "note", "source"])


def make_robustness_table():
    rows = []
    for ratio in ["0p0", "0p1", "0p2", "0p3"]:
        current = metrics_from_summary("artifacts/runs/binary_specialist_fusion/eval_mask_%s/metrics_summary.json" % ratio)
        noisebank = metrics_from_summary("artifacts/runs/noisebank_v2_binary_specialist_fusion/eval_mask_%s/metrics_summary.json" % ratio)
        label = ratio.replace("p", ".")
        rows.append({
            "mask_ratio": label,
            "model": "current_best",
            "accuracy": pct(current.get("accuracy")) if current else "",
            "macro_f1": pct(current.get("macro_f1")) if current else "",
            "weighted_f1": pct(current.get("weighted_f1")) if current else "",
            "c1_recall": pct(current.get("per_class_recall", {}).get("C1")) if current else "",
            "source": current.get("path", "") if current else "missing",
        })
        rows.append({
            "mask_ratio": label,
            "model": "noisebank_v2",
            "accuracy": pct(noisebank.get("accuracy")) if noisebank else "",
            "macro_f1": pct(noisebank.get("macro_f1")) if noisebank else "",
            "weighted_f1": pct(noisebank.get("weighted_f1")) if noisebank else "",
            "c1_recall": pct(noisebank.get("per_class_recall", {}).get("C1")) if noisebank else "",
            "source": noisebank.get("path", "") if noisebank else "missing",
        })
    write_table("mask_robustness", rows, ["mask_ratio", "model", "accuracy", "macro_f1", "weighted_f1", "c1_recall", "source"])


def make_domain_gap_table(results):
    rows = []
    sim_val_paths = [
        ("sim_only_base_validation", "artifacts/runs/sim_only_mainline/resnet50_base_seed_42_gpu0/validation_metrics.json"),
        ("sim_only_tri_specialist_validation", "artifacts/runs/sim_only_mainline/specialist_b_c1_c2_seed_42_gpu0/validation_metrics.json"),
        ("sim_only_B_vs_C1_validation", "artifacts/runs/sim_only_mainline/pair_B_vs_C1_seed_42_gpu0/validation_metrics.json"),
        ("sim_only_C1_vs_C2_validation", "artifacts/runs/sim_only_mainline/pair_C1_vs_C2_seed_42_gpu0/validation_metrics.json"),
    ]
    for name, path in sim_val_paths:
        payload = read_json(path)
        if payload:
            rows.append({
                "experiment": name,
                "validation_accuracy": pct(payload.get("accuracy")),
                "chime_test_accuracy": "",
                "chime_test_macro_f1": "",
                "note": "sim-only validation",
                "source": path,
            })
    rows.append({
        "experiment": "sim_only_base_on_chime_test",
        "validation_accuracy": "",
        "chime_test_accuracy": pct(results["sim_only_base"].get("accuracy")) if results["sim_only_base"] else "",
        "chime_test_macro_f1": pct(results["sim_only_base"].get("macro_f1")) if results["sim_only_base"] else "",
        "note": "simulation-to-real base transfer",
        "source": results["sim_only_base"].get("path", "") if results["sim_only_base"] else "",
    })
    rows.append({
        "experiment": "sim_only_fusion_on_chime_test",
        "validation_accuracy": "",
        "chime_test_accuracy": pct(results["sim_only_fusion"].get("accuracy")) if results["sim_only_fusion"] else "",
        "chime_test_macro_f1": pct(results["sim_only_fusion"].get("macro_f1")) if results["sim_only_fusion"] else "",
        "note": "simulation-to-real full pipeline transfer",
        "source": results["sim_only_fusion"].get("path", "") if results["sim_only_fusion"] else "",
    })
    write_table("domain_gap", rows, ["experiment", "validation_accuracy", "chime_test_accuracy", "chime_test_macro_f1", "note", "source"])


def make_per_class_table(results):
    rows = []
    for key, label in [
        ("paper_repro_single_cnn", "Single CNN reproduction"),
        ("resnet50_c1_repair", "ResNet50 C1 repair base"),
        ("tri_specialist", "Tri-specialist fusion"),
        ("current_best", "Current best binary-specialist fusion"),
        ("sim_only_fusion", "Sim-only fusion"),
    ]:
        result = results.get(key)
        if not result:
            continue
        rec = result.get("per_class_recall", {})
        rows.append({
            "model": label,
            "A_recall": pct(rec.get("A")),
            "B_recall": pct(rec.get("B")),
            "C_recall": pct(rec.get("C")),
            "C1_recall": pct(rec.get("C1")),
            "C2_recall": pct(rec.get("C2")),
            "source": result.get("path", ""),
        })
    write_table("per_class_recall", rows, ["model", "A_recall", "B_recall", "C_recall", "C1_recall", "C2_recall", "source"])


def copy_or_link_existing_notes():
    summary = []
    for path in [
        "artifacts/runs/multiseed/multiseed_summary.md",
        "artifacts/runs/multiseed/final_binary_fusion_multiseed_summary.md",
        "artifacts/runs/multiseed/bootstrap_ci_summary.md",
        str(OUT_DIR / "multiseed_summary.md"),
        str(OUT_DIR / "current_best_bootstrap_ci.csv"),
        "artifacts/runs/binary_specialist_fusion/eval_predicted_subset/bootstrap_ci.csv",
        str(OUT_DIR / "data_leakage_audit" / "audit_summary.md"),
        str(OUT_DIR / "inference_cost" / "inference_cost.md"),
        "src/hsfnet/ablation_notes.md",
    ]:
        if Path(path).exists():
            summary.append(path)
    return summary


def make_index(results):
    notes = copy_or_link_existing_notes()
    current = results["current_best"]
    with (OUT_DIR / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Publication Results\n\n")
        if current:
            f.write("Current best model: `Hierarchical ResNet50 + B/C1/C2 tri-specialist + B-vs-C1 and C1-vs-C2 binary specialists`.\n\n")
            f.write("- Accuracy: `%s%%`\n" % pct(current.get("accuracy")))
            f.write("- Macro-F1: `%s%%`\n" % pct(current.get("macro_f1")))
            f.write("- Weighted-F1: `%s%%`\n" % pct(current.get("weighted_f1")))
            f.write("- C1 recall: `%s%%`\n\n" % pct(current.get("per_class_recall", {}).get("C1")))
        f.write("Generated tables:\n\n")
        for name in ["main_comparison", "ablation_table", "mask_robustness", "domain_gap", "per_class_recall"]:
            f.write("- `%s.md` / `%s.csv`\n" % (name, name))
        f.write("\nSupporting outputs:\n\n")
        for path in notes:
            f.write("- `%s`\n" % path)
        f.write("\nInterpretation notes:\n\n")
        f.write("- The held-out CHIME paper-test half is not used for training, validation, early stopping, or hyperparameter fitting.\n")
        f.write("- Sim-only training drops substantially on CHIME test, supporting a simulation-to-real domain gap.\n")
        f.write("- Noisebank v2, class-balanced focal, and SigLIP2 ViT-B are negative ablations in the current setup.\n")


def main():
    global OUT_DIR
    args = parse_args()
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_all_results()
    make_main_comparison(results)
    make_ablation_table(results)
    make_robustness_table()
    make_domain_gap_table(results)
    make_per_class_table(results)
    make_index(results)
    print("Wrote publication result tables to %s" % OUT_DIR)


if __name__ == "__main__":
    main()
