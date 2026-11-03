# Innovation Pipeline

This directory contains a notebook-free training and evaluation pipeline for the FRABJOUS reproduction project.

The baseline paper uses simulated FRB dynamic spectra plus CHIME/FRB catalog bursts. This pipeline keeps the same paper-compatible data and metrics, then adds three changes:

1. Masked-channel aware training: random frequency channels are masked during training to mimic CHIME RFI/channel masking.
2. Real-noise augmentation: blank low-power time columns from real CHIME waterfall plots are sampled into a noise bank and added to simulated/training examples.
3. Switchable stronger backbone: the default is `resnet_small`, and the code also supports `simple_cnn`, `resnet18`, `resnet50`, `efficientnetb0`, and `convnext_tiny`.

## Train

From the repo root:

```bash
conda activate frabjous
python src/hsfnet/train_multiclass.py \
  --backbone resnet_small \
  --output-dir artifacts/runs/resnet_mask_noise \
  --epochs 50 \
  --batch-size 32
```

The script defaults to CPU because this machine has shown TensorFlow GPU memory allocation failures in notebook runs. To try GPU:

```bash
python src/hsfnet/train_multiclass.py --device gpu
```

Useful backbone switches:

```bash
python src/hsfnet/train_multiclass.py --backbone simple_cnn
python src/hsfnet/train_multiclass.py --backbone resnet18
python src/hsfnet/train_multiclass.py --backbone resnet50
python src/hsfnet/train_multiclass.py --backbone efficientnetb0
python src/hsfnet/train_multiclass.py --backbone convnext_tiny
```

## Evaluate On The Paper CHIME Split

```bash
python src/hsfnet/evaluate_multiclass.py \
  --model artifacts/runs/resnet_mask_noise/best_model \
  --output-dir artifacts/runs/resnet_mask_noise/eval_chime \
  --split paper_test
```

Outputs:

- `classification_report.json`: precision, recall, f1-score, support, accuracy, macro avg, weighted avg.
- `confusion_matrix.csv`: raw 5x5 confusion matrix.
- `confusion_matrix_counts.png`: count heatmap.
- `confusion_matrix_percent.png`: row-normalized percent heatmap.
- `metrics_summary.json`: compact metrics for comparison.
- `paper_comparison.json`: includes the paper approximate numbers and your already reproduced baseline JSON files when present.

## Evaluate On Simulated Data Too

```bash
python src/hsfnet/evaluate_multiclass.py \
  --model artifacts/runs/resnet_mask_noise/best_model \
  --output-dir artifacts/runs/resnet_mask_noise/eval_chime \
  --split paper_test \
  --eval-sim \
  --sim-samples-per-class 500
```

## Main Comparison Targets

Use these numbers as the first comparison table:

- Paper single multiclass CHIME accuracy: approximately `55%`.
- Your reproduced single multiclass CHIME accuracy: `57.8%`.
- Your reproduced one-vs-one CHIME accuracy: `51.8%`.
- New innovation pipeline: report `accuracy`, `macro-F1`, `weighted-F1`, and confusion matrix on the same CHIME paper test split.
