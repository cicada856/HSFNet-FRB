#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
SIM_DIR=${SIM_DIR:-data/simulated/simfrb}
DEVICE=${DEVICE:-gpu}

$PYTHON src/hsfnet/train_multiclass.py \
  --sim-dir "$SIM_DIR" --output-dir artifacts/runs/base_resnet50 \
  --backbone resnet50 --batch-size 8 --learning-rate 1e-4 --dropout 0.3 \
  --epochs 50 --early-stop-patience 10 --seed 42 --device "$DEVICE" \
  --mask-prob 0.7 --mask-max-fraction 0.25 \
  --noise-prob 0.6 --noise-scale 0.025 --noise-bank-size 512 \
  --class-aug-multipliers A:0.5,B:1.0,C:1.0,C1:0.4,C2:1.0 \
  --class-weights A:1.15,B:0.75,C:1.0,C1:3.0,C2:1.2

$PYTHON src/hsfnet/train_specialist_b_c1_c2.py \
  --sim-dir "$SIM_DIR" --output-dir artifacts/runs/tri_specialist \
  --backbone resnet18 --batch-size 16 --learning-rate 2e-4 --dropout 0.3 \
  --epochs 50 --early-stop-patience 10 --seed 42 --device "$DEVICE"

$PYTHON src/hsfnet/train_pair_specialist.py \
  --class-a B --class-b C1 --sim-dir "$SIM_DIR" \
  --output-dir artifacts/runs/pair_b_c1 --backbone resnet18 \
  --batch-size 16 --learning-rate 2e-4 --dropout 0.3 \
  --epochs 50 --early-stop-patience 10 --seed 42 --device "$DEVICE" \
  --class-a-aug-multiplier 1.0 --class-b-aug-multiplier 0.35 \
  --class-a-weight 0.7 --class-b-weight 4.0

$PYTHON src/hsfnet/train_pair_specialist.py \
  --class-a C1 --class-b C2 --sim-dir "$SIM_DIR" \
  --output-dir artifacts/runs/pair_c1_c2 --backbone resnet18 \
  --batch-size 16 --learning-rate 2e-4 --dropout 0.3 \
  --epochs 50 --early-stop-patience 10 --seed 42 --device "$DEVICE" \
  --class-a-aug-multiplier 0.35 --class-b-aug-multiplier 0.85 \
  --class-a-weight 3.5 --class-b-weight 1.3
