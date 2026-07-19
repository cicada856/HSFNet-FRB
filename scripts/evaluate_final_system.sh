#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-gpu}

$PYTHON src/hsfnet/evaluate_binary_specialist_fusion.py \
  --base-model artifacts/runs/base_resnet50/best_model \
  --tri-specialist-model artifacts/runs/tri_specialist/best_model \
  --b-c1-model artifacts/runs/pair_b_c1/best_model \
  --c1-c2-model artifacts/runs/pair_c1_c2/best_model \
  --split paper_test --route-mode predicted_subset --c1-vote-bonus 0.20 \
  --output-dir artifacts/evaluation/final_hsfnet \
  --device "$DEVICE"
