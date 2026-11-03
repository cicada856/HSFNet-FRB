#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/linzhizhen/miniconda3/envs/frabjous/bin/python}"
ENV_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"
REPORT_DIR="${REPORT_DIR:-reports/validation_beta_selection}"
MODEL_ROOT="${MODEL_ROOT:-artifacts/runs/validation_beta_selection}"

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${PROJECT_ROOT}"

if [[ -e "${REPORT_DIR}/final_paper_test_evaluation" ]]; then
  printf 'Refusing validation selection because a final paper-test evaluation already exists: %s\n' \
    "${REPORT_DIR}/final_paper_test_evaluation" >&2
  exit 1
fi
if [[ -e "${REPORT_DIR}/validation_bonus_selection.json" ]]; then
  printf 'Refusing to rerun a completed beta selection: %s\n' \
    "${REPORT_DIR}/validation_bonus_selection.json" >&2
  exit 1
fi

"${PYTHON_BIN}" -u src/hsfnet/experiments/generate_validation_oof_predictions.py \
  --sim-dir data/simulated/simfrb \
  --chime-data data/chime/chime_interp_frbs_data.npz \
  --chime-labels data/chime/chime_labels.txt \
  --development-indices artifacts/runs/resnet50_c1_repair_gpu0/paper_chime_split_indices.npz \
  --output-dir "${REPORT_DIR}" \
  --model-root "${MODEL_ROOT}" \
  --samples-per-class 1800 \
  --snrs 15,25,35,50,100 \
  --epochs 50 \
  --inner-val-fraction 0.2 \
  --early-stop-patience 10 \
  --seed 42 \
  --batch-size-base 8 \
  --batch-size-specialist 16 \
  --batch-size-inference 32 \
  --device gpu \
  --gpu-id "${GPU_ID}" \
  --resume

"${PYTHON_BIN}" src/hsfnet/experiments/select_vote_bonus_from_oof.py \
  --oof-csv "${REPORT_DIR}/oof_component_predictions.csv" \
  --output-dir "${REPORT_DIR}"

test -f "${REPORT_DIR}/validation_bonus_selection.json"
printf 'Validation-only beta selection complete: %s\n' "${REPORT_DIR}/validation_bonus_selection.json"
printf 'Paper-test evaluation has not been run by this script.\n'
