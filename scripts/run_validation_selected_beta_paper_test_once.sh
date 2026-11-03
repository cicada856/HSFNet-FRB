#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/linzhizhen/miniconda3/envs/frabjous/bin/python}"
ENV_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"
REPORT_DIR="${REPORT_DIR:-reports/validation_beta_selection}"

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-validation-beta}"

cd "${PROJECT_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

if [[ ! -f "${REPORT_DIR}/validation_bonus_selection.json" ]]; then
  printf 'Hard gate: validation_bonus_selection.json is missing; paper-test evaluation is forbidden.\n' >&2
  exit 1
fi

exec "${PYTHON_BIN}" -u src/hsfnet/experiments/evaluate_selected_beta_once.py \
  --selection-json "${REPORT_DIR}/validation_bonus_selection.json" \
  --base-model artifacts/models/final/base_resnet50 \
  --tri-model artifacts/models/final/tri_specialist_b_c1_c2 \
  --b-c1-model artifacts/models/final/binary_specialist_b_vs_c1 \
  --c1-c2-model artifacts/models/final/binary_specialist_c1_vs_c2 \
  --chime-data data/chime/chime_interp_frbs_data.npz \
  --chime-labels data/chime/chime_labels.txt \
  --output-dir "${REPORT_DIR}/final_paper_test_evaluation" \
  --batch-size 32 \
  --device gpu \
  --gpu-id "${GPU_ID}"
