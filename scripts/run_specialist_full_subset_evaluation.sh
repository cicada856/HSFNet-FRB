#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/linzhizhen/miniconda3/envs/frabjous/bin/python}"
ENV_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
GPU_ID="${GPU_ID:-1}"

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-hsfnet-specialists}"

cd "${PROJECT_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" -u src/hsfnet/experiments/evaluate_specialists_full_subsets.py \
  --tri-model artifacts/models/final/tri_specialist_b_c1_c2 \
  --b-c1-model artifacts/models/final/binary_specialist_b_vs_c1 \
  --c1-c2-model artifacts/models/final/binary_specialist_c1_vs_c2 \
  --chime-data data/chime/chime_interp_frbs_data.npz \
  --chime-labels data/chime/chime_labels.txt \
  --routed-confidence-csv artifacts/runs/binary_specialist_fusion/eval_predicted_subset/confidence_scores.csv \
  --routed-y-true artifacts/runs/binary_specialist_fusion/eval_predicted_subset/y_true.txt \
  --output-dir reports/specialist_full_true_subsets \
  --batch-size 32 \
  --device gpu \
  --gpu-id "${GPU_ID}"
