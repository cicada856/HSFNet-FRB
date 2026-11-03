#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/linzhizhen/miniconda3/envs/frabjous/bin/python}"
export PATH="/home/linzhizhen/miniconda3/envs/frabjous/bin:${PATH}"
export LD_LIBRARY_PATH="/home/linzhizhen/miniconda3/envs/frabjous/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/run_decoupled_cost_sensitive_hard_router.py" \
  --report-dir "${PROJECT_ROOT}/reports/decoupled_cost_sensitive_hard_router" \
  --artifact-dir "${PROJECT_ROOT}/artifacts/runs/decoupled_cost_sensitive_hard_router" \
  --device gpu --gpu-id "${CUDA_VISIBLE_DEVICES}"
