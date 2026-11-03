#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TF_ENV="${TF_ENV:-frabjous}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-reports}"
EVAL_DEVICE="${EVAL_DEVICE:-gpu}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Generating publication-ready result package"
log "Output dir: $OUT_DIR"
log "TensorFlow env: $TF_ENV"
log "GPU id: $GPU_ID"
log "Inference-cost device: $EVAL_DEVICE"

mkdir -p "$OUT_DIR"

log "1/5 Bootstrap CI for current best final model"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/experiments/bootstrap_ci.py \
  --eval-dir artifacts/runs/binary_specialist_fusion/eval_predicted_subset \
  --output "$OUT_DIR/current_best_bootstrap_ci" \
  --n-bootstrap 5000 \
  --seed 123

log "2/5 Data leakage / duplicate audit"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/experiments/data_leakage_audit.py \
  --output-dir "$OUT_DIR/data_leakage_audit"

log "3/5 Inference cost and parameter count"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/experiments/inference_cost.py \
  --device "$EVAL_DEVICE" \
  --gpu-id "$GPU_ID" \
  --output-dir "$OUT_DIR/inference_cost" \
  --repeats 5

log "4/5 Refresh multiseed summary from existing local seed results"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/experiments/summarize_multiseed.py \
  --base-dir artifacts/runs/multiseed \
  --output-prefix "$OUT_DIR/multiseed_summary"

log "5/5 Compile main comparison, ablation, robustness, domain-gap, and per-class tables"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/experiments/compile_publication_results.py \
  --output-dir "$OUT_DIR"

log "Done. Read: $OUT_DIR/README.md"
