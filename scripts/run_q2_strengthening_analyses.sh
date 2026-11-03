#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TF_ENV="${TF_ENV:-frabjous}"
GPU_ID="${GPU_ID:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-gpu}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RUN_BONUS_SWEEP="${RUN_BONUS_SWEEP:-1}"
BONUS_VALUES="${BONUS_VALUES:-0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.50}"
N_BOOTSTRAP="${N_BOOTSTRAP:-5000}"
OUT_DIR="${OUT_DIR:-reports/q2_strengthening}"
SWEEP_DIR="${SWEEP_DIR:-artifacts/runs/q2_strengthening/c1_bonus_sweep}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

bonus_tag() {
  printf '%s' "$1" | sed 's/\./p/g'
}

log "Q2/Q3 strengthening analysis package"
log "Output reports: $OUT_DIR"
log "Bonus sweep outputs: $SWEEP_DIR"
log "TensorFlow env: $TF_ENV"
log "Device: $EVAL_DEVICE, GPU id: $GPU_ID"

mkdir -p "$OUT_DIR" "$SWEEP_DIR"

if [[ "$RUN_BONUS_SWEEP" == "1" ]]; then
  log "1/2 Running C1 vote-bonus sensitivity sweep"
  for BONUS in $BONUS_VALUES; do
    TAG="$(bonus_tag "$BONUS")"
    EVAL_OUT="$SWEEP_DIR/bonus_${TAG}"
    log "Evaluating C1 vote bonus = $BONUS -> $EVAL_OUT"
    conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/evaluate_binary_specialist_fusion.py \
      --device "$EVAL_DEVICE" \
      --gpu-id "$GPU_ID" \
      --base-model artifacts/models/final/base_resnet50 \
      --tri-specialist-model artifacts/models/final/tri_specialist_b_c1_c2 \
      --b-c1-model artifacts/models/final/binary_specialist_b_vs_c1 \
      --c1-c2-model artifacts/models/final/binary_specialist_c1_vs_c2 \
      --output-dir "$EVAL_OUT" \
      --split paper_test \
      --route-mode predicted_subset \
      --c1-vote-bonus "$BONUS" \
      --batch-size "$BATCH_SIZE"
  done
else
  log "1/2 Skipping C1 vote-bonus sweep because RUN_BONUS_SWEEP=$RUN_BONUS_SWEEP"
fi

log "2/2 Compiling paired tests, baselines, ablations, and stratified analysis"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/experiments/q2_strengthening_analyses.py \
  --output-dir "$OUT_DIR" \
  --final-eval-dir artifacts/runs/binary_specialist_fusion/eval_predicted_subset \
  --bonus-sweep-dir "$SWEEP_DIR" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed 123

log "Done. Read: $OUT_DIR/README.md"
