#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TF_ENV="${TF_ENV:-frabjous}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-1800}"
EPOCHS="${EPOCHS:-50}"

RUN_ROOT="artifacts/runs/sim_only_mainline"
BASE_RUN="${RUN_ROOT}/resnet50_base_seed_${SEED}_gpu${GPU_ID}"
TRI_RUN="${RUN_ROOT}/specialist_b_c1_c2_seed_${SEED}_gpu${GPU_ID}"
BC1_RUN="${RUN_ROOT}/pair_B_vs_C1_seed_${SEED}_gpu${GPU_ID}"
C1C2_RUN="${RUN_ROOT}/pair_C1_vs_C2_seed_${SEED}_gpu${GPU_ID}"
FUSION_RUN="${RUN_ROOT}/fusion_eval_seed_${SEED}_gpu${GPU_ID}/eval_predicted_subset"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Sim-only mainline experiment"
log "Training/validation data: simulated FRBs only"
log "No CHIME train examples and no CHIME-derived real-noise bank are used for training"
log "Final test data: CHIME paper-test half only"
log "TF env: $TF_ENV"
log "GPU id: $GPU_ID"
log "Seed: $SEED"
log "Samples per simulated class: $SAMPLES_PER_CLASS"
log "Epochs: $EPOCHS"
log "Run root: $RUN_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
if [ -n "${CONDA_PREFIX:-}" ]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi

log "Training sim-only ResNet50 base classifier"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/train_multiclass.py \
  --device gpu \
  --gpu-id "$GPU_ID" \
  --backbone resnet50 \
  --output-dir "$BASE_RUN" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --epochs "$EPOCHS" \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --dropout 0.3 \
  --no-chime-train \
  --mask-prob 0.7 \
  --mask-max-fraction 0.25 \
  --noise-prob 0.0 \
  --noise-scale 0.0 \
  --class-aug-multipliers A:0.5,B:1.0,C:1.0,C1:0.4,C2:1.0 \
  --class-weights A:1.15,B:0.75,C:1.0,C1:3.0,C2:1.2 \
  --early-stop-patience 10 \
  --seed "$SEED"

log "Evaluating sim-only ResNet50 base on held-out CHIME paper-test"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/evaluate_multiclass.py \
  --device gpu \
  --gpu-id "$GPU_ID" \
  --model "$BASE_RUN/best_model" \
  --output-dir "$BASE_RUN/eval_paper_compare" \
  --split paper_test \
  --batch-size 32

log "Training sim-only B/C1/C2 tri-specialist"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/train_specialist_b_c1_c2.py \
  --device gpu \
  --gpu-id "$GPU_ID" \
  --backbone resnet18 \
  --output-dir "$TRI_RUN" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --dropout 0.3 \
  --no-chime-train \
  --mask-prob 0.65 \
  --mask-max-fraction 0.22 \
  --noise-prob 0.0 \
  --noise-scale 0.0 \
  --class-aug-multipliers B:1.0,C1:0.35,C2:0.85 \
  --class-weights B:0.7,C1:4.0,C2:1.3 \
  --early-stop-patience 10 \
  --seed "$SEED"

log "Training sim-only B-vs-C1 binary specialist"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/train_pair_specialist.py \
  --device gpu \
  --gpu-id "$GPU_ID" \
  --class-a B \
  --class-b C1 \
  --backbone resnet18 \
  --output-dir "$BC1_RUN" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --dropout 0.3 \
  --no-chime-train \
  --mask-prob 0.6 \
  --mask-max-fraction 0.2 \
  --noise-prob 0.0 \
  --noise-scale 0.0 \
  --class-a-aug-multiplier 1.0 \
  --class-b-aug-multiplier 0.35 \
  --class-a-weight 0.7 \
  --class-b-weight 4.0 \
  --early-stop-patience 10 \
  --seed "$SEED"

log "Training sim-only C1-vs-C2 binary specialist"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/train_pair_specialist.py \
  --device gpu \
  --gpu-id "$GPU_ID" \
  --class-a C1 \
  --class-b C2 \
  --backbone resnet18 \
  --output-dir "$C1C2_RUN" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --dropout 0.3 \
  --no-chime-train \
  --mask-prob 0.6 \
  --mask-max-fraction 0.2 \
  --noise-prob 0.0 \
  --noise-scale 0.0 \
  --class-a-aug-multiplier 0.35 \
  --class-b-aug-multiplier 0.85 \
  --class-a-weight 3.5 \
  --class-b-weight 1.3 \
  --early-stop-patience 10 \
  --seed "$SEED"

log "Evaluating full sim-only hierarchical/binary-specialist fusion on held-out CHIME paper-test"
conda run --no-capture-output -n "$TF_ENV" python -u src/hsfnet/evaluate_binary_specialist_fusion.py \
  --device gpu \
  --gpu-id "$GPU_ID" \
  --base-model "$BASE_RUN/best_model" \
  --tri-specialist-model "$TRI_RUN/best_model" \
  --b-c1-model "$BC1_RUN/best_model" \
  --c1-c2-model "$C1C2_RUN/best_model" \
  --output-dir "$FUSION_RUN" \
  --split paper_test \
  --route-mode predicted_subset \
  --c1-vote-bonus 0.15 \
  --batch-size 32

log "Summarizing sim-only experiment"
BASE_RUN="$BASE_RUN" FUSION_RUN="$FUSION_RUN" conda run --no-capture-output -n "$TF_ENV" python - <<'PY'
import json
import os
from pathlib import Path

current_best = Path("artifacts/runs/binary_specialist_fusion/eval_predicted_subset/metrics_summary.json")
paths = [
    ("current_best_with_chime_train", current_best),
    ("sim_only_base", Path(os.environ["BASE_RUN"]) / "eval_paper_compare" / "metrics_summary.json"),
    ("sim_only_fusion", Path(os.environ["FUSION_RUN"]) / "metrics_summary.json"),
]

for name, path in paths:
    data = json.loads(path.read_text())
    rec = data.get("per_class_recall", {})
    print(
        "%s: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f C1_recall=%.4f"
        % (name, data["accuracy"], data["macro_f1"], data["weighted_f1"], rec.get("C1", 0.0))
    )
PY

log "Done"

