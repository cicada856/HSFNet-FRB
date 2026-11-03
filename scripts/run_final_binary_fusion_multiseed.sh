#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TF_ENV="${TF_ENV:-frabjous}"
SEEDS="${SEEDS:-1 2 3 4 5}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-50}"
BASE_DIR="${BASE_DIR:-artifacts/runs/multiseed}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_py() {
  conda run --no-capture-output -n "$TF_ENV" python -u "$@"
}

train_if_missing() {
  local model_dir="$1"
  shift
  if [[ -f "${model_dir}/best_model/saved_model.pb" ]]; then
    log "[skip train] ${model_dir}"
  else
    log "[train] ${model_dir}"
    "$@"
  fi
}

eval_if_missing() {
  local eval_dir="$1"
  shift
  if [[ -f "${eval_dir}/metrics_summary.json" ]]; then
    log "[skip eval] ${eval_dir}"
  else
    log "[eval] ${eval_dir}"
    "$@"
  fi
}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi

log "Final binary-specialist fusion multiseed experiment"
log "Seeds: $SEEDS"
log "GPU id: $GPU_ID"
log "Epochs: $EPOCHS"
log "Base dir: $BASE_DIR"
log "This script reuses existing ResNet50 base and B/C1/C2 tri-specialist seed models when present."

for seed in ${SEEDS}; do
  log "========== seed ${seed} =========="

  repair_dir="${BASE_DIR}/resnet50_c1_repair_seed_${seed}"
  train_if_missing "${repair_dir}" \
    run_py src/hsfnet/train_multiclass.py \
      --device gpu \
      --gpu-id "$GPU_ID" \
      --seed "$seed" \
      --backbone resnet50 \
      --output-dir "$repair_dir" \
      --epochs "$EPOCHS" \
      --batch-size 8 \
      --learning-rate 1e-4 \
      --dropout 0.3 \
      --mask-prob 0.7 \
      --mask-max-fraction 0.25 \
      --noise-prob 0.6 \
      --noise-scale 0.025 \
      --class-aug-multipliers A:0.5,B:1.0,C:1.0,C1:0.4,C2:1.0 \
      --class-weights A:1.15,B:0.75,C:1.0,C1:3.0,C2:1.2 \
      --early-stop-patience 10

  specialist_dir="${BASE_DIR}/specialist_b_c1_c2_seed_${seed}"
  train_if_missing "${specialist_dir}" \
    run_py src/hsfnet/train_specialist_b_c1_c2.py \
      --device gpu \
      --gpu-id "$GPU_ID" \
      --seed "$seed" \
      --backbone resnet18 \
      --output-dir "$specialist_dir" \
      --epochs "$EPOCHS" \
      --batch-size 16 \
      --learning-rate 2e-4 \
      --dropout 0.3 \
      --mask-prob 0.65 \
      --mask-max-fraction 0.22 \
      --noise-prob 0.55 \
      --noise-scale 0.02 \
      --class-aug-multipliers B:1.0,C1:0.35,C2:0.85 \
      --class-weights B:0.7,C1:4.0,C2:1.3 \
      --early-stop-patience 10

  bc1_dir="${BASE_DIR}/pair_B_vs_C1_seed_${seed}"
  train_if_missing "${bc1_dir}" \
    run_py src/hsfnet/train_pair_specialist.py \
      --device gpu \
      --gpu-id "$GPU_ID" \
      --seed "$seed" \
      --class-a B \
      --class-b C1 \
      --backbone resnet18 \
      --output-dir "$bc1_dir" \
      --epochs "$EPOCHS" \
      --batch-size 16 \
      --learning-rate 2e-4 \
      --dropout 0.3 \
      --mask-prob 0.6 \
      --mask-max-fraction 0.2 \
      --noise-prob 0.5 \
      --noise-scale 0.02 \
      --class-a-aug-multiplier 1.0 \
      --class-b-aug-multiplier 0.35 \
      --class-a-weight 0.7 \
      --class-b-weight 4.0 \
      --early-stop-patience 10

  c1c2_dir="${BASE_DIR}/pair_C1_vs_C2_seed_${seed}"
  train_if_missing "${c1c2_dir}" \
    run_py src/hsfnet/train_pair_specialist.py \
      --device gpu \
      --gpu-id "$GPU_ID" \
      --seed "$seed" \
      --class-a C1 \
      --class-b C2 \
      --backbone resnet18 \
      --output-dir "$c1c2_dir" \
      --epochs "$EPOCHS" \
      --batch-size 16 \
      --learning-rate 2e-4 \
      --dropout 0.3 \
      --mask-prob 0.6 \
      --mask-max-fraction 0.2 \
      --noise-prob 0.5 \
      --noise-scale 0.02 \
      --class-a-aug-multiplier 0.35 \
      --class-b-aug-multiplier 0.85 \
      --class-a-weight 3.5 \
      --class-b-weight 1.3 \
      --early-stop-patience 10

  fusion_dir="${BASE_DIR}/final_binary_specialist_fusion_seed_${seed}/eval_predicted_subset"
  eval_if_missing "$fusion_dir" \
    run_py src/hsfnet/evaluate_binary_specialist_fusion.py \
      --device gpu \
      --gpu-id "$GPU_ID" \
      --base-model "${repair_dir}/best_model" \
      --tri-specialist-model "${specialist_dir}/best_model" \
      --b-c1-model "${bc1_dir}/best_model" \
      --c1-c2-model "${c1c2_dir}/best_model" \
      --output-dir "$fusion_dir" \
      --split paper_test \
      --route-mode predicted_subset \
      --c1-vote-bonus 0.15 \
      --batch-size 32
done

log "Summarizing final binary-specialist fusion multiseed results"
run_py src/hsfnet/experiments/summarize_final_binary_fusion_multiseed.py \
  --base-dir "$BASE_DIR" \
  --seeds ${SEEDS} \
  --output-prefix "${BASE_DIR}/final_binary_fusion_multiseed_summary"

log "Done. Summary:"
cat "${BASE_DIR}/final_binary_fusion_multiseed_summary.md"

