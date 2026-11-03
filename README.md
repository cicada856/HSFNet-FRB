# HSFNet-FRB-DLSR

HSFNet-FRB-DLSR classifies fast radio burst morphology into five classes: A, B, C, C1, and C2.

## Model

A ResNet-50 extracts global features and five-class probabilities. The Decoupled Cost-Sensitive Local-Subspace Router (DLSR) uses these outputs and confidence statistics to decide whether to retain the base prediction or route the input to three parallel ResNet-18 specialists: B/C1/C2, B-vs-C1, and C1-vs-C2. Specialist probabilities and binary votes are combined to produce the local prediction.

The component networks are frozen during router training. The selected router uses 64- and 32-unit hidden layers, cost-sensitive focal loss, a dual-confirmation threshold of 0.40, and local routing for base-predicted C1. The specialist vote weight is 0.20. Configuration selection uses five-fold out-of-fold (OOF) predictions on the development split.

## Experiments

The fixed CHIME split contains 267 development samples and 268 test samples. Router training uses seeds 42, 43, and 44.

| System | Accuracy (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| CHIME-adapted HSFNet-FRB-DLSR | 63.18 ± 0.22 | 53.07 ± 0.24 |
| Simulation-only DLSR control | 54.85 | 39.80 |

Values are means across the three router runs; ± gives the sample standard deviation. The simulation-only control gives the same scores across all three runs. Both systems use the same locked DLSR configuration and test split. The control trains its component networks and router on simulations; the adapted system also uses CHIME development data.

## Reproduction

Run the commands below from the repository root on Linux with a TensorFlow-compatible GPU environment.

### 1. Install dependencies

```bash
conda create -n hsfnet python=3.8 -y
conda activate hsfnet
python -m pip install -r requirements.txt
```

### 2. Prepare data and component models

```text
data/chime/chime_interp_frbs_data.npz
data/chime/chime_labels.txt
data/simulated/simfrb/simdata/type_{class}/SNR_{snr}/simulatefrbs_type_{class}.npz
data/simulated/simfrb/remove_type_C1_SNR_{snr}.txt
data/simulated/simfrb/type_C2_SNR_{snr}.txt

artifacts/models/final/base_resnet50/
artifacts/models/final/tri_specialist_b_c1_c2/
artifacts/models/final/binary_specialist_b_vs_c1/
artifacts/models/final/binary_specialist_c1_vs_c2/
```

Simulation classes are `A, B, C, C1, C2`; SNR values are `15, 25, 35, 50, 100`. CHIME labels are one class name per line in the array's sample order. Use the supplied split at `artifacts/runs/resnet50_c1_repair_gpu0/paper_chime_split_indices.npz`.

Component training entry points are `src/hsfnet/train_multiclass.py`, `train_specialist_b_c1_c2.py`, and `train_pair_specialist.py`. For the simulation-only control, place the four component models at the paths in `reports/matched_simulation_only_dlsr_control/checkpoint_manifest.json`. Data and trained weights are stored separately from this repository.

### 3. Generate development OOF predictions

```bash
python src/hsfnet/experiments/generate_validation_oof_predictions.py \
  --output-dir reports/reproduction_oof \
  --model-root artifacts/runs/validation_beta_selection \
  --samples-per-class 1800 --epochs 50 --seed 42 \
  --device gpu --gpu-id 0

python src/hsfnet/experiments/select_vote_bonus_from_oof.py \
  --oof-csv reports/reproduction_oof/oof_component_predictions.csv \
  --output-dir reports/reproduction_oof
```

This trains the five-fold component models and generates development predictions. The DLSR workflow uses the fold base models under `artifacts/runs/validation_beta_selection/fold_{0..4}/base/best_model/`.

### 4. Train and evaluate DLSR

```bash
python scripts/run_decoupled_cost_sensitive_hard_router.py \
  --oof-components reports/reproduction_oof/oof_component_predictions.csv \
  --report-dir reports/reproduction_dlsr \
  --artifact-dir artifacts/runs/reproduction_dlsr \
  --seeds 42,43,44 --epochs 25 --device gpu --gpu-id 0
```

The script selects the router and routing policy on development OOF predictions, locks the configuration, then trains and evaluates three routers. Outputs include the locked configuration, per-sample predictions, confusion matrices, and aggregate metrics.

### 5. Run the matched simulation-only control

With the simulation-only component models in place, run:

```bash
python scripts/run_matched_simulation_only_dlsr_control.py \
  --report-dir reports/reproduction_simulation_only \
  --artifact-dir artifacts/runs/reproduction_simulation_only \
  --seeds 42,43,44 --epochs 25 --bootstrap-replicates 10000 \
  --device gpu --gpu-id 0
```

This script uses the saved configuration and adapted-system predictions in `reports/decoupled_cost_sensitive_hard_router/`. It trains the control routers and computes paired McNemar tests, bootstrap intervals, and randomization tests against those predictions.

## Saved results

- `reports/decoupled_cost_sensitive_hard_router/`: DLSR configuration, selection records, and test results.
- `reports/matched_simulation_only_dlsr_control/`: control results and paired statistics.
- `reports/validation_beta_selection/`: development folds, OOF predictions, and vote-weight selection.
- `reports/specialist_full_true_subsets/`: specialist predictions and confusion matrices.
- `reports/data_leakage_audit/` and `reports/inference_cost/`: data audit and inference cost.

File checksums are listed in `SHA256SUMS.txt`. Source code is distributed under [LICENSE](LICENSE).
