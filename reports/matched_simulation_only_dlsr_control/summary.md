# Matched Simulation-Only DLSR Control

## Protocol

- The DLSR architecture, focal costs, dual-confirm policy, threshold 0.40, C1 protection, beta=0.20, test samples, preprocessing, and label mapping are fixed to the adapted system.
- Every trainable control checkpoint (base, tri-specialist, pair specialists, and router) receives simulation samples only.
- No CHIME development sample enters the control optimizer; no paper-test sample is used for tuning or selection.
- The configuration itself is inherited from the previously development-locked DLSR, so the precise claim is simulation-only training under a fixed DLSR configuration.

## Paper-Test Results

| System | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | C1 recall | Local balanced acc. |
|---|---:|---:|---:|---:|---:|---:|
| Simulation-only DLSR | 0.5485 +/- 0.0000 | 0.4195 +/- 0.0000 | 0.3980 +/- 0.0000 | 0.5500 +/- 0.0000 | 0.3000 +/- 0.0000 | 0.4654 +/- 0.0000 |
| CHIME-adapted DLSR | 0.6318 +/- 0.0022 | 0.5549 +/- 0.0009 | 0.5307 +/- 0.0024 | 0.6381 +/- 0.0019 | 0.3000 +/- 0.0000 | 0.4700 +/- 0.0014 |

## Matched Effect

- Accuracy difference (adapted - control): 8.33 percentage points.
- Macro-F1 difference (adapted - control): 13.27 percentage points.
- Mean wrong-to-correct changes per seed: 51.00.
- Mean correct-to-wrong changes per seed: 28.67.
- Exact McNemar and 10,000-replicate paired stratified bootstrap/randomization results are stored per seed.

This matched control supplies the missing per-sample evidence for the simulation-to-real adaptation claim.
