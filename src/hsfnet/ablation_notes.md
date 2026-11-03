# Ablation Notes

## 2026-07-09: Noisebank v2 augmentation

Purpose: test a stronger real-noise augmentation strategy for CHIME-like robustness.

Changes tested:
- Bucketed CHIME real-noise bank by burst SNR / mask-ratio related metadata.
- Additional corruption augmentations:
  - random frequency-band RFI masking
  - random time dropout
  - random local patch corruption
- Same downstream binary-specialist fusion evaluation as the current best model.

Main result: this did not improve the current best model.

Clean CHIME paper-test comparison:

| Model | Accuracy | Macro-F1 | Weighted-F1 | C1 recall |
| --- | ---: | ---: | ---: | ---: |
| Current best hierarchical ResNet50 + specialists | 63.43 | 53.28 | 64.01 | 30.00 |
| Noisebank v2 + specialists | 61.94 | 52.58 | 62.38 | 30.00 |

Mask robustness comparison:

| Fixed mask ratio | Current best accuracy | Current best macro-F1 | Noisebank v2 accuracy | Noisebank v2 macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 63.43 | 53.28 | 61.94 | 52.58 |
| 0.1 | 62.69 | 50.43 | 61.94 | 49.36 |
| 0.2 | 60.45 | 47.27 | 55.97 | 42.80 |
| 0.3 | 55.97 | 39.83 | 54.85 | 39.28 |

Interpretation:
- Noisebank v2 is a negative ablation for the current setup.
- It makes training more realistic, but the augmentation strength/distribution does not improve clean CHIME test performance or fixed-mask robustness.
- It should not replace the current best model.
- It can be reported as evidence that augmentation distribution matching matters.

Result paths:
- Current best clean: `artifacts/runs/binary_specialist_fusion/eval_predicted_subset/`
- Current best mask robustness: `artifacts/runs/binary_specialist_fusion/eval_mask_0p*/`
- Noisebank v2 clean: `artifacts/runs/noisebank_v2_binary_specialist_fusion/eval_predicted_subset/`
- Noisebank v2 mask robustness: `artifacts/runs/noisebank_v2_binary_specialist_fusion/eval_mask_0p*/`

