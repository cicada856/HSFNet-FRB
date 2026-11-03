# Decoupled Cost-Sensitive Hard Router

## Protocol

- The published five-class base and all specialists are frozen.
- A separate binary router predicts global A/C versus local B/C1/C2 from frozen base features.
- Router architecture, asymmetric costs, policy, and thresholds were selected with five-fold CHIME development-only OOF results.
- Paper-test samples were first indexed after `locked_config.json` was written.
- The existing beta=0.20 specialist fusion is unchanged.

## Locked Configuration

- Active: `True`
- Router: `{'batch_size': 256, 'learning_rate': 0.001, 'dropout': 0.15, 'l2': 1e-05, 'name': 'focal_global_h64', 'hidden_dim': 64, 'development_weight': 16.0, 'false_positive_cost': 2.0, 'false_negative_cost': 1.0, 'focal_gamma': 2.0}`
- Policy: `{'name': 'dual_confirm', 'confirm_threshold': 0.4, 'protect_base_c1': True}`

## Paper-Test Aggregate

| Model | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | C1 recall | Local balanced acc. |
|---|---:|---:|---:|---:|---:|---:|
| Current Top-1 | 0.6343 | 0.5559 | 0.5335 | 0.6403 | 0.3000 | 0.4716 |
| DLSR mean +/- std | 0.6318 +/- 0.0022 | 0.5549 +/- 0.0009 | 0.5307 +/- 0.0024 | 0.6381 +/- 0.0019 | 0.3000 +/- 0.0000 | 0.4700 +/- 0.0014 |

## Decision

- Criteria: `{'accuracy_within_1pp': True, 'macro_f1_within_0p5pp': True, 'c1_recall_not_lower': True, 'a_recall_drop_at_most_one_sample': True, 'c_recall_drop_at_most_one_sample': True, 'at_least_two_favorable_seeds': False, 'router_active_all_seeds': False, 'development_guard_passed': True, 'recommend_replacement': False}`
- Recommendation: retain current Top-1 routing; keep DLSR as an ablation.

The paper LaTeX, final checkpoints, and all prior reports were not modified.
