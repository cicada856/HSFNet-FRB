# Paired Statistical Analysis

All comparisons use the same 268 CHIME paper-test samples in identical order.
The simulation-only and CHIME-adapted DLSR predictions are paired by sample.

## Seed-Level Results

| Seed | Accuracy gain | Macro-F1 gain | Wrong to correct | Correct to wrong | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|
| 42 | +8.58 pp | +13.55 pp | 51 | 28 | 0.0128 |
| 43 | +8.21 pp | +13.14 pp | 51 | 29 | 0.0183 |
| 44 | +8.21 pp | +13.14 pp | 51 | 29 | 0.0183 |

The adapted system has significantly more wrong-to-correct than
correct-to-wrong changes for every seed under the exact McNemar test.

For seed 42, the paired accuracy gain is +8.58 percentage points with a
class-stratified bootstrap 95% confidence interval of +2.61 to +14.93
percentage points and a paired-randomization p-value of 0.0108.

The seed-42 Macro-F1 gain is +13.55 percentage points with a 95% confidence
interval of +7.37 to +20.25 percentage points and p=0.0002. Seeds 43 and 44
give the same substantive conclusion.

The local B/C1/C2 balanced-accuracy difference is small and not significant.
The main adaptation gain is driven by improved A and C recall, while C1 recall
remains 30% in both systems.

Full 10,000-replicate results are stored in each seed metrics JSON file.
