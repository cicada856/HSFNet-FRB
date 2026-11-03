# Complete true-label subsets versus routed subsets

The complete-subset evaluation measures each specialist on every applicable paper-test example. The routed comparison is conditional on the existing base-model predicted-subset routing and is therefore smaller and selection-dependent.

| Specialist | Full n | Full accuracy | Full balanced accuracy | Full macro-F1 | Routed applicable n | Routed accuracy | Routed balanced accuracy | Routed macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tri-specialist: B / C1 / C2 | 155 | 82.58% | 53.21% | 51.12% | 112 | 80.36% | 56.86% | 53.13% |
| Binary specialist: B / C1 | 145 | 73.79% | 62.78% | 52.57% | 103 | 74.76% | 66.59% | 54.21% |
| Binary specialist: C1 / C2 | 20 | 70.00% | 70.00% | 70.00% | 16 | 68.75% | 69.05% | 68.63% |

Routed-subset values should not be interpreted as independent test-set estimates: routing depends on the base model's predictions. The complete true-label subsets are the direct standalone expert evaluation requested here.
