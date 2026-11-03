# Inference Cost

| Component | Parameters | Mean ms/sample on CHIME test |
|---|---:|---:|
| base_resnet50 | 23597957 | 24.659 |
| tri_specialist_b_c1_c2 | 2801315 | 3.472 |
| binary_specialist_b_vs_c1 | 2801058 | 3.465 |
| binary_specialist_c1_vs_c2 | 2801058 | 3.395 |

Total loaded parameters: `32001388`
