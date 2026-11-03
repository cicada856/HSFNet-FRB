| model | training data | test data | n | accuracy | macro-F1 | weighted-F1 | C1 recall | C2 recall | note |
|---|---|---|---|---|---|---|---|---|---|
| Original paper single multi-class classifier (Fig. 13, recomputed) | sim + CHIME train half | CHIME half-test | 267 | 55.0562 | 38.3099 | 56.8130 | 10.0000 | 10.0000 | from paper confusion matrix; same task, close split size |
| Original paper One-vs-One binary framework (Fig. 12, recomputed) | sim only | all CHIME | 535 | 49.5327 | 32.6140 | 48.6100 | 20.0000 | 10.0000 | not paired with final split; context only |
| Local original notebook single CNN reproduction |  | CHIME paper-test | 268 | 57.8358 | 44.2040 | 58.9991 | 20.0000 | 30.0000 | baseline reproduction |
| Local original One-vs-One reproduction |  | CHIME paper-test | 535 | 51.7757 | 34.4028 | 50.2120 | 10.0000 | 25.0000 | baseline reproduction |
| ResNet small + mask/noise | sim + CHIME train | CHIME paper-test | 268 | 56.7164 | 46.8353 | 53.6257 | 30.0000 | 50.0000 | initial stronger pipeline |
| ResNet18 + mask/noise | sim + CHIME train | CHIME paper-test | 268 | 60.0746 | 43.8584 | 59.9114 | 0.0000 | 30.0000 | backbone comparison |
| ResNet50 + mask/noise | sim + CHIME train | CHIME paper-test | 268 | 61.9403 | 46.7186 | 62.6886 | 0.0000 | 50.0000 | backbone comparison |
| ResNet50 C1 repair base | sim + CHIME train | CHIME paper-test | 268 | 62.3134 | 48.7033 | 63.1631 | 10.0000 | 50.0000 | main base model |
| Hierarchical ResNet50 + B/C1/C2 specialist | sim + CHIME train | CHIME paper-test | 268 | 63.0597 | 51.3475 | 63.6533 | 20.0000 | 50.0000 | tri-specialist fusion |
| Final HSFNet-FRB binary specialist fusion | sim + CHIME train | CHIME paper-test | 268 | 63.4328 | 53.2774 | 64.0134 | 30.0000 | 50.0000 | current selected model |
| Uncertain-subset routing | sim + CHIME train | CHIME paper-test | 268 | 63.8060 | 52.9646 | 64.4869 | 30.0000 | 50.0000 | routing ablation |
| Noisebank v2 + binary specialists | sim + CHIME train | CHIME paper-test | 268 | 61.9403 | 52.5806 | 62.3797 | 30.0000 | 50.0000 | negative augmentation ablation |
| Class-balanced focal + binary specialists | sim + CHIME train | CHIME paper-test | 268 | 55.2239 | 44.3028 | 54.7174 | 30.0000 | 40.0000 | negative loss ablation |
| Sim-only final pipeline | sim only | CHIME paper-test | 268 | 54.8507 | 39.7970 | 54.9991 | 30.0000 | 40.0000 | domain-gap test |
