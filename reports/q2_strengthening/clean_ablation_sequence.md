| model | accuracy | macro-F1 | weighted-F1 | C1 recall | C2 recall | note | source |
|---|---|---|---|---|---|---|---|
| ResNet small + mask/noise | 56.7164 | 46.8353 | 53.6257 | 30.0000 | 50.0000 | initial stronger pipeline | artifacts/runs/resnet_mask_noise_gpu/eval_paper_compare |
| ResNet18 + mask/noise | 60.0746 | 43.8584 | 59.9114 | 0.0000 | 30.0000 | backbone comparison | artifacts/runs/resnet18_aug_gpu0/eval_paper_compare |
| ResNet50 + mask/noise | 61.9403 | 46.7186 | 62.6886 | 0.0000 | 50.0000 | backbone comparison | artifacts/runs/resnet50_aug_gpu0/eval_paper_compare |
| ResNet50 C1 repair base | 62.3134 | 48.7033 | 63.1631 | 10.0000 | 50.0000 | main base model | artifacts/runs/resnet50_c1_repair_gpu0/eval_paper_compare |
| Hierarchical ResNet50 + B/C1/C2 specialist | 63.0597 | 51.3475 | 63.6533 | 20.0000 | 50.0000 | tri-specialist fusion | artifacts/runs/hierarchical_resnet50_b_c1_c2/eval_predicted_subset |
| Final HSFNet-FRB binary specialist fusion | 63.4328 | 53.2774 | 64.0134 | 30.0000 | 50.0000 | current selected model | artifacts/runs/binary_specialist_fusion/eval_predicted_subset |
| Uncertain-subset routing | 63.8060 | 52.9646 | 64.4869 | 30.0000 | 50.0000 | routing ablation | artifacts/runs/binary_specialist_fusion/eval_uncertain_subset_m018 |
| Noisebank v2 + binary specialists | 61.9403 | 52.5806 | 62.3797 | 30.0000 | 50.0000 | negative augmentation ablation | artifacts/runs/noisebank_v2_binary_specialist_fusion/eval_predicted_subset |
| Class-balanced focal + binary specialists | 55.2239 | 44.3028 | 54.7174 | 30.0000 | 40.0000 | negative loss ablation | artifacts/runs/cb_focal_binary_specialist_fusion/eval_predicted_subset |
| Sim-only final pipeline | 54.8507 | 39.7970 | 54.9991 | 30.0000 | 40.0000 | domain-gap test | artifacts/runs/sim_only_mainline/fusion_eval_seed_42_gpu0/eval_predicted_subset |
