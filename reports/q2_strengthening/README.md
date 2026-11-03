# Q2/Q3 Strengthening Analyses

This directory contains reviewer-facing analyses generated from retained HSFNet-FRB runs.

Generated files:

- `paired_delta_tests.md`: paired bootstrap delta CIs and exact McNemar tests for final HSFNet-FRB versus available paired baselines.
- `paired_delta_tests_skipped.json`: comparisons skipped because paired predictions were unavailable or not on the same ordered test set.
- `c1_vote_bonus_sensitivity.md`: sensitivity of the final routing score to the C1 vote bonus.
- `strong_baselines.md`: original-paper, local reproduction, backbone, loss, augmentation, routing, and domain-gap baselines.
- `clean_ablation_sequence.md`: compact ablation table for the main technical story.
- `stratified_performance.md`: final-model performance stratified by true class, CHIME mSNR tertile, and non-positive intensity proxy tertile.
- `error_examples.csv`: misclassified CHIME paper-test examples with global CHIME index and metadata proxies.
- `fixed_mask_robustness.md`: retained fixed-mask robustness table copied into the Q2/Q3 package.

Important interpretation notes:

- C1 and C2 have only 10 samples each in the CHIME paper-test split, so minority-class recall should be described cautiously.
- McNemar tests only evaluate paired correctness changes on the same ordered test set.
- The non-positive intensity proxy is not a physical width or SNR measurement; it is a reproducible image-level proxy for blank/low-power regions in the retained CHIME array.
- No reliable burst-width metadata file was found in the cleaned project, so width-stratified claims should not be made from this package alone.

Run configuration:

- final eval dir: `artifacts/runs/binary_specialist_fusion/eval_predicted_subset`
- bonus sweep dir: `artifacts/runs/q2_strengthening/c1_bonus_sweep`
- bootstrap resamples: `5000`
- seed: `123`
