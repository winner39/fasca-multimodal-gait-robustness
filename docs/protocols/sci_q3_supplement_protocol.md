# SCI Q3 Supplementary Experiment Protocol

Locked on: 2026-07-02

## Status and interpretation

This protocol was fixed before inspecting the results of the new factorial
runs. The original main-dataset test folds had already been inspected during
earlier method development, so these experiments are post-hoc mechanistic
validation, not an independent confirmatory study. No hyperparameter will be
tuned from the new test results.

## Fixed data and training protocol

- Main dataset: `data/processed/windows_v2.npz`
- Subject-independent folds: 5
- Partition seed: 20260901
- Optimization seeds: 51, 52, 53
- Fold-local robust normalization
- Student epochs: at most 35
- Patience: 8
- Batch size: 128
- Learning rate: 0.001
- Weight decay: 0.0001
- Balanced sampling over all 15 non-empty modality combinations
- XTinyHAR full-input teacher from the matching fold and seed
- KD objective: 0.8 CE + 0.2 KL, temperature 3
- Model selection: 0.9 clean 15-combination validation score plus
  0.1 mixed-corruption validation score
- FASCA probability: 0.35
- Severity range: 0.15 to 0.75
- Clean-view auxiliary loss: 0.50
- No desynchronization augmentation in the locked `drop_gain` profile

## Experiment A: architecture-by-training fairness

All variants receive the same masks, teacher, FASCA corruption budget,
curriculum, optimizer, and model-selection rule.

1. Masked gated fusion + FASCA
2. EmbraceNet adaptation + FASCA
3. RAPID with uniform source averaging + FASCA
4. Full RAPID + FASCA (existing reference)

Questions:

- Does the strongest missing-modality baseline catch up when trained with the
  same fault-aware curriculum?
- Does proxy completion add value beyond masked fusion?
- Does learned source reliability add value beyond uniform source averaging?

## Experiment B: FASCA design controls

1. Full RAPID + FASCA (existing reference)
2. Full RAPID + the same structured corruption without curriculum warm-up
3. Full RAPID + matched-probability generic independent channel
   dropout/global gain augmentation

Questions:

- Is curriculum scheduling necessary?
- Are sensor-structure priors useful beyond merely exposing the model to
  an equal corruption probability and severity range?

## Endpoints

Primary clean/missing-modality endpoint:

- Across-fold pooled trial-level macro-F1 averaged over all 15 combinations.

Tail endpoint:

- Worst modality-combination pooled trial-level macro-F1.

Mechanistic secondary endpoints:

- Full-input macro-F1
- Mean of 14 incomplete combinations
- Mean of single-, two-, and three-modality strata
- Subject-level paired differences for the 15-combination mean

Fault-robustness primary endpoint:

- Subject-level macro-F1 averaged over the five joint all-modality corruption
  conditions.

Fault-robustness statistics:

- Average optimization seeds within each subject
- Paired subject bootstrap with 20,000 resamples
- Wilcoxon signed-rank test
- Paired rank-biserial effect size
- Holm adjustment across the reported endpoint family

## Claim rules

- A higher point estimate without a confidence interval excluding zero is
  described as an observed trend, not a significant improvement.
- Simulated sensor-node or channel removal is not described as a real hardware
  failure.
- If EmbraceNet + FASCA matches full RAPID + FASCA, the paper is framed around
  FASCA rather than architectural superiority.
- If generic augmentation matches FASCA, claims about sensor-structure priors
  are removed or weakened.
- Results from smoke-test directories are never reported.
