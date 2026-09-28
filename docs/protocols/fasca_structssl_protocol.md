# FASCA-StructSSL screening protocol

## Motivation

Generic clean-corrupt consistency did not improve the prespecified joint-fault
mean, but it showed an exploratory positive signal under joint structured
channel dropout. This screen tests whether matching the self-supervised view to
that fault family strengthens the signal without replacing the original FASCA
supervised augmentation.

## Method

- The supervised student branch retains the original FASCA drop/gain profile.
- A separate self-supervised view applies only electrode, muscle, grouped IMU
  sensor, and force-plate dropout.
- The clean EMA encoder and corrupted online encoder use the same
  modality-availability mask.
- Consistency is applied to available per-modality encoder tokens.
- `lambda_ssl=0.10`, structured corruption probability `0.70`, severity
  `[0.35, 0.75]`.

## Screening design

- Backbone: EmbraceNet
- Five fixed subject-disjoint folds, covering all 55 retained subjects
- Student seed: 51
- Baselines at the same seed: EmbraceNet+FASCA and generic
  EmbraceNet+FASCA-SSL
- Primary screening endpoint: joint structured-dropout subject macro-F1
- Guardrails: clean full-modality performance, 15-combination mean,
  worst combination, and standard joint-fault mean

This is a development screen rather than confirmatory evidence. The remaining
two seeds are run only if the targeted method improves structured dropout
without a material guardrail failure.
