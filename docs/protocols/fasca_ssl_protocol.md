# FASCA-SSL experiment protocol

## Hypothesis

Under the same modality-availability mask, aligning a corrupted student view
with a clean EMA-teacher view improves corruption robustness without requiring
the recovery of unobserved modalities.

## Controlled comparison

- Baseline: `embracenet_fasca_kd`
- Proposed variant: `embracenet_fasca_ssl_kd`
- Identical subject folds, seeds, preprocessing, XTinyHAR teacher, modality
  masks, FASCA corruption budget, clean-view loss, optimizer, model selection,
  and evaluation protocol.
- The only added training objective is clean-corrupt latent consistency with
  `lambda_ssl=0.10`.
- The clean EMA target and corrupted online view always receive the same
  modality-availability mask.
- The fixed SSL weight is not selected using test performance.

## Main-dataset run

- Five subject-disjoint folds
- Seeds: 51, 52, and 53
- 35 maximum epochs with patience 8
- Primary endpoints: 15-combination subject mean and joint-corruption subject
  mean
- Guardrails: full-modality clean performance and worst modality combination
- Paired inference is performed at subject level.

Because the main test folds informed earlier project decisions, this comparison
is treated as method-development evidence. A positive result must be confirmed
on HuGaDB or another untouched external cohort before it supports a
confirmatory generalization claim.

## Decision rule

Advance FASCA-SSL when it improves the prespecified corruption endpoint,
does not reduce clean full-modality macro-F1 by more than 0.5 percentage points,
and shows no material deterioration of the worst modality combination. Extend
the same fixed method to Masked Fusion only after this gate is met.
