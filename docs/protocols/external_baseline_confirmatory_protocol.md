# External robustness baselines: confirmatory protocol

Date locked: 2026-09-20

## Purpose

The fold-1/seed-51 structural screen showed that all three task adaptations
trained stably and that Centaur was a competitive within-modality-fault
baseline. This confirmatory extension estimates performance over the same five
participant-disjoint folds and optimization seeds 51--53 used by the primary
FASCA experiment. No architecture, loss weight, training budget, early-stopping
rule, or endpoint is changed after the screen.

## Methods

- Centaur-adaptation
- ADAPT-adaptation
- CIMSleepNet-adaptation
- EmbraceNet+FASCA, using the existing locked primary checkpoints

The external methods remain task adaptations rather than exact reproductions.
All methods use the same dataset, participant folds, fold-specific scaling,
trial aggregation, and class definition. External adaptations use the frozen
implementation and hyperparameters in `external_baseline_compare.py`.

## Endpoints

The two co-primary descriptive endpoints are:

1. **Incomplete-14 macro-F1:** the unweighted mean over all 14 non-complete
   modality-availability patterns.
2. **Fault-30 macro-F1:** the unweighted mean over six fixed fault scenarios
   and five affected-modality targets.

Supporting endpoints are complete-input macro-F1, the four equally weighted
availability-depth strata (one to four available modalities), the minimum over
15 availability patterns, and the all-modality channel- and structured-dropout
conditions. Results are summarized over 15 fold-seed runs. Paired inference,
when reported, resamples held-out participants from saved trial predictions.

## Interpretation rule

No single endpoint is used to declare universal superiority. A method may be
described as more balanced only when it performs competitively on both
co-primary endpoints without a material complete-input loss. Endpoint-specific
leaders are reported explicitly. The fold-1 ordering is not used to select a
method or FASCA variant.

