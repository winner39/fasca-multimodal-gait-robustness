# Dataset-observed acquisition-failure protocol

## Purpose

This protocol supplements synthetic corruption tests with failures already
present in the source dataset. It does not relabel normal signals as faults.

## Failure definition

A trial is included when it is absent from `windows_v2.npz` because at least
one source stream is:

1. not indexed by the released dataset layout;
2. empty; or
3. incompatible with the required acquisition schema.

Available streams are filtered, physically resampled, windowed, and scaled
using the same fold-only pipeline as complete trials. Unavailable streams are
zero-filled only for tensor construction and excluded by the availability
mask.

## Evaluation

- Twenty excluded trials (480 windows, ten subjects) are recovered.
- For subjects represented in the complete-case cohort, only their
  subject-disjoint test-fold model is used.
- Subjects with no complete trial are absent from every training fold. Their
  predictions are averaged across the five fixed fold models within each
  optimization seed and are reported as a separate fully-unseen subgroup.
- EmbraceNet and EmbraceNet+FASCA use identical preprocessing, splits, and
  seeds 51/52/53.

## Dataset-derived schema replay

Trial `S32_1` lacks required EEG fields `A1`, `Fp2`, `F4`, `Cz`, `T5`, `P4`,
`O1`, and `O2`. `A1` is a rereferencing channel and is not a direct model
input. The seven corresponding model-input channels define a frozen topology
replayed on complete test trials at 0%, 1%, 5%, and 10% residual amplitude.
A conservative condition rejects the entire EEG modality because the
rereferencing field is unavailable.

## Interpretation boundary

These are real acquisition/export failures and a semi-real replay of their
observed topology. They are not prospectively recorded hardware faults. The
failure audit and the refined FTER rule were developed after examining the
released source files, so they are exploratory rather than an independent
confirmatory validation.
