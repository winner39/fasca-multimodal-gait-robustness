# Reproducibility guide

## 1. Primary preprocessing

```powershell
python -m gait_robust.prepare_windows_v2 `
  --dataset-root data/raw/multimodal-gait-dataset-1.0.0/Dataset `
  --output data/processed/windows_v2.npz `
  --target-hz 100 --window-seconds 2 --common-seconds 48
```

The pipeline selects the centered 48 s common interval, resamples every stream
to 100 Hz, and creates 24 non-overlapping two-second windows per trial. EEG is
re-referenced to linked ears and filtered from 1 to 45 Hz before resampling.
No whole-dataset or per-trial standardization is performed at this stage.

## 2. Fold-specific scaling

For every outer fold, channel medians and interquartile ranges are fitted using
training participants only. Values are transformed as `(x - median) / IQR` and
clipped to `[-10, 10]`. The exact fitted statistics are preserved in
`scalers/primary/`.

## 3. Primary training

The locked settings are in `configs/primary_fasca.yaml`. The exact command
sequence is preserved in `scripts/run_sci_q3_supplements.ps1`. The core run is:

```powershell
python -m gait_robust.rapid_distill `
  --data data/processed/windows_v2.npz `
  --output-dir artifacts/fasca_primary `
  --teacher-root artifacts/full_input_teachers `
  --methods embracenet_fasca_kd `
  --folds 5 --seeds 51 52 53 --partition-seed 20260901 `
  --epochs-student 35 --patience 8 --batch-size 128 `
  --learning-rate 0.001 --weight-decay 0.0001 `
  --temperature 3 --alpha 0.2 `
  --augmentation-profile drop_gain --augmentation-probability 0.35 `
  --severity-min 0.15 --severity-max 0.75 `
  --augmentation-warmup-epochs 5 --desync-probability 0 `
  --lambda-clean 0.50 --corruption-selection-weight 0.10
```

## 4. External baselines

Run `gait_robust.external_baseline_compare` with
`configs/external_baselines.yaml`. All methods use the same processed windows,
outer folds, fold-specific scaling, labels, availability masks, optimization
seeds, trial aggregation, and checkpoint-selection rule.

## 5. Statistical summaries

- Availability endpoints: pool held-out trials across the five outer folds,
  compute one score per optimization seed, then report mean and sample SD over
  seeds.
- Paired effects: calculate participant-level metrics, average optimization
  seeds within participant, and resample participants 20,000 times.
- Calibration: average window probabilities within trial before Brier score,
  negative log likelihood, and 10-bin expected calibration error.
- The saved subject-level tables allow the confidence intervals to be
  regenerated without raw signals or checkpoints.

## 6. Outputs not stored in Git

Raw signals, processed arrays, and neural-network checkpoints are omitted to
respect source-dataset distribution terms and repository size. Every reported
aggregate and the participant-level numerical inputs needed for the reported
bootstrap analyses are included under `results/`.

