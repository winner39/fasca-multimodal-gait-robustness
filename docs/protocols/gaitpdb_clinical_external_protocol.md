# Locked protocol: study-held-out clinical external validation on Gait in Parkinson's Disease

Protocol version: 1.0  
Locked: 2026-07-06, before full waveform download, model training, or inspection of any model result  
Role in paper: independent retrospective clinical-cohort validation and deployment-oriented sensor-fault stress test

## 1. Rationale

The existing primary and HuGaDB experiments contain healthy participants, and the primary test folds were observed during method development. This experiment therefore uses a previously unused public clinical dataset and study-held-out evaluation. It does not repair the historical contamination of the primary benchmark. Instead, it provides an independent evidence stream with patient data and a stricter separation rule.

The dataset is PhysioNet's *Gait in Parkinson's Disease* v1.0.0 (DOI `10.13026/C24H3N`). It contains usual-walking plantar-force recordings from 93 people with idiopathic Parkinson's disease (PD) and 73 healthy controls collected in three studies at a movement-disorders research unit.

## 2. Information inspected before lock

Only the following were inspected:

- the public PhysioNet description;
- `format.txt`;
- `demographics.txt`;
- one control waveform, `GaCo01_01.txt`, to verify parsing and units.

No patient waveform, model output, split result, or fault result was inspected.

SHA-256:

```text
format.txt
8A91A8C4181E790474F247B59C8189D6AF915FD5AF45178A8B355C69A884BFF6

demographics.txt
EBE5E3C5C3055023B876225D3EB067A9351693E28CE47FA678872EAD8284F643

GaCo01_01.txt
81BCEDC0F72C1C6804D7830627431A6C1B76E9DDB42DE0CF1AE2F418848C13A4
```

## 3. Cohort and labels

Use all subjects in `demographics.txt` for whom a usual-walking `_01` waveform exists and passes the prespecified integrity checks.

Expected metadata counts before waveform quality control:

| Study | PD | Control | Total |
|---|---:|---:|---:|
| Ga | 29 | 18 | 47 |
| Ju | 29 | 26 | 55 |
| Si | 35 | 29 | 64 |
| Total | 93 | 73 | 166 |

Binary label:

- PD: `Group = 1`;
- control: `Group = 2`.

The task is retrospective patient-control discrimination. It is not described as clinical diagnosis, screening efficacy, or a medical-device claim.

## 4. Waveform integrity checks

Each `_01` file must:

1. contain exactly 19 numeric columns;
2. have finite time and sensor values;
3. have strictly increasing time with median interval within 1% of 0.01 s;
4. contain at least 20 s after trimming;
5. have no duplicate timestamps;
6. have non-negative VGRF values apart from numerical tolerance `-1e-6`;
7. agree with the supplied left/right total columns within a documented tolerance.

Failures are reported before training. No subject is removed based on model behavior.

## 5. Input representation

- Sampling rate: 100 Hz.
- Use columns 2--9 as the left-foot modality and columns 10--17 as the right-foot modality.
- Do not use the two supplied total-force columns as model inputs.
- Trim the first and last 5 s of each recording.
- Divide the 16 sensor channels by the median total force across the retained recording. This reduces body-mass scale differences without using class labels.
- Segment into non-overlapping 5 s windows (500 samples).
- Aggregate window probabilities by arithmetic mean to obtain one subject prediction.
- Training sampling is subject-balanced so longer recordings do not dominate.

## 6. Study-held-out evaluation

Use leave-one-study-out evaluation:

1. test Ga; train/validate on Ju+Si;
2. test Ju; train/validate on Ga+Si;
3. test Si; train/validate on Ga+Ju.

The held-out study is never used for normalization, early stopping, threshold selection, or model choice.

Within the two training studies, reserve 20% of subjects for validation using stratification by study and class with split seed 20260706. Optimization seeds are 101, 102, and 103. Validation identities are fixed before training and reused across methods.

## 7. Backbone and fair comparison

Use one two-modality masked-fusion network for both methods:

- shared foot encoder;
- Conv1d blocks with 32, 64, and 64 channels;
- kernel sizes 7, 5, and 3;
- batch normalization, ReLU, and temporal pooling;
- 64-dimensional foot embedding;
- masked mean across available feet;
- 64-to-32-to-2 classifier with dropout 0.20.

Both methods use balanced sampling of the three non-empty availability masks:

- left only;
- right only;
- both feet.

Comparison:

1. masked-fusion baseline: availability masking only;
2. masked-fusion + FASCA-pressure: identical training plus fault-aware corruption.

This comparison intentionally excludes distillation and curriculum scheduling so the incremental effect is attributable to the fault-aware training distribution.

## 8. FASCA-pressure training corruptions

For each training window, apply a corruption with probability 0.35 to an available foot modality. Corruption type is sampled uniformly:

1. complete sensor dropout;
2. near-dead sensor attenuation;
3. contiguous contact-loss segment;
4. sensor saturation/clipping;
5. baseline drift;
6. packet-loss segment across the foot;
7. foot-level gain miscalibration.

Severity is sampled uniformly from 0.20 to 0.80. Sensor and segment positions are sampled from the training RNG. The baseline receives no within-modality corruption.

## 9. Optimization

- Maximum epochs: 30.
- Early-stopping patience: 6.
- Batch size: 64 windows.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Loss: class-weighted cross-entropy with label smoothing 0.05.
- Gradient clipping: 5.0.
- Model selection: validation subject-level macro-F1.
- No test-based hyperparameter changes are permitted.

## 10. Locked fault suite

Faults are generated deterministically for every held-out test window using a seed derived from study, subject, window index, and fault name.

1. `sensor_dead_bilateral`: one sensor per foot set to zero for the full window.
2. `near_dead_bilateral`: one sensor per foot multiplied by 0.05.
3. `contact_loss_bilateral`: one sensor per foot set to zero for a contiguous 1 s segment.
4. `saturation_bilateral`: one sensor per foot clipped at 60% of its window maximum.
5. `drift_bilateral`: a linear offset from 0 to 0.15 normalized-force units added to one sensor per foot.
6. `packet_loss_bilateral`: all sensors in each foot set to zero for a contiguous 0.5 s segment.
7. `gain_miscalibration`: left foot multiplied by 0.70 and right foot by 1.30.

Also evaluate:

- clean both feet;
- left only;
- right only.

No severity or fault definition is changed after results are observed.

## 11. Endpoints

Primary endpoint:

- subject-level macro-F1 averaged over the seven locked bilateral fault conditions.

Secondary effectiveness endpoints:

- clean subject-level macro-F1;
- worst fault-condition macro-F1;
- balanced accuracy;
- AUROC;
- sensitivity and specificity;
- Brier score;
- ten-bin ECE.

System/deployment endpoints:

- fault-induced macro-F1 decline from clean;
- selective accuracy at 80% subject coverage using maximum predicted probability;
- area under the risk-coverage curve;
- exact-zero FTER detector trigger rate for clean, exact-dead, near-dead, contact-loss, saturation, drift, packet-loss, and gain conditions.

Subgroup reporting:

- held-out study;
- sex;
- PD Hoehn--Yahr stage `<=2` versus `>2`, where available.

Subgroup results are descriptive unless sample sizes support paired inference.

## 12. Statistical analysis

For each subject and method, average probabilities across the three optimization seeds. Each subject appears in exactly one held-out study.

- 20,000 paired subject bootstrap resamples.
- Bootstrap seed: 20260706.
- Report point difference and percentile 95% CI.
- Paired Wilcoxon signed-rank tests for subject-level additive endpoints where meaningful.
- Holm correction across the prespecified primary/secondary comparison family.
- The primary endpoint is interpreted first; secondary endpoints are supportive.

## 13. Claim rules

Allowed if supported:

- independent study-held-out evidence on a retrospective PD/control cohort;
- improved robustness to locked insole-sensor faults;
- preserved or changed clean performance with exact effect estimates;
- explicit detector sensitivity and false-trigger boundaries.

Not allowed:

- prospective clinical validation;
- diagnostic utility or patient benefit;
- repair of contamination in the original primary benchmark;
- clinical deployment readiness;
- universal sensor-fault robustness.

## 14. Stop and reporting rules

All completed runs are retained. Negative or null results are reported. If data quality invalidates a study or endpoint, the reason is documented and the protocol is not silently changed.
