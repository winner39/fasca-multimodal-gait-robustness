# FASCA multimodal gait robustness

Anonymous reproducibility package for the manuscript **Robust Multimodal Gait
Classification under Arbitrary Missing Modalities with Fault-Aware Training**.

The repository contains preprocessing, training, evaluation, baseline,
aggregation, and statistical-analysis code; locked configurations; the five
participant-level splits; fold-specific scaling parameters; and the numerical
tables used for the manuscript and Online Resource 1. Raw public datasets and
model checkpoints are not redistributed.

## Data

- Primary benchmark: PhysioNet multimodal gait dataset, version 1.0.0,
  <https://doi.org/10.13026/r0ea-7161>.
- Cross-layout benchmark: HuGaDB, obtained from its source release.
- Plantar-force benchmark: PhysioNet Gait in Parkinson's Disease,
  <https://doi.org/10.13026/C24H3N>.

Place downloaded data below `data/raw/`; processed arrays are written below
`data/processed/`. Neither directory is tracked by Git.

## Environment

Python 3.10 or newer is required. The reported runs used Python 3.12 and
PyTorch 2.6. Install an appropriate PyTorch build for the host first, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproduction map

| Manuscript evidence | Entry point | Preserved results |
|---|---|---|
| Primary preprocessing | `python -m gait_robust.prepare_windows_v2` | `results/primary_availability/` |
| FASCA training | `python -m gait_robust.rapid_distill` | `results/primary_availability/` |
| Centaur, ADAPT, CIMSleepNet | `python -m gait_robust.external_baseline_compare` | `results/external_baselines/` |
| Calibration and reliability | `python -m gait_robust.analyze_main_calibration` | `results/calibration/` |
| Fixed sensor faults | `python -m gait_robust.evaluate_corruptions` | `results/sensor_faults/` |
| HuGaDB cross-layout evaluation | `python -m gait_robust.hugadb_external` | `results/cross_dataset/` |
| Plantar-force evaluation | `python -m gait_robust.run_gaitpdb_clinical` | `results/plantar_force/` |
| Deterministic routing | `python -m gait_robust.evaluate_observed_failures` | `results/routing/` |

The PowerShell files in `scripts/` preserve the complete commands used for the
reported experiments. Machine-specific paths have been removed. Locked values
are also summarized in `configs/`.

## Splits and leakage controls

`splits/primary/fold_1.json` through `fold_5.json` contain the exact training,
validation, and test participant identifiers. Each participant belongs to one
outer test fold only. Scaling is fitted on outer-fold training participants;
the corresponding channel medians and interquartile ranges are in
`scalers/primary/`. Test participants are not used for scaling, early stopping,
checkpoint selection, or hyperparameter selection.

## Result regeneration

The complete 15-pattern summary and participant-bootstrap confidence intervals
can be recreated without checkpoints:

```powershell
python scripts/summarize_availability_patterns.py `
  --combination-results results/primary_availability/factorial_combination_metrics.csv `
  --subject-results results/primary_availability/factorial_subject_combination_metrics.csv `
  --output results/primary_availability/availability_pattern_summary_with_ci.csv
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the end-to-end sequence and
[MATERIALS_AUDIT.md](MATERIALS_AUDIT.md) for the reviewer-checklist mapping.

## Scope

The code and preserved numerical results support research reproduction only.
They are not intended for diagnosis, treatment selection, safety-critical
control, or direct clinical use.

