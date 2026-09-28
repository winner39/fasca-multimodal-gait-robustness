# Reviewer materials audit

| Requested material | Repository location | Status |
|---|---|---|
| All 15 modality combinations | `results/primary_availability/availability_pattern_summary_with_ci.csv` and seed-level source table | Complete |
| FASCA component ablation | Distribution/curriculum and SSL analyses are preserved in the protocols and scripts | Partial: a strict one-component-at-a-time ablation remains a separate experiment |
| Channel/sensor faults and severity | `src/gait_robust/augmentations.py`, `configs/primary_fasca.yaml`, Online Resource 1 | Complete |
| Hyperparameters and early stopping | `configs/` and `scripts/` | Complete |
| Preprocessing/windows/normalization | `REPRODUCIBILITY.md`, `prepare_windows_v2.py`, `data.py` | Complete |
| Five-fold participant IDs and seeds | `splits/primary/` and `configs/primary_fasca.yaml` | Complete |
| ECE, Brier, NLL, reliability bins | `results/calibration/` | Complete |
| Cross-dataset, plantar-force, routing, sensor faults | Corresponding `results/` subdirectories | Complete |
| Centaur, ADAPT, CIMSleepNet implementations/settings | `external_baseline_models.py`, `external_baseline_compare.py`, and `configs/external_baselines.yaml` | Complete |
| Participant-level numerical inputs for CIs | `results/primary_availability/` and external/plantar subdirectories | Complete for reported CIs |

The remaining component-ablation gap is explicitly identified rather than
being represented by a backbone comparison or a distribution-matching study.

