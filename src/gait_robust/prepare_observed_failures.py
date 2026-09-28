from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gait_robust.prepare_windows import MODALITIES, index_files
from gait_robust.prepare_windows_v2 import (
    read_eeg,
    read_emg,
    read_fp,
    read_imu,
    resample_center,
)


CHANNEL_COUNTS = {"eeg": 19, "emg": 12, "imu": 51, "fp": 16}
READERS = {
    "eeg": read_eeg,
    "emg": read_emg,
    "imu": read_imu,
    "fp": read_fp,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover trials excluded from the complete-case dataset because "
            "source files or required schemas were unavailable."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--complete-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=100.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--common-seconds", type=float, default=48.0)
    return parser.parse_args()


def trial_id(subject: int, speed: float) -> str:
    return f"S{subject}_{speed:g}"


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    maps = index_files(root)
    indexed_trials = set().union(*(set(paths) for paths in maps.values()))
    complete = np.load(args.complete_data, allow_pickle=False)
    complete_trial_ids = set(np.unique(complete["trial_id"]).astype(str))
    excluded_trials = sorted(
        key
        for key in indexed_trials
        if trial_id(*key) not in complete_trial_ids
    )
    windows_per_trial = int(round(args.common_seconds / args.window_seconds))
    samples_per_window = int(round(args.target_hz * args.window_seconds))

    output: dict[str, list[np.ndarray]] = {name: [] for name in MODALITIES}
    availability_rows: list[np.ndarray] = []
    subjects: list[int] = []
    speeds: list[float] = []
    trial_ids: list[str] = []
    window_starts: list[float] = []
    trial_manifest: list[dict[str, object]] = []

    for subject, speed in excluded_trials:
        available: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        trial_values: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            if (subject, speed) not in maps[modality]:
                available[modality] = False
                reasons[modality] = "source_file_not_indexed"
                continue
            try:
                times, values, _, rate = READERS[modality](
                    maps[modality][(subject, speed)]
                )
                resampled = resample_center(
                    times,
                    values,
                    source_rate=rate,
                    target_rate=args.target_hz,
                    common_seconds=args.common_seconds,
                )
                trial_values[modality] = resampled.reshape(
                    windows_per_trial,
                    samples_per_window,
                    CHANNEL_COUNTS[modality],
                ).transpose(0, 2, 1)
                available[modality] = True
                reasons[modality] = "available"
            except Exception as exc:
                available[modality] = False
                reasons[modality] = f"{type(exc).__name__}: {exc}"

        mask = np.asarray(
            [available[name] for name in MODALITIES], dtype=bool
        )
        if not mask.any():
            continue
        for modality in MODALITIES:
            if available[modality]:
                values = trial_values[modality].astype(np.float32)
            else:
                values = np.zeros(
                    (
                        windows_per_trial,
                        CHANNEL_COUNTS[modality],
                        samples_per_window,
                    ),
                    dtype=np.float32,
                )
            output[modality].append(values)
        availability_rows.append(
            np.repeat(mask[None, :], windows_per_trial, axis=0)
        )
        subjects.extend([subject] * windows_per_trial)
        speeds.extend([speed] * windows_per_trial)
        current_trial = trial_id(subject, speed)
        trial_ids.extend([current_trial] * windows_per_trial)
        window_starts.extend(
            np.arange(windows_per_trial, dtype=float) * args.window_seconds
        )
        trial_manifest.append(
            {
                "trial_id": current_trial,
                "subject": subject,
                "speed": speed,
                "available_modalities": [
                    name
                    for name, is_available in available.items()
                    if is_available
                ],
                "unavailable_modalities": [
                    name
                    for name, is_available in available.items()
                    if not is_available
                ],
                "availability_reasons": reasons,
            }
        )

    arrays = {
        modality: np.concatenate(output[modality], axis=0)
        for modality in MODALITIES
    }
    arrays.update(
        {
            "availability_mask": np.concatenate(
                availability_rows, axis=0
            ),
            "subject": np.asarray(subjects, dtype=np.int16),
            "speed": np.asarray(speeds, dtype=np.float32),
            "trial_id": np.asarray(trial_ids),
            "window_start_seconds": np.asarray(
                window_starts, dtype=np.float32
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "schema_version": 1,
        "dataset_root": str(root),
        "complete_data": str(args.complete_data.resolve()),
        "output": str(args.output.resolve()),
        "definition": (
            "Dataset-observed acquisition/export failures: trials omitted "
            "from the complete-case file because a source file was absent, "
            "empty, or incompatible with the required channel schema. "
            "Available streams are otherwise processed identically to v2."
        ),
        "excluded_trials_discovered": len(excluded_trials),
        "trials_recovered": len(trial_manifest),
        "windows_written": len(subjects),
        "modalities": list(MODALITIES),
        "trials": trial_manifest,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
