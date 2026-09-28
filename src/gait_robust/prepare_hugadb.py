from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


ACTIVITIES = {
    1: "walking",
    2: "running",
    3: "stairs_up",
    4: "stairs_down",
}
FILE_PATTERN = re.compile(
    r"^HuGaDB_v1_.+_(?P<subject>\d+)_(?P<repeat>\d+)\.txt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare subject-independent HuGaDB gait windows."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-samples", type=int, default=128)
    parser.add_argument("--stride-samples", type=int, default=64)
    return parser.parse_args()


def contiguous_runs(labels: np.ndarray) -> list[tuple[int, int]]:
    if len(labels) == 0:
        return []
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(labels)]
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def prepare(
    input_dir: Path,
    output: Path,
    window_samples: int,
    stride_samples: int,
) -> dict[str, object]:
    if window_samples < 16:
        raise ValueError("window_samples must be at least 16")
    if not 1 <= stride_samples <= window_samples:
        raise ValueError("stride_samples must be in [1, window_samples]")

    imu_windows: list[np.ndarray] = []
    emg_windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    trials: list[int] = []
    source_files: list[str] = []
    trial_lookup: dict[str, int] = {}
    skipped_short = Counter()

    paths = sorted(input_dir.glob("HuGaDB_v1_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No HuGaDB text files found in {input_dir}")

    for path in paths:
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        subject = int(match.group("subject"))
        values = np.genfromtxt(
            path,
            delimiter="\t",
            skip_header=4,
            dtype=np.float32,
        )
        if values.ndim != 2 or values.shape[1] != 39:
            raise ValueError(
                f"{path.name}: expected 39 columns, got {values.shape}"
            )
        activity_ids = values[:, 38].astype(np.int16)
        for run_index, (start, stop) in enumerate(
            contiguous_runs(activity_ids)
        ):
            activity_id = int(activity_ids[start])
            if activity_id not in ACTIVITIES:
                continue
            length = stop - start
            if length < window_samples:
                skipped_short[ACTIVITIES[activity_id]] += 1
                continue
            trial_key = f"{path.stem}:run{run_index:03d}"
            trial_id = trial_lookup.setdefault(trial_key, len(trial_lookup))
            for window_start in range(
                start, stop - window_samples + 1, stride_samples
            ):
                window = values[
                    window_start : window_start + window_samples, :38
                ].T
                # HuGaDB stores six physical IMU nodes consecutively, each
                # with accelerometer xyz followed by gyroscope xyz.
                imu_windows.append(window[:36].astype(np.float32))
                emg_windows.append(window[36:38].astype(np.float32))
                labels.append(activity_id - 1)
                subjects.append(subject)
                trials.append(trial_id)
                source_files.append(path.name)

    if not labels:
        raise RuntimeError("No gait windows were generated")
    arrays = {
        "imu": np.stack(imu_windows),
        "emg": np.stack(emg_windows),
        "label": np.asarray(labels, dtype=np.int64),
        "subject": np.asarray(subjects, dtype=np.int16),
        "trial_id": np.asarray(trials, dtype=np.int32),
        "source_file": np.asarray(source_files),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    class_counts = Counter(arrays["label"].tolist())
    metadata = {
        "source": "HuGaDB version 1",
        "source_directory": str(input_dir.resolve()),
        "output": str(output.resolve()),
        "activities": {
            str(index - 1): name for index, name in ACTIVITIES.items()
        },
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "nominal_sampling_hz": 56.35,
        "subjects": sorted(np.unique(arrays["subject"]).astype(int).tolist()),
        "subject_count": int(len(np.unique(arrays["subject"]))),
        "trial_count": int(len(np.unique(arrays["trial_id"]))),
        "window_count": int(len(arrays["label"])),
        "class_windows": {
            ACTIVITIES[index + 1]: int(class_counts[index])
            for index in range(len(ACTIVITIES))
        },
        "skipped_short_runs": dict(skipped_short),
        "modalities": {
            "imu": {
                "channels": 36,
                "layout": "6 nodes x (accelerometer xyz + gyroscope xyz)",
            },
            "emg": {
                "channels": 2,
                "layout": "right and left vastus lateralis",
            },
        },
        "boundary_policy": (
            "Windows are contained within contiguous single-label runs."
        ),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = prepare(
        args.input_dir,
        args.output,
        args.window_samples,
        args.stride_samples,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
