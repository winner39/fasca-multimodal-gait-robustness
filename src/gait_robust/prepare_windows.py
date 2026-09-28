from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


TRIAL_RE = re.compile(r"(?:^|_)S(?P<subject>\d+)_(?P<speed>0\.5|0\.75|1(?:\.0)?)(?:_|\.|$)")
MODALITIES = ("eeg", "emg", "imu", "fp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-hz", type=int, default=100)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--windows-per-trial", type=int, default=24)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Debug-only cap; omit for the complete dataset.",
    )
    return parser.parse_args()


def trial_key(path: Path) -> tuple[int, float] | None:
    match = TRIAL_RE.search(path.name)
    if match is None:
        match = TRIAL_RE.search(path.parent.name)
    if match is None:
        return None
    return int(match.group("subject")), float(match.group("speed"))


def index_files(root: Path) -> dict[str, dict[tuple[int, float], Path]]:
    eeg_csv = root / "EEG_Data" / "EEG.csv"
    maps: dict[str, dict[tuple[int, float], Path]] = {
        "eeg": {},
        "emg": {},
        "imu": {},
        "fp": {},
    }
    for path in eeg_csv.glob("*.csv"):
        key = trial_key(path)
        if key is not None:
            maps["eeg"][key] = path
    for path in (root / "EMG_Data").glob("*.csv"):
        key = trial_key(path)
        if key is not None:
            maps["emg"][key] = path
    for path in (root / "FP_Data").glob("*.csv"):
        key = trial_key(path)
        if key is not None:
            maps["fp"][key] = path

    imu_root = root / "IMU_Data"
    extracted_roots = [p for p in (imu_root, imu_root / "IMU") if p.exists()]
    for candidate_root in extracted_roots:
        for path in candidate_root.rglob("IMU_S*"):
            if not path.is_dir():
                continue
            key = trial_key(path)
            if key is not None:
                maps["imu"][key] = path
    return maps


def numeric_frame(path: Path, skiprows: int = 0) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, skiprows=skiprows, low_memory=False)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if numeric.empty:
        raise ValueError(f"No numeric samples found in {path}")
    return numeric.to_numpy(dtype=np.float32), [str(c) for c in numeric.columns]


def strip_time_column(values: np.ndarray, columns: list[str]) -> np.ndarray:
    if values.shape[1] <= 1:
        return values
    first = columns[0].lower()
    if "time" in first or first in {"t", "timestamp", "frame"}:
        return values[:, 1:]
    return values


def read_eeg(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, skiprows=15, low_memory=False)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
    required = (
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T3",
        "C3",
        "Cz",
        "C4",
        "T4",
        "T5",
        "P3",
        "P4",
        "T6",
        "O1",
        "O2",
        "A1",
        "A2",
    )
    missing = [channel for channel in required if channel not in numeric.columns]
    if missing:
        raise ValueError(f"Missing EEG columns in {path}: {missing}")
    linked_ears = (
        numeric["A1"].to_numpy(dtype=np.float32)
        + numeric["A2"].to_numpy(dtype=np.float32)
    ) / 2.0
    ordered = (
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T3",
        "C3",
        "Cz",
        "C4",
        "T4",
        "T5",
        "P3",
        "Pz",
        "P4",
        "T6",
        "O1",
        "O2",
    )
    rereferenced = []
    for channel in ordered:
        # Pz is the hardware reference and is therefore reconstructed as
        # zero minus the linked-ear reference, following the supplied MATLAB
        # script. All recorded scalp channels are referenced identically.
        if channel == "Pz":
            rereferenced.append(-linked_ears)
        else:
            rereferenced.append(
                numeric[channel].to_numpy(dtype=np.float32) - linked_ears
            )
    return np.stack(rereferenced, axis=1)


def read_emg(path: Path) -> np.ndarray:
    values, columns = numeric_frame(path)
    if values.shape[1] < 13:
        raise ValueError(f"Expected 12 EMG channels in {path}, got {values.shape[1]}")
    # The first column is named "X [s]" rather than "Time".
    return values[:, 1:13]


def read_fp(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, low_memory=False)
    signal_suffixes = ("Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy")
    selected = [
        column
        for column in frame.columns
        if any(str(column).endswith(suffix) for suffix in signal_suffixes)
    ]
    if len(selected) != 16:
        raise ValueError(
            f"Expected 16 bilateral force/moment/COP channels in {path}, "
            f"found {len(selected)}"
        )
    return (
        frame[selected]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )


def choose_imu_file(folder: Path) -> Path:
    patterns = (
        "*Sensor Free Acceleration*.csv",
        "*SensorFreeAcceleration*.csv",
        "*Sensor_Free_Acceleration*.csv",
        "*Segment Acceleration*.csv",
        "*SegmentAcceleration*.csv",
        "*Segment_Acceleration*.csv",
    )
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No IMU CSV files found under {folder}")
    raise FileNotFoundError(
        f"No acceleration file found under {folder}; available examples: "
        + ", ".join(p.name for p in csv_files[:5])
    )


def read_imu(folder: Path) -> np.ndarray:
    path = choose_imu_file(folder)
    values, columns = numeric_frame(path)
    values = strip_time_column(values, columns)
    # Retain all exported segment free-acceleration axes. The first Frame
    # column is removed by strip_time_column.
    return values


def fill_and_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    for channel in range(x.shape[1]):
        series = x[:, channel]
        finite = np.isfinite(series)
        if not finite.any():
            x[:, channel] = 0
            continue
        indices = np.arange(len(series))
        series[~finite] = np.interp(indices[~finite], indices[finite], series[finite])
        median = np.median(series)
        q25, q75 = np.percentile(series, [25, 75])
        scale = max(float(q75 - q25), 1e-6)
        x[:, channel] = np.clip((series - median) / scale, -10.0, 10.0)
    return x


def normalized_windows(
    x: np.ndarray,
    samples_per_window: int,
    windows_per_trial: int,
    crop_fraction: float = 0.1,
) -> np.ndarray:
    x = fill_and_standardize(x)
    lo = int(round(len(x) * crop_fraction))
    hi = int(round(len(x) * (1.0 - crop_fraction)))
    if hi - lo < 4:
        raise ValueError("Recording is too short after central crop")
    x = x[lo:hi]
    target_samples = samples_per_window * windows_per_trial
    old_grid = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    new_grid = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    resampled = np.empty((target_samples, x.shape[1]), dtype=np.float32)
    for channel in range(x.shape[1]):
        resampled[:, channel] = np.interp(new_grid, old_grid, x[:, channel])
    windows = resampled.reshape(windows_per_trial, samples_per_window, x.shape[1])
    return windows.transpose(0, 2, 1)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    maps = index_files(root)
    complete = set.intersection(*(set(maps[m]) for m in MODALITIES))
    keys = sorted(complete)
    if args.max_trials is not None:
        keys = keys[: args.max_trials]
    if not keys:
        counts = {m: len(maps[m]) for m in MODALITIES}
        raise RuntimeError(f"No complete trials found under {root}; discovered {counts}")

    samples_per_window = int(round(args.target_hz * args.window_seconds))
    output: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    subjects: list[int] = []
    speeds: list[float] = []
    trial_ids: list[str] = []
    failures: list[dict[str, str]] = []

    readers = {
        "eeg": read_eeg,
        "emg": read_emg,
        "imu": read_imu,
        "fp": read_fp,
    }
    for subject, speed in tqdm(keys, desc="Preparing complete trials"):
        trial_arrays: dict[str, np.ndarray] = {}
        try:
            for modality in MODALITIES:
                raw = readers[modality](maps[modality][(subject, speed)])
                trial_arrays[modality] = normalized_windows(
                    raw,
                    samples_per_window=samples_per_window,
                    windows_per_trial=args.windows_per_trial,
                )
        except Exception as exc:
            failures.append(
                {"subject": str(subject), "speed": str(speed), "error": repr(exc)}
            )
            continue

        for modality in MODALITIES:
            output[modality].append(trial_arrays[modality])
        subjects.extend([subject] * args.windows_per_trial)
        speeds.extend([speed] * args.windows_per_trial)
        trial_ids.extend([f"S{subject}_{speed:g}"] * args.windows_per_trial)

    if not output["eeg"]:
        raise RuntimeError(f"All trial conversions failed: {failures[:3]}")

    arrays = {m: np.concatenate(output[m], axis=0) for m in MODALITIES}
    arrays["subject"] = np.asarray(subjects, dtype=np.int16)
    arrays["speed"] = np.asarray(speeds, dtype=np.float32)
    arrays["trial_id"] = np.asarray(trial_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "dataset_root": str(root),
        "output": str(args.output.resolve()),
        "complete_trials_discovered": len(complete),
        "complete_trials_written": len(output["eeg"]),
        "windows_written": len(subjects),
        "subjects_written": len(set(subjects)),
        "shape": {m: list(arrays[m].shape) for m in MODALITIES},
        "target_hz": args.target_hz,
        "window_seconds": args.window_seconds,
        "windows_per_trial": args.windows_per_trial,
        "failed_trials": failures,
        "source_counts": {m: len(maps[m]) for m in MODALITIES},
        "alignment": (
            "Central 80% of each constant-speed recording aligned by normalized "
            "trial time; suitable for the speed-classification pipeline check, "
            "not for phase-sensitive kinetics."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
