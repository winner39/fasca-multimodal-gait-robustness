from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from gait_robust.prepare_windows import MODALITIES, index_files


EEG_CHANNELS = (
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

IMU_SEGMENTS = (
    "Pelvis",
    "T8",
    "Head",
    "Right Shoulder",
    "Right Upper Arm",
    "Right Forearm",
    "Right Hand",
    "Left Shoulder",
    "Left Upper Arm",
    "Left Forearm",
    "Left Hand",
    "Right Upper Leg",
    "Right Lower Leg",
    "Right Foot",
    "Left Upper Leg",
    "Left Lower Leg",
    "Left Foot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-hz", type=int, default=100)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--common-seconds", type=float, default=48.0)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser.parse_args()


def fill_nonfinite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    index = np.arange(len(values))
    for channel in range(values.shape[1]):
        column = values[:, channel]
        finite = np.isfinite(column)
        if not finite.any():
            values[:, channel] = 0.0
        elif not finite.all():
            values[~finite, channel] = np.interp(
                index[~finite], index[finite], column[finite]
            )
    return values


def read_eeg(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = [next(handle) for _ in range(2)]
    try:
        sampling_rate = float(header[1].split(",")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse EEG sampling rate from {path}") from exc
    frame = pd.read_csv(path, skiprows=15, low_memory=False)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(
        axis=0, how="all"
    )
    required = set(EEG_CHANNELS) - {"Pz"}
    required.update(("A1", "A2", "Time"))
    missing = sorted(required - set(numeric.columns))
    if missing:
        raise ValueError(f"Missing EEG columns in {path}: {missing}")
    linked_ears = (
        numeric["A1"].to_numpy(dtype=np.float64)
        + numeric["A2"].to_numpy(dtype=np.float64)
    ) / 2.0
    channels = []
    for channel in EEG_CHANNELS:
        if channel == "Pz":
            channels.append(-linked_ears)
        else:
            channels.append(
                numeric[channel].to_numpy(dtype=np.float64) - linked_ears
            )
    values = fill_nonfinite(np.stack(channels, axis=1))
    # The exported Time column is rounded to 4 decimals and therefore gives
    # an erroneous median rate of about 303 Hz. Use the acquisition header.
    times = np.arange(len(numeric), dtype=np.float64) / sampling_rate
    # The 1--45 Hz pass band removes drift and attenuates 50/60 Hz line noise
    # before downsampling to 100 Hz. This is not an ICA motion-artifact step.
    sos = signal.butter(
        4, (1.0, 45.0), btype="bandpass", fs=sampling_rate, output="sos"
    )
    values = signal.sosfiltfilt(sos, values, axis=0)
    return times, values, list(EEG_CHANNELS), sampling_rate


def read_emg(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    frame = pd.read_csv(path, low_memory=False)
    if frame.shape[1] < 13:
        raise ValueError(f"Expected time plus 12 EMG channels in {path}")
    numeric = frame.iloc[:, :13].apply(pd.to_numeric, errors="coerce")
    times = numeric.iloc[:, 0].to_numpy(dtype=np.float64)
    values = fill_nonfinite(numeric.iloc[:, 1:].to_numpy(dtype=np.float64))
    sampling_rate = float(1.0 / np.median(np.diff(times)))
    return times, values, [str(c) for c in frame.columns[1:13]], sampling_rate


def read_frame_rate(folder: Path) -> float:
    info_path = folder / "General_Information.csv"
    frame = pd.read_csv(info_path, header=None, names=["key", "value"])
    rate = frame.loc[frame["key"] == "Frame Rate", "value"]
    if rate.empty:
        raise ValueError(f"Frame Rate missing from {info_path}")
    return float(rate.iloc[0])


def read_imu(folder: Path) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    path = folder / "Sensor_Free_Acceleration.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    columns = [
        f"{segment} {axis}"
        for segment in IMU_SEGMENTS
        for axis in ("x", "y", "z")
    ]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing populated IMU columns in {path}: {missing}")
    rate = read_frame_rate(folder)
    frames = pd.to_numeric(frame["Frame"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    times = (frames - frames[0]) / rate
    values = fill_nonfinite(
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    return times, values, columns, rate


def read_fp(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    frame = pd.read_csv(path, low_memory=False)
    suffixes = ("Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy")
    columns = [
        column
        for column in frame.columns
        if any(str(column).endswith(suffix) for suffix in suffixes)
    ]
    if len(columns) != 16:
        raise ValueError(
            f"Expected 16 force/moment/COP channels in {path}, "
            f"found {len(columns)}"
        )
    rate = 1000.0
    times = np.arange(len(frame), dtype=np.float64) / rate
    values = fill_nonfinite(
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    return times, values, [str(c) for c in columns], rate


def resample_center(
    times: np.ndarray,
    values: np.ndarray,
    source_rate: float,
    target_rate: float,
    common_seconds: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    times = times - times[0]
    duration = float(times[-1])
    if duration < common_seconds:
        raise ValueError(
            f"Recording duration {duration:.3f}s is shorter than "
            f"{common_seconds:.3f}s"
        )
    if source_rate > target_rate * 1.05:
        cutoff = min(0.45 * target_rate, 0.45 * source_rate)
        sos = signal.butter(
            6, cutoff, btype="lowpass", fs=source_rate, output="sos"
        )
        values = signal.sosfiltfilt(sos, values, axis=0)
    start = 0.5 * (duration - common_seconds)
    target_count = int(round(common_seconds * target_rate))
    target_times = start + np.arange(target_count) / target_rate
    output = np.empty((target_count, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        output[:, channel] = np.interp(
            target_times, times, values[:, channel]
        ).astype(np.float32)
    return output


def main() -> None:
    args = parse_args()
    if args.common_seconds % args.window_seconds != 0:
        raise ValueError("common-seconds must be divisible by window-seconds")
    root = args.dataset_root.resolve()
    maps = index_files(root)
    complete = set.intersection(*(set(maps[m]) for m in MODALITIES))
    keys = sorted(complete)
    if args.max_trials is not None:
        keys = keys[: args.max_trials]
    if not keys:
        raise RuntimeError("No complete four-modality trials were found")

    readers = {
        "eeg": read_eeg,
        "emg": read_emg,
        "imu": read_imu,
        "fp": read_fp,
    }
    windows_per_trial = int(args.common_seconds / args.window_seconds)
    samples_per_window = int(round(args.target_hz * args.window_seconds))
    output: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    channel_names: dict[str, list[str]] = {}
    subjects: list[int] = []
    speeds: list[float] = []
    trial_ids: list[str] = []
    window_start_seconds: list[float] = []
    failures: list[dict[str, str]] = []
    source_rates: dict[str, list[float]] = {m: [] for m in MODALITIES}
    source_durations: dict[str, list[float]] = {m: [] for m in MODALITIES}

    for subject, speed in tqdm(keys, desc="Preparing physical-time windows"):
        trial_arrays: dict[str, np.ndarray] = {}
        try:
            for modality in MODALITIES:
                times, values, names, rate = readers[modality](
                    maps[modality][(subject, speed)]
                )
                if modality in channel_names and channel_names[modality] != names:
                    raise ValueError(
                        f"{modality} channel schema changed for "
                        f"S{subject}_{speed:g}"
                    )
                channel_names[modality] = names
                source_rates[modality].append(rate)
                source_durations[modality].append(
                    float(times[-1] - times[0])
                )
                resampled = resample_center(
                    times,
                    values,
                    source_rate=rate,
                    target_rate=args.target_hz,
                    common_seconds=args.common_seconds,
                )
                trial_arrays[modality] = (
                    resampled.reshape(
                        windows_per_trial,
                        samples_per_window,
                        resampled.shape[1],
                    ).transpose(0, 2, 1)
                )
        except Exception as exc:
            failures.append(
                {
                    "subject": str(subject),
                    "speed": str(speed),
                    "error": repr(exc),
                }
            )
            continue

        for modality in MODALITIES:
            output[modality].append(trial_arrays[modality])
        subjects.extend([subject] * windows_per_trial)
        speeds.extend([speed] * windows_per_trial)
        trial_ids.extend([f"S{subject}_{speed:g}"] * windows_per_trial)
        window_start_seconds.extend(
            [
                window * args.window_seconds
                for window in range(windows_per_trial)
            ]
        )

    if not output["eeg"]:
        raise RuntimeError(f"All trial conversions failed: {failures[:3]}")

    arrays = {
        modality: np.concatenate(output[modality], axis=0)
        for modality in MODALITIES
    }
    arrays.update(
        {
            "subject": np.asarray(subjects, dtype=np.int16),
            "speed": np.asarray(speeds, dtype=np.float32),
            "trial_id": np.asarray(trial_ids),
            "window_start_seconds": np.asarray(
                window_start_seconds, dtype=np.float32
            ),
        }
    )
    for modality, names in channel_names.items():
        arrays[f"{modality}_channels"] = np.asarray(names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "schema_version": 2,
        "dataset_root": str(root),
        "output": str(args.output.resolve()),
        "preprocessing_scope": (
            "Physical-time filtering/resampling only. No per-trial or "
            "full-dataset standardization is applied."
        ),
        "complete_trials_discovered": len(complete),
        "complete_trials_written": len(output["eeg"]),
        "windows_written": len(subjects),
        "subjects_written": len(set(subjects)),
        "shape": {m: list(arrays[m].shape) for m in MODALITIES},
        "channels": channel_names,
        "target_hz": args.target_hz,
        "window_seconds": args.window_seconds,
        "common_seconds": args.common_seconds,
        "windows_per_trial": windows_per_trial,
        "source_rate_summary_hz": {
            modality: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
            for modality, values in source_rates.items()
            if values
        },
        "source_duration_summary_seconds": {
            modality: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
            for modality, values in source_durations.items()
            if values
        },
        "failed_trials": failures,
        "known_alignment_limit": (
            "Each modality uses a centered physical-time interval. EEG, EMG, "
            "and IMU were trigger synchronized by the source protocol; force "
            "plate acquisition was manually initiated and may retain a fixed "
            "trial offset. Phase-sensitive claims require a separate alignment "
            "validation."
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
