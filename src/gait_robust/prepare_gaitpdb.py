from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_RATE = 100
WINDOW_SAMPLES = 500
TRIM_SECONDS = 5.0
SENSOR_COLUMNS = slice(1, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and preprocess the PhysioNet Gait in Parkinson's "
            "Disease usual-walking records."
        )
    )
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root
        / "data"
        / "raw"
        / "gaitpdb_full"
        / "gait-in-parkinsons-disease-1.0.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "gaitpdb_clinical_windows.npz",
    )
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=project_root / "artifacts" / "gaitpdb_data_quality",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_demographics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=range(20),
        engine="python",
        skip_blank_lines=True,
    )
    frame = frame.dropna(subset=["ID"]).copy()
    frame["ID"] = frame["ID"].astype(str)
    frame["Study"] = frame["Study"].astype(str)
    frame["Group"] = frame["Group"].astype(int)
    frame["Gender"] = frame["Gender"].astype(int)
    if len(frame) != 166:
        raise RuntimeError(
            f"Expected 166 metadata rows, found {len(frame)}"
        )
    if frame["ID"].duplicated().any():
        duplicate_ids = frame.loc[frame["ID"].duplicated(), "ID"].tolist()
        raise RuntimeError(f"Duplicate metadata IDs: {duplicate_ids}")
    if set(frame["Study"]) != {"Ga", "Ju", "Si"}:
        raise RuntimeError(
            f"Unexpected study values: {sorted(frame['Study'].unique())}"
        )
    if set(frame["Group"]) != {1, 2}:
        raise RuntimeError(
            f"Unexpected group values: {sorted(frame['Group'].unique())}"
        )
    return frame


def index_usual_walks(raw_dir: Path) -> dict[str, Path]:
    candidates = sorted(raw_dir.rglob("*_01.txt"))
    indexed: dict[str, Path] = {}
    for path in candidates:
        subject_id = path.stem.rsplit("_", 1)[0]
        if subject_id in indexed:
            raise RuntimeError(
                f"Multiple usual-walk files found for {subject_id}: "
                f"{indexed[subject_id]} and {path}"
            )
        indexed[subject_id] = path
    return indexed


def validate_record(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 19:
        raise ValueError(f"expected 19 columns, found shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("non-finite values")
    time = values[:, 0]
    intervals = np.diff(time)
    if len(intervals) == 0 or np.any(intervals <= 0):
        raise ValueError("timestamps are not strictly increasing")
    median_interval = float(np.median(intervals))
    if abs(median_interval - 0.01) > 0.0001:
        raise ValueError(
            f"median interval {median_interval:.8f} is outside 1% tolerance"
        )
    sensors = values[:, SENSOR_COLUMNS]
    negative_count = int((sensors < -1e-6).sum())
    if negative_count:
        raise ValueError(f"{negative_count} VGRF values below -1e-6")
    left_error = np.abs(sensors[:, :8].sum(axis=1) - values[:, 17])
    right_error = np.abs(sensors[:, 8:].sum(axis=1) - values[:, 18])
    maximum_total_error = float(max(left_error.max(), right_error.max()))
    if maximum_total_error > 1e-3:
        raise ValueError(
            f"sensor sums disagree with supplied totals by "
            f"{maximum_total_error:.6f} N"
        )
    duration = float(time[-1] - time[0])
    if duration - 2 * TRIM_SECONDS < 20.0:
        raise ValueError(
            f"only {duration:.2f} s before the prespecified trim"
        )
    report = {
        "rows": int(len(values)),
        "duration_seconds": duration,
        "median_interval_seconds": median_interval,
        "minimum_sensor_value": float(sensors.min()),
        "maximum_sensor_value": float(sensors.max()),
        "negative_count": negative_count,
        "maximum_total_error": maximum_total_error,
    }
    return values, report


def make_windows(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    time = values[:, 0]
    retained = (time >= time[0] + TRIM_SECONDS) & (
        time <= time[-1] - TRIM_SECONDS
    )
    sensors = values[retained, SENSOR_COLUMNS].astype(np.float32)
    total_force = sensors.sum(axis=1)
    force_scale = float(np.median(total_force))
    if not np.isfinite(force_scale) or force_scale <= 1e-6:
        raise ValueError(f"invalid median total-force scale {force_scale}")
    sensors = sensors / force_scale
    window_count = len(sensors) // WINDOW_SAMPLES
    if window_count < 4:
        raise ValueError(
            f"only {window_count} complete windows after trimming"
        )
    sensors = sensors[: window_count * WINDOW_SAMPLES]
    windows = sensors.reshape(window_count, WINDOW_SAMPLES, 2, 8)
    windows = windows.transpose(0, 2, 3, 1).copy()
    return windows, {
        "force_scale_newtons": force_scale,
        "window_count": int(window_count),
    }


def main() -> None:
    args = parse_args()
    demographics_path = args.raw_dir / "demographics.txt"
    format_path = args.raw_dir / "format.txt"
    if not demographics_path.exists() or not format_path.exists():
        raise FileNotFoundError(
            f"Missing demographics.txt or format.txt under {args.raw_dir}"
        )
    demographics = load_demographics(demographics_path)
    waveforms = index_usual_walks(args.raw_dir)

    all_windows: list[np.ndarray] = []
    subject_ids: list[str] = []
    studies: list[str] = []
    labels: list[int] = []
    genders: list[int] = []
    ages: list[float] = []
    stages: list[float] = []
    window_subject_index: list[np.ndarray] = []
    window_ordinals: list[np.ndarray] = []
    quality_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, str]] = []

    for _, row in demographics.sort_values("ID").iterrows():
        subject_id = row["ID"]
        path = waveforms.get(subject_id)
        if path is None:
            excluded_rows.append(
                {
                    "subject": subject_id,
                    "reason": "missing usual-walking _01 waveform",
                }
            )
            continue
        try:
            values, quality = validate_record(path)
            windows, window_info = make_windows(values)
        except (OSError, ValueError) as exc:
            excluded_rows.append(
                {"subject": subject_id, "reason": str(exc)}
            )
            continue
        subject_index = len(subject_ids)
        all_windows.append(windows)
        subject_ids.append(subject_id)
        studies.append(str(row["Study"]))
        labels.append(1 if int(row["Group"]) == 1 else 0)
        genders.append(int(row["Gender"]))
        ages.append(float(row["Age"]))
        stages.append(float(row["HoehnYahr"]))
        window_subject_index.append(
            np.full(len(windows), subject_index, dtype=np.int32)
        )
        window_ordinals.append(np.arange(len(windows), dtype=np.int32))
        quality_rows.append(
            {
                "subject": subject_id,
                "study": row["Study"],
                "label": "PD" if int(row["Group"]) == 1 else "control",
                "gender_code": int(row["Gender"]),
                "age": float(row["Age"]),
                "hoehen_yahr": float(row["HoehnYahr"]),
                "source_file": path.name,
                "source_sha256": sha256(path),
                **quality,
                **window_info,
            }
        )

    if not all_windows:
        raise RuntimeError("No valid usual-walking records were found")
    windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    subject_index = np.concatenate(window_subject_index)
    window_ordinal = np.concatenate(window_ordinals)
    if len(windows) != len(subject_index):
        raise RuntimeError("Window and subject-index lengths differ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        windows=windows,
        window_subject_index=subject_index,
        window_ordinal=window_ordinal,
        subject_ids=np.asarray(subject_ids, dtype="U16"),
        studies=np.asarray(studies, dtype="U2"),
        labels=np.asarray(labels, dtype=np.int64),
        genders=np.asarray(genders, dtype=np.int64),
        ages=np.asarray(ages, dtype=np.float32),
        hoehen_yahr=np.asarray(stages, dtype=np.float32),
        sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int64),
        window_samples=np.asarray(WINDOW_SAMPLES, dtype=np.int64),
    )

    args.quality_dir.mkdir(parents=True, exist_ok=True)
    quality_frame = pd.DataFrame(quality_rows)
    excluded_frame = pd.DataFrame(
        excluded_rows, columns=["subject", "reason"]
    )
    quality_frame.to_csv(
        args.quality_dir / "record_quality.csv", index=False
    )
    excluded_frame.to_csv(
        args.quality_dir / "excluded_records.csv", index=False
    )
    summary = {
        "dataset": "PhysioNet Gait in Parkinson's Disease v1.0.0",
        "source_doi": "10.13026/C24H3N",
        "metadata_sha256": sha256(demographics_path),
        "format_sha256": sha256(format_path),
        "subjects_in_metadata": int(len(demographics)),
        "subjects_included": int(len(subject_ids)),
        "subjects_excluded": int(len(excluded_rows)),
        "windows": int(len(windows)),
        "window_shape": list(windows.shape[1:]),
        "study_counts": quality_frame.groupby(["study", "label"])
        .size()
        .unstack(fill_value=0)
        .to_dict(),
        "integrity_rules": {
            "columns": 19,
            "median_interval_seconds": 0.01,
            "interval_tolerance_fraction": 0.01,
            "minimum_post_trim_seconds": 20,
            "negative_tolerance": -1e-6,
            "total_force_tolerance_newtons": 1e-3,
        },
    }
    (args.quality_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
