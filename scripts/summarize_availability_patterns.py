from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ORDER = (
    "eeg",
    "emg",
    "imu",
    "fp",
    "eeg+emg",
    "eeg+imu",
    "eeg+fp",
    "emg+imu",
    "emg+fp",
    "imu+fp",
    "eeg+emg+imu",
    "eeg+emg+fp",
    "eeg+imu+fp",
    "emg+imu+fp",
    "eeg+emg+imu+fp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combination-results", type=Path, required=True)
    parser.add_argument("--subject-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="embracenet")
    parser.add_argument("--fasca", default="rapid_embracenet_fasca_kd")
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combination = pd.read_csv(args.combination_results)
    subject = pd.read_csv(args.subject_results)
    rng = np.random.default_rng(args.bootstrap_seed)
    rows: list[dict[str, float | int | str]] = []

    for pattern in ORDER:
        pattern_runs = combination[
            combination["available_modalities"].eq(pattern)
        ]
        summary: dict[str, tuple[float, float]] = {}
        for method in (args.baseline, args.fasca):
            values = pattern_runs.loc[
                pattern_runs["method"].eq(method), "trial_macro_f1"
            ].to_numpy()
            summary[method] = (float(values.mean()), float(values.std(ddof=1)))

        paired = (
            subject[subject["available_modalities"].eq(pattern)]
            .groupby(["method", "subject"], as_index=False)["macro_f1"]
            .mean()
            .pivot(index="subject", columns="method", values="macro_f1")
            .dropna(subset=[args.baseline, args.fasca])
        )
        differences = (
            paired[args.fasca] - paired[args.baseline]
        ).to_numpy()
        indices = rng.integers(
            0,
            len(differences),
            size=(args.bootstrap_repeats, len(differences)),
        )
        bootstrap = differences[indices].mean(axis=1)
        low, high = np.quantile(bootstrap, (0.025, 0.975))
        rows.append(
            {
                "available_modalities": pattern,
                "modality_count": pattern.count("+") + 1,
                "baseline_mean": summary[args.baseline][0],
                "baseline_seed_sd": summary[args.baseline][1],
                "fasca_mean": summary[args.fasca][0],
                "fasca_seed_sd": summary[args.fasca][1],
                "paired_participant_gain": float(differences.mean()),
                "paired_ci_low": float(low),
                "paired_ci_high": float(high),
                "participants": len(differences),
                "bootstrap_repeats": args.bootstrap_repeats,
                "bootstrap_seed": args.bootstrap_seed,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

