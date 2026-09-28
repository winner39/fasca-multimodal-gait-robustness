from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import f1_score


CLASSES = (0, 1, 2)
PRIMARY_ENDPOINT = "joint_fault_mean"
STANDARD_SCENARIOS = {
    "noise_10db",
    "channel_dropout_30",
    "packet_loss_30",
    "shift_250ms",
    "gain_1.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path, nargs="+", required=True
    )
    parser.add_argument("--include-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def macro_f1(group: pd.DataFrame) -> float:
    probability_columns = [f"p_{index}" for index in CLASSES]
    return float(
        f1_score(
            group["truth"].to_numpy(dtype=int),
            group[probability_columns].to_numpy().argmax(axis=1),
            labels=list(CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return adjusted
    ordered = finite[np.argsort(p_values[finite])]
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def rank_biserial(difference: np.ndarray) -> float:
    nonzero = difference[difference != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    denominator = ranks.sum()
    return float(
        (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum())
        / denominator
    )


def add_composite_endpoints(values: pd.DataFrame) -> pd.DataFrame:
    rows = [values]
    corrupted = values[values["scenario"] != "clean"]
    standard_corrupted = corrupted[
        corrupted["scenario"].isin(STANDARD_SCENARIOS)
    ]
    definitions = {
        "single_modality_mean": standard_corrupted[
            standard_corrupted["target"] != "eeg+emg+imu+fp"
        ],
        "joint_fault_mean": standard_corrupted[
            standard_corrupted["target"] == "eeg+emg+imu+fp"
        ],
        "all_corruptions_mean": standard_corrupted,
    }
    for endpoint, selected in definitions.items():
        rows.append(
            selected.groupby(
                ["method", "seed", "subject"], as_index=False
            )["macro_f1"]
            .mean()
            .assign(endpoint=endpoint)
        )
    joint = corrupted[
        corrupted["target"] == "eeg+emg+imu+fp"
    ].copy()
    joint["endpoint"] = "joint_" + joint["scenario"]
    rows.append(
        joint[
            ["method", "seed", "subject", "endpoint", "macro_f1"]
        ]
    )
    clean = values[values["scenario"] == "clean"].copy()
    clean["endpoint"] = "clean"
    rows.append(
        clean[
            ["method", "seed", "subject", "endpoint", "macro_f1"]
        ]
    )
    return pd.concat(
        [
            row[
                ["method", "seed", "subject", "endpoint", "macro_f1"]
            ]
            for row in rows[1:]
        ],
        ignore_index=True,
    )


def paired_summaries(
    values: pd.DataFrame,
    reference: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    methods = sorted(set(values["method"]) - {reference})
    endpoints = list(dict.fromkeys(values["endpoint"].tolist()))
    rows = []
    rng = np.random.default_rng(seed)
    subject_values = (
        values.groupby(
            ["method", "subject", "endpoint"], as_index=False
        )["macro_f1"]
        .mean()
    )
    for comparison in methods:
        for endpoint in endpoints:
            selected = subject_values[
                subject_values["endpoint"] == endpoint
            ]
            pivot = selected.pivot(
                index="subject", columns="method", values="macro_f1"
            )
            if reference not in pivot or comparison not in pivot:
                continue
            pivot = pivot[[reference, comparison]].dropna()
            difference = (
                pivot[reference] - pivot[comparison]
            ).to_numpy()
            samples = rng.choice(
                difference,
                size=(repeats, len(difference)),
                replace=True,
            ).mean(axis=1)
            try:
                statistic, p_value = wilcoxon(difference)
            except ValueError:
                statistic, p_value = np.nan, np.nan
            rows.append(
                {
                    "reference_method": reference,
                    "comparison_method": comparison,
                    "endpoint": endpoint,
                    "primary_endpoint": endpoint == PRIMARY_ENDPOINT,
                    "subjects": len(difference),
                    "reference_subject_mean": pivot[reference].mean(),
                    "comparison_subject_mean": pivot[comparison].mean(),
                    "paired_delta": difference.mean(),
                    "bootstrap_ci_low": np.quantile(samples, 0.025),
                    "bootstrap_ci_high": np.quantile(samples, 0.975),
                    "subjects_improved": int((difference > 0).sum()),
                    "subjects_tied": int((difference == 0).sum()),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                    "rank_biserial": rank_biserial(difference),
                }
            )
    output = pd.DataFrame(rows)
    output["holm_p_within_comparison"] = np.nan
    for comparison, index in output.groupby(
        "comparison_method"
    ).groups.items():
        indices = np.asarray(list(index), dtype=int)
        output.loc[
            indices, "holm_p_within_comparison"
        ] = holm_adjust(output.loc[indices, "wilcoxon_p"].to_numpy())
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(
        [pd.read_csv(path) for path in args.predictions],
        ignore_index=True,
    )
    if args.include_seeds is not None:
        predictions = predictions[
            predictions["seed"].isin(args.include_seeds)
        ].copy()
    required = {
        "method",
        "seed",
        "subject",
        "scenario",
        "target",
        "truth",
        "p_0",
        "p_1",
        "p_2",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    condition_values = (
        predictions.groupby(
            ["method", "seed", "subject", "scenario", "target"],
            sort=False,
        )
        .apply(macro_f1, include_groups=False)
        .rename("macro_f1")
        .reset_index()
    )
    condition_values["endpoint"] = (
        condition_values["scenario"]
        + "|"
        + condition_values["target"]
    )
    condition_values.to_csv(
        args.output_dir / "subject_seed_condition_metrics.csv",
        index=False,
    )
    endpoints = add_composite_endpoints(condition_values)
    endpoints.to_csv(
        args.output_dir / "subject_seed_endpoint_metrics.csv",
        index=False,
    )
    paired = paired_summaries(
        endpoints,
        args.reference_method,
        args.bootstrap_repeats,
        args.bootstrap_seed,
    )
    paired.to_csv(
        args.output_dir / "paired_subject_endpoint_summary.csv",
        index=False,
    )
    primary = paired[paired["primary_endpoint"]]
    primary.to_csv(
        args.output_dir / "primary_endpoint_summary.csv",
        index=False,
    )
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
