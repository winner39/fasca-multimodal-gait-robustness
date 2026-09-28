from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


ACTIVITIES = ("walking", "running", "stairs_up", "stairs_down")
METHODS = ("balanced_kd", "fasca_kd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    return parser.parse_args()


def macro_f1(group: pd.DataFrame) -> float:
    probability_columns = [
        f"p_{activity}" for activity in ACTIVITIES
    ]
    return float(
        f1_score(
            group["true_class"].to_numpy(),
            group[probability_columns].to_numpy().argmax(axis=1),
            labels=list(range(len(ACTIVITIES))),
            average="macro",
            zero_division=0,
        )
    )


def paired_summary(
    values: pd.DataFrame,
    metric_column: str,
    group_column: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for group_name, group in values.groupby(group_column, sort=False):
        pivot = group.pivot(
            index="subject", columns="method", values=metric_column
        ).dropna()
        difference = (
            pivot["fasca_kd"] - pivot["balanced_kd"]
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
                group_column: group_name,
                "subjects": int(len(difference)),
                "baseline_subject_mean": float(
                    pivot["balanced_kd"].mean()
                ),
                "fasca_subject_mean": float(
                    pivot["fasca_kd"].mean()
                ),
                "paired_delta": float(difference.mean()),
                "bootstrap_ci_low": float(
                    np.quantile(samples, 0.025)
                ),
                "bootstrap_ci_high": float(
                    np.quantile(samples, 0.975)
                ),
                "subjects_improved": int((difference > 0).sum()),
                "subjects_tied": int((difference == 0).sum()),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths = sorted(
        args.experiment_dir.glob(
            "fold_*/seed_*/*/predictions.csv.gz"
        )
    )
    corruption_paths = sorted(
        args.experiment_dir.glob(
            "fold_*/seed_*/*/corruptions.csv"
        )
    )
    if not prediction_paths or not corruption_paths:
        raise FileNotFoundError("Experiment predictions are incomplete")

    predictions = pd.concat(
        (pd.read_csv(path) for path in prediction_paths),
        ignore_index=True,
    )
    expected_methods = set(METHODS)
    if set(predictions["method"].unique()) != expected_methods:
        raise ValueError("Both paired methods are required")
    per_subject_seed = (
        predictions.groupby(
            ["method", "seed", "subject", "available_modalities"],
            sort=False,
        )
        .apply(macro_f1, include_groups=False)
        .rename("macro_f1")
        .reset_index()
    )
    per_subject = (
        per_subject_seed.groupby(
            ["method", "subject", "available_modalities"],
            as_index=False,
        )["macro_f1"]
        .mean()
    )
    mean_three = (
        per_subject.groupby(["method", "subject"], as_index=False)[
            "macro_f1"
        ]
        .mean()
        .assign(available_modalities="mean_3")
    )
    per_subject_with_mean = pd.concat(
        [per_subject, mean_three], ignore_index=True
    )
    per_subject_with_mean.to_csv(
        args.output_dir / "subject_combination_metrics.csv", index=False
    )
    paired_summary(
        per_subject_with_mean,
        "macro_f1",
        "available_modalities",
        args.bootstrap_repeats,
        args.bootstrap_seed,
    ).to_csv(
        args.output_dir / "paired_subject_combination_summary.csv",
        index=False,
    )

    corruptions = pd.concat(
        (pd.read_csv(path) for path in corruption_paths),
        ignore_index=True,
    )
    per_corruption_subject_seed = (
        corruptions.groupby(
            ["method", "seed", "subject", "corruption"], sort=False
        )
        .apply(macro_f1, include_groups=False)
        .rename("macro_f1")
        .reset_index()
    )
    per_corruption_subject = (
        per_corruption_subject_seed.groupby(
            ["method", "subject", "corruption"], as_index=False
        )["macro_f1"]
        .mean()
    )
    per_corruption_subject.to_csv(
        args.output_dir / "subject_corruption_metrics.csv", index=False
    )
    paired_summary(
        per_corruption_subject,
        "macro_f1",
        "corruption",
        args.bootstrap_repeats,
        args.bootstrap_seed + 1,
    ).to_csv(
        args.output_dir / "paired_subject_corruption_summary.csv",
        index=False,
    )

    seed_corruptions = pd.read_csv(
        args.experiment_dir / "seed_pooled_corruptions.csv"
    )
    method_corruptions = (
        seed_corruptions.groupby(["method", "corruption"])
        .agg(
            mean=("window_macro_f1", "mean"),
            std=("window_macro_f1", "std"),
        )
        .reset_index()
    )
    comparison = method_corruptions.pivot(
        index="corruption", columns="method", values="mean"
    ).reset_index()
    comparison["delta"] = (
        comparison["fasca_kd"] - comparison["balanced_kd"]
    )
    comparison.to_csv(
        args.output_dir / "corruption_method_comparison.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
