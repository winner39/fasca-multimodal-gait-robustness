from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


PROBABILITY_COLUMNS = ("p_0.5", "p_0.75", "p_1")
FULL_MODALITIES = "eeg+emg+imu+fp"
DISPLAY_NAMES = {
    "centaur_adaptation": "Centaur-adaptation",
    "adapt_adaptation": "ADAPT-adaptation",
    "cimsleepnet_adaptation": "CIMSleepNet-adaptation",
    "rapid_sensor_aug_kd": "RAPID+FASCA",
    "rapid_moddrop_fasca_kd": "Masked fusion+FASCA",
    "rapid_moddrop_fasca_ssl_kd": "Masked fusion+FASCA-SSL",
    "rapid_embracenet_fasca_kd": "EmbraceNet+FASCA",
    "rapid_embracenet_fasca_ssl_kd": "EmbraceNet+FASCA-SSL",
    "rapid_embracenet_fasca_structssl_kd": (
        "EmbraceNet+FASCA-StructSSL"
    ),
    "rapid_embracenet_fasca_qstructssl_kd": (
        "EmbraceNet+FASCA-QStructSSL"
    ),
    "rapid_embracenet_fasca_mar_kd": "EmbraceNet+FASCA-MAR",
    "rapid_embracenet_fasca_qstructssl_mar_kd": (
        "EmbraceNet+FASCA-QStructSSL-MAR"
    ),
    "rapid_embracenet_quality_fasca_kd": (
        "Quality-EmbraceNet+FASCA"
    ),
    "rapid_actionmae_fasca_kd": "ActionMAE+FASCA",
    "rapid_uniform_rapid_fasca_kd": "RAPID-uniform+FASCA",
    "rapid_sensor_aug_no_curriculum_kd": "RAPID+FASCA (no curriculum)",
    "rapid_iid_aug_kd": "RAPID+generic augmentation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-roots", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--include-methods", nargs="+", default=None
    )
    parser.add_argument("--include-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    return parser.parse_args()


def macro_f1(group: pd.DataFrame) -> float:
    return float(
        f1_score(
            group["true_class"].to_numpy(dtype=int),
            group[list(PROBABILITY_COLUMNS)].to_numpy().argmax(axis=1),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )


def trial_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(
            [
                "method",
                "seed",
                "available_modalities",
                "subject",
                "trial_id",
            ],
            sort=False,
        )
        .agg(
            true_class=("true_class", "first"),
            **{
                column: (column, "mean")
                for column in PROBABILITY_COLUMNS
            },
        )
        .reset_index()
    )


def pooled_seed_metrics(trial: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in trial.groupby(
        ["method", "seed", "available_modalities"], sort=False
    ):
        rows.append(
            {
                "method": keys[0],
                "seed": int(keys[1]),
                "available_modalities": keys[2],
                "modality_count": keys[2].count("+") + 1,
                "trial_macro_f1": macro_f1(group),
            }
        )
    combinations = pd.DataFrame(rows)
    summaries = []
    for (method, seed), group in combinations.groupby(
        ["method", "seed"], sort=False
    ):
        full = group[
            group["available_modalities"] == FULL_MODALITIES
        ]["trial_macro_f1"]
        incomplete = group[
            group["available_modalities"] != FULL_MODALITIES
        ]["trial_macro_f1"]
        summaries.append(
            {
                "method": method,
                "seed": seed,
                "full": full.iloc[0],
                "incomplete_14": incomplete.mean(),
                "mean_15": group["trial_macro_f1"].mean(),
                "worst": group["trial_macro_f1"].min(),
                "single_modality_mean": group[
                    group["modality_count"] == 1
                ]["trial_macro_f1"].mean(),
                "two_modality_mean": group[
                    group["modality_count"] == 2
                ]["trial_macro_f1"].mean(),
                "three_modality_mean": group[
                    group["modality_count"] == 3
                ]["trial_macro_f1"].mean(),
            }
        )
    return pd.DataFrame(summaries), combinations


def subject_endpoints(trial: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in trial.groupby(
        ["method", "seed", "subject", "available_modalities"],
        sort=False,
    ):
        rows.append(
            {
                "method": keys[0],
                "seed": int(keys[1]),
                "subject": int(keys[2]),
                "available_modalities": keys[3],
                "modality_count": keys[3].count("+") + 1,
                "macro_f1": macro_f1(group),
            }
        )
    combinations = pd.DataFrame(rows)
    endpoint_rows = []
    for keys, group in combinations.groupby(
        ["method", "seed", "subject"], sort=False
    ):
        incomplete = group[
            group["available_modalities"] != FULL_MODALITIES
        ]
        full = group[
            group["available_modalities"] == FULL_MODALITIES
        ]
        values = {
            "mean_15": group["macro_f1"].mean(),
            "incomplete_14": incomplete["macro_f1"].mean(),
            "full": full["macro_f1"].iloc[0],
            "single_modality_mean": group[
                group["modality_count"] == 1
            ]["macro_f1"].mean(),
        }
        for endpoint, value in values.items():
            endpoint_rows.append(
                {
                    "method": keys[0],
                    "seed": keys[1],
                    "subject": keys[2],
                    "endpoint": endpoint,
                    "macro_f1": value,
                }
            )
    return pd.DataFrame(endpoint_rows), combinations


def paired_summary(
    values: pd.DataFrame,
    reference: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    averaged = (
        values.groupby(
            ["method", "subject", "endpoint"], as_index=False
        )["macro_f1"]
        .mean()
    )
    methods = sorted(set(averaged["method"]) - {reference})
    rows = []
    rng = np.random.default_rng(seed)
    for method in methods:
        for endpoint, group in averaged.groupby("endpoint", sort=False):
            pivot = group.pivot(
                index="subject", columns="method", values="macro_f1"
            )
            if reference not in pivot or method not in pivot:
                continue
            pivot = pivot[[reference, method]].dropna()
            difference = (
                pivot[reference] - pivot[method]
            ).to_numpy(dtype=float)
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
                    "comparison_method": method,
                    "endpoint": endpoint,
                    "subjects": len(difference),
                    "reference_subject_mean": pivot[reference].mean(),
                    "comparison_subject_mean": pivot[method].mean(),
                    "paired_delta": difference.mean(),
                    "bootstrap_ci_low": np.quantile(samples, 0.025),
                    "bootstrap_ci_high": np.quantile(samples, 0.975),
                    "subjects_improved": int((difference > 0).sum()),
                    "subjects_tied": int((difference == 0).sum()),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for root in args.prediction_roots:
        paths.extend(
            sorted(
                root.glob(
                    "fold_*/seed_*/*/window_predictions.csv.gz"
                )
            )
        )
    if not paths:
        raise FileNotFoundError("No window prediction files found")
    predictions = pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True
    )
    if args.include_seeds is not None:
        predictions = predictions[
            predictions["seed"].isin(args.include_seeds)
        ].copy()
    if args.include_methods is not None:
        predictions = predictions[
            predictions["method"].isin(args.include_methods)
        ].copy()
        missing_methods = set(args.include_methods) - set(
            predictions["method"].unique()
        )
        if missing_methods:
            raise ValueError(
                f"Missing requested methods: {sorted(missing_methods)}"
            )
    trial = trial_predictions(predictions)
    seed_metrics, combination_metrics = pooled_seed_metrics(trial)
    seed_metrics["display_name"] = seed_metrics["method"].map(
        DISPLAY_NAMES
    )
    seed_metrics.to_csv(
        args.output_dir / "factorial_seed_metrics.csv", index=False
    )
    combination_metrics.to_csv(
        args.output_dir / "factorial_combination_metrics.csv", index=False
    )
    method_summary = (
        seed_metrics.groupby(["method", "display_name"], dropna=False)
        .agg(
            full_mean=("full", "mean"),
            full_std=("full", "std"),
            incomplete_14_mean=("incomplete_14", "mean"),
            incomplete_14_std=("incomplete_14", "std"),
            mean_15=("mean_15", "mean"),
            mean_15_std=("mean_15", "std"),
            worst_mean=("worst", "mean"),
            worst_std=("worst", "std"),
            single_modality_mean=("single_modality_mean", "mean"),
            two_modality_mean=("two_modality_mean", "mean"),
            three_modality_mean=("three_modality_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_15", ascending=False)
    )
    method_summary.to_csv(
        args.output_dir / "factorial_method_summary.csv", index=False
    )
    endpoints, subject_combinations = subject_endpoints(trial)
    endpoints.to_csv(
        args.output_dir / "factorial_subject_endpoint_metrics.csv",
        index=False,
    )
    subject_combinations.to_csv(
        args.output_dir / "factorial_subject_combination_metrics.csv",
        index=False,
    )
    paired = paired_summary(
        endpoints,
        args.reference_method,
        args.bootstrap_repeats,
        args.bootstrap_seed,
    )
    paired.to_csv(
        args.output_dir / "factorial_paired_summary.csv", index=False
    )
    print(method_summary.to_string(index=False))


if __name__ == "__main__":
    main()
