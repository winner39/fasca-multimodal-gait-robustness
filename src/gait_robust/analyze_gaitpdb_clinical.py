from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from gait_robust.run_gaitpdb_clinical import (
    CONDITIONS,
    FAULT_CONDITIONS,
    METHODS,
    TEST_STUDIES,
)


DISPLAY_METHOD = {
    "baseline": "Masked fusion",
    "generic": "Generic augmentation",
    "iid": "IID channel corruption",
    "fasca": "FASCA-pressure",
}
GENDER_LABEL = {1: "male", 2: "female"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the locked study-held-out Gait in Parkinson's Disease "
            "clinical robustness experiment."
        )
    )
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_root
        / "artifacts"
        / "gaitpdb_clinical_external"
        / "subject_predictions.csv.gz",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "gaitpdb_clinical_windows.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "artifacts"
        / "gaitpdb_clinical_external"
        / "analysis",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260706)
    return parser.parse_args()


def load_subject_metadata(path: Path) -> pd.DataFrame:
    archive = np.load(path)
    return pd.DataFrame(
        {
            "subject": archive["subject_ids"].astype(str),
            "metadata_study": archive["studies"].astype(str),
            "metadata_truth": archive["labels"].astype(int),
            "gender": archive["genders"].astype(int),
            "age": archive["ages"].astype(float),
            "hoehen_yahr": archive["hoehen_yahr"].astype(float),
        }
    )


def cohort_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for study in (*TEST_STUDIES, "all"):
        study_frame = (
            metadata
            if study == "all"
            else metadata[metadata["metadata_study"] == study]
        )
        for group_name, truth in (
            ("all", None),
            ("PD", 1),
            ("control", 0),
        ):
            group = (
                study_frame
                if truth is None
                else study_frame[
                    study_frame["metadata_truth"] == truth
                ]
            )
            age = group["age"].dropna()
            stage = group["hoehen_yahr"].dropna()
            rows.append(
                {
                    "study": study,
                    "group": group_name,
                    "subjects": int(len(group)),
                    "male_subjects": int((group["gender"] == 1).sum()),
                    "female_subjects": int((group["gender"] == 2).sum()),
                    "age_mean": float(age.mean()) if len(age) else np.nan,
                    "age_sd": (
                        float(age.std(ddof=1)) if len(age) > 1 else np.nan
                    ),
                    "hoehen_yahr_mean": (
                        float(stage.mean()) if len(stage) else np.nan
                    ),
                    "hoehen_yahr_sd": (
                        float(stage.std(ddof=1))
                        if len(stage) > 1
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def validate_and_average(
    predictions: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "held_out_study",
        "seed",
        "method",
        "condition",
        "subject",
        "study",
        "truth",
        "p_control",
        "p_pd",
        "fter_exact_route_rate",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise RuntimeError(f"Missing prediction columns: {sorted(missing)}")
    if set(predictions["method"]) != set(METHODS):
        raise RuntimeError("Unexpected method values")
    if set(predictions["condition"]) != set(CONDITIONS):
        raise RuntimeError("Unexpected condition values")
    if set(predictions["held_out_study"]) != set(TEST_STUDIES):
        raise RuntimeError("Unexpected held-out study values")
    if not (
        predictions["held_out_study"] == predictions["study"]
    ).all():
        raise RuntimeError("Some predictions are not from the held-out study")
    seed_count = predictions.groupby(
        ["method", "condition", "subject"]
    )["seed"].nunique()
    expected_seed_count = predictions["seed"].nunique()
    if not (seed_count == expected_seed_count).all():
        raise RuntimeError(
            "Some subject-condition-method cells lack an optimization seed"
        )
    row_count = predictions.groupby(
        ["method", "condition", "subject", "seed"]
    ).size()
    if not (row_count == 1).all():
        raise RuntimeError("Duplicate subject-condition-seed predictions")
    averaged = (
        predictions.groupby(
            [
                "held_out_study",
                "method",
                "condition",
                "subject",
                "study",
            ],
            sort=False,
        )
        .agg(
            truth=("truth", "first"),
            windows=("windows", "first"),
            p_control=("p_control", "mean"),
            p_pd=("p_pd", "mean"),
            fter_exact_route_rate=("fter_exact_route_rate", "mean"),
        )
        .reset_index()
    )
    averaged = averaged.merge(metadata, on="subject", how="left")
    if averaged["metadata_study"].isna().any():
        raise RuntimeError("Predictions contain unknown subjects")
    if not (averaged["study"] == averaged["metadata_study"]).all():
        raise RuntimeError("Study metadata mismatch")
    if not (averaged["truth"] == averaged["metadata_truth"]).all():
        raise RuntimeError("Label metadata mismatch")
    if not np.allclose(
        averaged["p_control"] + averaged["p_pd"], 1.0, atol=1e-5
    ):
        raise RuntimeError("Probabilities do not sum to one")
    expected_rows = len(metadata) * len(METHODS) * len(CONDITIONS)
    if len(averaged) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} averaged rows, found {len(averaged)}"
        )
    return averaged


def ece10(truth: np.ndarray, probability_pd: np.ndarray) -> float:
    probabilities = np.column_stack([1.0 - probability_pd, probability_pd])
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == truth).astype(float)
    bin_id = np.minimum((confidence * 10).astype(int), 9)
    value = 0.0
    for current_bin in range(10):
        selected = bin_id == current_bin
        if selected.any():
            value += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(value)


def aurc(truth: np.ndarray, probability_pd: np.ndarray) -> float:
    confidence = np.maximum(probability_pd, 1.0 - probability_pd)
    predicted = (probability_pd >= 0.5).astype(int)
    error = (predicted != truth).astype(float)
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    return float(cumulative_risk.mean())


def selective_accuracy(
    truth: np.ndarray, probability_pd: np.ndarray, coverage: float = 0.8
) -> float:
    confidence = np.maximum(probability_pd, 1.0 - probability_pd)
    predicted = (probability_pd >= 0.5).astype(int)
    keep = max(1, int(np.ceil(coverage * len(truth))))
    order = np.argsort(-confidence, kind="stable")[:keep]
    return float((predicted[order] == truth[order]).mean())


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["truth"].to_numpy(dtype=int)
    probability_pd = frame["p_pd"].to_numpy(dtype=float)
    predicted = (probability_pd >= 0.5).astype(int)
    sensitivity = (
        float((predicted[truth == 1] == 1).mean())
        if (truth == 1).any()
        else np.nan
    )
    specificity = (
        float((predicted[truth == 0] == 0).mean())
        if (truth == 0).any()
        else np.nan
    )
    targets = np.column_stack([1 - truth, truth])
    probabilities = np.column_stack([1 - probability_pd, probability_pd])
    return {
        "subjects": int(len(frame)),
        "pd_subjects": int((truth == 1).sum()),
        "control_subjects": int((truth == 0).sum()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, predicted)
        ),
        "macro_f1": float(
            f1_score(truth, predicted, average="macro", zero_division=0)
        ),
        "auroc": float(roc_auc_score(truth, probability_pd)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": float(
            np.mean(np.sum((probabilities - targets) ** 2, axis=1))
        ),
        "ece10": ece10(truth, probability_pd),
        "nll": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(truth)), truth],
                    1e-12,
                    1.0,
                )
            ).mean()
        ),
        "selective_accuracy_80": selective_accuracy(
            truth, probability_pd, 0.8
        ),
        "aurc": aurc(truth, probability_pd),
        "mean_exact_route_rate": float(
            frame["fter_exact_route_rate"].mean()
        ),
    }


def condition_metrics(averaged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for condition in CONDITIONS:
            selected = averaged[
                (averaged["method"] == method)
                & (averaged["condition"] == condition)
            ]
            rows.append(
                {
                    "scope": "all_studies",
                    "study": "all",
                    "method": method,
                    "condition": condition,
                    **metric_values(selected),
                }
            )
            for study in TEST_STUDIES:
                study_frame = selected[selected["study"] == study]
                rows.append(
                    {
                        "scope": "held_out_study",
                        "study": study,
                        "method": method,
                        "condition": condition,
                        **metric_values(study_frame),
                    }
                )
    return pd.DataFrame(rows)


def seed_condition_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in sorted(predictions["seed"].unique()):
        seed_frame = predictions[predictions["seed"] == seed]
        for method in METHODS:
            for condition in CONDITIONS:
                selected = seed_frame[
                    (seed_frame["method"] == method)
                    & (seed_frame["condition"] == condition)
                ]
                rows.append(
                    {
                        "scope": "all_studies",
                        "study": "all",
                        "seed": int(seed),
                        "method": method,
                        "condition": condition,
                        **metric_values(selected),
                    }
                )
                for study in TEST_STUDIES:
                    study_frame = selected[selected["study"] == study]
                    rows.append(
                        {
                            "scope": "held_out_study",
                            "study": study,
                            "seed": int(seed),
                            "method": method,
                            "condition": condition,
                            **metric_values(study_frame),
                        }
                    )
    return pd.DataFrame(rows)


def study_robustness_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["scope"] == "held_out_study")
        & (metrics["condition"].isin(FAULT_CONDITIONS))
    ].copy()
    rows: list[dict[str, object]] = []
    for study in TEST_STUDIES:
        study_frame = selected[selected["study"] == study]
        method_values: dict[str, dict[str, float]] = {}
        for method in METHODS:
            method_frame = study_frame[study_frame["method"] == method]
            method_values[method] = {
                "fault_macro_f1_mean": float(
                    method_frame["macro_f1"].mean()
                ),
                "fault_macro_f1_worst": float(
                    method_frame["macro_f1"].min()
                ),
                "fault_balanced_accuracy_mean": float(
                    method_frame["balanced_accuracy"].mean()
                ),
                "fault_brier_mean": float(method_frame["brier"].mean()),
            }
        for endpoint in method_values["baseline"]:
            baseline = method_values["baseline"][endpoint]
            fasca = method_values["fasca"][endpoint]
            rows.append(
                {
                    "study": study,
                    "endpoint": endpoint,
                    "baseline": baseline,
                    "fasca": fasca,
                    "difference_fasca_minus_baseline": fasca - baseline,
                }
            )
    return pd.DataFrame(rows)


def study_method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["scope"] == "held_out_study"].copy()
    rows: list[dict[str, object]] = []
    for study in TEST_STUDIES:
        for method in METHODS:
            method_frame = selected[
                (selected["study"] == study)
                & (selected["method"] == method)
            ]
            fault_frame = method_frame[
                method_frame["condition"].isin(FAULT_CONDITIONS)
            ]
            clean = method_frame[
                method_frame["condition"] == "clean_both"
            ].iloc[0]
            rows.append(
                {
                    "study": study,
                    "method": method,
                    "subjects": int(clean["subjects"]),
                    "pd_subjects": int(clean["pd_subjects"]),
                    "control_subjects": int(clean["control_subjects"]),
                    "fault_macro_f1_mean": float(
                        fault_frame["macro_f1"].mean()
                    ),
                    "fault_macro_f1_worst": float(
                        fault_frame["macro_f1"].min()
                    ),
                    "fault_balanced_accuracy_mean": float(
                        fault_frame["balanced_accuracy"].mean()
                    ),
                    "fault_brier_mean": float(
                        fault_frame["brier"].mean()
                    ),
                    "clean_macro_f1": float(clean["macro_f1"]),
                    "clean_auroc": float(clean["auroc"]),
                }
            )
    return pd.DataFrame(rows)


def detector_trigger_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[
        (metrics["scope"] == "all_studies")
        & (metrics["method"] == "fasca")
    ][["condition", "mean_exact_route_rate"]].copy()
    selected = selected.rename(
        columns={"mean_exact_route_rate": "exact_zero_fter_trigger_rate"}
    )
    selected["detector_scope"] = (
        "exact-zero channel only; not a generic fault detector"
    )
    return selected


def binary_macro_f1_from_counts(
    true_positive: np.ndarray,
    true_negative: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
) -> np.ndarray:
    pd_precision_denominator = true_positive + false_positive
    pd_recall_denominator = true_positive + false_negative
    control_precision_denominator = true_negative + false_negative
    control_recall_denominator = true_negative + false_positive
    pd_precision = np.divide(
        true_positive,
        pd_precision_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=pd_precision_denominator > 0,
    )
    pd_recall = np.divide(
        true_positive,
        pd_recall_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=pd_recall_denominator > 0,
    )
    control_precision = np.divide(
        true_negative,
        control_precision_denominator,
        out=np.zeros_like(true_negative, dtype=float),
        where=control_precision_denominator > 0,
    )
    control_recall = np.divide(
        true_negative,
        control_recall_denominator,
        out=np.zeros_like(true_negative, dtype=float),
        where=control_recall_denominator > 0,
    )
    pd_f1 = np.divide(
        2 * pd_precision * pd_recall,
        pd_precision + pd_recall,
        out=np.zeros_like(pd_precision),
        where=(pd_precision + pd_recall) > 0,
    )
    control_f1 = np.divide(
        2 * control_precision * control_recall,
        control_precision + control_recall,
        out=np.zeros_like(control_precision),
        where=(control_precision + control_recall) > 0,
    )
    return (pd_f1 + control_f1) / 2


def macro_f1_bootstrap(
    truth: np.ndarray, predicted: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    selected_truth = truth[draws]
    selected_prediction = predicted[draws]
    tp = ((selected_truth == 1) & (selected_prediction == 1)).sum(axis=1)
    tn = ((selected_truth == 0) & (selected_prediction == 0)).sum(axis=1)
    fp = ((selected_truth == 0) & (selected_prediction == 1)).sum(axis=1)
    fn = ((selected_truth == 1) & (selected_prediction == 0)).sum(axis=1)
    return binary_macro_f1_from_counts(tp, tn, fp, fn)


def paired_bootstrap(
    averaged: pd.DataFrame,
    samples: int,
    seed: int,
    stratify_by_study: bool = False,
    comparator_method: str = "baseline",
    reference_method: str = "fasca",
) -> pd.DataFrame:
    subject_order = np.sort(averaged["subject"].unique())
    subject_metadata = (
        averaged[["subject", "truth", "study"]]
        .drop_duplicates()
        .set_index("subject")
        .loc[subject_order]
    )
    subject_truth = subject_metadata["truth"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    if stratify_by_study:
        study_draws = []
        study_values = subject_metadata["study"].to_numpy(dtype=str)
        for study in TEST_STUDIES:
            study_indices = np.flatnonzero(study_values == study)
            local_draws = rng.integers(
                0,
                len(study_indices),
                size=(samples, len(study_indices)),
            )
            study_draws.append(study_indices[local_draws])
        draws = np.concatenate(study_draws, axis=1)
        resampling_scheme = "subject_within_held_out_study"
    else:
        draws = rng.integers(
            0, len(subject_order), size=(samples, len(subject_order))
        )
        resampling_scheme = "subject_unstratified"
    predictions: dict[str, dict[str, np.ndarray]] = {}
    probability: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        predictions[method] = {}
        probability[method] = {}
        for condition in CONDITIONS:
            selected = (
                averaged[
                    (averaged["method"] == method)
                    & (averaged["condition"] == condition)
                ]
                .set_index("subject")
                .loc[subject_order]
            )
            probability[method][condition] = selected["p_pd"].to_numpy()
            predictions[method][condition] = (
                selected["p_pd"].to_numpy() >= 0.5
            ).astype(int)

    metric_bootstrap: dict[str, dict[str, np.ndarray]] = {
        method: {} for method in METHODS
    }
    point_values: dict[str, dict[str, float]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        fault_f1 = []
        fault_balanced = []
        for condition in FAULT_CONDITIONS:
            boot_f1 = macro_f1_bootstrap(
                subject_truth, predictions[method][condition], draws
            )
            fault_f1.append(boot_f1)
            selected_truth = subject_truth[draws]
            selected_prediction = predictions[method][condition][draws]
            sensitivity = np.divide(
                (
                    (selected_truth == 1) & (selected_prediction == 1)
                ).sum(axis=1),
                (selected_truth == 1).sum(axis=1),
            )
            specificity = np.divide(
                (
                    (selected_truth == 0) & (selected_prediction == 0)
                ).sum(axis=1),
                (selected_truth == 0).sum(axis=1),
            )
            fault_balanced.append((sensitivity + specificity) / 2)
        fault_f1_array = np.stack(fault_f1, axis=1)
        metric_bootstrap[method]["fault_macro_f1_mean"] = fault_f1_array.mean(
            axis=1
        )
        metric_bootstrap[method]["fault_macro_f1_worst"] = fault_f1_array.min(
            axis=1
        )
        metric_bootstrap[method][
            "fault_balanced_accuracy_mean"
        ] = np.stack(fault_balanced, axis=1).mean(axis=1)
        metric_bootstrap[method]["clean_macro_f1"] = macro_f1_bootstrap(
            subject_truth, predictions[method]["clean_both"], draws
        )
        target = subject_truth[:, None]
        fault_probability = np.stack(
            [probability[method][condition] for condition in FAULT_CONDITIONS],
            axis=1,
        )
        fault_brier = 2.0 * (fault_probability - target) ** 2
        metric_bootstrap[method]["fault_brier_mean"] = fault_brier[
            draws
        ].mean(axis=(1, 2))
        point_f1 = []
        point_balanced = []
        for condition in FAULT_CONDITIONS:
            predicted = predictions[method][condition]
            point_f1.append(
                f1_score(
                    subject_truth,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            )
            point_balanced.append(
                balanced_accuracy_score(subject_truth, predicted)
            )
        point_values[method]["fault_macro_f1_mean"] = float(
            np.mean(point_f1)
        )
        point_values[method]["fault_macro_f1_worst"] = float(
            np.min(point_f1)
        )
        point_values[method]["fault_balanced_accuracy_mean"] = float(
            np.mean(point_balanced)
        )
        point_values[method]["clean_macro_f1"] = float(
            f1_score(
                subject_truth,
                predictions[method]["clean_both"],
                average="macro",
                zero_division=0,
            )
        )
        point_values[method]["fault_brier_mean"] = float(fault_brier.mean())

    rows: list[dict[str, object]] = []
    if comparator_method not in point_values:
        raise KeyError(f"Unknown comparator method: {comparator_method}")
    if reference_method not in point_values:
        raise KeyError(f"Unknown reference method: {reference_method}")
    for metric in point_values[comparator_method]:
        comparator = metric_bootstrap[comparator_method][metric]
        reference = metric_bootstrap[reference_method][metric]
        difference = reference - comparator
        row = {
            "metric": metric,
            "comparator": comparator_method,
            "reference": reference_method,
            "comparator_value": point_values[comparator_method][metric],
            "reference_value": point_values[reference_method][metric],
            "difference_reference_minus_comparator": (
                point_values[reference_method][metric]
                - point_values[comparator_method][metric]
            ),
            "difference_ci_low": float(np.quantile(difference, 0.025)),
            "difference_ci_high": float(np.quantile(difference, 0.975)),
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "resampling_scheme": resampling_scheme,
            "subjects": len(subject_order),
        }
        if comparator_method == "baseline" and reference_method == "fasca":
            row.update(
                {
                    "baseline": point_values["baseline"][metric],
                    "fasca": point_values["fasca"][metric],
                    "difference_fasca_minus_baseline": (
                        point_values["fasca"][metric]
                        - point_values["baseline"][metric]
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def all_method_contrasts(
    averaged: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    frames = []
    for comparator in METHODS:
        if comparator == "fasca":
            continue
        frames.append(
            paired_bootstrap(
                averaged,
                samples,
                seed,
                comparator_method=comparator,
                reference_method="fasca",
            )
        )
    return pd.concat(frames, ignore_index=True)


def study_method_contrasts(
    averaged: pd.DataFrame, samples: int, seed: int
) -> pd.DataFrame:
    frames = []
    for study_index, study in enumerate(TEST_STUDIES):
        study_frame = averaged[averaged["study"] == study]
        for comparator_index, comparator in enumerate(METHODS):
            if comparator == "fasca":
                continue
            contrast = paired_bootstrap(
                study_frame,
                samples,
                seed + 100 * study_index + comparator_index,
                comparator_method=comparator,
                reference_method="fasca",
            )
            contrast.insert(0, "study", study)
            frames.append(contrast)
    return pd.concat(frames, ignore_index=True)


def seed_endpoint_summary(
    seed_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = seed_metrics[seed_metrics["scope"] == "all_studies"]
    rows: list[dict[str, object]] = []
    for seed in sorted(selected["seed"].unique()):
        for method in METHODS:
            method_frame = selected[
                (selected["seed"] == seed)
                & (selected["method"] == method)
            ]
            fault_frame = method_frame[
                method_frame["condition"].isin(FAULT_CONDITIONS)
            ]
            clean_frame = method_frame[
                method_frame["condition"] == "clean_both"
            ]
            rows.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "fault_macro_f1_mean": float(
                        fault_frame["macro_f1"].mean()
                    ),
                    "fault_macro_f1_worst": float(
                        fault_frame["macro_f1"].min()
                    ),
                    "fault_balanced_accuracy_mean": float(
                        fault_frame["balanced_accuracy"].mean()
                    ),
                    "fault_brier_mean": float(
                        fault_frame["brier"].mean()
                    ),
                    "clean_macro_f1": float(
                        clean_frame["macro_f1"].iloc[0]
                    ),
                }
            )
    endpoints = pd.DataFrame(rows)
    contrasts: list[dict[str, object]] = []
    endpoint_names = [
        column
        for column in endpoints.columns
        if column not in {"seed", "method"}
    ]
    for comparator in METHODS:
        if comparator == "fasca":
            continue
        comparator_frame = endpoints[
            endpoints["method"] == comparator
        ].set_index("seed")
        fasca_frame = endpoints[endpoints["method"] == "fasca"].set_index(
            "seed"
        )
        for endpoint in endpoint_names:
            difference = (
                fasca_frame[endpoint] - comparator_frame[endpoint]
            ).to_numpy(dtype=float)
            if np.allclose(difference, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(
                        difference,
                        zero_method="pratt",
                        alternative="two-sided",
                    ).pvalue
                )
            contrasts.append(
                {
                    "comparator": comparator,
                    "reference": "fasca",
                    "endpoint": endpoint,
                    "seeds": len(difference),
                    "mean_difference": float(difference.mean()),
                    "sd_difference": float(
                        difference.std(ddof=1)
                        if len(difference) > 1
                        else 0.0
                    ),
                    "median_difference": float(np.median(difference)),
                    "minimum_difference": float(difference.min()),
                    "maximum_difference": float(difference.max()),
                    "positive_seed_count": int((difference > 0).sum()),
                    "negative_seed_count": int((difference < 0).sum()),
                    "wilcoxon_p_descriptive": p_value,
                }
            )
    return endpoints, pd.DataFrame(contrasts)


def ensemble_diagnostics(
    predictions: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ensemble_all = ensemble_metrics[
        ensemble_metrics["scope"] == "all_studies"
    ]
    seed_all = seed_metrics[seed_metrics["scope"] == "all_studies"]
    for method in METHODS:
        method_predictions = predictions[
            predictions["method"] == method
        ]
        for group_name, conditions in (
            ("clean", ("clean_both",)),
            ("fault_mean", FAULT_CONDITIONS),
        ):
            ensemble_selected = ensemble_all[
                (ensemble_all["method"] == method)
                & (ensemble_all["condition"].isin(conditions))
            ]
            seed_selected = seed_all[
                (seed_all["method"] == method)
                & (seed_all["condition"].isin(conditions))
            ]
            pivot = method_predictions[
                method_predictions["condition"].isin(conditions)
            ].pivot_table(
                index=["condition", "subject"],
                columns="seed",
                values="p_pd",
            )
            decisions = (pivot.to_numpy() >= 0.5).astype(int)
            pair_disagreement = []
            for left in range(decisions.shape[1]):
                for right in range(left + 1, decisions.shape[1]):
                    pair_disagreement.append(
                        float(
                            (
                                decisions[:, left]
                                != decisions[:, right]
                            ).mean()
                        )
                    )
            rows.append(
                {
                    "method": method,
                    "condition_group": group_name,
                    "seeds": int(pivot.shape[1]),
                    "ensemble_macro_f1": float(
                        ensemble_selected["macro_f1"].mean()
                    ),
                    "mean_seed_macro_f1": float(
                        seed_selected["macro_f1"].mean()
                    ),
                    "ensemble_minus_mean_seed_macro_f1": float(
                        ensemble_selected["macro_f1"].mean()
                        - seed_selected["macro_f1"].mean()
                    ),
                    "ensemble_brier": float(
                        ensemble_selected["brier"].mean()
                    ),
                    "mean_seed_brier": float(
                        seed_selected["brier"].mean()
                    ),
                    "ensemble_minus_mean_seed_brier": float(
                        ensemble_selected["brier"].mean()
                        - seed_selected["brier"].mean()
                    ),
                    "mean_pairwise_seed_decision_disagreement": float(
                        np.mean(pair_disagreement)
                        if pair_disagreement
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def subject_additive_tests(averaged: pd.DataFrame) -> pd.DataFrame:
    subject_order = np.sort(averaged["subject"].unique())
    truth = (
        averaged[["subject", "truth"]]
        .drop_duplicates()
        .set_index("subject")
        .loc[subject_order, "truth"]
        .to_numpy(dtype=int)
    )
    values: dict[str, dict[str, np.ndarray]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        condition_probability = {}
        for condition in CONDITIONS:
            condition_probability[condition] = (
                averaged[
                    (averaged["method"] == method)
                    & (averaged["condition"] == condition)
                ]
                .set_index("subject")
                .loc[subject_order, "p_pd"]
                .to_numpy()
            )
        fault_probability = np.stack(
            [
                condition_probability[condition]
                for condition in FAULT_CONDITIONS
            ],
            axis=1,
        )
        fault_prediction = fault_probability >= 0.5
        values[method]["fault_correctness"] = (
            fault_prediction == truth[:, None]
        ).mean(axis=1)
        values[method]["fault_brier"] = (
            2.0 * (fault_probability - truth[:, None]) ** 2
        ).mean(axis=1)
        values[method]["fault_nll"] = (
            -truth[:, None]
            * np.log(np.clip(fault_probability, 1e-12, 1.0))
            - (1 - truth[:, None])
            * np.log(np.clip(1 - fault_probability, 1e-12, 1.0))
        ).mean(axis=1)
        clean_probability = condition_probability["clean_both"]
        values[method]["clean_correctness"] = (
            (clean_probability >= 0.5).astype(int) == truth
        ).astype(float)
    rows = []
    for endpoint in values["baseline"]:
        baseline = values["baseline"][endpoint]
        fasca = values["fasca"][endpoint]
        if np.allclose(baseline, fasca):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(
                    fasca,
                    baseline,
                    zero_method="pratt",
                    alternative="two-sided",
                ).pvalue
            )
        rows.append(
            {
                "endpoint": endpoint,
                "baseline_mean": float(baseline.mean()),
                "fasca_mean": float(fasca.mean()),
                "difference": float((fasca - baseline).mean()),
                "wilcoxon_p": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame["holm_p"] = holm_adjust(frame["wilcoxon_p"].to_numpy())
    return frame


def subgroup_metrics(averaged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for condition_group, conditions in (
            ("clean", ("clean_both",)),
            ("fault_mean", FAULT_CONDITIONS),
        ):
            for gender in sorted(averaged["gender"].unique()):
                scores = []
                for condition in conditions:
                    selected = averaged[
                        (averaged["method"] == method)
                        & (averaged["condition"] == condition)
                        & (averaged["gender"] == gender)
                    ]
                    scores.append(metric_values(selected)["macro_f1"])
                rows.append(
                    {
                        "method": method,
                        "condition_group": condition_group,
                        "subgroup_type": "gender",
                        "subgroup": GENDER_LABEL.get(
                            int(gender), f"code_{int(gender)}"
                        ),
                        "subjects": int(
                            averaged[averaged["gender"] == gender][
                                "subject"
                            ].nunique()
                        ),
                        "metric": "macro_f1",
                        "value": float(np.mean(scores)),
                    }
                )
            pd_rows = averaged[
                (averaged["method"] == method) & (averaged["truth"] == 1)
            ].copy()
            pd_rows["stage_group"] = np.where(
                pd_rows["hoehen_yahr"] <= 2.0,
                "<=2",
                ">2",
            )
            pd_rows.loc[pd_rows["hoehen_yahr"].isna(), "stage_group"] = "missing"
            for stage_group in ("<=2", ">2"):
                sensitivities = []
                for condition in conditions:
                    selected = pd_rows[
                        (pd_rows["condition"] == condition)
                        & (pd_rows["stage_group"] == stage_group)
                    ]
                    sensitivities.append(
                        float((selected["p_pd"] >= 0.5).mean())
                    )
                rows.append(
                    {
                        "method": method,
                        "condition_group": condition_group,
                        "subgroup_type": "hoehen_yahr",
                        "subgroup": stage_group,
                        "subjects": int(
                            pd_rows[pd_rows["stage_group"] == stage_group][
                                "subject"
                            ].nunique()
                        ),
                        "metric": "sensitivity",
                        "value": float(np.mean(sensitivities)),
                    }
                )
    return pd.DataFrame(rows)


def plot_condition_performance(
    metrics: pd.DataFrame, output_path: Path
) -> None:
    selected = metrics[metrics["scope"] == "all_studies"].copy()
    label_map = {
        "clean_both": "Clean",
        "left_only": "Left only",
        "right_only": "Right only",
        "sensor_dead_bilateral": "Dead sensor",
        "near_dead_bilateral": "Near-dead",
        "contact_loss_bilateral": "Contact loss",
        "saturation_bilateral": "Saturation",
        "drift_bilateral": "Drift",
        "packet_loss_bilateral": "Packet loss",
        "gain_miscalibration": "Gain",
    }
    x = np.arange(len(CONDITIONS))
    width = 0.82 / len(METHODS)
    fig, axis = plt.subplots(figsize=(12.2, 4.8))
    colors = {
        "baseline": "#577590",
        "generic": "#43AA8B",
        "iid": "#9C89B8",
        "fasca": "#F28E2B",
    }
    for method_index, method in enumerate(METHODS):
        method_frame = (
            selected[selected["method"] == method]
            .set_index("condition")
            .loc[list(CONDITIONS)]
        )
        offset = (
            method_index - (len(METHODS) - 1) / 2
        ) * width
        axis.bar(
            x + offset,
            method_frame["macro_f1"] * 100,
            width=width,
            label=DISPLAY_METHOD[method],
            color=colors[method],
        )
    axis.set_ylabel("Subject-level macro-F1 (%)")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [label_map[condition] for condition in CONDITIONS],
        rotation=28,
        ha="right",
    )
    axis.set_ylim(40, 100)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=min(4, len(METHODS)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global METHODS
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    present_methods = set(predictions["method"].astype(str))
    preferred_order = ("baseline", "generic", "iid", "fasca")
    METHODS = tuple(
        method for method in preferred_order if method in present_methods
    )
    unknown_methods = present_methods.difference(METHODS)
    if unknown_methods:
        raise RuntimeError(
            f"Unexpected method values: {sorted(unknown_methods)}"
        )
    if not {"baseline", "fasca"}.issubset(present_methods):
        raise RuntimeError("Analysis requires baseline and fasca methods")
    metadata = load_subject_metadata(args.data)
    cohort = cohort_summary(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_dir / "cohort_summary.csv", index=False)
    averaged = validate_and_average(predictions, metadata)
    averaged.to_csv(
        args.output_dir / "seed_averaged_subject_predictions.csv",
        index=False,
    )
    metrics = condition_metrics(averaged)
    metrics.to_csv(args.output_dir / "condition_metrics.csv", index=False)
    per_seed_metrics = seed_condition_metrics(predictions)
    per_seed_metrics.to_csv(
        args.output_dir / "seed_condition_metrics.csv", index=False
    )
    study_summary = study_robustness_summary(metrics)
    study_summary.to_csv(
        args.output_dir / "study_robustness_summary.csv", index=False
    )
    study_methods = study_method_summary(metrics)
    study_methods.to_csv(
        args.output_dir / "study_method_summary.csv", index=False
    )
    detector_summary = detector_trigger_summary(metrics)
    detector_summary.to_csv(
        args.output_dir / "detector_trigger_summary.csv", index=False
    )
    bootstrap = paired_bootstrap(
        averaged, args.bootstrap_samples, args.bootstrap_seed
    )
    bootstrap.to_csv(
        args.output_dir / "paired_bootstrap.csv", index=False
    )
    method_contrasts = all_method_contrasts(
        averaged, args.bootstrap_samples, args.bootstrap_seed
    )
    method_contrasts.to_csv(
        args.output_dir / "method_contrasts.csv", index=False
    )
    study_contrasts = study_method_contrasts(
        averaged, args.bootstrap_samples, args.bootstrap_seed + 10_000
    )
    study_contrasts.to_csv(
        args.output_dir / "study_method_contrasts.csv", index=False
    )
    stratified_bootstrap = paired_bootstrap(
        averaged,
        args.bootstrap_samples,
        args.bootstrap_seed + 1,
        stratify_by_study=True,
    )
    stratified_bootstrap.to_csv(
        args.output_dir / "paired_bootstrap_study_stratified_sensitivity.csv",
        index=False,
    )
    additive = subject_additive_tests(averaged)
    additive.to_csv(
        args.output_dir / "subject_additive_tests.csv", index=False
    )
    subgroup = subgroup_metrics(averaged)
    subgroup.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    seed_endpoints, seed_contrasts = seed_endpoint_summary(per_seed_metrics)
    seed_endpoints.to_csv(
        args.output_dir / "seed_endpoint_summary.csv", index=False
    )
    seed_contrasts.to_csv(
        args.output_dir / "seed_contrast_summary.csv", index=False
    )
    ensemble = ensemble_diagnostics(
        predictions, metrics, per_seed_metrics
    )
    ensemble.to_csv(
        args.output_dir / "ensemble_diagnostics.csv", index=False
    )
    plot_condition_performance(
        metrics, args.output_dir / "clinical_condition_performance.pdf"
    )
    plot_condition_performance(
        metrics, args.output_dir / "clinical_condition_performance.png"
    )
    summary = {
        "protocol": (
            "docs/gaitpdb_jmbe_extension_protocol.md"
            if len(METHODS) > 2
            else "docs/gaitpdb_clinical_external_protocol.md"
        ),
        "subjects": int(metadata["subject"].nunique()),
        "studies": TEST_STUDIES,
        "methods": METHODS,
        "conditions": CONDITIONS,
        "fault_conditions": FAULT_CONDITIONS,
        "probability_aggregation": (
            "mean across windows within subject during evaluation, then mean "
            "across optimization seeds for analysis"
        ),
        "training_stability": (
            "seed_condition_metrics.csv reports performance before "
            "optimization-seed probability averaging"
        ),
        "study_summary_note": (
            "Study-specific estimates are descriptive and expose "
            "cross-study heterogeneity; no study-level significance claim is "
            "made."
        ),
        "detector_scope": (
            "The exact-zero FTER detector is expected to trigger only for "
            "the locked exact-dead condition."
        ),
        "bootstrap": {
            "unit": "subject",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "interval": "percentile 95%",
        },
        "post_hoc_sensitivity_bootstrap": {
            "unit": "subject within held-out study",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed + 1,
            "interval": "percentile 95%",
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
