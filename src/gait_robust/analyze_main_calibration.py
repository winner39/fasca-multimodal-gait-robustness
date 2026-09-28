from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["svg.fonttype"] = "none"

PROBABILITY_COLUMNS = ["p_0.5", "p_0.75", "p_1"]
FULL_MODALITIES = "eeg+emg+imu+fp"
METHODS = {
    "EmbraceNet": {
        "relative_root": Path("multimodel_confirmatory/main"),
        "directory": "embracenet",
    },
    "EmbraceNet+FASCA": {
        "relative_root": Path("fasca_factorial_confirmatory"),
        "directory": "embracenet_fasca_kd",
    },
}
ENDPOINTS = {
    "full": "Complete input",
    "all_15": "All 15 availability patterns",
}
METRICS = ("brier", "ece10", "nll")


@dataclass(frozen=True)
class SubjectSufficientStatistics:
    subjects: np.ndarray
    sample_count: np.ndarray
    brier_sum: np.ndarray
    nll_sum: np.ndarray
    bin_count: np.ndarray
    bin_correct_sum: np.ndarray
    bin_confidence_sum: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze trial-level calibration for EmbraceNet and "
            "EmbraceNet+FASCA on the primary five-fold test predictions."
        )
    )
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=project_root / "artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts" / "main_calibration",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260704)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def prediction_paths(
    artifact_root: Path,
    relative_root: Path,
    method_directory: str,
) -> list[Path]:
    root = artifact_root / relative_root
    paths = sorted(
        root.glob(
            f"fold_*/seed_*/{method_directory}/window_predictions.csv.gz"
        )
    )
    if len(paths) != 15:
        raise RuntimeError(
            f"Expected 15 prediction files under {root}, found {len(paths)}"
        )
    return paths


def load_trial_predictions(
    artifact_root: Path,
    display_name: str,
    relative_root: Path,
    method_directory: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in prediction_paths(
        artifact_root, relative_root, method_directory
    ):
        frame = pd.read_csv(path)
        required = {
            "seed",
            "fold",
            "available_modalities",
            "subject",
            "trial_id",
            "true_class",
            *PROBABILITY_COLUMNS,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame[list(required)])
    windows = pd.concat(frames, ignore_index=True)
    group_columns = [
        "seed",
        "fold",
        "available_modalities",
        "subject",
        "trial_id",
    ]
    trials = (
        windows.groupby(group_columns, sort=False)
        .agg(
            true_class=("true_class", "first"),
            **{
                column: (column, "mean") for column in PROBABILITY_COLUMNS
            },
        )
        .reset_index()
    )
    trials.insert(0, "method", display_name)
    probability_sum = trials[PROBABILITY_COLUMNS].sum(axis=1).to_numpy()
    if not np.allclose(probability_sum, 1.0, atol=1e-5):
        raise RuntimeError(f"{display_name} probabilities do not sum to one")
    expected_rows = 3 * 157 * 15
    if len(trials) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} trial-mask rows for {display_name}, "
            f"found {len(trials)}"
        )
    if trials["subject"].nunique() != 55:
        raise RuntimeError(
            f"Expected 55 subjects for {display_name}, "
            f"found {trials['subject'].nunique()}"
        )
    return trials


def endpoint_frame(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    if endpoint == "full":
        return frame[frame["available_modalities"] == FULL_MODALITIES].copy()
    if endpoint == "all_15":
        return frame.copy()
    raise KeyError(endpoint)


def calibration_values(
    frame: pd.DataFrame, bins: int
) -> dict[str, float | int]:
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    truth = frame["true_class"].to_numpy(dtype=int)
    targets = np.eye(len(PROBABILITY_COLUMNS), dtype=float)[truth]
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == truth).astype(float)
    bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = 0.0
    for current_bin in range(bins):
        selected = bin_id == current_bin
        if not selected.any():
            continue
        ece += selected.mean() * abs(
            correct[selected].mean() - confidence[selected].mean()
        )
    return {
        "trials": int(len(frame)),
        "subjects": int(frame["subject"].nunique()),
        "brier": float(
            np.mean(np.sum((probabilities - targets) ** 2, axis=1))
        ),
        "ece10": float(ece),
        "nll": float(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(truth)), truth],
                    1e-12,
                    1.0,
                )
            ).mean()
        ),
    }


def average_probabilities_across_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        "method",
        "fold",
        "available_modalities",
        "subject",
        "trial_id",
    ]
    averaged = (
        frame.groupby(identity_columns, sort=False)
        .agg(
            true_class=("true_class", "first"),
            seed_count=("seed", "nunique"),
            **{
                column: (column, "mean") for column in PROBABILITY_COLUMNS
            },
        )
        .reset_index()
    )
    if not (averaged["seed_count"] == 3).all():
        raise RuntimeError("Some trial-mask rows do not contain all three seeds")
    return averaged.drop(columns="seed_count")


def subject_statistics(
    frame: pd.DataFrame, bins: int
) -> SubjectSufficientStatistics:
    subjects = np.sort(frame["subject"].unique())
    sample_count = np.zeros(len(subjects), dtype=float)
    brier_sum = np.zeros(len(subjects), dtype=float)
    nll_sum = np.zeros(len(subjects), dtype=float)
    bin_count = np.zeros((len(subjects), bins), dtype=float)
    bin_correct_sum = np.zeros_like(bin_count)
    bin_confidence_sum = np.zeros_like(bin_count)
    for subject_index, subject in enumerate(subjects):
        selected = frame[frame["subject"] == subject]
        probabilities = selected[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        truth = selected["true_class"].to_numpy(dtype=int)
        targets = np.eye(len(PROBABILITY_COLUMNS), dtype=float)[truth]
        confidence = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        correct = (predicted == truth).astype(float)
        bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
        sample_count[subject_index] = len(selected)
        brier_sum[subject_index] = np.sum(
            np.sum((probabilities - targets) ** 2, axis=1)
        )
        nll_sum[subject_index] = np.sum(
            -np.log(
                np.clip(
                    probabilities[np.arange(len(truth)), truth],
                    1e-12,
                    1.0,
                )
            )
        )
        for current_bin in range(bins):
            in_bin = bin_id == current_bin
            bin_count[subject_index, current_bin] = in_bin.sum()
            bin_correct_sum[subject_index, current_bin] = correct[in_bin].sum()
            bin_confidence_sum[
                subject_index, current_bin
            ] = confidence[in_bin].sum()
    return SubjectSufficientStatistics(
        subjects=subjects,
        sample_count=sample_count,
        brier_sum=brier_sum,
        nll_sum=nll_sum,
        bin_count=bin_count,
        bin_correct_sum=bin_correct_sum,
        bin_confidence_sum=bin_confidence_sum,
    )


def metrics_from_bootstrap_counts(
    statistics: SubjectSufficientStatistics,
    draws: np.ndarray,
) -> dict[str, np.ndarray]:
    total_count = statistics.sample_count[draws].sum(axis=1)
    brier = statistics.brier_sum[draws].sum(axis=1) / total_count
    nll = statistics.nll_sum[draws].sum(axis=1) / total_count
    bin_count = statistics.bin_count[draws].sum(axis=1)
    bin_correct = statistics.bin_correct_sum[draws].sum(axis=1)
    bin_confidence = statistics.bin_confidence_sum[draws].sum(axis=1)
    accuracy = np.divide(
        bin_correct,
        bin_count,
        out=np.zeros_like(bin_correct),
        where=bin_count > 0,
    )
    confidence = np.divide(
        bin_confidence,
        bin_count,
        out=np.zeros_like(bin_confidence),
        where=bin_count > 0,
    )
    ece = (
        bin_count
        * np.abs(accuracy - confidence)
        / total_count[:, np.newaxis]
    ).sum(axis=1)
    return {"brier": brier, "ece10": ece, "nll": nll}


def reliability_rows(
    frame: pd.DataFrame,
    endpoint: str,
    method: str,
    bins: int,
) -> list[dict[str, float | int | str]]:
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    truth = frame["true_class"].to_numpy(dtype=int)
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == truth).astype(float)
    bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
    rows: list[dict[str, float | int | str]] = []
    for current_bin in range(bins):
        selected = bin_id == current_bin
        count = int(selected.sum())
        rows.append(
            {
                "endpoint": endpoint,
                "method": method,
                "bin": current_bin + 1,
                "lower": current_bin / bins,
                "upper": (current_bin + 1) / bins,
                "count": count,
                "mean_confidence": (
                    float(confidence[selected].mean()) if count else np.nan
                ),
                "accuracy": (
                    float(correct[selected].mean()) if count else np.nan
                ),
            }
        )
    return rows


def plot_reliability(
    reliability: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True, sharey=True)
    colors = {
        "EmbraceNet": "#577590",
        "EmbraceNet+FASCA": "#F28E2B",
    }
    styles = {
        "EmbraceNet": {"linestyle": "-", "marker": "o"},
        "EmbraceNet+FASCA": {"linestyle": "--", "marker": "s"},
    }
    for axis, endpoint in zip(axes, ENDPOINTS, strict=True):
        axis.plot(
            [0, 1],
            [0, 1],
            linestyle=":",
            color="#777777",
            linewidth=1,
            label="Perfect calibration",
        )
        selected_endpoint = reliability[
            reliability["endpoint"] == endpoint
        ]
        for method in METHODS:
            selected = selected_endpoint[
                selected_endpoint["method"] == method
            ].dropna(subset=["mean_confidence", "accuracy"])
            axis.plot(
                selected["mean_confidence"],
                selected["accuracy"],
                linewidth=1.8,
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
                label=method,
                color=colors[method],
                **styles[method],
            )
        axis.set_title(ENDPOINTS[endpoint], fontsize=10)
        axis.set_xlabel("Mean confidence")
        axis.grid(alpha=0.2)
        axis.set_xlim(0.35, 1.01)
        axis.set_ylim(0.35, 1.01)
    axes[0].set_ylabel("Empirical accuracy")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bins != 10:
        raise ValueError(
            "This analysis names the endpoint ece10 and therefore requires "
            "--bins 10."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_frames: dict[str, pd.DataFrame] = {}
    for method, config in METHODS.items():
        method_frames[method] = load_trial_predictions(
            artifact_root=args.artifact_root,
            display_name=method,
            relative_root=config["relative_root"],
            method_directory=config["directory"],
        )

    seed_rows: list[dict[str, float | int | str]] = []
    averaged_frames: dict[str, pd.DataFrame] = {}
    for method, frame in method_frames.items():
        for seed in sorted(frame["seed"].unique()):
            seed_frame = frame[frame["seed"] == seed]
            for endpoint in ENDPOINTS:
                values = calibration_values(
                    endpoint_frame(seed_frame, endpoint), args.bins
                )
                seed_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "endpoint": endpoint,
                        **values,
                    }
                )
        averaged_frames[method] = average_probabilities_across_seeds(frame)
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(args.output_dir / "seed_metrics.csv", index=False)

    seed_summary = (
        seed_metrics.groupby(["method", "endpoint"], sort=False)
        .agg(
            seeds=("seed", "nunique"),
            brier_mean=("brier", "mean"),
            brier_sd=("brier", "std"),
            ece10_mean=("ece10", "mean"),
            ece10_sd=("ece10", "std"),
            nll_mean=("nll", "mean"),
            nll_sd=("nll", "std"),
        )
        .reset_index()
    )
    seed_summary.to_csv(args.output_dir / "seed_summary.csv", index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows: list[dict[str, float | int | str]] = []
    reliability: list[dict[str, float | int | str]] = []
    for endpoint in ENDPOINTS:
        endpoint_frames = {
            method: endpoint_frame(frame, endpoint)
            for method, frame in averaged_frames.items()
        }
        identities = {}
        for method, frame in endpoint_frames.items():
            identities[method] = frame[
                [
                    "fold",
                    "available_modalities",
                    "subject",
                    "trial_id",
                    "true_class",
                ]
            ].sort_values(
                ["fold", "available_modalities", "subject", "trial_id"]
            )
        first_method, second_method = METHODS
        if not identities[first_method].reset_index(drop=True).equals(
            identities[second_method].reset_index(drop=True)
        ):
            raise RuntimeError(
                f"Prediction identities do not match for endpoint {endpoint}"
            )
        statistics = {
            method: subject_statistics(frame, args.bins)
            for method, frame in endpoint_frames.items()
        }
        if not np.array_equal(
            statistics[first_method].subjects,
            statistics[second_method].subjects,
        ):
            raise RuntimeError("Subject identities differ between methods")
        subject_count = len(statistics[first_method].subjects)
        draws = rng.integers(
            0,
            subject_count,
            size=(args.bootstrap_samples, subject_count),
        )
        bootstrap_metrics = {
            method: metrics_from_bootstrap_counts(current, draws)
            for method, current in statistics.items()
        }
        point_metrics = {
            method: calibration_values(frame, args.bins)
            for method, frame in endpoint_frames.items()
        }
        for metric in METRICS:
            baseline_values = bootstrap_metrics[first_method][metric]
            fasca_values = bootstrap_metrics[second_method][metric]
            difference = fasca_values - baseline_values
            bootstrap_rows.append(
                {
                    "endpoint": endpoint,
                    "metric": metric,
                    "baseline": first_method,
                    "method": second_method,
                    "baseline_value": point_metrics[first_method][metric],
                    "method_value": point_metrics[second_method][metric],
                    "difference_method_minus_baseline": (
                        point_metrics[second_method][metric]
                        - point_metrics[first_method][metric]
                    ),
                    "baseline_ci_low": np.quantile(
                        baseline_values, 0.025
                    ),
                    "baseline_ci_high": np.quantile(
                        baseline_values, 0.975
                    ),
                    "method_ci_low": np.quantile(fasca_values, 0.025),
                    "method_ci_high": np.quantile(fasca_values, 0.975),
                    "difference_ci_low": np.quantile(difference, 0.025),
                    "difference_ci_high": np.quantile(difference, 0.975),
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_seed": args.bootstrap_seed,
                    "subjects": subject_count,
                    "trial_mask_rows": len(endpoint_frames[first_method]),
                }
            )
        for method, frame in endpoint_frames.items():
            reliability.extend(
                reliability_rows(frame, endpoint, method, args.bins)
            )

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(
        args.output_dir / "seed_averaged_bootstrap_metrics.csv",
        index=False,
    )
    reliability_frame = pd.DataFrame(reliability)
    reliability_frame.to_csv(
        args.output_dir / "reliability_bins.csv", index=False
    )
    plot_reliability(
        reliability_frame, args.output_dir / "reliability_diagram.pdf"
    )
    plot_reliability(
        reliability_frame, args.output_dir / "reliability_diagram.png"
    )
    plot_reliability(
        reliability_frame, args.output_dir / "reliability_diagram.svg"
    )

    summary = {
        "analysis": "primary trial-level calibration",
        "probability_aggregation": (
            "mean across windows within trial, then mean across three seeds "
            "for bootstrap analysis"
        ),
        "endpoints": ENDPOINTS,
        "metrics": {
            "brier": (
                "multiclass Brier score: mean sum of squared probability "
                "errors across classes"
            ),
            "ece10": (
                "top-label expected calibration error with 10 equal-width "
                "confidence bins"
            ),
            "nll": "mean negative log likelihood of the true class",
        },
        "bootstrap": {
            "unit": "subject",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "interval": "percentile 95%",
        },
        "source_roots": {
            method: str(args.artifact_root / config["relative_root"])
            for method, config in METHODS.items()
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
