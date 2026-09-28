from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score

from gait_robust.cross_validate_full import subject_folds
from gait_robust.data import FoldRobustScaler
from gait_robust.robust_models import EmbraceNetLite
from gait_robust.sci_cross_validate import (
    MODALITIES,
    SPEEDS,
    load_arrays,
    move_inputs,
)


OBSERVED_EEG_INPUT_CHANNELS = (
    "Fp2",
    "F4",
    "Cz",
    "T5",
    "P4",
    "O1",
    "O2",
)
REPLAY_SCENARIOS = {
    "observed_schema_zero": 0.0,
    "observed_schema_1pct": 0.01,
    "observed_schema_5pct": 0.05,
    "observed_schema_10pct": 0.10,
    "observed_eeg_rejected": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete-data", type=Path, required=True)
    parser.add_argument("--observed-data", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--fasca-root", type=Path, required=True)
    parser.add_argument("--specialist-seed51-root", type=Path, required=True)
    parser.add_argument("--specialist-confirm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53])
    parser.add_argument("--partition-seed", type=int, default=20260901)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


class ModelStore:
    def __init__(
        self,
        args: argparse.Namespace,
        channels: dict[str, int],
        device: torch.device,
    ):
        self.args = args
        self.channels = channels
        self.device = device
        self.cache: dict[tuple[str, int, int], EmbraceNetLite] = {}

    def _location(self, method: str, fold: int, seed: int) -> Path:
        if method == "embracenet":
            return (
                self.args.baseline_root
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "embracenet"
                / "best_model.pt"
            )
        if method == "fasca":
            return (
                self.args.fasca_root
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "embracenet_fasca_kd"
                / "best_model.pt"
            )
        if method == "specialist":
            root = (
                self.args.specialist_seed51_root
                if seed == 51
                else self.args.specialist_confirm_root
            )
            return (
                root
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "embracenet_fasca_qstructssl_mar_kd"
                / "best_model.pt"
            )
        raise ValueError(method)

    def get(self, method: str, fold: int, seed: int) -> EmbraceNetLite:
        key = (method, fold, seed)
        if key not in self.cache:
            checkpoint = self._location(method, fold, seed)
            payload = torch.load(
                checkpoint, map_location=self.device, weights_only=True
            )
            model = EmbraceNetLite(
                channels=self.channels, classes=len(SPEEDS)
            ).to(self.device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self.cache[key] = model
        return self.cache[key]


def scaled_inputs(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    scaler: FoldRobustScaler,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        modality: torch.from_numpy(
            scaler.transform(modality, arrays[modality][indices])
        ).to(device)
        for modality in MODALITIES
    }


@torch.no_grad()
def model_probabilities(
    model: EmbraceNetLite,
    inputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> np.ndarray:
    return (
        model(inputs, modality_mask=mask)["logits"]
        .softmax(dim=1)
        .cpu()
        .numpy()
    )


def trial_probability_rows(
    probabilities: np.ndarray,
    truth: np.ndarray,
    trial_ids: np.ndarray,
    subjects: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trial_id": trial_ids.astype(str),
            "subject": subjects.astype(int),
            "truth": truth.astype(int),
            **{
                f"p_{index}": probabilities[:, index]
                for index in range(len(SPEEDS))
            },
        }
    )
    return (
        frame.groupby(["trial_id", "subject"], sort=False)
        .agg(
            truth=("truth", "first"),
            **{
                f"p_{index}": (f"p_{index}", "mean")
                for index in range(len(SPEEDS))
            },
        )
        .reset_index()
    )


def evaluate_observed_trials(
    args: argparse.Namespace,
    complete_arrays: dict[str, np.ndarray],
    observed_arrays: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    scalers: list[FoldRobustScaler],
    store: ModelStore,
    device: torch.device,
) -> pd.DataFrame:
    complete_subjects = set(complete_arrays["subject"].astype(int))
    test_fold_by_subject = {
        int(subject): fold
        for fold, (_, _, test_indices) in enumerate(folds, start=1)
        for subject in np.unique(
            complete_arrays["subject"][test_indices]
        )
    }
    rows: list[pd.DataFrame] = []
    for current_trial in np.unique(observed_arrays["trial_id"]):
        indices = np.flatnonzero(
            observed_arrays["trial_id"] == current_trial
        )
        subject = int(observed_arrays["subject"][indices[0]])
        truth = int(
            SPEEDS.index(
                round(float(observed_arrays["speed"][indices[0]]), 2)
            )
        )
        available = observed_arrays["availability_mask"][indices[0]].astype(
            bool
        )
        evaluation_folds = (
            [test_fold_by_subject[subject]]
            if subject in test_fold_by_subject
            else list(range(1, len(folds) + 1))
        )
        for seed in args.seeds:
            for method in ("embracenet", "fasca"):
                fold_predictions = []
                for fold in evaluation_folds:
                    inputs = scaled_inputs(
                        observed_arrays,
                        indices,
                        scalers[fold - 1],
                        device,
                    )
                    mask = torch.from_numpy(
                        np.repeat(
                            available[None, :], len(indices), axis=0
                        )
                    ).to(device)
                    fold_predictions.append(
                        model_probabilities(
                            store.get(method, fold, seed), inputs, mask
                        ).mean(axis=0)
                    )
                probability = np.mean(fold_predictions, axis=0)
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "evaluation": "observed_incomplete_trials",
                                "scenario": "source_observed",
                                "method": method,
                                "seed": seed,
                                "fold": (
                                    evaluation_folds[0]
                                    if len(evaluation_folds) == 1
                                    else 0
                                ),
                                "trial_id": str(current_trial),
                                "subject": subject,
                                "truth": truth,
                                "subject_in_complete_cv": (
                                    subject in complete_subjects
                                ),
                                "available_modalities": "+".join(
                                    modality
                                    for modality, keep in zip(
                                        MODALITIES, available
                                    )
                                    if keep
                                ),
                                **{
                                    f"p_{index}": probability[index]
                                    for index in range(len(SPEEDS))
                                },
                            }
                        ]
                    )
                )
    return pd.concat(rows, ignore_index=True)


@torch.no_grad()
def predict_replay_condition(
    model: EmbraceNetLite,
    scaled: dict[str, torch.Tensor],
    device: torch.device,
    eeg_indices: list[int],
    attenuation: float | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    probability_rows = []
    samples = len(scaled["truth"])
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        inputs = move_inputs(
            {
                modality: scaled[modality][start:stop]
                for modality in MODALITIES
            },
            device,
        )
        mask = torch.ones(
            stop - start,
            len(MODALITIES),
            dtype=torch.bool,
            device=device,
        )
        if attenuation is None:
            mask[:, MODALITIES.index("eeg")] = False
            inputs["eeg"] = torch.zeros_like(inputs["eeg"])
        else:
            inputs["eeg"][:, eeg_indices] *= attenuation
        probability_rows.append(
            model(inputs, modality_mask=mask)["logits"]
            .softmax(dim=1)
            .cpu()
            .numpy()
        )
    return scaled["truth"].numpy(), np.concatenate(probability_rows)


def evaluate_schema_replay(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    scalers: list[FoldRobustScaler],
    store: ModelStore,
    device: torch.device,
) -> pd.DataFrame:
    eeg_names = arrays["eeg_channels"].astype(str).tolist()
    eeg_indices = [eeg_names.index(name) for name in OBSERVED_EEG_INPUT_CHANNELS]
    rows: list[pd.DataFrame] = []
    for fold, indices in enumerate(folds, start=1):
        _, _, test_indices = indices
        scaled = {
            modality: torch.from_numpy(
                scalers[fold - 1].transform(
                    modality, arrays[modality][test_indices]
                )
            )
            for modality in MODALITIES
        }
        scaled["truth"] = torch.from_numpy(
            arrays["label"][test_indices].astype(np.int64)
        )
        trial_ids = arrays["trial_id"][test_indices]
        subjects = arrays["subject"][test_indices]
        for seed in args.seeds:
            for scenario, attenuation in REPLAY_SCENARIOS.items():
                base_method = {
                    "embracenet": "embracenet",
                    "fasca": "fasca",
                    # Refined FTER requires dead channels in at least two
                    # available modalities; this replay affects EEG only.
                    "fter": "fasca",
                }
                for output_method, checkpoint_method in base_method.items():
                    truth, probabilities = predict_replay_condition(
                        store.get(checkpoint_method, fold, seed),
                        scaled,
                        device,
                        eeg_indices,
                        attenuation,
                        args.batch_size,
                    )
                    trial_rows = trial_probability_rows(
                        probabilities,
                        truth,
                        trial_ids,
                        subjects,
                    )
                    trial_rows = trial_rows.assign(
                        evaluation="observed_schema_replay",
                        scenario=scenario,
                        method=output_method,
                        seed=seed,
                        fold=fold,
                        subject_in_complete_cv=True,
                        available_modalities=(
                            "emg+imu+fp"
                            if attenuation is None
                            else "+".join(MODALITIES)
                        ),
                    )
                    rows.append(trial_rows)
    return pd.concat(rows, ignore_index=True)


def averaged_trials(frame: pd.DataFrame) -> pd.DataFrame:
    probability_columns = [f"p_{index}" for index in range(len(SPEEDS))]
    group_columns = [
        "evaluation",
        "scenario",
        "method",
        "trial_id",
        "subject",
        "subject_in_complete_cv",
        "available_modalities",
    ]
    return (
        frame.groupby(group_columns, sort=False)
        .agg(
            truth=("truth", "first"),
            **{
                column: (column, "mean") for column in probability_columns
            },
        )
        .reset_index()
    )


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    probability_columns = [f"p_{index}" for index in range(len(SPEEDS))]
    probabilities = frame[probability_columns].to_numpy()
    truth = frame["truth"].to_numpy(dtype=int)
    predicted = probabilities.argmax(axis=1)
    targets = np.eye(len(SPEEDS), dtype=float)[truth]
    return {
        "trials": int(len(frame)),
        "subjects": int(frame["subject"].nunique()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=np.arange(len(SPEEDS)),
                average="macro",
                zero_division=0,
            )
        ),
        "brier": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
    }


def paired_comparison(
    averaged: pd.DataFrame,
    evaluation: str,
    scenario: str,
    method: str,
    reference: str,
    bootstrap_samples: int,
    internal_only: bool = False,
) -> dict[str, object]:
    selected = averaged[
        (averaged["evaluation"] == evaluation)
        & (averaged["scenario"] == scenario)
        & (averaged["method"].isin([method, reference]))
    ].copy()
    if internal_only:
        selected = selected[selected["subject_in_complete_cv"]]
    method_frame = selected[selected["method"] == method].sort_values(
        "trial_id"
    )
    reference_frame = selected[
        selected["method"] == reference
    ].sort_values("trial_id")
    if method_frame["trial_id"].tolist() != reference_frame[
        "trial_id"
    ].tolist():
        raise RuntimeError("Paired trial identifiers differ")
    method_metric = metric_values(method_frame)
    reference_metric = metric_values(reference_frame)
    subjects = np.unique(method_frame["subject"])
    rng = np.random.default_rng(20260703)
    probability_columns = [f"p_{index}" for index in range(len(SPEEDS))]
    method_confusions = np.zeros(
        (len(subjects), len(SPEEDS), len(SPEEDS)), dtype=float
    )
    reference_confusions = np.zeros_like(method_confusions)
    for subject_index, subject in enumerate(subjects):
        for current_frame, destination in (
            (method_frame, method_confusions),
            (reference_frame, reference_confusions),
        ):
            values = current_frame[current_frame["subject"] == subject]
            predicted = values[probability_columns].to_numpy().argmax(axis=1)
            truth = values["truth"].to_numpy(dtype=int)
            np.add.at(
                destination[subject_index],
                (truth, predicted),
                1.0,
            )
    weights = rng.multinomial(
        len(subjects),
        np.full(len(subjects), 1.0 / len(subjects)),
        size=bootstrap_samples,
    )
    method_bootstrap = np.einsum(
        "bs,sij->bij", weights, method_confusions
    )
    reference_bootstrap = np.einsum(
        "bs,sij->bij", weights, reference_confusions
    )

    def macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
        true_positive = np.diagonal(confusion, axis1=1, axis2=2)
        false_positive = confusion.sum(axis=1) - true_positive
        false_negative = confusion.sum(axis=2) - true_positive
        denominator = (
            2.0 * true_positive + false_positive + false_negative
        )
        per_class = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0,
        )
        return per_class.mean(axis=1)

    bootstrap = macro_f1_from_confusion(
        method_bootstrap
    ) - macro_f1_from_confusion(reference_bootstrap)
    subject_scores = []
    for current_method, current_frame in (
        (method, method_frame),
        (reference, reference_frame),
    ):
        values = current_frame.copy()
        values["correct"] = (
            values[probability_columns].to_numpy().argmax(axis=1)
            == values["truth"].to_numpy()
        ).astype(float)
        subject_score = values.groupby("subject")["correct"].mean()
        subject_scores.append(subject_score.rename(current_method))
    paired_subjects = pd.concat(subject_scores, axis=1).dropna()
    differences = (
        paired_subjects[method] - paired_subjects[reference]
    ).to_numpy()
    p_value = (
        1.0
        if np.allclose(differences, 0.0)
        else float(wilcoxon(differences).pvalue)
    )
    return {
        "evaluation": evaluation,
        "scenario": scenario,
        "method": method,
        "reference": reference,
        "internal_only": internal_only,
        "trials": method_metric["trials"],
        "subjects": method_metric["subjects"],
        "method_macro_f1": method_metric["macro_f1"],
        "reference_macro_f1": reference_metric["macro_f1"],
        "macro_f1_difference": (
            method_metric["macro_f1"] - reference_metric["macro_f1"]
        ),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "method_accuracy": method_metric["accuracy"],
        "reference_accuracy": reference_metric["accuracy"],
        "subject_accuracy_wilcoxon_p": p_value,
    }


def holm_adjust(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return pd.Series(adjusted, index=frame.index)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete_arrays = load_arrays(args.complete_data)
    observed_npz = np.load(args.observed_data, allow_pickle=False)
    observed_arrays = {
        key: observed_npz[key] for key in observed_npz.files
    }
    folds = subject_folds(
        complete_arrays["subject"], args.folds, args.partition_seed
    )
    scalers = [
        FoldRobustScaler.fit(complete_arrays, train_indices)
        for train_indices, _, _ in folds
    ]
    channels = {
        modality: int(complete_arrays[modality].shape[1])
        for modality in MODALITIES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = ModelStore(args, channels, device)
    observed = evaluate_observed_trials(
        args,
        complete_arrays,
        observed_arrays,
        folds,
        scalers,
        store,
        device,
    )
    replay = evaluate_schema_replay(
        args,
        complete_arrays,
        folds,
        scalers,
        store,
        device,
    )
    predictions = pd.concat([observed, replay], ignore_index=True)
    predictions.to_csv(
        args.output_dir / "seed_trial_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    seed_summary_rows = []
    for keys, group in predictions.groupby(
        ["evaluation", "scenario", "method", "seed"], sort=False
    ):
        seed_summary_rows.append(
            {
                "evaluation": keys[0],
                "scenario": keys[1],
                "method": keys[2],
                "seed": keys[3],
                "scope": "all",
                **metric_values(group),
            }
        )
        if keys[0] == "observed_incomplete_trials":
            for scope, scoped in (
                (
                    "heldout_fold_subjects",
                    group[group["subject_in_complete_cv"]],
                ),
                (
                    "fully_unseen_subjects",
                    group[~group["subject_in_complete_cv"]],
                ),
            ):
                seed_summary_rows.append(
                    {
                        "evaluation": keys[0],
                        "scenario": keys[1],
                        "method": keys[2],
                        "seed": keys[3],
                        "scope": scope,
                        **metric_values(scoped),
                    }
                )
    seed_summary = pd.DataFrame(seed_summary_rows)
    seed_summary.to_csv(
        args.output_dir / "seed_metric_summary.csv", index=False
    )
    averaged = averaged_trials(predictions)
    averaged.to_csv(
        args.output_dir / "seed_averaged_trial_predictions.csv",
        index=False,
    )
    summary_rows = []
    for keys, group in averaged.groupby(
        ["evaluation", "scenario", "method"], sort=False
    ):
        summary_rows.append(
            {
                "evaluation": keys[0],
                "scenario": keys[1],
                "method": keys[2],
                **metric_values(group),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "metric_summary.csv", index=False)

    comparisons = [
        paired_comparison(
            averaged,
            "observed_incomplete_trials",
            "source_observed",
            "fasca",
            "embracenet",
            args.bootstrap_samples,
            internal_only=True,
        )
    ]
    for scenario in REPLAY_SCENARIOS:
        comparisons.append(
            paired_comparison(
                averaged,
                "observed_schema_replay",
                scenario,
                "fasca",
                "embracenet",
                args.bootstrap_samples,
            )
        )
    comparisons.append(
        paired_comparison(
            averaged,
            "observed_schema_replay",
            "observed_schema_zero",
            "fter",
            "fasca",
            args.bootstrap_samples,
        )
    )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["holm_p"] = holm_adjust(
        comparison_frame, "subject_accuracy_wilcoxon_p"
    )
    comparison_frame.to_csv(
        args.output_dir / "paired_comparisons.csv", index=False
    )
    protocol = {
        "device": str(device),
        "seeds": args.seeds,
        "partition_seed": args.partition_seed,
        "observed_failure_definition": (
            "Trials absent from the complete-case dataset because at least "
            "one source stream was not indexed, empty, or incompatible with "
            "the required schema. No test labels select failure cases."
        ),
        "schema_replay_definition": {
            "source_trial": "S32_1",
            "observed_missing_required_columns": [
                "A1",
                *OBSERVED_EEG_INPUT_CHANNELS,
            ],
            "direct_model_input_channels_replayed": list(
                OBSERVED_EEG_INPUT_CHANNELS
            ),
            "note": (
                "A1 is a rereferencing channel rather than a model input. "
                "The conservative observed_eeg_rejected condition marks the "
                "whole EEG stream unavailable; the attenuation sweep replays "
                "only the seven directly corresponding model-input channels."
            ),
        },
        "inference": (
            "For subjects present in the complete-case cohort, only their "
            "subject-disjoint held-out fold is used. A subject absent from "
            "all complete folds is evaluated by a fixed five-fold ensemble "
            "and reported separately from the internal comparison."
        ),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
