from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


MAIN_PROBABILITIES = ("p_0.5", "p_0.75", "p_1")
HUGADB_PROBABILITIES = (
    "p_walking",
    "p_running",
    "p_stairs_up",
    "p_stairs_down",
)
DISPLAY_NAMES = {
    "dropout": "ModDrop",
    "moddrop": "ModDrop",
    "embracenet": "EmbraceNet-adaptation",
    "actionmae": "ActionMAE-style",
    "compass": "COMPASS-style",
    "xtinyhar_dropout": "XTinyHAR+Dropout",
    "xtinyhar_original_kd": "XTinyHAR+KD",
    "rapid_balanced_kd": "RAPID-Gait+KD",
    "rapid_sensor_aug_kd": "RAPID-Gait+FASCA",
    "fasca_kd": "RAPID-Gait+FASCA",
    "balanced_kd": "RAPID-Gait+KD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def macro_f1(
    truth: np.ndarray,
    probabilities: np.ndarray,
    classes: int,
) -> float:
    return float(
        f1_score(
            truth,
            probabilities.argmax(axis=1),
            labels=list(range(classes)),
            average="macro",
            zero_division=0,
        )
    )


def main_new_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(
        root.glob("fold_*/seed_*/*/window_predictions.csv.gz")
    )
    predictions = pd.concat(
        (pd.read_csv(path) for path in paths), ignore_index=True
    )
    trial = (
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
                for column in MAIN_PROBABILITIES
            },
        )
        .reset_index()
    )
    rows = []
    for keys, group in trial.groupby(
        ["method", "seed", "available_modalities"], sort=False
    ):
        rows.append(
            {
                "method": keys[0],
                "seed": int(keys[1]),
                "available_modalities": keys[2],
                "trial_macro_f1": macro_f1(
                    group["true_class"].to_numpy(),
                    group[list(MAIN_PROBABILITIES)].to_numpy(),
                    3,
                ),
            }
        )
    combinations = pd.DataFrame(rows)
    summaries = []
    for (method, seed), group in combinations.groupby(
        ["method", "seed"], sort=False
    ):
        full = float(
            group.loc[
                group["available_modalities"].eq("eeg+emg+imu+fp"),
                "trial_macro_f1",
            ].iloc[0]
        )
        incomplete = group.loc[
            ~group["available_modalities"].eq("eeg+emg+imu+fp"),
            "trial_macro_f1",
        ]
        summaries.append(
            {
                "method": method,
                "seed": int(seed),
                "full": full,
                "incomplete": float(incomplete.mean()),
                "mean_all": float(group["trial_macro_f1"].mean()),
                "worst": float(group["trial_macro_f1"].min()),
            }
        )
    return pd.DataFrame(summaries), trial


def external_main_metrics(project_root: Path) -> pd.DataFrame:
    xtiny = pd.read_csv(
        project_root
        / "artifacts"
        / "xtinyhar_confirmatory"
        / "seed_pooled_trial_metrics.csv"
    )
    xtiny = xtiny.loc[
        xtiny["method"].isin(
            ["xtinyhar_dropout", "xtinyhar_original_kd"]
        ),
        ["method", "seed", "full", "incomplete", "worst"],
    ].copy()
    xtiny["mean_all"] = (
        xtiny["full"] + 14.0 * xtiny["incomplete"]
    ) / 15.0
    rapid = pd.concat(
        [
            pd.read_csv(
                project_root
                / "artifacts"
                / "rapid_distill_5fold_seed51"
                / "seed_pooled_trial_metrics.csv"
            ),
            pd.read_csv(
                project_root
                / "artifacts"
                / "sensor_aug_dev_v2_focus"
                / "seed_pooled_trial_metrics.csv"
            ),
        ],
        ignore_index=True,
    ).rename(columns={"mean_15": "mean_all"})
    return pd.concat(
        [
            xtiny[
                [
                    "method",
                    "seed",
                    "full",
                    "incomplete",
                    "mean_all",
                    "worst",
                ]
            ],
            rapid[
                [
                    "method",
                    "seed",
                    "full",
                    "incomplete",
                    "mean_all",
                    "worst",
                ]
            ],
        ],
        ignore_index=True,
    )


def method_summary(
    seed_metrics: pd.DataFrame, parameter_map: dict[str, int]
) -> pd.DataFrame:
    output = (
        seed_metrics.groupby("method")
        .agg(
            full_mean=("full", "mean"),
            full_std=("full", "std"),
            incomplete_mean=("incomplete", "mean"),
            incomplete_std=("incomplete", "std"),
            mean_all=("mean_all", "mean"),
            mean_all_std=("mean_all", "std"),
            worst_mean=("worst", "mean"),
            worst_std=("worst", "std"),
        )
        .reset_index()
    )
    output["display_name"] = output["method"].map(DISPLAY_NAMES)
    output["parameter_count"] = output["method"].map(parameter_map)
    return output.sort_values("mean_all", ascending=False)


def checkpoint_parameter_map(root: Path) -> dict[str, int]:
    values: dict[str, list[int]] = {}
    for path in root.glob("fold_*/seed_*/*/best_model.pt"):
        if path.parent.name == "teacher":
            continue
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "parameter_count" in payload:
            count = int(payload["parameter_count"])
        else:
            state = payload["state_dict"]
            count = int(sum(value.numel() for value in state.values()))
        values.setdefault(path.parent.name, []).append(count)
    return {
        method: int(np.median(counts))
        for method, counts in values.items()
    }


def bootstrap_pair(
    values: pd.DataFrame,
    baseline: str,
    proposed: str,
    repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    pivot = values.pivot(
        index="subject", columns="method", values="score"
    ).dropna()
    difference = (
        pivot[proposed] - pivot[baseline]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        difference,
        size=(repeats, len(difference)),
        replace=True,
    ).mean(axis=1)
    statistic, p_value = wilcoxon(difference)
    return {
        "baseline": baseline,
        "proposed": proposed,
        "subjects": int(len(difference)),
        "baseline_mean": float(pivot[baseline].mean()),
        "proposed_mean": float(pivot[proposed].mean()),
        "paired_delta": float(difference.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "improved": int((difference > 0).sum()),
        "tied": int((difference == 0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def main_subject_scores(
    trial: pd.DataFrame, method: str
) -> pd.DataFrame:
    rows = []
    for keys, group in trial.groupby(
        ["seed", "subject", "available_modalities"], sort=False
    ):
        rows.append(
            {
                "method": method,
                "seed": int(keys[0]),
                "subject": int(keys[1]),
                "available_modalities": keys[2],
                "score": macro_f1(
                    group["true_class"].to_numpy(),
                    group[list(MAIN_PROBABILITIES)].to_numpy(),
                    3,
                ),
            }
        )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["method", "seed", "subject"], as_index=False)[
            "score"
        ]
        .mean()
        .groupby(["method", "subject"], as_index=False)["score"]
        .mean()
    )


def load_main_fasca_trial(project_root: Path) -> pd.DataFrame:
    paths = sorted(
        (
            project_root
            / "artifacts"
            / "sensor_aug_dev_v2_focus"
        ).glob("fold_*/seed_*/sensor_aug_kd/window_predictions.csv.gz")
    )
    predictions = pd.concat(
        (pd.read_csv(path) for path in paths), ignore_index=True
    )
    return (
        predictions.groupby(
            [
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
                for column in MAIN_PROBABILITIES
            },
        )
        .reset_index()
    )


def hugadb_subject_scores(
    paths: list[Path],
    method_override: str | None = None,
) -> pd.DataFrame:
    predictions = pd.concat(
        (pd.read_csv(path) for path in paths), ignore_index=True
    )
    if method_override is not None:
        predictions["method"] = method_override
    rows = []
    for keys, group in predictions.groupby(
        ["method", "seed", "subject", "available_modalities"],
        sort=False,
    ):
        rows.append(
            {
                "method": keys[0],
                "seed": int(keys[1]),
                "subject": int(keys[2]),
                "score": macro_f1(
                    group["true_class"].to_numpy(),
                    group[list(HUGADB_PROBABILITIES)].to_numpy(),
                    4,
                ),
            }
        )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["method", "seed", "subject"], as_index=False)[
            "score"
        ]
        .mean()
        .groupby(["method", "subject"], as_index=False)["score"]
        .mean()
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_root = (
        args.project_root
        / "artifacts"
        / "multimodel_confirmatory"
        / "main"
    )
    hugadb_root = (
        args.project_root
        / "artifacts"
        / "multimodel_confirmatory"
        / "hugadb"
    )
    main_new, main_trial = main_new_metrics(main_root)
    main_seeds = pd.concat(
        [main_new, external_main_metrics(args.project_root)],
        ignore_index=True,
    )
    main_seeds["display_name"] = main_seeds["method"].map(DISPLAY_NAMES)
    main_seeds.to_csv(
        args.output_dir / "main_seed_metrics.csv", index=False
    )
    main_parameter_frame = pd.read_csv(
        main_root / "all_combination_metrics.csv"
    )
    main_parameters = (
        main_parameter_frame.groupby("method")["parameter_count"]
        .first()
        .astype(int)
        .to_dict()
    )
    main_parameters.update(
        {
            "xtinyhar_dropout": 518147,
            "xtinyhar_original_kd": 518147,
            "rapid_balanced_kd": 134317,
            "rapid_sensor_aug_kd": 134317,
        }
    )
    main_summary = method_summary(main_seeds, main_parameters)
    main_summary.to_csv(
        args.output_dir / "main_method_summary.csv", index=False
    )

    hugadb_new = pd.read_csv(hugadb_root / "seed_summary.csv").rename(
        columns={
            "incomplete_mean": "incomplete",
            "mean_3": "mean_all",
        }
    )
    hugadb_rapid = pd.read_csv(
        args.project_root
        / "artifacts"
        / "hugadb_external_paired_v1"
        / "seed_summary.csv"
    ).rename(
        columns={
            "incomplete_mean": "incomplete",
            "mean_3": "mean_all",
        }
    )
    hugadb_seeds = pd.concat(
        [hugadb_new, hugadb_rapid], ignore_index=True
    )
    hugadb_seeds["display_name"] = hugadb_seeds["method"].map(
        DISPLAY_NAMES
    )
    hugadb_seeds.to_csv(
        args.output_dir / "hugadb_seed_metrics.csv", index=False
    )
    hugadb_parameters = checkpoint_parameter_map(hugadb_root)
    rapid_checkpoint = next(
        (
            args.project_root
            / "artifacts"
            / "hugadb_external_paired_v1"
        ).glob("fold_*/seed_*/fasca_kd/best_model.pt")
    )
    import torch

    rapid_payload = torch.load(
        rapid_checkpoint, map_location="cpu", weights_only=True
    )
    rapid_parameters = int(rapid_payload["parameter_count"])
    hugadb_parameters.update(
        {"fasca_kd": rapid_parameters, "balanced_kd": rapid_parameters}
    )
    hugadb_summary = method_summary(
        hugadb_seeds, hugadb_parameters
    )
    hugadb_summary.to_csv(
        args.output_dir / "hugadb_method_summary.csv", index=False
    )

    corruption_new = pd.read_csv(
        hugadb_root / "seed_pooled_corruptions.csv"
    )
    corruption_rapid = pd.read_csv(
        args.project_root
        / "artifacts"
        / "hugadb_external_paired_v1"
        / "seed_pooled_corruptions.csv"
    )
    corruptions = pd.concat(
        [corruption_new, corruption_rapid], ignore_index=True
    )
    corruptions["display_name"] = corruptions["method"].map(
        DISPLAY_NAMES
    )
    corruption_summary = (
        corruptions.groupby(
            ["method", "display_name", "corruption"]
        )["window_macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    corruption_summary.to_csv(
        args.output_dir / "hugadb_corruption_summary.csv",
        index=False,
    )

    embrace_trial = main_trial.loc[
        main_trial["method"].eq("embracenet")
    ].copy()
    embrace_subject = main_subject_scores(
        embrace_trial, "embracenet"
    )
    fasca_trial = load_main_fasca_trial(args.project_root)
    fasca_subject = main_subject_scores(
        fasca_trial, "rapid_sensor_aug_kd"
    )
    main_pair_values = pd.concat(
        [embrace_subject, fasca_subject], ignore_index=True
    )
    main_pair_values.to_csv(
        args.output_dir / "main_subject_pair_values.csv", index=False
    )
    pd.DataFrame(
        [
            bootstrap_pair(
                main_pair_values,
                "embracenet",
                "rapid_sensor_aug_kd",
                args.bootstrap_repeats,
                args.bootstrap_seed,
            )
        ]
    ).to_csv(
        args.output_dir / "main_subject_pair_summary.csv", index=False
    )

    action_paths = sorted(
        hugadb_root.glob(
            "fold_*/seed_*/actionmae/predictions.csv.gz"
        )
    )
    fasca_paths = sorted(
        (
            args.project_root
            / "artifacts"
            / "hugadb_external_paired_v1"
        ).glob("fold_*/seed_*/fasca_kd/predictions.csv.gz")
    )
    hugadb_pair_values = pd.concat(
        [
            hugadb_subject_scores(action_paths),
            hugadb_subject_scores(
                fasca_paths, method_override="fasca_kd"
            ),
        ],
        ignore_index=True,
    )
    hugadb_pair_values.to_csv(
        args.output_dir / "hugadb_subject_pair_values.csv",
        index=False,
    )
    pd.DataFrame(
        [
            bootstrap_pair(
                hugadb_pair_values,
                "actionmae",
                "fasca_kd",
                args.bootstrap_repeats,
                args.bootstrap_seed + 1,
            )
        ]
    ).to_csv(
        args.output_dir / "hugadb_subject_pair_summary.csv",
        index=False,
    )

    chart_rows = []
    for dataset, table in (
        ("Main four-modality gait", main_summary),
        ("HuGaDB EMG–IMU", hugadb_summary),
    ):
        for _, row in table.iterrows():
            for metric, label in (
                ("full_mean", "Complete input"),
                ("mean_all", "All combinations"),
            ):
                chart_rows.append(
                    {
                        "dataset": dataset,
                        "method": row["display_name"],
                        "metric": label,
                        "macro_f1": float(row[metric]),
                        "parameter_count": (
                            int(row["parameter_count"])
                            if pd.notna(row["parameter_count"])
                            else None
                        ),
                    }
                )
    pd.DataFrame(chart_rows).to_csv(
        args.output_dir / "report_chart_rows.csv", index=False
    )


if __name__ == "__main__":
    main()
