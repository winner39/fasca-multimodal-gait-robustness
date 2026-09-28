from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


CELLS = {
    "structured_curriculum": ("structured", "curriculum"),
    "structured_constant": ("structured", "constant"),
    "iid_curriculum": ("iid", "curriculum"),
    "iid_constant": ("iid", "constant"),
}
STANDARD_SCENARIOS = {
    "noise_10db",
    "channel_dropout_30",
    "packet_loss_30",
    "shift_250ms",
    "gain_1.5",
}
PROBABILITY_COLUMNS = ["p_0", "p_1", "p_2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--cell",
        nargs=3,
        action="append",
        metavar=("NAME", "RUN_ROOT", "METHOD"),
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser.parse_args()


def macro_f1(truth: np.ndarray, probability: np.ndarray) -> float:
    return float(
        f1_score(
            truth.astype(int),
            probability.argmax(axis=1),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )


def availability_metrics(
    cell_specs: dict[str, tuple[Path, str]],
    seeds: list[int],
    max_folds: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_count = max_folds or 5
    for cell, (root, method) in cell_specs.items():
        for fold in range(1, fold_count + 1):
            for seed in seeds:
                path = (
                    root
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                    / method
                    / "window_predictions.csv.gz"
                )
                frame = pd.read_csv(path)
                probabilities = ["p_0.5", "p_0.75", "p_1"]
                trials = (
                    frame.groupby(
                        [
                            "subject",
                            "available_modalities",
                            "trial_id",
                            "true_class",
                        ],
                        as_index=False,
                    )[probabilities]
                    .mean()
                )
                for (subject, pattern), group in trials.groupby(
                    ["subject", "available_modalities"], sort=False
                ):
                    rows.append(
                        {
                            "cell": cell,
                            "seed": seed,
                            "subject": int(subject),
                            "endpoint": f"availability|{pattern}",
                            "macro_f1": macro_f1(
                                group["true_class"].to_numpy(),
                                group[probabilities].to_numpy(),
                            ),
                        }
                    )
    values = pd.DataFrame(rows)
    mean_15 = (
        values.groupby(["cell", "seed", "subject"], as_index=False)[
            "macro_f1"
        ]
        .mean()
        .assign(endpoint="mean_15")
    )
    return pd.concat([values, mean_15], ignore_index=True)


def corruption_metrics(
    experiment_root: Path,
    seeds: list[int],
) -> pd.DataFrame:
    predictions = []
    for cell in CELLS:
        path = (
            experiment_root
            / "evaluation"
            / f"{cell}_predictions.csv.gz"
        )
        frame = pd.read_csv(path)
        frame = frame[frame["seed"].isin(seeds)].copy()
        predictions.append(frame)
    data = pd.concat(predictions, ignore_index=True)
    rows: list[dict[str, object]] = []
    grouping = ["method", "seed", "subject", "scenario", "target"]
    for key, group in data.groupby(grouping, sort=False):
        cell, seed, subject, scenario, target = key
        rows.append(
            {
                "cell": cell,
                "seed": int(seed),
                "subject": int(subject),
                "scenario": scenario,
                "target": target,
                "macro_f1": macro_f1(
                    group["truth"].to_numpy(),
                    group[PROBABILITY_COLUMNS].to_numpy(),
                ),
            }
        )
    conditions = pd.DataFrame(rows)
    conditions["endpoint"] = (
        conditions["scenario"] + "|" + conditions["target"]
    )
    output = [
        conditions[["cell", "seed", "subject", "endpoint", "macro_f1"]]
    ]
    joint = conditions[
        conditions["target"] == "eeg+emg+imu+fp"
    ].copy()
    joint["endpoint"] = "joint_" + joint["scenario"]
    output.append(
        joint[["cell", "seed", "subject", "endpoint", "macro_f1"]]
    )
    standard_joint = joint[joint["scenario"].isin(STANDARD_SCENARIOS)]
    joint_mean = (
        standard_joint.groupby(
            ["cell", "seed", "subject"], as_index=False
        )["macro_f1"]
        .mean()
        .assign(endpoint="joint_fault_mean")
    )
    output.append(joint_mean)
    return pd.concat(output, ignore_index=True)


def bootstrap_summary(
    difference: pd.Series,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    values = difference.dropna().to_numpy(dtype=float)
    samples = rng.choice(
        values, size=(repeats, len(values)), replace=True
    ).mean(axis=1)
    try:
        statistic, p_value = wilcoxon(values)
    except ValueError:
        statistic, p_value = np.nan, np.nan
    return {
        "subjects": len(values),
        "effect": values.mean(),
        "ci_low": np.quantile(samples, 0.025),
        "ci_high": np.quantile(samples, 0.975),
        "subjects_positive": int((values > 0).sum()),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
    }


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


def planned_contrasts(
    values: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    averaged = (
        values.groupby(["cell", "subject", "endpoint"], as_index=False)[
            "macro_f1"
        ]
        .mean()
    )
    pivot = averaged.pivot(
        index="subject", columns=["cell", "endpoint"], values="macro_f1"
    )
    sc = "structured_curriculum"
    sn = "structured_constant"
    ic = "iid_curriculum"
    inn = "iid_constant"
    structured = "joint_structured_dropout_30"
    iid = "joint_channel_dropout_30"
    definitions = {
        "structured_vs_iid_on_structured_endpoint": (
            0.5
            * (
                pivot[(sc, structured)]
                - pivot[(ic, structured)]
                + pivot[(sn, structured)]
                - pivot[(inn, structured)]
            )
        ),
        "iid_vs_structured_on_iid_endpoint": (
            0.5
            * (
                pivot[(ic, iid)]
                - pivot[(sc, iid)]
                + pivot[(inn, iid)]
                - pivot[(sn, iid)]
            )
        ),
        "distribution_test_matching_interaction": 0.25
        * (
            pivot[(sc, structured)]
            - pivot[(ic, structured)]
            + pivot[(ic, iid)]
            - pivot[(sc, iid)]
            + pivot[(sn, structured)]
            - pivot[(inn, structured)]
            + pivot[(inn, iid)]
            - pivot[(sn, iid)]
        ),
        "curriculum_on_structured_endpoint": (
            pivot[(sc, structured)] - pivot[(sn, structured)]
        ),
        "curriculum_on_iid_endpoint": (
            pivot[(ic, iid)] - pivot[(inn, iid)]
        ),
        "curriculum_on_mean_15_structured": (
            pivot[(sc, "mean_15")] - pivot[(sn, "mean_15")]
        ),
        "curriculum_on_mean_15_iid": (
            pivot[(ic, "mean_15")] - pivot[(inn, "mean_15")]
        ),
    }
    rng = np.random.default_rng(seed)
    rows = []
    for name, difference in definitions.items():
        rows.append(
            {
                "contrast": name,
                "family": (
                    "curriculum"
                    if name.startswith("curriculum_")
                    else "distribution_matching"
                ),
                **bootstrap_summary(difference, repeats, rng),
            }
        )
    output = pd.DataFrame(rows)
    output["holm_p_within_family"] = np.nan
    for indices in output.groupby("family").groups.values():
        index = np.asarray(list(indices), dtype=int)
        output.loc[index, "holm_p_within_family"] = holm_adjust(
            output.loc[index, "wilcoxon_p"].to_numpy()
        )
    return output


def cell_summary(values: pd.DataFrame) -> pd.DataFrame:
    selected = values[
        values["endpoint"].isin(
            {
                "mean_15",
                "joint_fault_mean",
                "joint_structured_dropout_30",
                "joint_channel_dropout_30",
            }
        )
    ]
    per_seed = (
        selected.groupby(["cell", "seed", "endpoint"], as_index=False)[
            "macro_f1"
        ]
        .mean()
    )
    summary = (
        per_seed.groupby(["cell", "endpoint"])["macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary[["distribution", "curriculum"]] = summary["cell"].apply(
        lambda value: pd.Series(CELLS[value])
    )
    return summary


def main() -> None:
    args = parse_args()
    cell_specs = {
        name: (Path(root), method) for name, root, method in args.cell
    }
    if set(cell_specs) != set(CELLS):
        raise ValueError(
            "The four required cells are: " + ", ".join(CELLS)
        )
    output_dir = args.experiment_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    availability = availability_metrics(
        cell_specs, args.seeds, args.max_folds
    )
    corruptions = corruption_metrics(args.experiment_root, args.seeds)
    values = pd.concat([availability, corruptions], ignore_index=True)
    values.to_csv(output_dir / "subject_seed_endpoint_metrics.csv", index=False)
    summary = cell_summary(values)
    summary.to_csv(output_dir / "cell_endpoint_summary.csv", index=False)
    contrasts = planned_contrasts(
        values, args.bootstrap_repeats, args.bootstrap_seed
    )
    contrasts.to_csv(output_dir / "planned_contrasts.csv", index=False)
    print(summary.to_string(index=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
