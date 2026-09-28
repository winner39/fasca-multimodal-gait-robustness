from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def corruption_paths(root: Path) -> dict[str, Path]:
    return {
        "RAPID-Gait/FASCA": (
            root
            / "artifacts"
            / "quality_rapid_comparison"
            / "sensor_focus_5x3.csv"
        ),
        "EmbraceNet": (
            root
            / "artifacts"
            / "multimodel_confirmatory"
            / "main_corruptions"
            / "embracenet.csv"
        ),
        "ActionMAE": (
            root
            / "artifacts"
            / "multimodel_confirmatory"
            / "main_corruptions"
            / "actionmae.csv"
        ),
    }


def seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, seed_frame in frame.groupby("seed"):
        clean = seed_frame[seed_frame["scenario"].eq("clean")]
        corrupted = seed_frame[~seed_frame["scenario"].eq("clean")]
        single = corrupted[~corrupted["target"].str.contains(r"\+")]
        all_modal = corrupted[corrupted["target"].str.contains(r"\+")]
        condition_means = corrupted.groupby(
            ["scenario", "target"]
        )["trial_macro_f1"].mean()
        rows.extend(
            [
                {
                    "seed": seed,
                    "metric": "clean",
                    "trial_macro_f1_pct": 100
                    * clean["trial_macro_f1"].mean(),
                },
                {
                    "seed": seed,
                    "metric": "single_target_corruption_mean",
                    "trial_macro_f1_pct": 100
                    * single["trial_macro_f1"].mean(),
                },
                {
                    "seed": seed,
                    "metric": "all_target_corruption_mean",
                    "trial_macro_f1_pct": 100
                    * all_modal["trial_macro_f1"].mean(),
                },
                {
                    "seed": seed,
                    "metric": "all_corruption_mean",
                    "trial_macro_f1_pct": 100
                    * corrupted["trial_macro_f1"].mean(),
                },
                {
                    "seed": seed,
                    "metric": "worst_corruption_condition",
                    "trial_macro_f1_pct": 100 * condition_means.min(),
                },
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = (
        root
        / "artifacts"
        / "multimodel_confirmatory"
        / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)

    pooled_frames = {}
    metric_frames = []
    for method, path in corruption_paths(root).items():
        frame = pd.read_csv(path)
        expected_rows = 468
        if len(frame) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows}, got {len(frame)}")
        pooled = frame[frame["level"].eq("pooled")].copy()
        if len(pooled) != 78:
            raise ValueError(f"{path}: expected 78 pooled rows")
        if pooled["trial_macro_f1"].isna().any():
            raise ValueError(f"{path}: missing trial macro-F1")
        pooled_frames[method] = pooled
        metrics = seed_metrics(pooled)
        metrics.insert(0, "method", method)
        metric_frames.append(metrics)

    by_seed = pd.concat(metric_frames, ignore_index=True)
    by_seed.to_csv(output / "corruption_metrics_by_seed.csv", index=False)
    summary = (
        by_seed.groupby(["method", "metric"], sort=False)[
            "trial_macro_f1_pct"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_pct",
                "std": "sd_across_seeds_pct",
                "count": "n_seeds",
            }
        )
    )
    summary.to_csv(output / "corruption_summary.csv", index=False)

    conditions = []
    for method, pooled in pooled_frames.items():
        corrupted = pooled[~pooled["scenario"].eq("clean")]
        condition = (
            corrupted.groupby(["scenario", "target"], as_index=False)[
                "trial_macro_f1"
            ]
            .mean()
            .rename(columns={"trial_macro_f1": method})
        )
        condition[method] *= 100
        conditions.append(condition)
    comparison = conditions[0]
    for condition in conditions[1:]:
        comparison = comparison.merge(
            condition, on=["scenario", "target"], validate="one_to_one"
        )
    for baseline in ("EmbraceNet", "ActionMAE"):
        comparison[f"rapid_minus_{baseline.lower()}_pct"] = (
            comparison["RAPID-Gait/FASCA"] - comparison[baseline]
        )
    comparison.to_csv(
        output / "corruption_condition_comparison.csv", index=False
    )

    efficiency_path = (
        root
        / "artifacts"
        / "multimodel_confirmatory"
        / "efficiency"
        / "efficiency_summary.csv"
    )
    efficiency = pd.read_csv(efficiency_path)
    efficiency["macs_m"] = efficiency["profiled_macs"] / 1e6
    efficiency.to_csv(output / "efficiency_summary.csv", index=False)

    win_counts = {}
    for baseline in ("EmbraceNet", "ActionMAE"):
        delta = comparison[f"rapid_minus_{baseline.lower()}_pct"]
        win_counts[baseline] = {
            "rapid_wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "rapid_losses": int((delta < 0).sum()),
            "mean_delta_pct": float(delta.mean()),
            "min_delta_pct": float(delta.min()),
            "max_delta_pct": float(delta.max()),
        }
    payload = {
        "validation": {
            "models": list(pooled_frames),
            "rows_per_model": 468,
            "fold_rows_per_model": 390,
            "pooled_rows_per_model": 78,
            "seeds": [51, 52, 53],
            "corrupted_conditions": 25,
        },
        "condition_win_counts": win_counts,
        "notes": [
            "All corruption comparisons use pooled trial-level macro-F1.",
            "Means and SDs use three training seeds; no independent-subject "
            "confidence interval is claimed from these aggregate CSVs.",
            "MACs are profiler-covered MAC estimates, not hardware-independent "
            "theoretical totals.",
        ],
    }
    (output / "corruption_efficiency_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
