from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from gait_robust.cross_validate_full import subject_folds
from gait_robust.models import MultimodalLiteNet
from gait_robust.train_distill import relation_matrix, sample_mask
from gait_robust.train_full import MODALITIES, SPEEDS, WindowDataset, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lambda-kd", type=float, default=0.8)
    parser.add_argument("--lambda-feature", type=float, default=0.3)
    parser.add_argument("--lambda-relation", type=float, default=0.1)
    parser.add_argument("--max-drop-probability", type=float, default=0.65)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--include-dropout-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    arrays: dict[str, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> dict[str, DataLoader]:
    train_idx, validation_idx, test_idx = indices
    return {
        "train": DataLoader(
            WindowDataset(arrays, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            WindowDataset(arrays, validation_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            WindowDataset(arrays, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
    }


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_teacher(
    channels: dict[str, int],
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MultimodalLiteNet, int]:
    model = MultimodalLiteNet(
        channels=channels,
        classes=len(SPEEDS),
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, target in loaders["train"]:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in inputs.items()
            }
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(inputs)["logits"], target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        validation_loss, _, _, _ = evaluate(
            model, loaders["validation"], device
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = clone_state(model)
        elif epoch - best_epoch >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Teacher training did not select a checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch


@torch.no_grad()
def selection_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    masks = [torch.ones(1, len(MODALITIES), dtype=torch.bool)]
    masks.extend(
        torch.nn.functional.one_hot(
            torch.arange(len(MODALITIES)), num_classes=len(MODALITIES)
        ).bool().unsqueeze(1)
    )
    accuracies = []
    for mask in masks:
        _, truth, predicted, _ = evaluate(
            model, loader, device, modality_mask=mask
        )
        accuracies.append(float(accuracy_score(truth, predicted)))
    full_accuracy = accuracies[0]
    mean_single_accuracy = float(np.mean(accuracies[1:]))
    return (
        0.5 * (full_accuracy + mean_single_accuracy),
        full_accuracy,
        mean_single_accuracy,
    )


def train_mask_aware(
    channels: dict[str, int],
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    teacher: MultimodalLiteNet | None,
) -> tuple[MultimodalLiteNet, int]:
    student = MultimodalLiteNet(
        channels=channels,
        classes=len(SPEEDS),
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    supervised = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None

    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    for epoch in range(1, args.epochs + 1):
        student.train()
        fraction = (epoch - 1) / max(args.epochs - 1, 1)
        drop_probability = (
            0.1 + fraction * (args.max_drop_probability - 0.1)
        )
        for inputs, target in loaders["train"]:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in inputs.items()
            }
            target = target.to(device)
            mask = sample_mask(
                target.shape[0],
                len(MODALITIES),
                drop_probability,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            teacher_output = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_output = teacher(inputs)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                student_output = student(inputs, modality_mask=mask)
                loss = supervised(student_output["logits"], target)
                if teacher_output is not None:
                    kd_loss = F.kl_div(
                        F.log_softmax(
                            student_output["logits"] / args.temperature,
                            dim=1,
                        ),
                        F.softmax(
                            teacher_output["logits"] / args.temperature,
                            dim=1,
                        ),
                        reduction="batchmean",
                    ) * (args.temperature**2)
                    feature_loss = 1.0 - F.cosine_similarity(
                        student_output["fused"],
                        teacher_output["fused"],
                        dim=1,
                    ).mean()
                    relation_loss = F.mse_loss(
                        relation_matrix(student_output["fused"]),
                        relation_matrix(teacher_output["fused"]),
                    )
                    loss = (
                        loss
                        + args.lambda_kd * kd_loss
                        + args.lambda_feature * feature_loss
                        + args.lambda_relation * relation_loss
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        score, _, _ = selection_accuracy(
            student, loaders["validation"], device
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = clone_state(student)
        elif epoch - best_epoch >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Mask-aware training did not select a checkpoint")
    student.load_state_dict(best_state)
    return student, best_epoch


@torch.no_grad()
def evaluate_all_masks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    rows = []
    for bits in itertools.product((False, True), repeat=len(MODALITIES)):
        if not any(bits):
            continue
        mask = torch.tensor(bits, dtype=torch.bool).unsqueeze(0)
        loss, truth, predicted, _ = evaluate(
            model, loader, device, modality_mask=mask
        )
        available = [
            modality
            for modality, present in zip(MODALITIES, bits)
            if present
        ]
        rows.append(
            {
                "seed": seed,
                "fold": fold,
                "method": method,
                "available_modalities": "+".join(available),
                "n_modalities": len(available),
                "test_windows": len(truth),
                "loss": loss,
                "accuracy": accuracy_score(truth, predicted),
                "macro_f1": f1_score(
                    truth, predicted, average="macro", zero_division=0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_modalities", "available_modalities"],
        ascending=[False, True],
    )


def summarize_run(
    results: pd.DataFrame,
    seed: int,
    fold: int,
    method: str,
    best_epoch: int,
) -> dict[str, float | int | str]:
    full_name = "+".join(MODALITIES)
    full = results.loc[
        results["available_modalities"] == full_name
    ].iloc[0]
    no_fp = results.loc[
        results["available_modalities"] == "eeg+emg+imu"
    ].iloc[0]
    imu = results.loc[
        results["available_modalities"] == "imu"
    ].iloc[0]
    return {
        "seed": seed,
        "fold": fold,
        "method": method,
        "best_epoch": best_epoch,
        "test_windows": int(full["test_windows"]),
        "full_accuracy": float(full["accuracy"]),
        "full_macro_f1": float(full["macro_f1"]),
        "mean_15_accuracy": float(results["accuracy"].mean()),
        "mean_15_macro_f1": float(results["macro_f1"].mean()),
        "mean_single_accuracy": float(
            results.loc[results["n_modalities"] == 1, "accuracy"].mean()
        ),
        "worst_accuracy": float(results["accuracy"].min()),
        "no_fp_accuracy": float(no_fp["accuracy"]),
        "imu_only_accuracy": float(imu["accuracy"]),
        "largest_drop": float(full["accuracy"] - results["accuracy"].min()),
    }


def mean_sd_ci(values: np.ndarray) -> dict[str, float | list[float]]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) == 1:
        return {"mean": mean, "sd": 0.0, "t_95ci": [mean, mean]}
    sd = float(values.std(ddof=1))
    half_width = float(
        stats.t.ppf(0.975, df=len(values) - 1) * stats.sem(values)
    )
    return {
        "mean": mean,
        "sd": sd,
        "t_95ci": [mean - half_width, mean + half_width],
    }


def build_final_summary(
    combination_table: pd.DataFrame,
    run_table: pd.DataFrame,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    channels: dict[str, int],
) -> dict[str, object]:
    metric_columns = [
        "full_accuracy",
        "full_macro_f1",
        "mean_15_accuracy",
        "mean_15_macro_f1",
        "mean_single_accuracy",
        "worst_accuracy",
        "no_fp_accuracy",
        "imu_only_accuracy",
        "largest_drop",
    ]
    methods: dict[str, object] = {}
    per_method_seed_tables: dict[str, pd.DataFrame] = {}
    for method, method_runs in run_table.groupby("method"):
        seed_rows = []
        for seed, seed_runs in method_runs.groupby("seed"):
            weights = seed_runs["test_windows"].to_numpy(dtype=float)
            row: dict[str, float | int | str] = {
                "method": method,
                "seed": int(seed),
            }
            for metric in metric_columns:
                row[metric] = float(
                    np.average(seed_runs[metric], weights=weights)
                )
            seed_combinations = combination_table.loc[
                (combination_table["method"] == method)
                & (combination_table["seed"] == seed)
            ]
            pooled_combination_accuracy = {}
            for combination, combination_rows in seed_combinations.groupby(
                "available_modalities"
            ):
                pooled_combination_accuracy[combination] = float(
                    np.average(
                        combination_rows["accuracy"].to_numpy(),
                        weights=combination_rows["test_windows"].to_numpy(),
                    )
                )
            worst_combination = min(
                pooled_combination_accuracy,
                key=pooled_combination_accuracy.get,
            )
            row["worst_accuracy"] = pooled_combination_accuracy[
                worst_combination
            ]
            row["worst_combination"] = worst_combination
            row["largest_drop"] = (
                float(row["full_accuracy"]) - float(row["worst_accuracy"])
            )
            seed_rows.append(row)
        seed_table = pd.DataFrame(seed_rows)
        per_method_seed_tables[method] = seed_table.set_index("seed")
        best_seed = seed_table.sort_values(
            ["mean_15_accuracy", "worst_accuracy", "full_accuracy"],
            ascending=False,
        ).iloc[0].to_dict()
        methods[method] = {
            "across_seed_pooled_fold_summary": {
                metric: mean_sd_ci(seed_table[metric].to_numpy())
                for metric in metric_columns
            },
            "per_seed_pooled_fold_metrics": seed_rows,
            "best_seed_by_mean_15_accuracy": best_seed,
            "best_single_fold_by_mean_15_accuracy": method_runs.sort_values(
                ["mean_15_accuracy", "worst_accuracy", "full_accuracy"],
                ascending=False,
            ).iloc[0].to_dict(),
        }

    comparisons: dict[str, object] = {}
    if {"baseline", "distilled"}.issubset(per_method_seed_tables):
        delta = (
            per_method_seed_tables["distilled"][metric_columns]
            - per_method_seed_tables["baseline"][metric_columns]
        )
        comparisons["distilled_minus_baseline_across_seeds"] = {
            metric: mean_sd_ci(delta[metric].to_numpy())
            for metric in metric_columns
        }
    if {"baseline", "dropout_only"}.issubset(per_method_seed_tables):
        delta = (
            per_method_seed_tables["dropout_only"][metric_columns]
            - per_method_seed_tables["baseline"][metric_columns]
        )
        comparisons["dropout_only_minus_baseline_across_seeds"] = {
            metric: mean_sd_ci(delta[metric].to_numpy())
            for metric in metric_columns
        }
    if {"dropout_only", "distilled"}.issubset(per_method_seed_tables):
        delta = (
            per_method_seed_tables["distilled"][metric_columns]
            - per_method_seed_tables["dropout_only"][metric_columns]
        )
        comparisons["distilled_minus_dropout_only_across_seeds"] = {
            metric: mean_sd_ci(delta[metric].to_numpy())
            for metric in metric_columns
        }

    return {
        "protocol": (
            f"Fixed subject-disjoint {args.folds}-fold partition shared by all "
            f"methods; {len(args.seeds)} optimization seed(s); each subject is "
            "tested exactly once per seed; model selection uses validation "
            "subjects only."
        ),
        "reporting_policy": (
            "Primary: mean, SD and 95% t interval over seed-level pooled "
            "five-fold estimates. Secondary: the best complete five-fold seed. "
            "The best single fold is descriptive only and must not replace the "
            "primary result. Accuracy is pooled exactly by test-window counts; "
            "macro-F1 is a test-window-weighted mean of fold-level macro-F1."
        ),
        "partition_seed": args.partition_seed,
        "training_seeds": args.seeds,
        "folds": args.folds,
        "subjects": int(len(np.unique(arrays["subject"]))),
        "windows": int(len(arrays["subject"])),
        "modalities": list(MODALITIES),
        "evaluated_nonempty_combinations": int(
            combination_table["available_modalities"].nunique()
        ),
        "parameter_count": sum(
            parameter.numel()
            for parameter in MultimodalLiteNet(
                channels=channels,
                classes=len(SPEEDS),
                embedding_dim=args.embedding_dim,
            ).parameters()
        ),
        "methods": methods,
        "paired_comparisons": comparisons,
    }


def write_status(
    path: Path,
    stage: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "completed_fold_seed_runs": completed,
                "total_fold_seed_runs": total,
                "message": message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    loaded = np.load(args.data, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    arrays["label"] = np.asarray(
        [SPEEDS.index(round(float(value), 2)) for value in arrays["speed"]],
        dtype=np.int64,
    )
    channels = {
        modality: int(arrays[modality].shape[1])
        for modality in MODALITIES
    }
    folds = subject_folds(
        arrays["subject"], args.folds, args.partition_seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_runs = len(args.seeds) * len(folds)
    completed = 0
    all_combinations = []
    all_run_summaries = []
    split_records = []
    write_status(
        status_path,
        "running",
        completed,
        total_runs,
        f"Starting on {device}",
    )

    for fold_number, indices in enumerate(folds, start=1):
        train_idx, validation_idx, test_idx = indices
        split_records.append(
            {
                "fold": fold_number,
                "train_subjects": sorted(
                    np.unique(arrays["subject"][train_idx]).astype(int).tolist()
                ),
                "validation_subjects": sorted(
                    np.unique(arrays["subject"][validation_idx])
                    .astype(int)
                    .tolist()
                ),
                "test_subjects": sorted(
                    np.unique(arrays["subject"][test_idx]).astype(int).tolist()
                ),
            }
        )
        for seed in args.seeds:
            run_dir = (
                args.output_dir
                / "runs"
                / f"seed_{seed}"
                / f"fold_{fold_number}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            run_combinations_path = run_dir / "combination_metrics.csv"
            run_summary_path = run_dir / "run_summary.csv"
            if run_combinations_path.exists() and run_summary_path.exists():
                all_combinations.append(
                    pd.read_csv(run_combinations_path)
                )
                all_run_summaries.append(pd.read_csv(run_summary_path))
                completed += 1
                write_status(
                    status_path,
                    "running",
                    completed,
                    total_runs,
                    f"Reused seed {seed}, fold {fold_number}",
                )
                continue

            training_seed = int(seed * 100 + fold_number)
            seed_everything(training_seed)
            loaders = make_loaders(
                arrays, indices, args.batch_size, device
            )
            teacher, teacher_epoch = train_teacher(
                channels, loaders, args, device
            )
            torch.save(
                {
                    "state_dict": clone_state(teacher),
                    "channels": channels,
                    "embedding_dim": args.embedding_dim,
                    "seed": seed,
                    "fold": fold_number,
                    "best_epoch": teacher_epoch,
                },
                run_dir / "teacher.pt",
            )
            result_tables = [
                evaluate_all_masks(
                    teacher,
                    loaders["test"],
                    device,
                    "baseline",
                    seed,
                    fold_number,
                )
            ]
            run_rows = [
                summarize_run(
                    result_tables[-1],
                    seed,
                    fold_number,
                    "baseline",
                    teacher_epoch,
                )
            ]

            if args.include_dropout_only:
                seed_everything(training_seed + 10_000)
                dropout_model, dropout_epoch = train_mask_aware(
                    channels, loaders, args, device, teacher=None
                )
                torch.save(
                    {
                        "state_dict": clone_state(dropout_model),
                        "channels": channels,
                        "embedding_dim": args.embedding_dim,
                        "seed": seed,
                        "fold": fold_number,
                        "best_epoch": dropout_epoch,
                    },
                    run_dir / "dropout_only.pt",
                )
                result_tables.append(
                    evaluate_all_masks(
                        dropout_model,
                        loaders["test"],
                        device,
                        "dropout_only",
                        seed,
                        fold_number,
                    )
                )
                run_rows.append(
                    summarize_run(
                        result_tables[-1],
                        seed,
                        fold_number,
                        "dropout_only",
                        dropout_epoch,
                    )
                )
                del dropout_model

            # Use the same initialization and minibatch/mask RNG as the
            # dropout-only ablation so their paired difference isolates the
            # distillation losses rather than a favorable random seed.
            seed_everything(training_seed + 10_000)
            distilled, distilled_epoch = train_mask_aware(
                channels, loaders, args, device, teacher=teacher
            )
            torch.save(
                {
                    "state_dict": clone_state(distilled),
                    "channels": channels,
                    "embedding_dim": args.embedding_dim,
                    "seed": seed,
                    "fold": fold_number,
                    "best_epoch": distilled_epoch,
                },
                run_dir / "distilled.pt",
            )
            result_tables.append(
                evaluate_all_masks(
                    distilled,
                    loaders["test"],
                    device,
                    "distilled",
                    seed,
                    fold_number,
                )
            )
            run_rows.append(
                summarize_run(
                    result_tables[-1],
                    seed,
                    fold_number,
                    "distilled",
                    distilled_epoch,
                )
            )

            run_combinations = pd.concat(
                result_tables, ignore_index=True
            )
            run_summary = pd.DataFrame(run_rows)
            run_combinations.to_csv(run_combinations_path, index=False)
            run_summary.to_csv(run_summary_path, index=False)
            all_combinations.append(run_combinations)
            all_run_summaries.append(run_summary)
            completed += 1
            write_status(
                status_path,
                "running",
                completed,
                total_runs,
                f"Completed seed {seed}, fold {fold_number}",
            )
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": total_runs,
                        "seed": seed,
                        "fold": fold_number,
                        "rows": run_rows,
                    }
                ),
                flush=True,
            )
            del teacher, distilled
            if device.type == "cuda":
                torch.cuda.empty_cache()

    combination_table = pd.concat(all_combinations, ignore_index=True)
    run_table = pd.concat(all_run_summaries, ignore_index=True)
    combination_table.to_csv(
        args.output_dir / "all_combination_metrics.csv", index=False
    )
    run_table.to_csv(args.output_dir / "all_run_metrics.csv", index=False)
    pooled_combination_rows = []
    for (
        method,
        seed,
        available_modalities,
        n_modalities,
    ), rows in combination_table.groupby(
        ["method", "seed", "available_modalities", "n_modalities"]
    ):
        weights = rows["test_windows"].to_numpy(dtype=float)
        pooled_combination_rows.append(
            {
                "method": method,
                "seed": int(seed),
                "available_modalities": available_modalities,
                "n_modalities": int(n_modalities),
                "test_windows": int(weights.sum()),
                "accuracy": float(
                    np.average(
                        rows["accuracy"].to_numpy(), weights=weights
                    )
                ),
                "fold_weighted_macro_f1": float(
                    np.average(
                        rows["macro_f1"].to_numpy(), weights=weights
                    )
                ),
            }
        )
    pd.DataFrame(pooled_combination_rows).to_csv(
        args.output_dir / "seed_pooled_combination_metrics.csv",
        index=False,
    )
    (args.output_dir / "subject_splits.json").write_text(
        json.dumps(split_records, indent=2), encoding="utf-8"
    )
    summary = build_final_summary(
        combination_table, run_table, args, arrays, channels
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_status(
        status_path,
        "complete",
        completed,
        total_runs,
        "Robust cross-validation complete",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
