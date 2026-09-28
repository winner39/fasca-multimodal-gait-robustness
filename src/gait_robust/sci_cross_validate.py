from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from gait_robust.cross_validate_full import subject_folds
from gait_robust.data import FoldRobustScaler, ScaledWindowDataset
from gait_robust.models import MultimodalLiteNet
from gait_robust.robust_models import (
    ActionMAELite,
    CompassLite,
    EmbraceNetLite,
    RapidGait,
)
from gait_robust.train_full import MODALITIES, SPEEDS
from gait_robust.xtinyhar import XTinyHARAdaptation


METHODS = (
    "complete",
    "dropout",
    "dropout_ema",
    "embracenet",
    "actionmae",
    "compass",
    "rapid",
    "xtinyhar",
    "xtinyhar_dropout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53])
    parser.add_argument("--partition-seed", type=int, default=20260701)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--missing-probability", type=float, default=0.7)
    parser.add_argument("--lambda-reconstruction", type=float, default=0.5)
    parser.add_argument("--lambda-alignment", type=float, default=0.2)
    parser.add_argument("--lambda-proxy", type=float, default=0.5)
    parser.add_argument("--lambda-reliability", type=float, default=0.1)
    parser.add_argument("--lambda-kd", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-confidence", type=float, default=0.6)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    arrays["label"] = np.asarray(
        [SPEEDS.index(round(float(value), 2)) for value in arrays["speed"]],
        dtype=np.int64,
    )
    return arrays


def make_model(
    method: str,
    channels: dict[str, int],
    embedding_dim: int,
) -> nn.Module:
    if method in {"complete", "dropout", "dropout_ema"}:
        return MultimodalLiteNet(
            channels=channels,
            classes=len(SPEEDS),
            embedding_dim=embedding_dim,
        )
    if method == "embracenet":
        return EmbraceNetLite(
            channels=channels,
            classes=len(SPEEDS),
            embedding_dim=embedding_dim,
        )
    if method == "actionmae":
        return ActionMAELite(
            channels=channels,
            classes=len(SPEEDS),
            embedding_dim=embedding_dim,
        )
    if method == "compass":
        return CompassLite(
            channels=channels,
            classes=len(SPEEDS),
            embedding_dim=embedding_dim,
        )
    if method == "rapid":
        return RapidGait(
            channels=channels,
            classes=len(SPEEDS),
            embedding_dim=embedding_dim,
        )
    if method in {"xtinyhar", "xtinyhar_dropout"}:
        return XTinyHARAdaptation(
            channels=channels,
            classes=len(SPEEDS),
            embedding_dim=128,
        )
    raise ValueError(method)


def make_loaders(
    arrays: dict[str, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    scaler: FoldRobustScaler,
    batch_size: int,
    device: torch.device,
    loader_seed: int,
) -> dict[str, DataLoader]:
    train_idx, validation_idx, test_idx = indices
    generator = torch.Generator()
    generator.manual_seed(loader_seed)
    return {
        "train": DataLoader(
            ScaledWindowDataset(arrays, train_idx, scaler),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            ScaledWindowDataset(arrays, validation_idx, scaler),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            ScaledWindowDataset(arrays, test_idx, scaler),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
    }


def sample_uniform_mask(
    batch_size: int,
    missing_probability: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = torch.ones(
        batch_size, len(MODALITIES), dtype=torch.bool, device=device
    )
    apply_missing = torch.rand(
        batch_size, device=device, generator=generator
    ) < missing_probability
    affected = torch.nonzero(apply_missing, as_tuple=False).flatten()
    if len(affected) == 0:
        return mask
    available_count = torch.randint(
        1,
        len(MODALITIES),
        (len(affected),),
        device=device,
        generator=generator,
    )
    random_scores = torch.rand(
        len(affected),
        len(MODALITIES),
        device=device,
        generator=generator,
    )
    order = random_scores.argsort(dim=1)
    mask[affected] = False
    for row, count in enumerate(available_count.tolist()):
        mask[affected[row], order[row, :count]] = True
    return mask


def move_inputs(
    inputs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in inputs.items()
    }


def training_loss(
    method: str,
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    task = criterion(output["logits"], target)
    loss = task
    components = {"task": float(task.detach())}
    if method == "actionmae":
        reconstruction = output["reconstruction_loss"]
        loss = loss + args.lambda_reconstruction * reconstruction
        components["reconstruction"] = float(reconstruction.detach())
    elif method in {"compass", "rapid"}:
        alignment = output["alignment_loss"]
        missing = ~output["modality_mask"]
        if missing.any():
            expanded_target = target[:, None].expand(-1, len(MODALITIES))
            proxy = F.cross_entropy(
                output["proxy_logits"][missing],
                expanded_target[missing],
                label_smoothing=0.05,
            )
        else:
            proxy = task.new_zeros(())
        loss = (
            loss
            + args.lambda_alignment * alignment
            + args.lambda_proxy * proxy
        )
        components["alignment"] = float(alignment.detach())
        components["proxy"] = float(proxy.detach())
        if method == "rapid":
            reliability = output["reliability_loss"]
            loss = loss + args.lambda_reliability * reliability
            components["reliability"] = float(reliability.detach())
    return loss, components


def confidence_weighted_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    threshold: float,
) -> torch.Tensor:
    teacher_probability = F.softmax(teacher_logits / temperature, dim=1)
    confidence = F.softmax(teacher_logits, dim=1).max(dim=1).values
    weight = ((confidence - threshold) / max(1.0 - threshold, 1e-6)).clamp(
        0.0, 1.0
    )
    per_sample = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        teacher_probability,
        reduction="none",
    ).sum(dim=1) * (temperature**2)
    denominator = weight.sum().clamp_min(1.0)
    return (per_sample * weight).sum() / denominator


@torch.no_grad()
def update_ema(
    teacher: nn.Module, student: nn.Module, decay: float
) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.mul_(decay).add_(
            student_parameter.detach(), alpha=1.0 - decay
        )
    for teacher_buffer, student_buffer in zip(
        teacher.buffers(), student.buffers()
    ):
        teacher_buffer.copy_(student_buffer)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth = []
    probabilities = []
    for inputs, target in loader:
        inputs = move_inputs(inputs, device)
        target = target.to(device)
        batch_mask = modality_mask.to(device).expand(target.shape[0], -1)
        logits = model(inputs, modality_mask=batch_mask)["logits"]
        truth.append(target.cpu().numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(truth), np.concatenate(probabilities)


def classification_metrics(
    truth: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, average="macro", zero_division=0)
        ),
        "brier": float(
            np.mean(
                np.sum(
                    (
                        probabilities
                        - np.eye(len(SPEEDS), dtype=np.float32)[truth]
                    )
                    ** 2,
                    axis=1,
                )
            )
        ),
    }


def trial_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    trial_ids: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "trial_id": trial_ids,
            "truth": truth,
            **{
                f"p_{index}": probabilities[:, index]
                for index in range(len(SPEEDS))
            },
        }
    )
    probability_columns = [f"p_{index}" for index in range(len(SPEEDS))]
    grouped = (
        frame.groupby("trial_id", sort=False)
        .agg(
            truth=("truth", "first"),
            **{
                column: (column, "mean") for column in probability_columns
            },
        )
        .reset_index()
    )
    grouped_probabilities = grouped[probability_columns].to_numpy()
    return (
        classification_metrics(
            grouped["truth"].to_numpy(dtype=int), grouped_probabilities
        ),
        grouped,
    )


def selection_score(
    model: nn.Module,
    loader: DataLoader,
    validation_trial_ids: np.ndarray,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    mask_rows = [
        bits
        for bits in itertools.product(
            (False, True), repeat=len(MODALITIES)
        )
        if any(bits)
    ]
    full_key = tuple(True for _ in MODALITIES)
    trial_f1: dict[tuple[bool, ...], float] = {}
    for bits in mask_rows:
        mask = torch.tensor(bits, dtype=torch.bool).unsqueeze(0)
        truth, probabilities = predict(model, loader, device, mask)
        metrics, _ = trial_metrics(
            truth, probabilities, validation_trial_ids
        )
        trial_f1[bits] = metrics["macro_f1"]
    full = trial_f1[full_key]
    incomplete = float(
        np.mean(
            [
                value
                for bits, value in trial_f1.items()
                if bits != full_key
            ]
        )
    )
    worst = float(
        min(
            value
            for bits, value in trial_f1.items()
            if bits != full_key
        )
    )
    return 0.5 * (full + incomplete), {
        "validation_full_trial_macro_f1": full,
        "validation_mean_incomplete_trial_macro_f1": incomplete,
        "validation_worst_incomplete_trial_macro_f1": worst,
    }


def train_model(
    method: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    validation_trial_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    run_seed: int,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(run_seed + 91_337)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    history = []
    teacher = None
    if method in {"rapid", "dropout_ema"}:
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            if method in {"complete", "xtinyhar"}:
                mask = torch.ones(
                    target.shape[0],
                    len(MODALITIES),
                    dtype=torch.bool,
                    device=device,
                )
            else:
                mask = sample_uniform_mask(
                    target.shape[0],
                    args.missing_probability,
                    device,
                    mask_generator,
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(inputs, modality_mask=mask)
                loss, _ = training_loss(
                    method, output, target, criterion, args
                )
                if teacher is not None:
                    with torch.no_grad():
                        full_mask = torch.ones_like(mask)
                        teacher_logits = teacher(
                            inputs, modality_mask=full_mask
                        )["logits"]
                    kd = confidence_weighted_kd(
                        output["logits"],
                        teacher_logits,
                        temperature=args.kd_temperature,
                        threshold=args.teacher_confidence,
                    )
                    loss = loss + args.lambda_kd * kd
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                update_ema(teacher, model, args.ema_decay)
            epoch_losses.append(float(loss.detach()))
        scheduler.step()
        score, validation = selection_score(
            model,
            loaders["validation"],
            validation_trial_ids,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "selection_score": score,
            **validation,
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{method}: no checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def evaluate_all_masks(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    arrays: dict[str, np.ndarray],
    test_indices: np.ndarray,
    seed: int,
    fold: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    prediction_rows = []
    trial_ids = arrays["trial_id"][test_indices]
    subjects = arrays["subject"][test_indices]
    window_starts = arrays["window_start_seconds"][test_indices]
    for bits in itertools.product((False, True), repeat=len(MODALITIES)):
        if not any(bits):
            continue
        mask = torch.tensor(bits, dtype=torch.bool).unsqueeze(0)
        truth, probabilities = predict(model, loader, device, mask)
        available = [
            modality
            for modality, present in zip(MODALITIES, bits)
            if present
        ]
        combination = "+".join(available)
        window = classification_metrics(truth, probabilities)
        trial, trial_table = trial_metrics(
            truth, probabilities, trial_ids
        )
        metrics_rows.append(
            {
                "method": method,
                "seed": seed,
                "fold": fold,
                "available_modalities": combination,
                "n_modalities": len(available),
                "test_windows": len(truth),
                "test_trials": len(trial_table),
                **{f"window_{key}": value for key, value in window.items()},
                **{f"trial_{key}": value for key, value in trial.items()},
            }
        )
        predicted = probabilities.argmax(axis=1)
        for index in range(len(truth)):
            prediction_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "fold": fold,
                    "available_modalities": combination,
                    "subject": int(subjects[index]),
                    "trial_id": str(trial_ids[index]),
                    "window_start_seconds": float(window_starts[index]),
                    "true_class": int(truth[index]),
                    "predicted_class": int(predicted[index]),
                    **{
                        f"p_{speed:g}": float(probabilities[index, class_index])
                        for class_index, speed in enumerate(SPEEDS)
                    },
                }
            )
    return pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows)


def write_status(
    path: Path, completed: int, total: int, message: str, stage: str = "running"
) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "completed_method_fold_seed_runs": completed,
                "total_method_fold_seed_runs": total,
                "message": message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args.data)
    all_folds = subject_folds(
        arrays["subject"], args.folds, args.partition_seed
    )
    if args.max_folds is not None:
        all_folds = all_folds[: args.max_folds]
    channels = {
        modality: int(arrays[modality].shape[1])
        for modality in MODALITIES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(all_folds) * len(args.seeds) * len(args.methods)
    completed = 0
    status_path = args.output_dir / "status.json"
    write_status(status_path, completed, total, f"Starting on {device}")
    all_metrics = []

    for fold, indices in enumerate(all_folds, start=1):
        train_idx, validation_idx, test_idx = indices
        scaler = FoldRobustScaler.fit(arrays, train_idx)
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "scaler.json").write_text(
            json.dumps(scaler.as_serializable()),
            encoding="utf-8",
        )
        split_payload = {
            "fold": fold,
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
        (fold_dir / "subjects.json").write_text(
            json.dumps(split_payload, indent=2), encoding="utf-8"
        )
        for seed in args.seeds:
            for method in args.methods:
                run_dir = fold_dir / f"seed_{seed}" / method
                metrics_path = run_dir / "combination_metrics.csv"
                predictions_path = run_dir / "window_predictions.csv.gz"
                if metrics_path.exists() and predictions_path.exists():
                    all_metrics.append(pd.read_csv(metrics_path))
                    completed += 1
                    write_status(
                        status_path,
                        completed,
                        total,
                        f"Reused fold {fold}, seed {seed}, {method}",
                    )
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                run_seed = int(seed * 100 + fold)
                seed_everything(run_seed)
                loaders = make_loaders(
                    arrays,
                    indices,
                    scaler,
                    args.batch_size,
                    device,
                    loader_seed=run_seed + 77,
                )
                model = make_model(
                    method, channels, args.embedding_dim
                ).to(device)
                model, best_epoch, history = train_model(
                    method,
                    model,
                    loaders,
                    arrays["trial_id"][validation_idx],
                    args,
                    device,
                    run_seed,
                )
                metrics, predictions = evaluate_all_masks(
                    method,
                    model,
                    loaders["test"],
                    arrays,
                    test_idx,
                    seed,
                    fold,
                    device,
                )
                metrics["best_epoch"] = best_epoch
                metrics["parameter_count"] = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                metrics.to_csv(metrics_path, index=False)
                predictions.to_csv(
                    predictions_path, index=False, compression="gzip"
                )
                pd.DataFrame(history).to_csv(
                    run_dir / "history.csv", index=False
                )
                torch.save(
                    {
                        "method": method,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "channels": channels,
                        "embedding_dim": args.embedding_dim,
                        "best_epoch": best_epoch,
                        "seed": seed,
                        "fold": fold,
                    },
                    run_dir / "best_model.pt",
                )
                all_metrics.append(metrics)
                completed += 1
                write_status(
                    status_path,
                    completed,
                    total,
                    f"Completed fold {fold}, seed {seed}, {method}",
                )
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": total,
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "best_epoch": best_epoch,
                            "parameter_count": int(
                                metrics["parameter_count"].iloc[0]
                            ),
                            "full_trial_macro_f1": float(
                                metrics.loc[
                                    metrics["available_modalities"]
                                    == "+".join(MODALITIES),
                                    "trial_macro_f1",
                                ].iloc[0]
                            ),
                            "mean_15_trial_macro_f1": float(
                                metrics["trial_macro_f1"].mean()
                            ),
                        }
                    ),
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(
        args.output_dir / "all_combination_metrics.csv", index=False
    )
    write_status(
        status_path,
        completed,
        total,
        "SCI baseline cross-validation complete",
        stage="complete",
    )


if __name__ == "__main__":
    main()
