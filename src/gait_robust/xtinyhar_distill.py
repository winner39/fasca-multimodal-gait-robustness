from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from gait_robust.cross_validate_full import subject_folds
from gait_robust.data import FoldRobustScaler
from gait_robust.sci_cross_validate import (
    MODALITIES,
    SPEEDS,
    evaluate_all_masks,
    load_arrays,
    make_loaders,
    move_inputs,
    sample_uniform_mask,
    seed_everything,
    selection_score,
    trial_metrics,
    write_status,
)
from gait_robust.xtinyhar import (
    XTinyHARAdaptation,
    XTinyHARTeacherAdaptation,
)


METHODS = ("original_kd", "selective_kd")


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
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs-teacher", type=int, default=50)
    parser.add_argument("--epochs-student", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--missing-probability", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lambda-kd", type=float, default=0.7)
    parser.add_argument("--lambda-consistency", type=float, default=0.25)
    parser.add_argument("--lambda-full", type=float, default=0.25)
    parser.add_argument("--teacher-confidence", type=float, default=0.55)
    return parser.parse_args()


@torch.no_grad()
def clean_validation_score(
    model: nn.Module,
    loader,
    trial_ids: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    truth, probabilities = [], []
    full_mask = torch.ones(
        1, len(MODALITIES), dtype=torch.bool, device=device
    )
    for inputs, target in loader:
        inputs = move_inputs(inputs, device)
        target = target.to(device)
        mask = full_mask.expand(target.shape[0], -1)
        logits = model(inputs, modality_mask=mask)["logits"]
        truth.append(target.cpu().numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    metrics, _ = trial_metrics(
        np.concatenate(truth),
        np.concatenate(probabilities),
        trial_ids,
    )
    return float(metrics["macro_f1"])


def train_teacher(
    model: nn.Module,
    loaders,
    validation_trial_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_teacher
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(1, args.epochs_teacher + 1):
        model.train()
        losses = []
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            mask = torch.ones(
                target.shape[0],
                len(MODALITIES),
                dtype=torch.bool,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(inputs, modality_mask=mask)["logits"]
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        score = clean_validation_score(
            model, loaders["validation"], validation_trial_ids, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_full_trial_macro_f1": score,
            }
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No teacher checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def per_sample_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="none",
    ).sum(dim=1) * (temperature**2)


def train_student(
    method: str,
    model: nn.Module,
    teacher: nn.Module,
    loaders,
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
        optimizer, T_max=args.epochs_student
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device=device)
    generator.manual_seed(run_seed + 193_337)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    best_score, best_epoch, best_state = -float("inf"), 0, None
    history = []
    for epoch in range(1, args.epochs_student + 1):
        model.train()
        losses = []
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            mask = sample_uniform_mask(
                target.shape[0],
                args.missing_probability,
                device,
                generator,
            )
            full_mask = torch.ones_like(mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                teacher_logits = teacher(
                    inputs, modality_mask=full_mask
                )["logits"]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                masked_logits = model(inputs, modality_mask=mask)["logits"]
                hard = criterion(masked_logits, target)
                kd_each = per_sample_kd(
                    masked_logits, teacher_logits, args.temperature
                )
                if method == "original_kd":
                    kd = kd_each.mean()
                    loss = (1.0 - args.alpha) * hard + args.alpha * kd
                else:
                    teacher_probability = teacher_logits.softmax(dim=1)
                    confidence, teacher_prediction = (
                        teacher_probability.max(dim=1)
                    )
                    correct_weight = torch.where(
                        teacher_prediction.eq(target),
                        torch.ones_like(confidence),
                        torch.full_like(confidence, 0.25),
                    )
                    confidence_weight = (
                        (confidence - args.teacher_confidence)
                        / max(1.0 - args.teacher_confidence, 1e-6)
                    ).clamp(0.0, 1.0)
                    severity = 1.0 - mask.float().mean(dim=1)
                    weight = correct_weight * confidence_weight * (
                        0.5 + severity
                    )
                    kd = (kd_each * weight).sum() / weight.sum().clamp_min(1.0)
                    full_logits = model(
                        inputs, modality_mask=full_mask
                    )["logits"]
                    consistency = per_sample_kd(
                        masked_logits,
                        full_logits.detach(),
                        args.temperature,
                    ).mean()
                    loss = (
                        hard
                        + args.lambda_kd * kd
                        + args.lambda_consistency * consistency
                        + args.lambda_full * criterion(full_logits, target)
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        score, validation = selection_score(
            model,
            loaders["validation"],
            validation_trial_ids,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_score": score,
                **validation,
            }
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(f"{method}: no student checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args.data)
    folds = subject_folds(arrays["subject"], args.folds, args.partition_seed)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    channels = {
        modality: int(arrays[modality].shape[1])
        for modality in MODALITIES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(folds) * len(args.seeds) * len(args.methods)
    completed = 0
    status_path = args.output_dir / "status.json"
    write_status(status_path, completed, total, f"Starting on {device}")
    all_metrics = []
    for fold, indices in enumerate(folds, start=1):
        train_idx, validation_idx, test_idx = indices
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaler = FoldRobustScaler.fit(arrays, train_idx)
        (fold_dir / "scaler.json").write_text(
            json.dumps(scaler.as_serializable()), encoding="utf-8"
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
            run_seed = int(seed * 100 + fold)
            seed_everything(run_seed)
            loaders = make_loaders(
                arrays,
                indices,
                scaler,
                args.batch_size,
                device,
                run_seed + 77,
            )
            teacher_dir = fold_dir / f"seed_{seed}" / "teacher"
            teacher_dir.mkdir(parents=True, exist_ok=True)
            teacher_path = teacher_dir / "best_model.pt"
            teacher = XTinyHARTeacherAdaptation(
                channels=channels, classes=len(SPEEDS)
            ).to(device)
            if teacher_path.exists():
                teacher_payload = torch.load(
                    teacher_path, map_location=device, weights_only=True
                )
                teacher.load_state_dict(teacher_payload["state_dict"])
            else:
                teacher, teacher_epoch, teacher_history = train_teacher(
                    teacher,
                    loaders,
                    arrays["trial_id"][validation_idx],
                    args,
                    device,
                )
                pd.DataFrame(teacher_history).to_csv(
                    teacher_dir / "history.csv", index=False
                )
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in teacher.state_dict().items()
                        },
                        "best_epoch": teacher_epoch,
                        "channels": channels,
                    },
                    teacher_path,
                )
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
                seed_everything(run_seed)
                student = XTinyHARAdaptation(
                    channels=channels, classes=len(SPEEDS)
                ).to(device)
                student, best_epoch, history = train_student(
                    method,
                    student,
                    teacher,
                    loaders,
                    arrays["trial_id"][validation_idx],
                    args,
                    device,
                    run_seed,
                )
                metrics, predictions = evaluate_all_masks(
                    f"xtinyhar_{method}",
                    student,
                    loaders["test"],
                    arrays,
                    test_idx,
                    seed,
                    fold,
                    device,
                )
                metrics["best_epoch"] = best_epoch
                metrics["parameter_count"] = sum(
                    parameter.numel()
                    for parameter in student.parameters()
                )
                metrics["teacher_parameter_count"] = sum(
                    parameter.numel()
                    for parameter in teacher.parameters()
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
                            for key, value in student.state_dict().items()
                        },
                        "best_epoch": best_epoch,
                        "channels": channels,
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
                            "student_parameters": int(
                                metrics["parameter_count"].iloc[0]
                            ),
                            "teacher_parameters": int(
                                metrics["teacher_parameter_count"].iloc[0]
                            ),
                        }
                    ),
                    flush=True,
                )
                del student
                torch.cuda.empty_cache()
            del teacher
            torch.cuda.empty_cache()
    pd.concat(all_metrics, ignore_index=True).to_csv(
        args.output_dir / "all_combination_metrics.csv", index=False
    )
    write_status(
        status_path,
        completed,
        total,
        "XTinyHAR teacher distillation complete",
        stage="complete",
    )


if __name__ == "__main__":
    main()
