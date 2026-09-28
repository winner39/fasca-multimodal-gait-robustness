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
from gait_robust.external_baseline_models import (
    ADAPTAdaptation,
    CIMSleepNetAdaptation,
    CentaurAdaptation,
)
from gait_robust.rapid_distill import balanced_masks, validation_score
from gait_robust.sci_cross_validate import (
    MODALITIES,
    SPEEDS,
    evaluate_all_masks,
    load_arrays,
    make_loaders,
    move_inputs,
    seed_everything,
)


METHODS = ("centaur_adaptation", "adapt_adaptation", "cimsleepnet_adaptation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51])
    parser.add_argument("--partition-seed", type=int, default=20260901)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--worst-weight", type=float, default=0.2)
    parser.add_argument("--lambda-centaur-reconstruction", type=float, default=0.25)
    parser.add_argument("--lambda-adapt-alignment", type=float, default=0.20)
    parser.add_argument("--lambda-cim-imagination", type=float, default=0.50)
    parser.add_argument("--lambda-cim-contrastive", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.10)
    return parser.parse_args()


def make_model(
    method: str,
    channels: dict[str, int],
    embedding_dim: int,
) -> nn.Module:
    common = {
        "channels": channels,
        "classes": len(SPEEDS),
        "embedding_dim": embedding_dim,
    }
    if method == "centaur_adaptation":
        return CentaurAdaptation(**common)
    if method == "adapt_adaptation":
        return ADAPTAdaptation(**common, anchor_modality="imu")
    if method == "cimsleepnet_adaptation":
        return CIMSleepNetAdaptation(**common)
    raise ValueError(method)


def centaur_corrupt(
    inputs: dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Generic stochastic corruption; deliberately differs from test cases."""

    output = {key: value.clone() for key, value in inputs.items()}
    for index, modality in enumerate(MODALITIES):
        signal = output[modality]
        batch, channels, time = signal.shape
        for row in range(batch):
            if not bool(modality_mask[row, index]):
                signal[row].zero_()
                continue
            draw = float(torch.rand((), device=signal.device, generator=generator))
            if draw < 0.34:
                # Random channel loss with a continuously sampled fraction.
                fraction = 0.10 + 0.40 * float(
                    torch.rand((), device=signal.device, generator=generator)
                )
                count = max(1, int(round(channels * fraction)))
                selected = torch.randperm(
                    channels, device=signal.device, generator=generator
                )[:count]
                signal[row, selected] = 0.0
            elif draw < 0.67:
                # Randomly located consecutive temporal loss.
                fraction = 0.10 + 0.30 * float(
                    torch.rand((), device=signal.device, generator=generator)
                )
                length = max(1, int(round(time * fraction)))
                start = int(
                    torch.randint(
                        max(time - length + 1, 1),
                        (),
                        device=signal.device,
                        generator=generator,
                    )
                )
                signal[row, :, start : start + length] = 0.0
            else:
                # Signal-relative Gaussian noise with a continuous scale.
                scale = 0.05 + 0.20 * float(
                    torch.rand((), device=signal.device, generator=generator)
                )
                rms = signal[row].square().mean().sqrt().clamp_min(0.05)
                noise = torch.randn(
                    signal[row].shape,
                    device=signal.device,
                    dtype=signal.dtype,
                    generator=generator,
                )
                signal[row].add_(noise * rms * scale)
    return output


def anchor_alignment_loss(
    projected: torch.Tensor,
    modality_mask: torch.Tensor,
    anchor_index: int,
    temperature: float,
) -> torch.Tensor:
    losses = []
    anchor = F.normalize(projected[:, anchor_index], dim=-1)
    for index in range(projected.shape[1]):
        if index == anchor_index:
            continue
        valid = modality_mask[:, anchor_index] & modality_mask[:, index]
        if int(valid.sum()) < 2:
            continue
        left = anchor[valid]
        right = F.normalize(projected[valid, index], dim=-1)
        logits = left @ right.T / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append(
            0.5
            * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )
        )
    if not losses:
        return projected.new_zeros(())
    return torch.stack(losses).mean()


def supervised_contrastive_loss(
    views: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Two-view supervised contrastive loss used for CIM calibration."""

    batch, n_views, dimension = views.shape
    features = F.normalize(views, dim=-1).reshape(batch * n_views, dimension)
    repeated_labels = labels.repeat_interleave(n_views)
    logits = features @ features.T / temperature
    identity = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    positive = repeated_labels[:, None].eq(repeated_labels[None, :]) & ~identity
    exp_logits = torch.exp(logits) * (~identity)
    log_probability = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return logits.new_zeros(())
    mean_positive = (log_probability * positive).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_positive[valid].mean()


def train_model(
    method: str,
    model: nn.Module,
    loaders,
    validation_trial_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    run_seed: int,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(run_seed + 91_337)
    corruption_generator = torch.Generator(device=device)
    corruption_generator.manual_seed(run_seed + 187_331)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rows = []
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            mask, _ = balanced_masks(target.shape[0], device, mask_generator)
            model_inputs = inputs
            forward_kwargs = {}
            if method == "centaur_adaptation":
                model_inputs = centaur_corrupt(inputs, mask, corruption_generator)
                forward_kwargs["reconstruction_targets"] = inputs
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(
                    model_inputs, modality_mask=mask, **forward_kwargs
                )
                task_loss = F.cross_entropy(
                    output["logits"], target, label_smoothing=0.05
                )
                auxiliary = task_loss.new_zeros(())
                if method == "centaur_adaptation":
                    auxiliary = output["reconstruction_loss"]
                    loss = task_loss + args.lambda_centaur_reconstruction * auxiliary
                elif method == "adapt_adaptation":
                    auxiliary = anchor_alignment_loss(
                        output["projected_tokens"],
                        mask,
                        model.anchor_index,
                        args.temperature,
                    )
                    loss = task_loss + args.lambda_adapt_alignment * auxiliary
                elif method == "cimsleepnet_adaptation":
                    contrastive = supervised_contrastive_loss(
                        output["contrastive_views"], target, args.temperature
                    )
                    auxiliary = output["imagination_loss"]
                    loss = (
                        task_loss
                        + args.lambda_cim_imagination * auxiliary
                        + args.lambda_cim_contrastive * contrastive
                    )
                else:
                    raise ValueError(method)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_rows.append(
                (
                    float(loss.detach()),
                    float(task_loss.detach()),
                    float(auxiliary.detach()),
                )
            )
        scheduler.step()
        score, validation = validation_score(
            model,
            loaders["validation"],
            validation_trial_ids,
            device,
            args.worst_weight,
        )
        losses = np.asarray(epoch_rows)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(losses[:, 0].mean()),
                "train_task_loss": float(losses[:, 1].mean()),
                "train_primary_auxiliary_loss": float(losses[:, 2].mean()),
                "selection_score": score,
                **validation,
            }
        )
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
        raise RuntimeError(f"No checkpoint selected for {method}")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def write_status(path: Path, **payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    arrays = load_arrays(args.data)
    folds = subject_folds(arrays["subject"], args.folds, args.partition_seed)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    channels = {
        modality: int(arrays[modality].shape[1]) for modality in MODALITIES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(folds) * len(args.seeds) * len(args.methods)
    completed = 0
    status_path = args.output_dir / "status.json"
    write_status(
        status_path,
        stage="running",
        completed=completed,
        total=total,
        message=f"Starting on {device}",
    )
    all_metrics = []

    for fold, indices in enumerate(folds, start=1):
        train_idx, validation_idx, test_idx = indices
        scaler = FoldRobustScaler.fit(arrays, train_idx)
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "scaler.json").write_text(
            json.dumps(scaler.as_serializable()), encoding="utf-8"
        )
        split_payload = {
            "fold": fold,
            "train_subjects": sorted(
                np.unique(arrays["subject"][train_idx]).astype(int).tolist()
            ),
            "validation_subjects": sorted(
                np.unique(arrays["subject"][validation_idx]).astype(int).tolist()
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
                model = make_model(method, channels, args.embedding_dim).to(device)
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
                parameter_count = sum(p.numel() for p in model.parameters())
                metrics["best_epoch"] = best_epoch
                metrics["parameter_count"] = parameter_count
                metrics.to_csv(metrics_path, index=False)
                predictions.to_csv(
                    predictions_path, index=False, compression="gzip"
                )
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
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
                        "adaptation": True,
                    },
                    run_dir / "best_model.pt",
                )
                all_metrics.append(metrics)
                completed += 1
                summary = {
                    "completed": completed,
                    "total": total,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "best_epoch": best_epoch,
                    "parameter_count": parameter_count,
                    "full_trial_macro_f1": float(
                        metrics.loc[
                            metrics["available_modalities"] == "+".join(MODALITIES),
                            "trial_macro_f1",
                        ].iloc[0]
                    ),
                    "mean_15_trial_macro_f1": float(
                        metrics["trial_macro_f1"].mean()
                    ),
                    "worst_15_trial_macro_f1": float(
                        metrics["trial_macro_f1"].min()
                    ),
                }
                print(json.dumps(summary), flush=True)
                write_status(
                    status_path,
                    stage="running",
                    completed=completed,
                    total=total,
                    message=f"Completed fold {fold}, seed {seed}, {method}",
                    latest=summary,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    pd.concat(all_metrics, ignore_index=True).to_csv(
        args.output_dir / "all_combination_metrics.csv", index=False
    )
    write_status(
        status_path,
        stage="complete",
        completed=completed,
        total=total,
        message="External baseline adaptation comparison complete",
    )


if __name__ == "__main__":
    main()

