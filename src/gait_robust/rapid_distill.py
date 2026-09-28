from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from gait_robust.augmentations import augment_modalities
from gait_robust.cross_validate_full import subject_folds
from gait_robust.data import FoldRobustScaler
from gait_robust.models import MultimodalLiteNet
from gait_robust.robust_models import (
    ActionMAELite,
    EmbraceNetLite,
    QualityRapidGait,
    QualityEmbraceNetLite,
    RapidGait,
)
from gait_robust.sci_cross_validate import (
    MODALITIES,
    SPEEDS,
    evaluate_all_masks,
    load_arrays,
    make_loaders,
    move_inputs,
    predict,
    seed_everything,
    trial_metrics,
    write_status,
)
from gait_robust.xtinyhar import XTinyHARTeacherAdaptation
from gait_robust.xtinyhar_distill import per_sample_kd, train_teacher


METHODS = (
    "balanced_ce",
    "balanced_kd",
    "groupdro_kd",
    "sensor_aug_kd",
    "sensor_aug_no_curriculum_kd",
    "iid_aug_kd",
    "jepa_aug_kd",
    "shared_jepa_kd",
    "shared_jepa_embrace_kd",
    "quality_aug_kd",
    "moddrop_fasca_kd",
    "moddrop_fasca_ssl_kd",
    "embracenet_fasca_kd",
    "embracenet_fasca_ssl_kd",
    "embracenet_fasca_structssl_kd",
    "embracenet_fasca_qstructssl_kd",
    "embracenet_fasca_mar_kd",
    "embracenet_fasca_qstructssl_mar_kd",
    "embracenet_quality_fasca_kd",
    "actionmae_fasca_kd",
    "uniform_rapid_fasca_kd",
)
MASK_ROWS = tuple(
    bits
    for bits in itertools.product((False, True), repeat=len(MODALITIES))
    if any(bits)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, default=None)
    parser.add_argument("--embracenet-root", type=Path, default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53])
    parser.add_argument("--partition-seed", type=int, default=20260901)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs-teacher", type=int, default=50)
    parser.add_argument("--epochs-student", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--groupdro-eta", type=float, default=0.05)
    parser.add_argument("--lambda-alignment", type=float, default=0.0)
    parser.add_argument("--lambda-proxy", type=float, default=0.0)
    parser.add_argument("--lambda-reliability", type=float, default=0.0)
    parser.add_argument("--worst-weight", type=float, default=0.0)
    parser.add_argument("--augmentation-probability", type=float, default=0.55)
    parser.add_argument("--severity-min", type=float, default=0.15)
    parser.add_argument("--severity-max", type=float, default=0.75)
    parser.add_argument("--desync-probability", type=float, default=0.15)
    parser.add_argument("--max-shift-samples", type=int, default=25)
    parser.add_argument("--lambda-quality", type=float, default=0.10)
    parser.add_argument("--lambda-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-clean", type=float, default=0.0)
    parser.add_argument("--lambda-jepa", type=float, default=0.10)
    parser.add_argument("--jepa-ema-decay", type=float, default=0.99)
    parser.add_argument("--lambda-ssl", type=float, default=0.10)
    parser.add_argument("--ssl-ema-decay", type=float, default=0.99)
    parser.add_argument("--ssl-projector-dim", type=int, default=64)
    parser.add_argument(
        "--ssl-structured-probability", type=float, default=0.70
    )
    parser.add_argument(
        "--ssl-structured-severity-min", type=float, default=0.35
    )
    parser.add_argument(
        "--ssl-structured-severity-max", type=float, default=0.75
    )
    parser.add_argument("--mar-warmup-epochs", type=int, default=5)
    parser.add_argument("--mar-rho", type=float, default=1.0)
    parser.add_argument("--mar-weight-floor", type=float, default=0.25)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument(
        "--reliability-weight-floor", type=float, default=0.25
    )
    parser.add_argument(
        "--augmentation-warmup-epochs", type=int, default=0
    )
    parser.add_argument("--quality-contrast", type=float, default=1.0)
    parser.add_argument(
        "--augmentation-profile",
        choices=("all", "drop_gain", "iid_drop_gain"),
        default="all",
    )
    parser.add_argument(
        "--quality-routing-strength", type=float, default=0.5
    )
    parser.add_argument(
        "--corruption-selection-weight", type=float, default=0.20
    )
    return parser.parse_args()


def balanced_masks(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw nearly equal counts from all 15 non-empty modality masks."""
    mask_bank = torch.tensor(MASK_ROWS, dtype=torch.bool, device=device)
    repeats = (batch_size + len(MASK_ROWS) - 1) // len(MASK_ROWS)
    group_ids = torch.arange(
        len(MASK_ROWS), device=device, dtype=torch.long
    ).repeat(repeats)[:batch_size]
    order = torch.randperm(batch_size, device=device, generator=generator)
    group_ids = group_ids[order]
    return mask_bank[group_ids], group_ids


@torch.no_grad()
def validation_score(
    model: nn.Module,
    loader,
    validation_trial_ids: np.ndarray,
    device: torch.device,
    worst_weight: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    values = []
    full = None
    for bits in MASK_ROWS:
        mask = torch.tensor(bits, dtype=torch.bool).unsqueeze(0)
        truth, probabilities = predict(model, loader, device, mask)
        metrics, _ = trial_metrics(
            truth, probabilities, validation_trial_ids
        )
        value = float(metrics["macro_f1"])
        values.append(value)
        if all(bits):
            full = value
    mean_15 = float(np.mean(values))
    worst = float(np.min(values))
    score = (1.0 - worst_weight) * mean_15 + worst_weight * worst
    return score, {
        "validation_mean_15_trial_macro_f1": mean_15,
        "validation_full_trial_macro_f1": float(full),
        "validation_worst_trial_macro_f1": worst,
    }


@torch.no_grad()
def corruption_validation_score(
    model: nn.Module,
    loader,
    validation_trial_ids: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(804_221)
    truth, probabilities = [], []
    quality_errors = []
    for inputs, target in loader:
        inputs = move_inputs(inputs, device)
        target = target.to(device)
        mask = torch.ones(
            target.shape[0],
            len(MODALITIES),
            dtype=torch.bool,
            device=device,
        )
        corrupted, quality_target, _ = augment_modalities(
            inputs,
            mask,
            generator,
            probability=max(args.augmentation_probability, 0.7),
            severity_min=max(args.severity_min, 0.35),
            severity_max=max(args.severity_max, 0.75),
            desync_probability=(
                0.0
                if args.augmentation_profile == "drop_gain"
                else max(args.desync_probability, 0.20)
            ),
            max_shift_samples=args.max_shift_samples,
            quality_contrast=args.quality_contrast,
            profile=args.augmentation_profile,
        )
        output = model(corrupted, modality_mask=mask)
        truth.append(target.cpu().numpy())
        probabilities.append(
            output["logits"].softmax(dim=1).cpu().numpy()
        )
        if "predicted_quality" in output:
            quality_errors.append(
                (output["predicted_quality"] - quality_target)
                .abs()
                .mean()
                .item()
            )
    metrics, _ = trial_metrics(
        np.concatenate(truth),
        np.concatenate(probabilities),
        validation_trial_ids,
    )
    quality_mae = (
        float(np.mean(quality_errors)) if quality_errors else float("nan")
    )
    return float(metrics["macro_f1"]), quality_mae


def auxiliary_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    loss = output["logits"].new_zeros(())
    if args.lambda_alignment:
        loss = loss + args.lambda_alignment * output["alignment_loss"]
    missing = ~output["modality_mask"]
    if args.lambda_proxy and missing.any():
        expanded_target = target[:, None].expand(-1, len(MODALITIES))
        proxy_loss = nn.functional.cross_entropy(
            output["proxy_logits"][missing],
            expanded_target[missing],
            label_smoothing=0.05,
        )
        loss = loss + args.lambda_proxy * proxy_loss
    if args.lambda_reliability:
        loss = (
            loss
            + args.lambda_reliability * output["reliability_loss"]
        )
    return loss


class SharedLatentProjector(nn.Module):
    """Training-only bottleneck for modality-invariant latent targets."""

    def __init__(self, embedding_dim: int, shared_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, shared_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@torch.no_grad()
def update_ema_module(
    target: nn.Module,
    online: nn.Module,
    decay: float,
) -> None:
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters()
    ):
        target_parameter.mul_(decay).add_(
            online_parameter.detach(), alpha=1.0 - decay
        )
    for target_buffer, online_buffer in zip(
        target.buffers(), online.buffers()
    ):
        target_buffer.copy_(online_buffer)


def grouped_objective(
    per_sample: torch.Tensor,
    group_ids: torch.Tensor,
    group_weights: torch.Tensor,
    method: str,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_losses = []
    present_groups = []
    for group_index in range(len(MASK_ROWS)):
        selected = group_ids.eq(group_index)
        present_groups.append(selected.any())
        if selected.any():
            group_losses.append(per_sample[selected].mean())
        else:
            group_losses.append(per_sample.new_zeros(()))
    stacked = torch.stack(group_losses)
    present = torch.stack(present_groups)
    if method == "groupdro_kd":
        with torch.no_grad():
            updated = group_weights * torch.exp(
                eta * stacked.detach() * present
            )
            group_weights = updated / updated.sum()
        active_weights = group_weights * present
        active_weights = active_weights / active_weights.sum()
        loss = (active_weights * stacked).sum()
    else:
        loss = stacked[present].mean()
    return loss, group_weights, stacked.detach()


def train_student(
    method: str,
    model: nn.Module,
    teacher: nn.Module,
    loaders,
    validation_trial_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    run_seed: int,
) -> tuple[nn.Module, int, list[dict[str, float]], torch.Tensor]:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    jepa_method = method == "jepa_aug_kd"
    shared_jepa_method = method in {
        "shared_jepa_kd",
        "shared_jepa_embrace_kd",
    }
    ssl_method = method in {
        "moddrop_fasca_ssl_kd",
        "embracenet_fasca_ssl_kd",
        "embracenet_fasca_structssl_kd",
        "embracenet_fasca_qstructssl_kd",
        "embracenet_fasca_qstructssl_mar_kd",
    }
    structured_ssl_method = method in {
        "embracenet_fasca_structssl_kd",
        "embracenet_fasca_qstructssl_kd",
        "embracenet_fasca_qstructssl_mar_kd",
    }
    quality_gated_structured_ssl = (
        method
        in {
            "embracenet_fasca_qstructssl_kd",
            "embracenet_fasca_qstructssl_mar_kd",
        }
    )
    mar_method = method in {
        "embracenet_fasca_mar_kd",
        "embracenet_fasca_qstructssl_mar_kd",
    }
    target_encoder = None
    online_projector = None
    target_projector = None
    shared_predictor = None
    ssl_target_model = None
    structured_ssl_target_encoder = None
    ssl_online_projector = None
    ssl_target_projector = None
    ssl_predictor = None
    if jepa_method or shared_jepa_method:
        if not hasattr(model, "encoder_bank"):
            raise TypeError("JEPA training requires a modality encoder bank")
        target_encoder = copy.deepcopy(model.encoder_bank).to(device).eval()
        for parameter in target_encoder.parameters():
            parameter.requires_grad_(False)
    trainable_parameters = list(model.parameters())
    if shared_jepa_method:
        if not 0.0 <= args.reliability_weight_floor <= 1.0:
            raise ValueError(
                "reliability_weight_floor must be between zero and one"
            )
        online_projector = SharedLatentProjector(
            args.embedding_dim, args.shared_dim
        ).to(device)
        target_projector = copy.deepcopy(online_projector).to(device).eval()
        for parameter in target_projector.parameters():
            parameter.requires_grad_(False)
        shared_predictor = nn.Sequential(
            nn.LayerNorm(args.shared_dim),
            nn.Linear(args.shared_dim, args.shared_dim * 2),
            nn.GELU(),
            nn.Linear(args.shared_dim * 2, args.shared_dim),
        ).to(device)
        trainable_parameters.extend(online_projector.parameters())
        trainable_parameters.extend(shared_predictor.parameters())
    if ssl_method:
        if not 0.0 <= args.ssl_ema_decay < 1.0:
            raise ValueError("ssl_ema_decay must be in [0, 1)")
        if args.lambda_ssl < 0.0:
            raise ValueError("lambda_ssl must be non-negative")
        if args.ssl_projector_dim <= 0:
            raise ValueError("ssl_projector_dim must be positive")
        if structured_ssl_method:
            if not hasattr(model, "encoder_bank"):
                raise TypeError(
                    "Structured SSL requires a modality encoder bank"
                )
            structured_ssl_target_encoder = copy.deepcopy(
                model.encoder_bank
            ).to(device).eval()
            target_module = structured_ssl_target_encoder
        else:
            ssl_target_model = copy.deepcopy(model).to(device).eval()
            target_module = ssl_target_model
        for parameter in target_module.parameters():
            parameter.requires_grad_(False)
        ssl_online_projector = SharedLatentProjector(
            args.embedding_dim, args.ssl_projector_dim
        ).to(device)
        ssl_target_projector = copy.deepcopy(
            ssl_online_projector
        ).to(device).eval()
        for parameter in ssl_target_projector.parameters():
            parameter.requires_grad_(False)
        ssl_predictor = nn.Sequential(
            nn.LayerNorm(args.ssl_projector_dim),
            nn.Linear(
                args.ssl_projector_dim, args.ssl_projector_dim * 2
            ),
            nn.GELU(),
            nn.Linear(
                args.ssl_projector_dim * 2, args.ssl_projector_dim
            ),
        ).to(device)
        trainable_parameters.extend(ssl_online_projector.parameters())
        trainable_parameters.extend(ssl_predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_student
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(run_seed + 293_337)
    structured_ssl_generator = torch.Generator(device=device)
    structured_ssl_generator.manual_seed(run_seed + 493_337)
    group_weights = torch.full(
        (len(MASK_ROWS),), 1.0 / len(MASK_ROWS), device=device
    )
    mar_gap_sum = torch.zeros(len(MASK_ROWS), device=device)
    mar_gap_count = torch.zeros(len(MASK_ROWS), device=device)
    mar_weights = torch.ones(len(MASK_ROWS), device=device)
    if mar_method:
        if args.mar_warmup_epochs <= 0:
            raise ValueError("mar_warmup_epochs must be positive")
        if args.mar_rho <= 0.0:
            raise ValueError("mar_rho must be positive")
        if not 0.0 <= args.mar_weight_floor <= 1.0:
            raise ValueError("mar_weight_floor must be in [0, 1]")
    best_score, best_epoch, best_state = -float("inf"), 0, None
    best_group_weights = group_weights.detach().cpu()
    history: list[dict[str, float]] = []
    quality_method = method in {
        "quality_aug_kd",
        "embracenet_quality_fasca_kd",
    }
    augmentation_method = method in {
        "sensor_aug_kd",
        "sensor_aug_no_curriculum_kd",
        "iid_aug_kd",
        "jepa_aug_kd",
        "shared_jepa_kd",
        "shared_jepa_embrace_kd",
        "quality_aug_kd",
        "moddrop_fasca_kd",
        "moddrop_fasca_ssl_kd",
        "embracenet_fasca_kd",
        "embracenet_fasca_ssl_kd",
        "embracenet_fasca_structssl_kd",
        "embracenet_fasca_qstructssl_kd",
        "embracenet_fasca_mar_kd",
        "embracenet_fasca_qstructssl_mar_kd",
        "embracenet_quality_fasca_kd",
        "actionmae_fasca_kd",
        "uniform_rapid_fasca_kd",
        "top3_ensemble_aug_kd",
        "hybrid_top3_aug_kd",
    }
    for epoch in range(1, args.epochs_student + 1):
        model.train()
        epoch_losses = []
        epoch_quality_losses = []
        epoch_consistency_losses = []
        epoch_clean_losses = []
        epoch_jepa_losses = []
        epoch_ssl_losses = []
        group_loss_sum = torch.zeros(len(MASK_ROWS), device=device)
        group_loss_batches = 0
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            mask, group_ids = balanced_masks(
                target.shape[0], device, generator
            )
            full_mask = torch.ones_like(mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                teacher_mask = (
                    mask
                    if getattr(teacher, "uses_student_mask", False)
                    else full_mask
                )
                teacher_output = teacher(
                    inputs, modality_mask=teacher_mask
                )
                teacher_logits = teacher_output["logits"]
                auxiliary_teacher_logits = teacher_output.get(
                    "auxiliary_logits"
                )
                target_embeddings = (
                    target_encoder(inputs)
                    if target_encoder is not None
                    else None
                )
                target_shared = None
                if (
                    target_embeddings is not None
                    and target_projector is not None
                ):
                    target_shared_tokens = nn.functional.normalize(
                        target_projector(target_embeddings.float()),
                        dim=-1,
                    )
                    target_shared = nn.functional.normalize(
                        target_shared_tokens.mean(dim=1), dim=-1
                    )
                ssl_target = None
                if ssl_target_model is not None:
                    clean_target_output = ssl_target_model(
                        inputs, modality_mask=mask
                    )
                    ssl_target = nn.functional.normalize(
                        ssl_target_projector(
                            clean_target_output["fused"].float()
                        ),
                        dim=-1,
                    )
                elif structured_ssl_target_encoder is not None:
                    clean_target_tokens = structured_ssl_target_encoder(
                        inputs
                    )
                    ssl_target = nn.functional.normalize(
                        ssl_target_projector(
                            clean_target_tokens.float()
                        ),
                        dim=-1,
                    )
            student_inputs = inputs
            quality_target = None
            if augmentation_method:
                if args.augmentation_warmup_epochs > 0:
                    curriculum = min(
                        1.0,
                        epoch / args.augmentation_warmup_epochs,
                    )
                else:
                    curriculum = 1.0
                epoch_probability = (
                    args.augmentation_probability * curriculum
                )
                epoch_severity_max = (
                    args.severity_min
                    + curriculum
                    * (args.severity_max - args.severity_min)
                )
                student_inputs, quality_target, _ = augment_modalities(
                    inputs,
                    mask,
                    generator,
                    probability=epoch_probability,
                    severity_min=args.severity_min,
                    severity_max=epoch_severity_max,
                    desync_probability=(
                        args.desync_probability * curriculum
                    ),
                    max_shift_samples=args.max_shift_samples,
                    quality_contrast=args.quality_contrast,
                    profile=args.augmentation_profile,
                )
            structured_ssl_inputs = None
            structured_ssl_quality = None
            if structured_ssl_method:
                (
                    structured_ssl_inputs,
                    structured_ssl_quality,
                    _,
                ) = augment_modalities(
                    inputs,
                    mask,
                    structured_ssl_generator,
                    probability=args.ssl_structured_probability,
                    severity_min=args.ssl_structured_severity_min,
                    severity_max=args.ssl_structured_severity_max,
                    desync_probability=0.0,
                    max_shift_samples=args.max_shift_samples,
                    quality_contrast=args.quality_contrast,
                    profile="structured_dropout",
                )
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(student_inputs, modality_mask=mask)
                hard_each = nn.functional.cross_entropy(
                    output["logits"],
                    target,
                    reduction="none",
                    label_smoothing=0.05,
                )
                teacher_hard_each = nn.functional.cross_entropy(
                    teacher_logits,
                    target,
                    reduction="none",
                    label_smoothing=0.05,
                )
                if method == "balanced_ce":
                    per_sample = hard_each
                else:
                    kd_each = per_sample_kd(
                        output["logits"],
                        teacher_logits,
                        args.temperature,
                    )
                    auxiliary_alpha = (
                        float(getattr(args, "aux_alpha", 0.0))
                        if auxiliary_teacher_logits is not None
                        else 0.0
                    )
                    if not 0.0 <= auxiliary_alpha <= args.alpha:
                        raise ValueError(
                            "aux_alpha must be between zero and alpha"
                        )
                    per_sample = (
                        (1.0 - args.alpha) * hard_each
                        + (args.alpha - auxiliary_alpha) * kd_each
                    )
                    if auxiliary_teacher_logits is not None:
                        auxiliary_kd_each = per_sample_kd(
                            output["logits"],
                            auxiliary_teacher_logits,
                            args.temperature,
                        )
                        per_sample = (
                            per_sample
                            + auxiliary_alpha * auxiliary_kd_each
                        )
                loss, group_weights, group_losses = grouped_objective(
                    per_sample,
                    group_ids,
                    group_weights,
                    method,
                    args.groupdro_eta,
                )
                if mar_method:
                    if epoch <= args.mar_warmup_epochs:
                        with torch.no_grad():
                            difficulty_gap = (
                                hard_each.detach()
                                - teacher_hard_each.detach()
                            ).clamp_min(0.0)
                            mar_gap_sum.scatter_add_(
                                0, group_ids, difficulty_gap
                            )
                            mar_gap_count.scatter_add_(
                                0,
                                group_ids,
                                torch.ones_like(difficulty_gap),
                            )
                    else:
                        sample_weights = mar_weights[group_ids]
                        loss = (
                            per_sample * sample_weights
                        ).sum() / sample_weights.sum().clamp_min(1e-8)
                        group_weights = (
                            mar_weights / mar_weights.sum()
                        ).detach()
                loss = loss + auxiliary_loss(output, target, args)
                quality_loss = output["logits"].new_zeros(())
                consistency_loss = output["logits"].new_zeros(())
                clean_loss = output["logits"].new_zeros(())
                jepa_loss = output["logits"].new_zeros(())
                ssl_loss = output["logits"].new_zeros(())
                if target_embeddings is not None and jepa_method:
                    missing = ~mask
                    if missing.any():
                        predicted = nn.functional.normalize(
                            output["proxy_tokens"][missing].float(),
                            dim=-1,
                        )
                        target_latent = nn.functional.normalize(
                            target_embeddings[missing].float(),
                            dim=-1,
                        )
                        jepa_loss = (
                            1.0
                            - nn.functional.cosine_similarity(
                                predicted, target_latent, dim=-1
                            ).mean()
                        )
                        loss = loss + args.lambda_jepa * jepa_loss
                if target_shared is not None and shared_jepa_method:
                    missing = ~mask
                    if missing.any():
                        predicted_shared = nn.functional.normalize(
                            shared_predictor(
                                online_projector(
                                    output["proxy_tokens"].float()
                                )
                            ),
                            dim=-1,
                        )
                        expanded_target = target_shared[:, None, :].expand(
                            -1, len(MODALITIES), -1
                        )
                        per_slot_jepa = (
                            1.0
                            - nn.functional.cosine_similarity(
                                predicted_shared,
                                expanded_target,
                                dim=-1,
                            )
                        )
                        source_quality = quality_target[:, None, :]
                        route_quality = (
                            output["reliability_weights"].detach()
                            * source_quality
                        ).sum(dim=-1)
                        reliability_weight = (
                            args.reliability_weight_floor
                            + (1.0 - args.reliability_weight_floor)
                            * route_quality
                        )
                        selected_weights = reliability_weight[missing]
                        selected_weights = selected_weights / (
                            selected_weights.mean().detach() + 1e-6
                        )
                        jepa_loss = (
                            per_slot_jepa[missing] * selected_weights
                        ).mean()
                        loss = loss + args.lambda_jepa * jepa_loss
                if structured_ssl_inputs is not None:
                    structured_tokens = model.encoder_bank(
                        structured_ssl_inputs
                    )
                    ssl_prediction = nn.functional.normalize(
                        ssl_predictor(
                            ssl_online_projector(
                                structured_tokens.float()
                            )
                        ),
                        dim=-1,
                    )
                    per_token_ssl = (
                        1.0
                        - nn.functional.cosine_similarity(
                            ssl_prediction,
                            ssl_target.detach(),
                            dim=-1,
                        )
                    )
                    if quality_gated_structured_ssl:
                        corruption_weight = (
                            1.0 - structured_ssl_quality
                        ) * mask.to(structured_ssl_quality.dtype)
                        denominator = corruption_weight.sum()
                        if denominator > 0:
                            ssl_loss = (
                                per_token_ssl * corruption_weight
                            ).sum() / denominator
                        else:
                            ssl_loss = per_token_ssl.new_zeros(())
                    else:
                        ssl_loss = per_token_ssl[mask].mean()
                    loss = loss + args.lambda_ssl * ssl_loss
                elif ssl_target is not None:
                    ssl_prediction = nn.functional.normalize(
                        ssl_predictor(
                            ssl_online_projector(
                                output["fused"].float()
                            )
                        ),
                        dim=-1,
                    )
                    ssl_loss = (
                        1.0
                        - nn.functional.cosine_similarity(
                            ssl_prediction,
                            ssl_target.detach(),
                            dim=-1,
                        ).mean()
                    )
                    loss = loss + args.lambda_ssl * ssl_loss
                if augmentation_method:
                    if args.lambda_clean:
                        clean_logits = model(
                            inputs, modality_mask=mask
                        )["logits"]
                        clean_loss = nn.functional.cross_entropy(
                            clean_logits,
                            target,
                            label_smoothing=0.05,
                        )
                        loss = loss + args.lambda_clean * clean_loss
                if quality_method:
                    quality_loss = (
                        nn.functional.binary_cross_entropy_with_logits(
                        output["quality_logits"][mask],
                        quality_target[mask],
                    )
                    )
                    loss = loss + args.lambda_quality * quality_loss
                    if args.lambda_consistency:
                        clean_logits = model(
                            inputs, modality_mask=mask
                        )["logits"]
                        consistency_loss = per_sample_kd(
                            output["logits"],
                            clean_logits.detach(),
                            args.temperature,
                        ).mean()
                        loss = (
                            loss
                            + args.lambda_consistency * consistency_loss
                        )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            if target_encoder is not None:
                with torch.no_grad():
                    decay = float(args.jepa_ema_decay)
                    for target_parameter, online_parameter in zip(
                        target_encoder.parameters(),
                        model.encoder_bank.parameters(),
                    ):
                        target_parameter.mul_(decay).add_(
                            online_parameter.detach(),
                            alpha=1.0 - decay,
                        )
                    for target_buffer, online_buffer in zip(
                        target_encoder.buffers(),
                        model.encoder_bank.buffers(),
                    ):
                        target_buffer.copy_(online_buffer)
                    if target_projector is not None:
                        for target_parameter, online_parameter in zip(
                            target_projector.parameters(),
                            online_projector.parameters(),
                        ):
                            target_parameter.mul_(decay).add_(
                                online_parameter.detach(),
                                alpha=1.0 - decay,
                            )
                        for target_buffer, online_buffer in zip(
                            target_projector.buffers(),
                            online_projector.buffers(),
                        ):
                            target_buffer.copy_(online_buffer)
            if ssl_target_model is not None:
                update_ema_module(
                    ssl_target_model, model, args.ssl_ema_decay
                )
                update_ema_module(
                    ssl_target_projector,
                    ssl_online_projector,
                    args.ssl_ema_decay,
                )
            elif structured_ssl_target_encoder is not None:
                update_ema_module(
                    structured_ssl_target_encoder,
                    model.encoder_bank,
                    args.ssl_ema_decay,
                )
                update_ema_module(
                    ssl_target_projector,
                    ssl_online_projector,
                    args.ssl_ema_decay,
                )
            epoch_losses.append(float(loss.detach()))
            epoch_quality_losses.append(float(quality_loss.detach()))
            epoch_consistency_losses.append(
                float(consistency_loss.detach())
            )
            epoch_clean_losses.append(float(clean_loss.detach()))
            epoch_jepa_losses.append(float(jepa_loss.detach()))
            epoch_ssl_losses.append(float(ssl_loss.detach()))
            group_loss_sum += group_losses
            group_loss_batches += 1
        scheduler.step()
        if mar_method and epoch == args.mar_warmup_epochs:
            average_gap = mar_gap_sum / mar_gap_count.clamp_min(1.0)
            normalized_gap = average_gap / average_gap.max().clamp_min(
                1e-8
            )
            mar_weights = args.mar_rho * normalized_gap.square()
            mar_weights = (
                args.mar_weight_floor
                + (1.0 - args.mar_weight_floor) * mar_weights
            )
            mar_weights = mar_weights / mar_weights.mean().clamp_min(
                1e-8
            )
            group_weights = mar_weights / mar_weights.sum()
        clean_score, validation = validation_score(
            model,
            loaders["validation"],
            validation_trial_ids,
            device,
            args.worst_weight,
        )
        corruption_score = float("nan")
        quality_mae = float("nan")
        score = clean_score
        if augmentation_method:
            corruption_score, quality_mae = corruption_validation_score(
                model,
                loaders["validation"],
                validation_trial_ids,
                device,
                args,
            )
            weight = args.corruption_selection_weight
            score = (
                (1.0 - weight) * clean_score
                + weight * corruption_score
            )
        average_group_losses = group_loss_sum / max(group_loss_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "train_quality_loss": float(
                    np.mean(epoch_quality_losses)
                ),
                "train_consistency_loss": float(
                    np.mean(epoch_consistency_losses)
                ),
                "train_clean_loss": float(
                    np.mean(epoch_clean_losses)
                ),
                "train_jepa_loss": float(
                    np.mean(epoch_jepa_losses)
                ),
                "train_ssl_loss": float(
                    np.mean(epoch_ssl_losses)
                ),
                "selection_score": score,
                "clean_selection_score": clean_score,
                "validation_mixed_corruption_trial_macro_f1": (
                    corruption_score
                ),
                "validation_quality_mae": quality_mae,
                **validation,
                **{
                    f"train_group_loss_{index:02d}": float(value)
                    for index, value in enumerate(average_group_losses)
                },
                **{
                    f"group_weight_{index:02d}": float(value)
                    for index, value in enumerate(group_weights)
                },
            }
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_group_weights = group_weights.detach().cpu().clone()
        elif epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(f"{method}: no student checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history, best_group_weights


def teacher_checkpoint(
    root: Path | None,
    fold: int,
    seed: int,
    expected_split: dict[str, object],
) -> Path | None:
    if root is None:
        return None
    split_path = root / f"fold_{fold}" / "subjects.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Teacher split metadata is missing: {split_path}"
        )
    teacher_split = json.loads(split_path.read_text(encoding="utf-8"))
    split_keys = (
        "train_subjects",
        "validation_subjects",
        "test_subjects",
    )
    mismatched = [
        key
        for key in split_keys
        if teacher_split.get(key) != expected_split.get(key)
    ]
    if mismatched:
        raise ValueError(
            "Teacher/student subject split mismatch for "
            f"fold {fold}: {', '.join(mismatched)}. "
            "Refusing to reuse a potentially leaking teacher."
        )
    candidate = (
        root / f"fold_{fold}" / f"seed_{seed}" / "teacher" / "best_model.pt"
    )
    return candidate if candidate.exists() else None


def embracenet_checkpoint(
    root: Path | None,
    fold: int,
    seed: int,
    expected_split: dict[str, object],
) -> Path:
    if root is None:
        raise ValueError(
            "--embracenet-root is required for shared_jepa_embrace_kd"
        )
    split_path = root / f"fold_{fold}" / "subjects.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"EmbraceNet split metadata is missing: {split_path}"
        )
    actual_split = json.loads(split_path.read_text(encoding="utf-8"))
    split_keys = (
        "train_subjects",
        "validation_subjects",
        "test_subjects",
    )
    mismatched = [
        key
        for key in split_keys
        if actual_split.get(key) != expected_split.get(key)
    ]
    if mismatched:
        raise ValueError(
            "EmbraceNet/student subject split mismatch for "
            f"fold {fold}: {', '.join(mismatched)}. "
            "Refusing to reuse a potentially leaking teacher."
        )
    candidate = (
        root
        / f"fold_{fold}"
        / f"seed_{seed}"
        / "embracenet"
        / "best_model.pt"
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"EmbraceNet checkpoint is missing: {candidate}"
        )
    return candidate


def summarize_seed_predictions(output_dir: Path) -> None:
    prediction_paths = sorted(
        output_dir.glob(
            "fold_*/seed_*/*/window_predictions.csv.gz"
        )
    )
    if not prediction_paths:
        return
    predictions = pd.concat(
        (pd.read_csv(path) for path in prediction_paths),
        ignore_index=True,
    )
    probability_columns = [f"p_{speed:g}" for speed in SPEEDS]
    trial = (
        predictions.groupby(
            [
                "method",
                "seed",
                "available_modalities",
                "trial_id",
            ],
            sort=False,
        )
        .agg(
            true_class=("true_class", "first"),
            **{
                column: (column, "mean")
                for column in probability_columns
            },
        )
        .reset_index()
    )
    rows = []
    for keys, group in trial.groupby(
        ["method", "seed", "available_modalities"], sort=False
    ):
        probabilities = group[probability_columns].to_numpy()
        prediction = probabilities.argmax(axis=1)
        rows.append(
            {
                "method": keys[0],
                "seed": int(keys[1]),
                "available_modalities": keys[2],
                "n_modalities": len(str(keys[2]).split("+")),
                "test_trials": int(len(group)),
                "trial_macro_f1": float(
                    f1_score(
                        group["true_class"].to_numpy(dtype=int),
                        prediction,
                        labels=list(range(len(SPEEDS))),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    pooled = pd.DataFrame(rows)
    pooled.to_csv(
        output_dir / "seed_pooled_combination_metrics.csv", index=False
    )
    summaries = []
    full_name = "+".join(MODALITIES)
    for (method, seed), group in pooled.groupby(
        ["method", "seed"], sort=False
    ):
        full = float(
            group.loc[
                group["available_modalities"].eq(full_name),
                "trial_macro_f1",
            ].iloc[0]
        )
        incomplete = group.loc[
            ~group["available_modalities"].eq(full_name)
        ]
        worst_index = incomplete["trial_macro_f1"].idxmin()
        summaries.append(
            {
                "method": method,
                "seed": int(seed),
                "full": full,
                "incomplete": float(
                    incomplete["trial_macro_f1"].mean()
                ),
                "mean_15": float(group["trial_macro_f1"].mean()),
                "worst": float(
                    pooled.loc[worst_index, "trial_macro_f1"]
                ),
                "worst_combo": pooled.loc[
                    worst_index, "available_modalities"
                ],
            }
        )
    pd.DataFrame(summaries).to_csv(
        output_dir / "seed_pooled_trial_metrics.csv", index=False
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
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
            teacher_dir = fold_dir / f"seed_{seed}" / "teacher"
            teacher_dir.mkdir(parents=True, exist_ok=True)
            teacher_path = teacher_dir / "best_model.pt"
            source_teacher = teacher_checkpoint(
                args.teacher_root, fold, seed, split_payload
            )
            if source_teacher is not None:
                teacher_path = source_teacher
            teacher = XTinyHARTeacherAdaptation(
                channels=channels, classes=len(SPEEDS)
            ).to(device)
            teacher_loaders = make_loaders(
                arrays,
                indices,
                scaler,
                args.batch_size,
                device,
                run_seed + 77,
            )
            if teacher_path.exists():
                payload = torch.load(
                    teacher_path, map_location=device, weights_only=True
                )
                teacher.load_state_dict(payload["state_dict"])
            else:
                seed_everything(run_seed)
                teacher, teacher_epoch, teacher_history = train_teacher(
                    teacher,
                    teacher_loaders,
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
                    teacher_dir / "best_model.pt",
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
                loaders = make_loaders(
                    arrays,
                    indices,
                    scaler,
                    args.batch_size,
                    device,
                    run_seed + 77,
                )
                if method == "quality_aug_kd":
                    student = QualityRapidGait(
                        channels=channels,
                        classes=len(SPEEDS),
                        embedding_dim=args.embedding_dim,
                        quality_routing_strength=(
                            args.quality_routing_strength
                        ),
                    ).to(device)
                elif method == "embracenet_quality_fasca_kd":
                    student = QualityEmbraceNetLite(
                        channels=channels,
                        classes=len(SPEEDS),
                        embedding_dim=args.embedding_dim,
                        quality_routing_strength=(
                            args.quality_routing_strength
                        ),
                    ).to(device)
                elif method in {
                    "moddrop_fasca_kd",
                    "moddrop_fasca_ssl_kd",
                }:
                    student = MultimodalLiteNet(
                        channels=channels,
                        classes=len(SPEEDS),
                        embedding_dim=args.embedding_dim,
                    ).to(device)
                elif method in {
                    "embracenet_fasca_kd",
                    "embracenet_fasca_ssl_kd",
                    "embracenet_fasca_structssl_kd",
                    "embracenet_fasca_qstructssl_kd",
                    "embracenet_fasca_mar_kd",
                    "embracenet_fasca_qstructssl_mar_kd",
                }:
                    student = EmbraceNetLite(
                        channels=channels,
                        classes=len(SPEEDS),
                        embedding_dim=args.embedding_dim,
                    ).to(device)
                elif method == "actionmae_fasca_kd":
                    student = ActionMAELite(
                        channels=channels,
                        classes=len(SPEEDS),
                        embedding_dim=args.embedding_dim,
                    ).to(device)
                else:
                    student = RapidGait(
                        channels=channels,
                        classes=len(SPEEDS),
                        embedding_dim=args.embedding_dim,
                        uniform_reliability=(
                            method == "uniform_rapid_fasca_kd"
                        ),
                    ).to(device)
                method_teacher = teacher
                if method == "shared_jepa_embrace_kd":
                    embrace_path = embracenet_checkpoint(
                        args.embracenet_root,
                        fold,
                        seed,
                        split_payload,
                    )
                    embrace_payload = torch.load(
                        embrace_path,
                        map_location=device,
                        weights_only=True,
                    )
                    if embrace_payload.get("channels") != channels:
                        raise ValueError(
                            "EmbraceNet/student channel definitions differ"
                        )
                    method_teacher = EmbraceNetLite(
                        channels=channels,
                        classes=len(SPEEDS),
                        embedding_dim=int(
                            embrace_payload.get("embedding_dim", 64)
                        ),
                    ).to(device)
                    method_teacher.load_state_dict(
                        embrace_payload["state_dict"]
                    )
                student, best_epoch, history, group_weights = train_student(
                    method,
                    student,
                    method_teacher,
                    loaders,
                    arrays["trial_id"][validation_idx],
                    args,
                    device,
                    run_seed,
                )
                metrics, predictions = evaluate_all_masks(
                    f"rapid_{method}",
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
                    for parameter in method_teacher.parameters()
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
                        "mask_rows": MASK_ROWS,
                        "group_weights": group_weights,
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
                            "mean_15_trial_macro_f1": float(
                                metrics["trial_macro_f1"].mean()
                            ),
                            "worst_trial_macro_f1": float(
                                metrics["trial_macro_f1"].min()
                            ),
                        }
                    ),
                    flush=True,
                )
                if method_teacher is not teacher:
                    del method_teacher
                del student
                torch.cuda.empty_cache()
            del teacher
            torch.cuda.empty_cache()
    pd.concat(all_metrics, ignore_index=True).to_csv(
        args.output_dir / "all_combination_metrics.csv", index=False
    )
    summarize_seed_predictions(args.output_dir)
    write_status(
        status_path,
        completed,
        total,
        "RAPID-Gait distillation complete",
        stage="complete",
    )


if __name__ == "__main__":
    main()
