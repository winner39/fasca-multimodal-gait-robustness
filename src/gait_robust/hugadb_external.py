from __future__ import annotations

import argparse
import itertools
import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gait_robust.cross_validate_full import subject_folds
from gait_robust.models import GatedFusion, TemporalLiteEncoder


MODALITIES = ("emg", "imu")
ACTIVITIES = ("walking", "running", "stairs_up", "stairs_down")
MASK_ROWS = tuple(
    row
    for row in itertools.product((False, True), repeat=len(MODALITIES))
    if any(row)
)
METHODS = ("balanced_kd", "fasca_kd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External HuGaDB validation of balanced KD and FASCA."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 72, 73])
    parser.add_argument("--partition-seed", type=int, default=20260902)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument("--epochs-teacher", type=int, default=25)
    parser.add_argument("--epochs-student", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lambda-alignment", type=float, default=0.05)
    parser.add_argument("--lambda-proxy", type=float, default=0.05)
    parser.add_argument("--augmentation-probability", type=float, default=0.35)
    parser.add_argument(
        "--emg-augmentation-probability", type=float, default=None
    )
    parser.add_argument(
        "--imu-augmentation-probability", type=float, default=None
    )
    parser.add_argument("--severity-min", type=float, default=0.15)
    parser.add_argument("--severity-max", type=float, default=0.75)
    parser.add_argument("--augmentation-warmup-epochs", type=int, default=5)
    parser.add_argument("--lambda-clean", type=float, default=0.50)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_name(bits: tuple[bool, ...]) -> str:
    return "+".join(
        modality
        for modality, available in zip(MODALITIES, bits, strict=True)
        if available
    )


class ArrayDataset(Dataset):
    def __init__(
        self, arrays: dict[str, np.ndarray], indices: np.ndarray
    ) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, item: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        index = int(self.indices[item])
        inputs = {
            modality: torch.from_numpy(self.arrays[modality][index])
            for modality in MODALITIES
        }
        return inputs, torch.tensor(
            self.arrays["label"][index], dtype=torch.long
        )


def robust_scale(
    source: dict[str, np.ndarray], train_indices: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    scaled = {
        key: value
        for key, value in source.items()
        if key not in MODALITIES
    }
    metadata: dict[str, object] = {"clip": 10.0}
    for modality in MODALITIES:
        train = source[modality][train_indices]
        channel_values = train.transpose(1, 0, 2).reshape(
            train.shape[1], -1
        )
        median = np.median(channel_values, axis=1).astype(np.float32)
        q25, q75 = np.percentile(
            channel_values, [25.0, 75.0], axis=1
        ).astype(np.float32)
        scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0).astype(
            np.float32
        )
        values = (
            source[modality] - median[None, :, None]
        ) / scale[None, :, None]
        scaled[modality] = np.clip(values, -10.0, 10.0).astype(
            np.float32
        )
        metadata[modality] = {
            "median": median.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
        }
    return scaled, metadata


def make_loaders(
    arrays: dict[str, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    keys = ("train", "validation", "test")
    return {
        key: DataLoader(
            ArrayDataset(arrays, selected),
            batch_size=batch_size,
            shuffle=key == "train",
            generator=generator if key == "train" else None,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for key, selected in zip(keys, indices, strict=True)
    }


def move_inputs(
    inputs: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in inputs.items()
    }


class HuGaDBTeacher(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                "emg": TemporalLiteEncoder(
                    2, embedding_dim, width=48, dropout=0.15
                ),
                "imu": TemporalLiteEncoder(
                    36, embedding_dim, width=64, dropout=0.15
                ),
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, len(ACTIVITIES)),
        )

    def forward(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        embeddings = [
            self.encoders[modality](inputs[modality])
            for modality in MODALITIES
        ]
        return {"logits": self.classifier(torch.cat(embeddings, dim=1))}


class HuGaDBRapid(nn.Module):
    def __init__(
        self, embedding_dim: int = 48, identity_dim: int = 8
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                "emg": TemporalLiteEncoder(
                    2, embedding_dim, width=32, dropout=0.15
                ),
                "imu": TemporalLiteEncoder(
                    36, embedding_dim, width=40, dropout=0.15
                ),
            }
        )
        self.modality_tokens = nn.Parameter(
            torch.zeros(len(MODALITIES), embedding_dim)
        )
        nn.init.normal_(self.modality_tokens, std=0.02)
        self.source_identity = nn.Embedding(len(MODALITIES), identity_dim)
        self.target_identity = nn.Embedding(len(MODALITIES), identity_dim)
        conditional_dim = embedding_dim + 2 * identity_dim
        self.proxy_generator = nn.Sequential(
            nn.LayerNorm(conditional_dim),
            nn.Linear(conditional_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.reliability = nn.Sequential(
            nn.LayerNorm(conditional_dim),
            nn.Linear(conditional_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.fusion = GatedFusion(embedding_dim, len(MODALITIES))
        self.classifier = nn.Linear(embedding_dim, len(ACTIVITIES))

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        real = torch.stack(
            [
                self.encoders[modality](inputs[modality])
                + self.modality_tokens[index]
                for index, modality in enumerate(MODALITIES)
            ],
            dim=1,
        )
        batch = real.shape[0]
        ids = torch.arange(len(MODALITIES), device=real.device)
        source_token = real[:, None, :, :].expand(
            batch, len(MODALITIES), len(MODALITIES), -1
        )
        source_id = self.source_identity(ids)[None, None, :, :].expand(
            batch, len(MODALITIES), -1, -1
        )
        target_id = self.target_identity(ids)[None, :, None, :].expand(
            batch, -1, len(MODALITIES), -1
        )
        conditional = torch.cat(
            (source_token, source_id, target_id), dim=-1
        )
        candidates = self.proxy_generator(conditional)
        reliability_logits = self.reliability(conditional).squeeze(-1)
        source_available = modality_mask[:, None, :].expand(
            -1, len(MODALITIES), -1
        )
        reliability_logits = reliability_logits.masked_fill(
            ~source_available, -1e4
        )
        reliability_weights = reliability_logits.softmax(dim=-1)
        proxy = (
            candidates * reliability_weights.unsqueeze(-1)
        ).sum(dim=2)
        completed = torch.where(
            modality_mask.unsqueeze(-1), real, proxy
        )
        fused, fusion_weights = self.fusion(
            completed, torch.ones_like(modality_mask)
        )
        missing = ~modality_mask
        alignment = (
            F.smooth_l1_loss(proxy[missing], real.detach()[missing])
            if missing.any()
            else fused.new_zeros(())
        )
        proxy_logits = self.classifier(
            proxy.reshape(-1, proxy.shape[-1])
        ).reshape(batch, len(MODALITIES), -1)
        return {
            "logits": self.classifier(fused),
            "proxy_logits": proxy_logits,
            "alignment_loss": alignment,
            "modality_mask": modality_mask,
            "fusion_weights": fusion_weights,
        }


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(ACTIVITIES)).astype(float)
    weights = counts.sum() / (len(ACTIVITIES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def balanced_masks(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    bank = torch.tensor(MASK_ROWS, dtype=torch.bool, device=device)
    repeats = (batch_size + len(bank) - 1) // len(bank)
    group_ids = torch.arange(
        len(bank), device=device
    ).repeat(repeats)[:batch_size]
    order = torch.randperm(
        batch_size, generator=generator, device=device
    )
    group_ids = group_ids[order]
    return bank[group_ids], group_ids


def kd_per_sample(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float
) -> torch.Tensor:
    return (
        F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="none",
        ).sum(dim=1)
        * temperature**2
    )


def grouped_mean(loss: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    values = [
        loss[group_ids.eq(index)].mean()
        for index in range(len(MASK_ROWS))
        if group_ids.eq(index).any()
    ]
    return torch.stack(values).mean()


def fasca_augment(
    inputs: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    generator: torch.Generator,
    emg_probability: float,
    imu_probability: float,
    severity_min: float,
    severity_max: float,
) -> dict[str, torch.Tensor]:
    """HuGaDB-specific structured failures: EMG leads and six IMU nodes."""
    output = {key: value.clone() for key, value in inputs.items()}
    device = output["imu"].device
    batch = output["imu"].shape[0]
    for row in range(batch):
        severity = float(
            (
                severity_min
                + (severity_max - severity_min)
                * torch.rand((), generator=generator, device=device)
            ).item()
        )
        if bool(mask[row, 0]) and float(
            torch.rand((), generator=generator, device=device).item()
        ) < emg_probability:
            if float(
                torch.rand((), generator=generator, device=device).item()
            ) < 0.5:
                channel = int(
                    torch.randint(
                        2, (), generator=generator, device=device
                    ).item()
                )
                output["emg"][row, channel] = 0.0
            else:
                log_gain = (
                    2.0
                    * float(
                        torch.rand(
                            (), generator=generator, device=device
                        ).item()
                    )
                    - 1.0
                ) * 0.55 * severity
                output["emg"][row] *= float(np.exp(log_gain))
        if bool(mask[row, 1]) and float(
            torch.rand((), generator=generator, device=device).item()
        ) < imu_probability:
            if float(
                torch.rand((), generator=generator, device=device).item()
            ) < 0.5:
                drop_count = 1 if severity < 0.55 else 2
                nodes = torch.randperm(
                    6, generator=generator, device=device
                )[:drop_count]
                for node in nodes.tolist():
                    output["imu"][row, node * 6 : (node + 1) * 6] = 0.0
            else:
                sign = (
                    -1.0
                    if float(
                        torch.rand(
                            (), generator=generator, device=device
                        ).item()
                    )
                    < 0.5
                    else 1.0
                )
                ramp = torch.linspace(
                    -1.0,
                    1.0,
                    output["imu"].shape[-1],
                    device=device,
                )
                scale = output["imu"][row].std(
                    dim=-1, keepdim=True
                ).clamp_min(0.05)
                output["imu"][row] += (
                    sign * (0.08 + 0.40 * severity) * scale * ramp
                )
    return output


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bits: tuple[bool, ...],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth, probabilities = [], []
    for inputs, target in loader:
        inputs = move_inputs(inputs, device)
        mask = torch.tensor(
            bits, dtype=torch.bool, device=device
        ).unsqueeze(0).expand(target.shape[0], -1)
        output = model(inputs, modality_mask=mask)
        truth.append(target.numpy())
        probabilities.append(
            output["logits"].softmax(dim=1).cpu().numpy()
        )
    return np.concatenate(truth), np.concatenate(probabilities)


@torch.no_grad()
def teacher_predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth, probabilities = [], []
    for inputs, target in loader:
        output = model(move_inputs(inputs, device))
        truth.append(target.numpy())
        probabilities.append(
            output["logits"].softmax(dim=1).cpu().numpy()
        )
    return np.concatenate(truth), np.concatenate(probabilities)


def macro_f1(truth: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            truth,
            probabilities.argmax(axis=1),
            labels=list(range(len(ACTIVITIES))),
            average="macro",
            zero_division=0,
        )
    )


def train_teacher(
    model: HuGaDBTeacher,
    loaders: dict[str, DataLoader],
    weights: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[HuGaDBTeacher, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs_teacher
    )
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, args.epochs_teacher + 1):
        model.train()
        losses = []
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                logits = model(inputs)["logits"]
                loss = F.cross_entropy(
                    logits,
                    target,
                    weight=weights,
                    label_smoothing=0.05,
                )
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp.step(optimizer)
            amp.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        truth, probabilities = teacher_predict(
            model, loaders["validation"], device
        )
        score = macro_f1(truth, probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": score,
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


def validation_score(
    model: HuGaDBRapid,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    scores = {}
    for bits in MASK_ROWS:
        truth, probabilities = predict(model, loader, device, bits)
        scores[mask_name(bits)] = macro_f1(truth, probabilities)
    return float(np.mean(list(scores.values()))), scores


def train_student(
    method: str,
    model: HuGaDBRapid,
    teacher: HuGaDBTeacher,
    loaders: dict[str, DataLoader],
    weights: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    run_seed: int,
) -> tuple[HuGaDBRapid, int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs_student
    )
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(run_seed + 193_337)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, args.epochs_student + 1):
        model.train()
        losses, clean_losses = [], []
        curriculum = min(
            1.0, epoch / max(args.augmentation_warmup_epochs, 1)
        )
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            mask, group_ids = balanced_masks(
                len(target), device, generator
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                teacher_logits = teacher(inputs)["logits"]
            student_inputs = inputs
            if method == "fasca_kd":
                student_inputs = fasca_augment(
                    inputs,
                    mask,
                    generator,
                    (
                        args.augmentation_probability
                        if args.emg_augmentation_probability is None
                        else args.emg_augmentation_probability
                    )
                    * curriculum,
                    (
                        args.augmentation_probability
                        if args.imu_augmentation_probability is None
                        else args.imu_augmentation_probability
                    )
                    * curriculum,
                    args.severity_min,
                    args.severity_min
                    + curriculum
                    * (args.severity_max - args.severity_min),
                )
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                output = model(student_inputs, modality_mask=mask)
                hard = F.cross_entropy(
                    output["logits"],
                    target,
                    weight=weights,
                    reduction="none",
                    label_smoothing=0.05,
                )
                kd = kd_per_sample(
                    output["logits"], teacher_logits, args.temperature
                )
                loss = grouped_mean(
                    (1.0 - args.alpha) * hard + args.alpha * kd,
                    group_ids,
                )
                loss = loss + args.lambda_alignment * output[
                    "alignment_loss"
                ]
                missing = ~mask
                if args.lambda_proxy and missing.any():
                    expanded = target[:, None].expand(
                        -1, len(MODALITIES)
                    )
                    proxy_loss = F.cross_entropy(
                        output["proxy_logits"][missing],
                        expanded[missing],
                        weight=weights,
                        label_smoothing=0.05,
                    )
                    loss = loss + args.lambda_proxy * proxy_loss
                clean_loss = loss.new_zeros(())
                if method == "fasca_kd" and args.lambda_clean:
                    clean_logits = model(
                        inputs, modality_mask=mask
                    )["logits"]
                    clean_loss = F.cross_entropy(
                        clean_logits,
                        target,
                        weight=weights,
                        label_smoothing=0.05,
                    )
                    loss = loss + args.lambda_clean * clean_loss
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp.step(optimizer)
            amp.update()
            losses.append(float(loss.detach()))
            clean_losses.append(float(clean_loss.detach()))
        scheduler.step()
        score, validation = validation_score(
            model, loaders["validation"], device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_clean_loss": float(np.mean(clean_losses)),
                "validation_mean_3_macro_f1": score,
                **{
                    f"validation_{name}_macro_f1": value
                    for name, value in validation.items()
                },
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
        raise RuntimeError(f"{method}: no checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def apply_corruption(
    inputs: Mapping[str, torch.Tensor],
    corruption: str,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    output = {key: value.clone() for key, value in inputs.items()}
    batch = output["imu"].shape[0]
    device = output["imu"].device
    if corruption == "clean":
        return output
    if corruption == "imu_node_dropout_2of6":
        for row in range(batch):
            nodes = torch.randperm(
                6, generator=generator, device=device
            )[:2]
            for node in nodes.tolist():
                output["imu"][row, node * 6 : (node + 1) * 6] = 0.0
    elif corruption == "emg_lead_dropout_1of2":
        channels = torch.randint(
            2, (batch,), generator=generator, device=device
        )
        output["emg"][torch.arange(batch, device=device), channels] = 0.0
    elif corruption == "all_gain_1.5":
        output = {key: value * 1.5 for key, value in output.items()}
    elif corruption == "all_packet_loss_30":
        for value in output.values():
            length = round(value.shape[-1] * 0.30)
            starts = torch.randint(
                value.shape[-1] - length + 1,
                (batch,),
                generator=generator,
                device=device,
            )
            for row, start in enumerate(starts.tolist()):
                value[row, :, start : start + length] = 0.0
    elif corruption == "all_noise_10db":
        for key, value in output.items():
            power = value.pow(2).mean(dim=(1, 2), keepdim=True)
            noise = torch.randn(
                value.shape,
                generator=generator,
                device=device,
                dtype=value.dtype,
            )
            noise_power = noise.pow(2).mean(
                dim=(1, 2), keepdim=True
            ).clamp_min(1e-8)
            output[key] = value + noise * (
                power / (10.0 * noise_power)
            ).sqrt()
    elif corruption == "all_shift_250ms":
        shift = 14
        for value in output.values():
            value[:, :, shift:] = value[:, :, :-shift].clone()
            value[:, :, :shift] = 0.0
    else:
        raise ValueError(f"Unknown corruption: {corruption}")
    return output


@torch.no_grad()
def evaluate_corruptions(
    model: HuGaDBRapid,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, float | str]]:
    corruptions = (
        "clean",
        "imu_node_dropout_2of6",
        "emg_lead_dropout_1of2",
        "all_gain_1.5",
        "all_packet_loss_30",
        "all_noise_10db",
        "all_shift_250ms",
    )
    rows = []
    model.eval()
    for corruption_index, corruption in enumerate(corruptions):
        truth, probabilities = [], []
        generator = torch.Generator(device=device)
        generator.manual_seed(918_221 + corruption_index)
        for inputs, target in loader:
            inputs = move_inputs(inputs, device)
            corrupted = apply_corruption(
                inputs, corruption, generator
            )
            mask = torch.ones(
                len(target),
                len(MODALITIES),
                dtype=torch.bool,
                device=device,
            )
            logits = model(
                corrupted, modality_mask=mask
            )["logits"]
            truth.append(target.numpy())
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
        rows.append(
            {
                "corruption": corruption,
                "window_macro_f1": macro_f1(
                    np.concatenate(truth),
                    np.concatenate(probabilities),
                ),
            }
        )
    return rows


def save_status(
    path: Path, completed: int, total: int, message: str
) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": "complete" if completed == total else "running",
                "completed_runs": completed,
                "total_runs": total,
                "message": message,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize(output_dir: Path) -> None:
    prediction_paths = sorted(
        output_dir.glob("fold_*/seed_*/*/predictions.csv.gz")
    )
    corruption_paths = sorted(
        output_dir.glob("fold_*/seed_*/*/corruptions.csv")
    )
    if not prediction_paths:
        return
    predictions = pd.concat(
        (pd.read_csv(path) for path in prediction_paths),
        ignore_index=True,
    )
    probability_columns = [
        f"p_{activity}" for activity in ACTIVITIES
    ]
    metrics = []
    for keys, group in predictions.groupby(
        ["method", "seed", "available_modalities"], sort=False
    ):
        metrics.append(
            {
                "method": keys[0],
                "seed": int(keys[1]),
                "available_modalities": keys[2],
                "test_windows": int(len(group)),
                "window_macro_f1": macro_f1(
                    group["true_class"].to_numpy(),
                    group[probability_columns].to_numpy(),
                ),
            }
        )
    pooled = pd.DataFrame(metrics)
    pooled.to_csv(
        output_dir / "seed_pooled_combination_metrics.csv", index=False
    )
    summaries = []
    for (method, seed), group in pooled.groupby(
        ["method", "seed"], sort=False
    ):
        full = float(
            group.loc[
                group["available_modalities"].eq("emg+imu"),
                "window_macro_f1",
            ].iloc[0]
        )
        incomplete = group.loc[
            ~group["available_modalities"].eq("emg+imu"),
            "window_macro_f1",
        ]
        summaries.append(
            {
                "method": method,
                "seed": int(seed),
                "full": full,
                "incomplete_mean": float(incomplete.mean()),
                "mean_3": float(group["window_macro_f1"].mean()),
                "worst": float(group["window_macro_f1"].min()),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "seed_summary.csv", index=False)
    descriptive = (
        summary.groupby("method")
        .agg(
            full_mean=("full", "mean"),
            full_std=("full", "std"),
            incomplete_mean=("incomplete_mean", "mean"),
            incomplete_std=("incomplete_mean", "std"),
            mean_3=("mean_3", "mean"),
            mean_3_std=("mean_3", "std"),
            worst_mean=("worst", "mean"),
            worst_std=("worst", "std"),
        )
        .reset_index()
    )
    descriptive.to_csv(output_dir / "method_summary.csv", index=False)
    if corruption_paths:
        corruptions = pd.concat(
            (pd.read_csv(path) for path in corruption_paths),
            ignore_index=True,
        )
        pooled_corruptions = (
            corruptions.groupby(["method", "seed", "corruption"])
            .apply(
                lambda group: pd.Series(
                    {
                        "window_macro_f1": macro_f1(
                            group["true_class"].to_numpy(),
                            group[probability_columns].to_numpy(),
                        )
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        pooled_corruptions.to_csv(
            output_dir / "seed_pooled_corruptions.csv", index=False
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    folds = subject_folds(
        arrays["subject"], args.folds, args.partition_seed
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(folds) * len(args.seeds) * len(args.methods)
    completed = 0
    status_path = args.output_dir / "status.json"
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **{
                    key: str(value)
                    if isinstance(value, Path)
                    else value
                    for key, value in vars(args).items()
                },
                "device": str(device),
                "modalities": MODALITIES,
                "activities": ACTIVITIES,
                "mask_rows": MASK_ROWS,
                "note": (
                    "IMU dropout respects six physical 6-channel nodes; "
                    "EMG dropout respects two physical leads."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_status(status_path, completed, total, f"Starting on {device}")

    for fold, indices in enumerate(folds, start=1):
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaled, scaler_metadata = robust_scale(arrays, indices[0])
        (fold_dir / "scaler.json").write_text(
            json.dumps(scaler_metadata, indent=2),
            encoding="utf-8",
        )
        split = {
            "train_subjects": sorted(
                np.unique(arrays["subject"][indices[0]])
                .astype(int)
                .tolist()
            ),
            "validation_subjects": sorted(
                np.unique(arrays["subject"][indices[1]])
                .astype(int)
                .tolist()
            ),
            "test_subjects": sorted(
                np.unique(arrays["subject"][indices[2]])
                .astype(int)
                .tolist()
            ),
        }
        (fold_dir / "subjects.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8"
        )
        for seed in args.seeds:
            run_seed = seed * 100 + fold
            seed_dir = fold_dir / f"seed_{seed}"
            teacher_dir = seed_dir / "teacher"
            teacher_dir.mkdir(parents=True, exist_ok=True)
            teacher_loaders = make_loaders(
                scaled,
                indices,
                args.batch_size,
                device,
                run_seed + 77,
            )
            weights = class_weights(
                arrays["label"][indices[0]], device
            )
            seed_everything(run_seed)
            teacher = HuGaDBTeacher().to(device)
            teacher_path = teacher_dir / "best_model.pt"
            if args.teacher_root is not None:
                source_split_path = (
                    args.teacher_root / f"fold_{fold}" / "subjects.json"
                )
                source_split = json.loads(
                    source_split_path.read_text(encoding="utf-8")
                )
                if source_split != split:
                    raise ValueError(
                        f"Teacher split mismatch in fold {fold}"
                    )
                teacher_path = (
                    args.teacher_root
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                    / "teacher"
                    / "best_model.pt"
                )
                if not teacher_path.exists():
                    raise FileNotFoundError(
                        f"Teacher checkpoint missing: {teacher_path}"
                    )
            if teacher_path.exists():
                teacher.load_state_dict(
                    torch.load(
                        teacher_path,
                        map_location=device,
                        weights_only=True,
                    )["state_dict"]
                )
            else:
                teacher, teacher_epoch, teacher_history = train_teacher(
                    teacher, teacher_loaders, weights, args, device
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
                    },
                    teacher_path,
                )
            for method in args.methods:
                run_dir = seed_dir / method
                predictions_path = run_dir / "predictions.csv.gz"
                corruptions_path = run_dir / "corruptions.csv"
                if predictions_path.exists() and corruptions_path.exists():
                    completed += 1
                    save_status(
                        status_path,
                        completed,
                        total,
                        f"Reused fold {fold}, seed {seed}, {method}",
                    )
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                seed_everything(run_seed)
                # Recreate the loader with the same seed for every method.
                # This keeps batch order paired instead of advancing one
                # shared generator according to method execution order.
                loaders = make_loaders(
                    scaled,
                    indices,
                    args.batch_size,
                    device,
                    run_seed + 177,
                )
                student = HuGaDBRapid(args.embedding_dim).to(device)
                student, best_epoch, history = train_student(
                    method,
                    student,
                    teacher,
                    loaders,
                    weights,
                    args,
                    device,
                    run_seed,
                )
                pd.DataFrame(history).to_csv(
                    run_dir / "history.csv", index=False
                )
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in student.state_dict().items()
                        },
                        "best_epoch": best_epoch,
                        "parameter_count": sum(
                            parameter.numel()
                            for parameter in student.parameters()
                        ),
                    },
                    run_dir / "best_model.pt",
                )
                prediction_frames = []
                test_indices = indices[2]
                for bits in MASK_ROWS:
                    truth, probabilities = predict(
                        student, loaders["test"], device, bits
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "method": method,
                                "fold": fold,
                                "seed": seed,
                                "available_modalities": mask_name(bits),
                                "subject": arrays["subject"][test_indices],
                                "trial_id": arrays["trial_id"][test_indices],
                                "true_class": truth,
                                **{
                                    f"p_{activity}": probabilities[
                                        :, index
                                    ]
                                    for index, activity in enumerate(
                                        ACTIVITIES
                                    )
                                },
                            }
                        )
                    )
                pd.concat(prediction_frames, ignore_index=True).to_csv(
                    predictions_path, index=False, compression="gzip"
                )
                corruption_metrics = evaluate_corruptions(
                    student, loaders["test"], device
                )
                corruption_predictions = []
                # Preserve fold-level probabilities for correct pooled F1.
                for corruption_index, row in enumerate(
                    corruption_metrics
                ):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(918_221 + corruption_index)
                    truth_parts, probability_parts = [], []
                    for inputs, target in loaders["test"]:
                        inputs = move_inputs(inputs, device)
                        corrupted = apply_corruption(
                            inputs, str(row["corruption"]), generator
                        )
                        mask = torch.ones(
                            len(target),
                            len(MODALITIES),
                            dtype=torch.bool,
                            device=device,
                        )
                        logits = student(
                            corrupted, modality_mask=mask
                        )["logits"]
                        truth_parts.append(target.numpy())
                        probability_parts.append(
                            logits.softmax(dim=1).detach().cpu().numpy()
                        )
                    truth = np.concatenate(truth_parts)
                    probabilities = np.concatenate(probability_parts)
                    corruption_predictions.append(
                        pd.DataFrame(
                            {
                                "method": method,
                                "fold": fold,
                                "seed": seed,
                                "corruption": row["corruption"],
                                "subject": arrays["subject"][test_indices],
                                "true_class": truth,
                                **{
                                    f"p_{activity}": probabilities[
                                        :, index
                                    ]
                                    for index, activity in enumerate(
                                        ACTIVITIES
                                    )
                                },
                            }
                        )
                    )
                pd.concat(
                    corruption_predictions, ignore_index=True
                ).to_csv(corruptions_path, index=False)
                completed += 1
                save_status(
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
                        }
                    ),
                    flush=True,
                )
        del scaled
    summarize(args.output_dir)
    save_status(
        status_path,
        completed,
        total,
        "HuGaDB external validation complete",
    )


if __name__ == "__main__":
    main()
