from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


TEST_STUDIES = ("Ga", "Ju", "Si")
METHODS = ("baseline", "generic", "iid", "fasca")
CONDITIONS = (
    "clean_both",
    "left_only",
    "right_only",
    "sensor_dead_bilateral",
    "near_dead_bilateral",
    "contact_loss_bilateral",
    "saturation_bilateral",
    "drift_bilateral",
    "packet_loss_bilateral",
    "gain_miscalibration",
)
FAULT_CONDITIONS = CONDITIONS[3:]
MASKS = {
    "clean_both": np.asarray([1.0, 1.0], dtype=np.float32),
    "left_only": np.asarray([1.0, 0.0], dtype=np.float32),
    "right_only": np.asarray([0.0, 1.0], dtype=np.float32),
}


@dataclass(frozen=True)
class ClinicalData:
    windows: np.ndarray
    window_subject_index: np.ndarray
    window_ordinal: np.ndarray
    subject_ids: np.ndarray
    studies: np.ndarray
    labels: np.ndarray
    genders: np.ndarray
    ages: np.ndarray
    hoehen_yahr: np.ndarray


class WindowDataset(Dataset):
    def __init__(self, data: ClinicalData, window_indices: np.ndarray):
        self.data = data
        self.window_indices = np.asarray(window_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_index = int(self.window_indices[item])
        subject_index = int(self.data.window_subject_index[window_index])
        return (
            torch.from_numpy(self.data.windows[window_index]),
            torch.tensor(self.data.labels[subject_index], dtype=torch.long),
            torch.tensor(window_index, dtype=torch.long),
        )


class FootEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class MaskedFootFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FootEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 2),
        )

    def forward(
        self, inputs: torch.Tensor, modality_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        encoded = self.encoder(inputs.reshape(batch_size * 2, 8, 500))
        encoded = encoded.reshape(batch_size, 2, 64)
        mask = modality_mask.to(encoded.dtype).unsqueeze(-1)
        fused = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(fused)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked study-held-out Gait in Parkinson's Disease "
            "clinical robustness experiment."
        )
    )
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "gaitpdb_clinical_windows.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts" / "gaitpdb_clinical_external",
    )
    parser.add_argument("--split-seed", type=int, default=20260706)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[101, 102, 103]
    )
    parser.add_argument(
        "--methods",
        choices=METHODS,
        nargs="+",
        default=["baseline", "fasca"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--windows-per-subject", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path: Path) -> ClinicalData:
    archive = np.load(path)
    data = ClinicalData(
        windows=archive["windows"].astype(np.float32),
        window_subject_index=archive["window_subject_index"].astype(np.int64),
        window_ordinal=archive["window_ordinal"].astype(np.int64),
        subject_ids=archive["subject_ids"].astype(str),
        studies=archive["studies"].astype(str),
        labels=archive["labels"].astype(np.int64),
        genders=archive["genders"].astype(np.int64),
        ages=archive["ages"].astype(np.float32),
        hoehen_yahr=archive["hoehen_yahr"].astype(np.float32),
    )
    if data.windows.ndim != 4 or data.windows.shape[1:] != (2, 8, 500):
        raise RuntimeError(
            f"Unexpected window shape {data.windows.shape}"
        )
    if len(data.subject_ids) != len(np.unique(data.subject_ids)):
        raise RuntimeError("Subject identifiers are not unique")
    if set(data.studies) != set(TEST_STUDIES):
        raise RuntimeError(
            f"Unexpected studies: {sorted(np.unique(data.studies))}"
        )
    if set(data.labels) != {0, 1}:
        raise RuntimeError(
            f"Unexpected labels: {sorted(np.unique(data.labels))}"
        )
    return data


def subject_split(
    data: ClinicalData, test_study: str, split_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_subjects = np.flatnonzero(data.studies == test_study)
    development = np.flatnonzero(data.studies != test_study)
    strata = np.asarray(
        [
            f"{data.studies[index]}_{data.labels[index]}"
            for index in development
        ]
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=split_seed
    )
    train_local, validation_local = next(
        splitter.split(development, strata)
    )
    train_subjects = development[train_local]
    validation_subjects = development[validation_local]
    for first, second, name in (
        (train_subjects, validation_subjects, "train/validation"),
        (train_subjects, test_subjects, "train/test"),
        (validation_subjects, test_subjects, "validation/test"),
    ):
        if np.intersect1d(first, second).size:
            raise RuntimeError(f"Subject overlap in {name} split")
    return train_subjects, validation_subjects, test_subjects


def windows_for_subjects(
    data: ClinicalData, subjects: np.ndarray
) -> np.ndarray:
    return np.flatnonzero(
        np.isin(data.window_subject_index, np.asarray(subjects))
    )


def training_sampler(
    data: ClinicalData,
    train_subjects: np.ndarray,
    window_indices: np.ndarray,
    windows_per_subject: int,
    seed: int,
) -> WeightedRandomSampler:
    subject_window_counts = np.bincount(
        data.window_subject_index[window_indices],
        minlength=len(data.subject_ids),
    )
    class_subject_counts = np.bincount(
        data.labels[train_subjects], minlength=2
    )
    weights = []
    for window_index in window_indices:
        subject_index = data.window_subject_index[window_index]
        label = data.labels[subject_index]
        weight = 1.0 / (
            subject_window_counts[subject_index] * class_subject_counts[label]
        )
        weights.append(weight)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(len(train_subjects) * windows_per_subject),
        replacement=True,
        generator=generator,
    )


def sample_availability_masks(
    batch_size: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    choices = torch.randint(
        0, 3, (batch_size,), generator=generator, device=device
    )
    masks = torch.ones((batch_size, 2), dtype=torch.float32, device=device)
    masks[choices == 0, 1] = 0.0
    masks[choices == 1, 0] = 0.0
    return masks


def random_integer(
    low: int,
    high: int,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    return int(
        torch.randint(
            low, high, (1,), generator=generator, device=device
        ).item()
    )


def augment_fasca_pressure(
    inputs: torch.Tensor,
    modality_mask: torch.Tensor,
    generator: torch.Generator,
    probability: float = 0.35,
) -> torch.Tensor:
    augmented = inputs.clone()
    batch_size = inputs.shape[0]
    apply = torch.rand(
        batch_size, generator=generator, device=inputs.device
    ) < probability
    corruption_type = torch.randint(
        0, 7, (batch_size,), generator=generator, device=inputs.device
    )
    severity = (
        0.20
        + 0.60
        * torch.rand(
            batch_size, generator=generator, device=inputs.device
        )
    )
    ramp = torch.linspace(
        0.0, 1.0, inputs.shape[-1], device=inputs.device
    )
    for index in torch.nonzero(apply, as_tuple=False).flatten().tolist():
        available = (
            torch.nonzero(
                modality_mask[index] > 0, as_tuple=False
            )
            .flatten()
            .tolist()
        )
        foot = available[
            random_integer(
                0, len(available), generator=generator, device=inputs.device
            )
        ]
        sensor = random_integer(
            0, 8, generator=generator, device=inputs.device
        )
        current_type = int(corruption_type[index].item())
        current_severity = float(severity[index].item())
        if current_type == 0:
            augmented[index, foot, sensor] = 0.0
        elif current_type == 1:
            factor = 0.20 - 0.19 * current_severity
            augmented[index, foot, sensor] *= factor
        elif current_type == 2:
            duration = int((0.25 + 0.75 * current_severity) * 100)
            start = random_integer(
                0,
                inputs.shape[-1] - duration + 1,
                generator=generator,
                device=inputs.device,
            )
            augmented[index, foot, sensor, start : start + duration] = 0.0
        elif current_type == 3:
            quantile = 0.90 - 0.40 * current_severity
            cap = torch.quantile(
                augmented[index, foot, sensor], quantile
            )
            augmented[index, foot, sensor] = torch.minimum(
                augmented[index, foot, sensor], cap
            )
        elif current_type == 4:
            augmented[index, foot, sensor] += (
                0.05 + 0.15 * current_severity
            ) * ramp
        elif current_type == 5:
            duration = int((0.10 + 0.40 * current_severity) * 100)
            start = random_integer(
                0,
                inputs.shape[-1] - duration + 1,
                generator=generator,
                device=inputs.device,
            )
            augmented[index, foot, :, start : start + duration] = 0.0
        elif current_type == 6:
            sign = (
                -1.0
                if random_integer(
                    0, 2, generator=generator, device=inputs.device
                )
                == 0
                else 1.0
            )
            factor = 1.0 + sign * (0.10 + 0.40 * current_severity)
            augmented[index, foot] *= factor
    return augmented


def augment_generic_timeseries(
    inputs: torch.Tensor,
    modality_mask: torch.Tensor,
    generator: torch.Generator,
    probability: float = 0.35,
) -> torch.Tensor:
    augmented = inputs.clone()
    batch_size = inputs.shape[0]
    apply = torch.rand(
        batch_size, generator=generator, device=inputs.device
    ) < probability
    transform_type = torch.randint(
        0, 4, (batch_size,), generator=generator, device=inputs.device
    )
    severity = 0.20 + 0.60 * torch.rand(
        batch_size, generator=generator, device=inputs.device
    )
    for index in torch.nonzero(apply, as_tuple=False).flatten().tolist():
        available = (
            torch.nonzero(
                modality_mask[index] > 0, as_tuple=False
            )
            .flatten()
            .tolist()
        )
        foot = available[
            random_integer(
                0, len(available), generator=generator, device=inputs.device
            )
        ]
        current_type = int(transform_type[index].item())
        current_severity = float(severity[index].item())
        branch = augmented[index, foot]
        if current_type == 0:
            scale = branch.std().clamp_min(1e-4)
            sigma = (0.01 + 0.09 * current_severity) * scale
            noise = torch.randn(
                branch.shape,
                generator=generator,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            augmented[index, foot] = branch + sigma * noise
        elif current_type == 1:
            direction = (
                -1.0
                if random_integer(
                    0, 2, generator=generator, device=inputs.device
                )
                == 0
                else 1.0
            )
            factor = 1.0 + direction * (
                0.05 + 0.25 * current_severity
            )
            augmented[index, foot] = branch * factor
        elif current_type == 2:
            duration = max(
                1,
                int(
                    (0.05 + 0.20 * current_severity)
                    * inputs.shape[-1]
                ),
            )
            start = random_integer(
                0,
                inputs.shape[-1] - duration + 1,
                generator=generator,
                device=inputs.device,
            )
            augmented[index, foot, :, start : start + duration] = 0.0
        elif current_type == 3:
            shift = max(
                1,
                int(
                    (0.02 + 0.08 * current_severity)
                    * inputs.shape[-1]
                ),
            )
            direction = (
                -1
                if random_integer(
                    0, 2, generator=generator, device=inputs.device
                )
                == 0
                else 1
            )
            shifted = torch.roll(branch, shifts=direction * shift, dims=-1)
            if direction > 0:
                shifted[:, :shift] = 0.0
            else:
                shifted[:, -shift:] = 0.0
            augmented[index, foot] = shifted
    return augmented


def augment_iid_channels(
    inputs: torch.Tensor,
    modality_mask: torch.Tensor,
    generator: torch.Generator,
    probability: float = 0.35,
) -> torch.Tensor:
    augmented = inputs.clone()
    batch_size = inputs.shape[0]
    apply = torch.rand(
        batch_size, generator=generator, device=inputs.device
    ) < probability
    corruption_type = torch.randint(
        0, 7, (batch_size,), generator=generator, device=inputs.device
    )
    severity = 0.20 + 0.60 * torch.rand(
        batch_size, generator=generator, device=inputs.device
    )
    for index in torch.nonzero(apply, as_tuple=False).flatten().tolist():
        available = (
            torch.nonzero(
                modality_mask[index] > 0, as_tuple=False
            )
            .flatten()
            .tolist()
        )
        foot = available[
            random_integer(
                0, len(available), generator=generator, device=inputs.device
            )
        ]
        current_type = int(corruption_type[index].item())
        current_severity = float(severity[index].item())
        branch = augmented[index, foot]
        channel_std = branch.std(dim=-1, keepdim=True).clamp_min(1e-4)
        if current_type == 0:
            sigma = (
                0.01
                + 0.09
                * torch.rand(
                    (8, 1),
                    generator=generator,
                    device=inputs.device,
                )
                * current_severity
            )
            noise = torch.randn(
                branch.shape,
                generator=generator,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            augmented[index, foot] = branch + sigma * channel_std * noise
        elif current_type in (1, 2):
            selected = (
                torch.rand(
                    8, generator=generator, device=inputs.device
                )
                < (0.05 + 0.35 * current_severity)
            )
            if not selected.any():
                selected[
                    random_integer(
                        0, 8, generator=generator, device=inputs.device
                    )
                ] = True
            if current_type == 1:
                augmented[index, foot, selected] = 0.0
            else:
                factors = 0.01 + 0.19 * torch.rand(
                    int(selected.sum().item()),
                    generator=generator,
                    device=inputs.device,
                )
                augmented[index, foot, selected] *= factors[:, None]
        elif current_type == 3:
            gains = 1.0 + (
                2.0
                * torch.rand(
                    (8, 1),
                    generator=generator,
                    device=inputs.device,
                )
                - 1.0
            ) * (0.05 + 0.35 * current_severity)
            augmented[index, foot] = branch * gains
        elif current_type == 4:
            offsets = (
                2.0
                * torch.rand(
                    (8, 1),
                    generator=generator,
                    device=inputs.device,
                )
                - 1.0
            ) * (0.02 + 0.13 * current_severity)
            augmented[index, foot] = branch + offsets * channel_std
        elif current_type == 5:
            for sensor in range(8):
                quantile = float(
                    0.90
                    - 0.40
                    * current_severity
                    * torch.rand(
                        1,
                        generator=generator,
                        device=inputs.device,
                    ).item()
                )
                cap = torch.quantile(branch[sensor], quantile)
                augmented[index, foot, sensor] = torch.minimum(
                    branch[sensor], cap
                )
        elif current_type == 6:
            duration = max(
                1,
                int((0.02 + 0.18 * current_severity) * inputs.shape[-1]),
            )
            for sensor in range(8):
                start = random_integer(
                    0,
                    inputs.shape[-1] - duration + 1,
                    generator=generator,
                    device=inputs.device,
                )
                augmented[
                    index, foot, sensor, start : start + duration
                ] = 0.0
    return augmented


def stable_rng(
    subject_id: str, window_ordinal: int, condition: str
) -> np.random.Generator:
    payload = f"{subject_id}|{window_ordinal}|{condition}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def apply_locked_fault(
    inputs: torch.Tensor,
    data: ClinicalData,
    window_indices: np.ndarray,
    condition: str,
) -> torch.Tensor:
    if condition in MASKS:
        return inputs
    faulted = inputs.clone()
    time_length = inputs.shape[-1]
    ramp = torch.linspace(
        0.0, 0.15, time_length, device=inputs.device
    )
    for batch_index, window_index in enumerate(window_indices):
        subject_index = int(data.window_subject_index[window_index])
        subject_id = data.subject_ids[subject_index]
        ordinal = int(data.window_ordinal[window_index])
        rng = stable_rng(subject_id, ordinal, condition)
        sensors = rng.integers(0, 8, size=2)
        if condition == "sensor_dead_bilateral":
            for foot in range(2):
                faulted[batch_index, foot, sensors[foot]] = 0.0
        elif condition == "near_dead_bilateral":
            for foot in range(2):
                faulted[batch_index, foot, sensors[foot]] *= 0.05
        elif condition == "contact_loss_bilateral":
            for foot in range(2):
                start = int(rng.integers(0, time_length - 100 + 1))
                faulted[
                    batch_index,
                    foot,
                    sensors[foot],
                    start : start + 100,
                ] = 0.0
        elif condition == "saturation_bilateral":
            for foot in range(2):
                signal = faulted[batch_index, foot, sensors[foot]]
                cap = 0.60 * signal.max()
                faulted[batch_index, foot, sensors[foot]] = torch.minimum(
                    signal, cap
                )
        elif condition == "drift_bilateral":
            for foot in range(2):
                faulted[batch_index, foot, sensors[foot]] += ramp
        elif condition == "packet_loss_bilateral":
            start = int(rng.integers(0, time_length - 50 + 1))
            faulted[batch_index, :, :, start : start + 50] = 0.0
        elif condition == "gain_miscalibration":
            faulted[batch_index, 0] *= 0.70
            faulted[batch_index, 1] *= 1.30
        else:
            raise KeyError(condition)
    return faulted


def exact_zero_route(
    inputs: torch.Tensor, modality_mask: torch.Tensor
) -> torch.Tensor:
    channel_amplitude = inputs.abs().amax(dim=-1)
    foot_flag = (channel_amplitude <= 1e-8).any(dim=-1)
    foot_flag = foot_flag & (modality_mask > 0)
    return foot_flag.sum(dim=1) >= 2


@torch.no_grad()
def predict_windows(
    model: nn.Module,
    loader: DataLoader,
    data: ClinicalData,
    device: torch.device,
    condition: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[pd.DataFrame] = []
    fixed_mask = torch.from_numpy(
        MASKS.get(condition, MASKS["clean_both"])
    ).to(device)
    for inputs, target, window_index in loader:
        indices = window_index.numpy()
        inputs = inputs.to(device=device, dtype=torch.float32)
        target = target.to(device)
        inputs = apply_locked_fault(inputs, data, indices, condition)
        mask = fixed_mask.expand(len(inputs), -1)
        logits = model(inputs, mask)
        probabilities = logits.softmax(dim=1).cpu().numpy()
        route = exact_zero_route(inputs, mask).cpu().numpy()
        subject_index = data.window_subject_index[indices]
        rows.append(
            pd.DataFrame(
                {
                    "subject": data.subject_ids[subject_index],
                    "study": data.studies[subject_index],
                    "truth": target.cpu().numpy(),
                    "window_ordinal": data.window_ordinal[indices],
                    "p_control": probabilities[:, 0],
                    "p_pd": probabilities[:, 1],
                    "fter_exact_route": route.astype(int),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def aggregate_subject_predictions(windows: pd.DataFrame) -> pd.DataFrame:
    return (
        windows.groupby(["subject", "study"], sort=False)
        .agg(
            truth=("truth", "first"),
            windows=("window_ordinal", "size"),
            p_control=("p_control", "mean"),
            p_pd=("p_pd", "mean"),
            fter_exact_route_rate=("fter_exact_route", "mean"),
        )
        .reset_index()
    )


@torch.no_grad()
def validation_macro_f1(
    model: nn.Module,
    loader: DataLoader,
    data: ClinicalData,
    device: torch.device,
) -> float:
    windows = predict_windows(
        model=model,
        loader=loader,
        data=data,
        device=device,
        condition="clean_both",
    )
    subjects = aggregate_subject_predictions(windows)
    predicted = (
        subjects[["p_control", "p_pd"]].to_numpy().argmax(axis=1)
    )
    return float(
        f1_score(
            subjects["truth"].to_numpy(),
            predicted,
            average="macro",
            zero_division=0,
        )
    )


def train_model(
    data: ClinicalData,
    train_subjects: np.ndarray,
    validation_subjects: np.ndarray,
    method: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, int, float]:
    set_seed(seed)
    train_windows = windows_for_subjects(data, train_subjects)
    validation_windows = windows_for_subjects(data, validation_subjects)
    train_dataset = WindowDataset(data, train_windows)
    validation_loader = DataLoader(
        WindowDataset(data, validation_windows),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    sampler = training_sampler(
        data=data,
        train_subjects=train_subjects,
        window_indices=train_windows,
        windows_per_subject=args.windows_per_subject,
        seed=seed + 10_000,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    model = MaskedFootFusion().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    class_counts = np.bincount(data.labels[train_subjects], minlength=2)
    class_weights = len(train_subjects) / (2.0 * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(
            class_weights, dtype=torch.float32, device=device
        ),
        label_smoothing=0.05,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(seed + 20_000)
    augmentation_generator = torch.Generator(device=device)
    augmentation_generator.manual_seed(seed + 30_000)

    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, target, _ in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            target = target.to(device)
            mask = sample_availability_masks(
                len(inputs), mask_generator, device
            )
            if method == "fasca":
                inputs = augment_fasca_pressure(
                    inputs, mask, augmentation_generator
                )
            elif method == "generic":
                inputs = augment_generic_timeseries(
                    inputs, mask, augmentation_generator
                )
            elif method == "iid":
                inputs = augment_iid_channels(
                    inputs, mask, augmentation_generator
                )
            elif method != "baseline":
                raise KeyError(f"Unknown training method: {method}")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                logits = model(inputs, mask)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_score = validation_macro_f1(
            model, validation_loader, data, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": validation_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_score > best_score + 1e-8:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No best model state was selected")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, float(best_score)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(choice)


def main() -> None:
    args = parse_args()
    data = load_data(args.data)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []

    for test_study in TEST_STUDIES:
        train_subjects, validation_subjects, test_subjects = subject_split(
            data, test_study, args.split_seed
        )
        split_dir = args.output_dir / f"test_{test_study}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_payload = {
            "test_study": test_study,
            "train_subjects": data.subject_ids[train_subjects].tolist(),
            "validation_subjects": data.subject_ids[
                validation_subjects
            ].tolist(),
            "test_subjects": data.subject_ids[test_subjects].tolist(),
        }
        (split_dir / "subjects.json").write_text(
            json.dumps(split_payload, indent=2), encoding="utf-8"
        )
        test_loader = DataLoader(
            WindowDataset(data, windows_for_subjects(data, test_subjects)),
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )
        for seed in args.seeds:
            for method in args.methods:
                run_dir = split_dir / f"seed_{seed}" / method
                run_dir.mkdir(parents=True, exist_ok=True)
                model, history, best_epoch, best_score = train_model(
                    data=data,
                    train_subjects=train_subjects,
                    validation_subjects=validation_subjects,
                    method=method,
                    seed=seed,
                    args=args,
                    device=device,
                )
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                history.to_csv(run_dir / "history.csv", index=False)
                for condition in CONDITIONS:
                    window_predictions = predict_windows(
                        model=model,
                        loader=test_loader,
                        data=data,
                        device=device,
                        condition=condition,
                    )
                    subject_predictions = aggregate_subject_predictions(
                        window_predictions
                    )
                    subject_predictions.insert(0, "condition", condition)
                    subject_predictions.insert(0, "method", method)
                    subject_predictions.insert(0, "seed", seed)
                    subject_predictions.insert(
                        0, "held_out_study", test_study
                    )
                    prediction_frames.append(subject_predictions)
                run_rows.append(
                    {
                        "held_out_study": test_study,
                        "seed": seed,
                        "method": method,
                        "best_epoch": best_epoch,
                        "best_validation_macro_f1": best_score,
                        "train_subjects": len(train_subjects),
                        "validation_subjects": len(validation_subjects),
                        "test_subjects": len(test_subjects),
                        "device": str(device),
                    }
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        args.output_dir / "subject_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(run_rows).to_csv(
        args.output_dir / "run_summary.csv", index=False
    )
    config = {
        "protocol": "docs/gaitpdb_clinical_external_protocol.md",
        "data": str(args.data),
        "split_seed": args.split_seed,
        "seeds": args.seeds,
        "test_studies": TEST_STUDIES,
        "methods": args.methods,
        "conditions": CONDITIONS,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "windows_per_subject": args.windows_per_subject,
        "device": str(device),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
