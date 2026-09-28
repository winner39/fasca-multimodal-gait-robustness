from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gait_robust.models import MultimodalLiteNet


MODALITIES = ("eeg", "emg", "imu", "fp")
SPEEDS = (0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        index = int(self.indices[item])
        inputs = {
            modality: torch.from_numpy(self.arrays[modality][index]).float()
            for modality in MODALITIES
        }
        target = torch.tensor(self.arrays["label"][index], dtype=torch.long)
        return inputs, target


def subject_split(
    subjects: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(subjects))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val, test = next(outer.split(indices, groups=subjects))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(train_val, groups=subjects[train_val])
    )
    return train_val[train_rel], train_val[val_rel], test


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality_mask: torch.Tensor | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    truth: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()
    for inputs, target in loader:
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        target = target.to(device)
        batch_mask = None
        if modality_mask is not None:
            batch_mask = modality_mask.to(device).expand(target.shape[0], -1)
        output = model(inputs, modality_mask=batch_mask)
        loss = criterion(output["logits"], target)
        probs = output["logits"].softmax(dim=1)
        losses.append(float(loss))
        truth.append(target.cpu().numpy())
        predicted.append(probs.argmax(dim=1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(truth),
        np.concatenate(predicted),
        np.concatenate(probabilities),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.data, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    arrays["label"] = np.asarray(
        [SPEEDS.index(round(float(value), 2)) for value in arrays["speed"]],
        dtype=np.int64,
    )
    train_idx, val_idx, test_idx = subject_split(arrays["subject"], args.seed)

    loaders = {
        "train": DataLoader(
            WindowDataset(arrays, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            WindowDataset(arrays, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            WindowDataset(arrays, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    channels = {modality: int(arrays[modality].shape[1]) for modality in MODALITIES}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalLiteNet(
        channels=channels,
        classes=len(SPEEDS),
        embedding_dim=args.embedding_dim,
        modality_dropout=args.modality_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    checkpoint = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: list[float] = []
        for inputs, target in loaders["train"]:
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in inputs.items()
            }
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(inputs)
                loss = criterion(output["logits"], target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach()))
        scheduler.step()
        val_loss, y_val, p_val, _ = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "val_loss": val_loss,
            "val_accuracy": accuracy_score(y_val, p_val),
            "val_macro_f1": f1_score(y_val, p_val, average="macro"),
        }
        history.append(row)
        print(json.dumps(row))
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "channels": channels,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                },
                checkpoint,
            )
        elif epoch - best_epoch >= args.patience:
            break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    test_loss, y_true, y_pred, y_prob = evaluate(model, loaders["test"], device)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(SPEEDS))),
        target_names=[str(v) for v in SPEEDS],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "task": "three-speed classification pipeline validation",
        "test_loss": test_loss,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "best_epoch": best_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "channels": channels,
        "split_subjects": {
            "train": sorted(np.unique(arrays["subject"][train_idx]).astype(int).tolist()),
            "validation": sorted(
                np.unique(arrays["subject"][val_idx]).astype(int).tolist()
            ),
            "test": sorted(np.unique(arrays["subject"][test_idx]).astype(int).tolist()),
        },
        "classification_report": report,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    predictions = pd.DataFrame(
        {
            "subject": arrays["subject"][test_idx],
            "trial_id": arrays["trial_id"][test_idx],
            "true_speed": [SPEEDS[i] for i in y_true],
            "predicted_speed": [SPEEDS[i] for i in y_pred],
            **{f"p_{speed}": y_prob[:, i] for i, speed in enumerate(SPEEDS)},
        }
    )
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(SPEEDS))))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=[str(v) for v in SPEEDS]
    )
    display.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

