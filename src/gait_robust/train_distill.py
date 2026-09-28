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
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from gait_robust.models import MultimodalLiteNet
from gait_robust.train_full import (
    MODALITIES,
    SPEEDS,
    WindowDataset,
    evaluate,
    subject_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lambda-kd", type=float, default=0.8)
    parser.add_argument("--lambda-feature", type=float, default=0.3)
    parser.add_argument("--lambda-relation", type=float, default=0.1)
    parser.add_argument("--max-drop-probability", type=float, default=0.65)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_mask(
    batch_size: int, modalities: int, drop_probability: float, device: torch.device
) -> torch.Tensor:
    mask = torch.rand(batch_size, modalities, device=device) >= drop_probability
    empty = ~mask.any(dim=1)
    if empty.any():
        keep = torch.randint(0, modalities, (int(empty.sum()),), device=device)
        mask[empty] = False
        mask[empty, keep] = True
    return mask


def relation_matrix(features: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    return features @ features.T


def all_mask_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> pd.DataFrame:
    rows = []
    for bits in itertools.product((False, True), repeat=len(MODALITIES)):
        if not any(bits):
            continue
        mask = torch.tensor(bits, dtype=torch.bool).unsqueeze(0)
        loss, truth, predicted, _ = evaluate(
            model, loader, device, modality_mask=mask
        )
        available = [m for m, present in zip(MODALITIES, bits) if present]
        rows.append(
            {
                "available_modalities": "+".join(available),
                "n_modalities": len(available),
                "loss": loss,
                "accuracy": accuracy_score(truth, predicted),
                "macro_f1": f1_score(truth, predicted, average="macro"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_modalities", "available_modalities"], ascending=[False, True]
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
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            WindowDataset(arrays, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            WindowDataset(arrays, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_saved = torch.load(args.teacher, map_location=device, weights_only=False)
    channels = teacher_saved["channels"]
    embedding_dim = int(teacher_saved["args"]["embedding_dim"])
    teacher = MultimodalLiteNet(
        channels=channels,
        classes=len(SPEEDS),
        embedding_dim=embedding_dim,
    ).to(device)
    teacher.load_state_dict(teacher_saved["state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = MultimodalLiteNet(
        channels=channels,
        classes=len(SPEEDS),
        embedding_dim=embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    supervised = nn.CrossEntropyLoss(label_smoothing=0.05)
    checkpoint = args.output_dir / "best_student.pt"
    best_score = -float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        student.train()
        fraction = (epoch - 1) / max(args.epochs - 1, 1)
        drop_probability = 0.1 + fraction * (args.max_drop_probability - 0.1)
        running: list[float] = []
        for inputs, target in loaders["train"]:
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in inputs.items()
            }
            target = target.to(device)
            mask = sample_mask(
                target.shape[0], len(MODALITIES), drop_probability, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_output = teacher(inputs)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                student_output = student(inputs, modality_mask=mask)
                task_loss = supervised(student_output["logits"], target)
                kd_loss = F.kl_div(
                    F.log_softmax(
                        student_output["logits"] / args.temperature, dim=1
                    ),
                    F.softmax(teacher_output["logits"] / args.temperature, dim=1),
                    reduction="batchmean",
                ) * (args.temperature**2)
                feature_loss = 1.0 - F.cosine_similarity(
                    student_output["fused"], teacher_output["fused"], dim=1
                ).mean()
                relation_loss = F.mse_loss(
                    relation_matrix(student_output["fused"]),
                    relation_matrix(teacher_output["fused"]),
                )
                loss = (
                    task_loss
                    + args.lambda_kd * kd_loss
                    + args.lambda_feature * feature_loss
                    + args.lambda_relation * relation_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach()))
        scheduler.step()

        # Model selection averages the full-input case and all single-modality
        # cases, preventing a superficially robust model that sacrifices the
        # clean operating point.
        validation = all_mask_metrics(student, loaders["val"], device)
        full_acc = float(
            validation.loc[
                validation["available_modalities"] == "+".join(MODALITIES),
                "accuracy",
            ].iloc[0]
        )
        single_acc = float(
            validation.loc[validation["n_modalities"] == 1, "accuracy"].mean()
        )
        selection_score = 0.5 * (full_acc + single_acc)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "drop_probability": drop_probability,
            "validation_full_accuracy": full_acc,
            "validation_mean_single_accuracy": single_acc,
            "selection_score": selection_score,
        }
        history.append(row)
        print(json.dumps(row))
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": student.state_dict(),
                    "channels": channels,
                    "args": vars(args),
                    "embedding_dim": embedding_dim,
                    "best_epoch": best_epoch,
                },
                checkpoint,
            )
        elif epoch - best_epoch >= args.patience:
            break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    student.load_state_dict(saved["state_dict"])
    results = all_mask_metrics(student, loaders["test"], device)
    results.to_csv(args.output_dir / "missing_modality_results.csv", index=False)
    full = results.loc[
        results["available_modalities"] == "+".join(MODALITIES)
    ].iloc[0]
    summary = {
        "best_epoch": best_epoch,
        "parameter_count": sum(p.numel() for p in student.parameters()),
        "test_full_accuracy": float(full["accuracy"]),
        "test_worst_accuracy": float(results["accuracy"].min()),
        "test_mean_accuracy": float(results["accuracy"].mean()),
        "test_mean_single_modality_accuracy": float(
            results.loc[results["n_modalities"] == 1, "accuracy"].mean()
        ),
        "test_largest_drop": float(full["accuracy"] - results["accuracy"].min()),
        "teacher_checkpoint": str(args.teacher.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

