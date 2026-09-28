from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from gait_robust.models import MultimodalLiteNet
from gait_robust.train_full import MODALITIES, SPEEDS, WindowDataset, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_folds(
    subjects: np.ndarray, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Match the published Experiment-3 subject-wise split structure."""
    unique_subjects = np.unique(subjects).copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)
    test_groups = np.array_split(unique_subjects, n_folds)
    output = []
    for test_subjects in test_groups:
        remaining = np.setdiff1d(unique_subjects, test_subjects)
        rng.shuffle(remaining)
        n_validation = max(1, len(remaining) // 5)
        validation_subjects = remaining[:n_validation]
        train_subjects = remaining[n_validation:]
        train_idx = np.flatnonzero(np.isin(subjects, train_subjects))
        validation_idx = np.flatnonzero(np.isin(subjects, validation_subjects))
        test_idx = np.flatnonzero(np.isin(subjects, test_subjects))
        assert not (
            set(train_subjects) & set(validation_subjects)
            or set(train_subjects) & set(test_subjects)
            or set(validation_subjects) & set(test_subjects)
        )
        output.append((train_idx, validation_idx, test_idx))
    test_union = np.concatenate([subjects[test] for _, _, test in output])
    observed_counts = pd.Series(test_union).groupby(level=0).size()
    # Each window appears in the test set once because each subject appears in
    # exactly one test fold.
    assert len(test_union) == len(subjects)
    assert observed_counts.eq(1).all()
    return output


def train_fold(
    fold: int,
    arrays: dict[str, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame]:
    train_idx, validation_idx, test_idx = indices
    loaders = {
        "train": DataLoader(
            WindowDataset(arrays, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            WindowDataset(arrays, validation_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            WindowDataset(arrays, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        ),
    }
    channels = {modality: int(arrays[modality].shape[1]) for modality in MODALITIES}
    model = MultimodalLiteNet(
        channels=channels,
        classes=len(SPEEDS),
        embedding_dim=args.embedding_dim,
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
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, target in loaders["train"]:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(inputs)["logits"]
                loss = criterion(logits, target)
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
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold}: no checkpoint was selected")
    model.load_state_dict(best_state)
    test_loss, truth, predicted, probabilities = evaluate(
        model, loaders["test"], device
    )
    prediction_table = pd.DataFrame(
        {
            "fold": fold,
            "subject": arrays["subject"][test_idx],
            "trial_id": arrays["trial_id"][test_idx],
            "true_speed": [SPEEDS[index] for index in truth],
            "predicted_speed": [SPEEDS[index] for index in predicted],
            **{
                f"p_{speed}": probabilities[:, index]
                for index, speed in enumerate(SPEEDS)
            },
        }
    )
    prediction_table["correct"] = (
        prediction_table["true_speed"] == prediction_table["predicted_speed"]
    )
    subject_accuracy = prediction_table.groupby("subject")["correct"].mean()
    trial_vote = (
        prediction_table.groupby(["subject", "trial_id"])
        .agg(
            true_speed=("true_speed", "first"),
            predicted_speed=(
                "predicted_speed",
                lambda values: values.value_counts().index[0],
            ),
        )
        .assign(correct=lambda frame: frame.true_speed == frame.predicted_speed)
    )
    metrics: dict[str, object] = {
        "fold": fold,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "window_accuracy": accuracy_score(truth, predicted),
        "window_macro_f1": f1_score(truth, predicted, average="macro"),
        "mean_subject_accuracy": float(subject_accuracy.mean()),
        "minimum_subject_accuracy": float(subject_accuracy.min()),
        "trial_majority_accuracy": float(trial_vote["correct"].mean()),
        "train_subjects": sorted(
            np.unique(arrays["subject"][train_idx]).astype(int).tolist()
        ),
        "validation_subjects": sorted(
            np.unique(arrays["subject"][validation_idx]).astype(int).tolist()
        ),
        "test_subjects": sorted(
            np.unique(arrays["subject"][test_idx]).astype(int).tolist()
        ),
        "test_windows": int(len(test_idx)),
    }
    return metrics, prediction_table


def mean_sd_ci(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half_width = float(
        stats.t.ppf(0.975, df=len(array) - 1) * stats.sem(array)
    )
    return {
        "mean": mean,
        "sd": sd,
        "fold_t_95ci": [mean - half_width, mean + half_width],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    loaded = np.load(args.data, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    arrays["label"] = np.asarray(
        [SPEEDS.index(round(float(value), 2)) for value in arrays["speed"]],
        dtype=np.int64,
    )
    folds = subject_folds(arrays["subject"], args.folds, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics = []
    prediction_tables = []
    for fold_number, indices in enumerate(folds, start=1):
        seed_everything(args.seed + fold_number)
        metrics, predictions = train_fold(
            fold_number, arrays, indices, args, device
        )
        fold_metrics.append(metrics)
        prediction_tables.append(predictions)
        print(json.dumps(metrics))

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_predictions.to_csv(args.output_dir / "fold_predictions.csv", index=False)
    pooled_truth = all_predictions["true_speed"].map(
        {speed: index for index, speed in enumerate(SPEEDS)}
    )
    pooled_predicted = all_predictions["predicted_speed"].map(
        {speed: index for index, speed in enumerate(SPEEDS)}
    )
    test_subject_lists = [
        set(metrics["test_subjects"]) for metrics in fold_metrics
    ]
    pairwise_overlap = any(
        test_subject_lists[i] & test_subject_lists[j]
        for i in range(len(test_subject_lists))
        for j in range(i + 1, len(test_subject_lists))
    )
    summary = {
        "protocol": (
            "Five disjoint subject-wise test folds after one seeded subject "
            "shuffle; validation subjects are drawn only from each fold's "
            "remaining training subjects."
        ),
        "task": "three-speed classification",
        "subjects": int(len(np.unique(arrays["subject"]))),
        "windows": int(len(arrays["subject"])),
        "parameter_count": sum(
            parameter.numel()
            for parameter in MultimodalLiteNet(
                channels={
                    modality: int(arrays[modality].shape[1])
                    for modality in MODALITIES
                },
                classes=len(SPEEDS),
                embedding_dim=args.embedding_dim,
            ).parameters()
        ),
        "test_subject_overlap_between_folds": pairwise_overlap,
        "window_accuracy": mean_sd_ci(
            [float(metrics["window_accuracy"]) for metrics in fold_metrics]
        ),
        "window_macro_f1": mean_sd_ci(
            [float(metrics["window_macro_f1"]) for metrics in fold_metrics]
        ),
        "mean_subject_accuracy": mean_sd_ci(
            [float(metrics["mean_subject_accuracy"]) for metrics in fold_metrics]
        ),
        "trial_majority_accuracy": mean_sd_ci(
            [float(metrics["trial_majority_accuracy"]) for metrics in fold_metrics]
        ),
        "pooled_out_of_fold_accuracy": accuracy_score(
            pooled_truth, pooled_predicted
        ),
        "pooled_out_of_fold_macro_f1": f1_score(
            pooled_truth, pooled_predicted, average="macro"
        ),
        "folds": fold_metrics,
    }
    (args.output_dir / "cross_validation_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

