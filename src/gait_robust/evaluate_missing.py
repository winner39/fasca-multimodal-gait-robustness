from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = np.load(args.data, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    arrays["label"] = np.asarray(
        [SPEEDS.index(round(float(value), 2)) for value in arrays["speed"]],
        dtype=np.int64,
    )
    _, _, test_idx = subject_split(arrays["subject"], args.seed)
    loader = DataLoader(
        WindowDataset(arrays, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultimodalLiteNet(
        channels=saved["channels"],
        classes=len(SPEEDS),
        embedding_dim=int(saved["args"]["embedding_dim"]),
        modality_dropout=float(saved["args"]["modality_dropout"]),
    ).to(device)
    model.load_state_dict(saved["state_dict"])

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
                "test_loss": loss,
                "accuracy": accuracy_score(truth, predicted),
                "macro_f1": f1_score(truth, predicted, average="macro"),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["n_modalities", "available_modalities"], ascending=[False, True]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    full = table.loc[table["available_modalities"] == "+".join(MODALITIES)].iloc[0]
    summary = {
        "full_accuracy": float(full["accuracy"]),
        "worst_accuracy": float(table["accuracy"].min()),
        "mean_accuracy_all_nonempty_masks": float(table["accuracy"].mean()),
        "mean_single_modality_accuracy": float(
            table.loc[table["n_modalities"] == 1, "accuracy"].mean()
        ),
        "largest_accuracy_drop": float(full["accuracy"] - table["accuracy"].min()),
        "rows": table.to_dict(orient="records"),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

