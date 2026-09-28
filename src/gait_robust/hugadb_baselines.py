from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from gait_robust.cross_validate_full import subject_folds
from gait_robust.hugadb_external import (
    ACTIVITIES,
    MASK_ROWS,
    MODALITIES,
    apply_corruption,
    class_weights,
    macro_f1,
    make_loaders,
    mask_name,
    move_inputs,
    robust_scale,
    save_status,
    summarize,
)
from gait_robust.models import GatedFusion, TemporalLiteEncoder


METHODS = ("moddrop", "embracenet", "actionmae", "compass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HuGaDB missing-modality architecture baselines."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 72, 73])
    parser.add_argument("--partition-seed", type=int, default=20260902)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--missing-probability", type=float, default=0.70)
    parser.add_argument("--lambda-reconstruction", type=float, default=0.50)
    parser.add_argument("--lambda-alignment", type=float, default=0.20)
    parser.add_argument("--lambda-proxy", type=float, default=0.50)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HuGaDBEncoderBank(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
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

    def forward(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.stack(
            [
                self.encoders[modality](inputs[modality])
                + self.modality_tokens[index]
                for index, modality in enumerate(MODALITIES)
            ],
            dim=1,
        )


class HuGaDBGated(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.encoder_bank = HuGaDBEncoderBank(embedding_dim)
        self.fusion = GatedFusion(embedding_dim, len(MODALITIES))
        self.classifier = nn.Linear(embedding_dim, len(ACTIVITIES))

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.encoder_bank(inputs)
        fused, weights = self.fusion(tokens, modality_mask)
        return {
            "logits": self.classifier(fused),
            "modality_mask": modality_mask,
            "fusion_weights": weights,
        }


class HuGaDBEmbraceNet(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.encoder_bank = HuGaDBEncoderBank(embedding_dim)
        self.docking = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.GELU(),
                )
                for _ in MODALITIES
            ]
        )
        self.selection_logits = nn.Parameter(
            torch.zeros(len(MODALITIES))
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, len(ACTIVITIES))

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        real = self.encoder_bank(inputs)
        docked = torch.stack(
            [
                layer(real[:, index])
                for index, layer in enumerate(self.docking)
            ],
            dim=1,
        )
        logits = self.selection_logits.unsqueeze(0).expand(
            real.shape[0], -1
        )
        probabilities = logits.masked_fill(
            ~modality_mask, -1e4
        ).softmax(dim=1)
        if self.training:
            selected = torch.multinomial(
                probabilities,
                num_samples=docked.shape[-1],
                replacement=True,
            )
            fused = torch.gather(
                docked.permute(0, 2, 1),
                2,
                selected.unsqueeze(-1),
            ).squeeze(-1)
        else:
            fused = (
                docked * probabilities.unsqueeze(-1)
            ).sum(dim=1)
        fused = self.output_norm(fused)
        return {
            "logits": self.classifier(fused),
            "modality_mask": modality_mask,
            "selection_probabilities": probabilities,
        }


class HuGaDBActionMAE(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.encoder_bank = HuGaDBEncoderBank(embedding_dim)
        self.mask_tokens = nn.Parameter(
            torch.zeros(len(MODALITIES), embedding_dim)
        )
        nn.init.normal_(self.mask_tokens, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, len(ACTIVITIES))

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        real = self.encoder_bank(inputs)
        masked = torch.where(
            modality_mask.unsqueeze(-1),
            real,
            self.mask_tokens.unsqueeze(0),
        )
        completed = self.transformer(masked)
        fused = self.output_norm(completed.mean(dim=1))
        missing = ~modality_mask
        reconstruction = (
            F.smooth_l1_loss(
                completed[missing], real.detach()[missing]
            )
            if missing.any()
            else fused.new_zeros(())
        )
        return {
            "logits": self.classifier(fused),
            "modality_mask": modality_mask,
            "reconstruction_loss": reconstruction,
        }


class PairGenerator(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class HuGaDBCompass(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.encoder_bank = HuGaDBEncoderBank(embedding_dim)
        self.generators = nn.ModuleDict(
            {
                f"{source}_to_{target}": PairGenerator(embedding_dim)
                for source in MODALITIES
                for target in MODALITIES
                if source != target
            }
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, len(ACTIVITIES))

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        real = self.encoder_bank(inputs)
        proxies = []
        for target_index, target in enumerate(MODALITIES):
            candidates, candidate_masks = [], []
            for source_index, source in enumerate(MODALITIES):
                if source == target:
                    continue
                candidates.append(
                    self.generators[f"{source}_to_{target}"](
                        real[:, source_index]
                    )
                )
                candidate_masks.append(modality_mask[:, source_index])
            stacked = torch.stack(candidates, dim=1)
            available = torch.stack(candidate_masks, dim=1).to(
                stacked.dtype
            )
            proxy = (
                stacked * available.unsqueeze(-1)
            ).sum(dim=1) / available.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            proxies.append(proxy)
        proxy = torch.stack(proxies, dim=1)
        completed = torch.where(
            modality_mask.unsqueeze(-1), real, proxy
        )
        fused = self.output_norm(completed.sum(dim=1))
        missing = ~modality_mask
        alignment = (
            F.mse_loss(proxy[missing], real.detach()[missing])
            if missing.any()
            else fused.new_zeros(())
        )
        proxy_logits = self.classifier(
            self.output_norm(proxy.reshape(-1, proxy.shape[-1]))
        ).reshape(proxy.shape[0], len(MODALITIES), -1)
        return {
            "logits": self.classifier(fused),
            "modality_mask": modality_mask,
            "proxy_logits": proxy_logits,
            "alignment_loss": alignment,
        }


def make_model(method: str, embedding_dim: int) -> nn.Module:
    if method == "moddrop":
        return HuGaDBGated(embedding_dim)
    if method == "embracenet":
        return HuGaDBEmbraceNet(embedding_dim)
    if method == "actionmae":
        return HuGaDBActionMAE(embedding_dim)
    if method == "compass":
        return HuGaDBCompass(embedding_dim)
    raise ValueError(method)


def sample_mask(
    batch_size: int,
    probability: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = torch.ones(
        batch_size, len(MODALITIES), dtype=torch.bool, device=device
    )
    affected = torch.nonzero(
        torch.rand(
            batch_size, generator=generator, device=device
        )
        < probability,
        as_tuple=False,
    ).flatten()
    if len(affected):
        scores = torch.rand(
            len(affected),
            len(MODALITIES),
            generator=generator,
            device=device,
        )
        keep = scores.argmax(dim=1)
        mask[affected] = False
        mask[affected, keep] = True
    return mask


@torch.no_grad()
def predict(
    model: nn.Module,
    loader,
    device: torch.device,
    bits: tuple[bool, ...],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truth, probabilities = [], []
    for inputs, target in loader:
        inputs = move_inputs(inputs, device)
        mask = torch.tensor(
            bits, dtype=torch.bool, device=device
        ).unsqueeze(0).expand(len(target), -1)
        logits = model(inputs, modality_mask=mask)["logits"]
        truth.append(target.numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(truth), np.concatenate(probabilities)


def validation_score(
    model: nn.Module, loader, device: torch.device
) -> tuple[float, dict[str, float]]:
    scores = {}
    for bits in MASK_ROWS:
        truth, probabilities = predict(model, loader, device, bits)
        scores[mask_name(bits)] = macro_f1(truth, probabilities)
    full = scores["emg+imu"]
    incomplete = np.mean(
        [value for name, value in scores.items() if name != "emg+imu"]
    )
    return float(0.5 * (full + incomplete)), scores


def train_model(
    method: str,
    model: nn.Module,
    loaders,
    weights: torch.Tensor,
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
        optimizer, args.epochs
    )
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(run_seed + 91_337)
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, target in loaders["train"]:
            inputs = move_inputs(inputs, device)
            target = target.to(device)
            mask = sample_mask(
                len(target),
                args.missing_probability,
                device,
                generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                output = model(inputs, modality_mask=mask)
                task = F.cross_entropy(
                    output["logits"],
                    target,
                    weight=weights,
                    label_smoothing=0.05,
                )
                loss = task
                if method == "actionmae":
                    loss = (
                        loss
                        + args.lambda_reconstruction
                        * output["reconstruction_loss"]
                    )
                elif method == "compass":
                    missing = ~mask
                    expanded = target[:, None].expand(
                        -1, len(MODALITIES)
                    )
                    proxy_loss = F.cross_entropy(
                        output["proxy_logits"][missing],
                        expanded[missing],
                        weight=weights,
                        label_smoothing=0.05,
                    )
                    loss = (
                        loss
                        + args.lambda_alignment
                        * output["alignment_loss"]
                        + args.lambda_proxy * proxy_loss
                    )
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp.step(optimizer)
            amp.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        score, values = validation_score(
            model, loaders["validation"], device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_score": score,
                **{
                    f"validation_{name}_macro_f1": value
                    for name, value in values.items()
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


def condition_predictions(
    method: str,
    fold: int,
    seed: int,
    model: nn.Module,
    loader,
    subjects: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    corruptions = (
        "clean",
        "imu_node_dropout_2of6",
        "emg_lead_dropout_1of2",
        "all_gain_1.5",
        "all_packet_loss_30",
        "all_noise_10db",
        "all_shift_250ms",
    )
    frames = []
    model.eval()
    for corruption_index, corruption in enumerate(corruptions):
        generator = torch.Generator(device=device)
        generator.manual_seed(918_221 + corruption_index)
        truth_parts, probability_parts = [], []
        for inputs, target in loader:
            inputs = move_inputs(inputs, device)
            corrupted = apply_corruption(inputs, corruption, generator)
            mask = torch.ones(
                len(target),
                len(MODALITIES),
                dtype=torch.bool,
                device=device,
            )
            with torch.no_grad():
                logits = model(
                    corrupted, modality_mask=mask
                )["logits"]
            truth_parts.append(target.numpy())
            probability_parts.append(logits.softmax(dim=1).cpu().numpy())
        truth = np.concatenate(truth_parts)
        probabilities = np.concatenate(probability_parts)
        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "corruption": corruption,
                    "subject": subjects,
                    "true_class": truth,
                    **{
                        f"p_{activity}": probabilities[:, index]
                        for index, activity in enumerate(ACTIVITIES)
                    },
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


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
                "activities": ACTIVITIES,
                "modalities": MODALITIES,
            },
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
            json.dumps(scaler_metadata, indent=2), encoding="utf-8"
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
            weights = class_weights(
                arrays["label"][indices[0]], device
            )
            for method in args.methods:
                run_dir = fold_dir / f"seed_{seed}" / method
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
                loaders = make_loaders(
                    scaled,
                    indices,
                    args.batch_size,
                    device,
                    run_seed + 77,
                )
                model = make_model(
                    method, args.embedding_dim
                ).to(device)
                model, best_epoch, history = train_model(
                    method,
                    model,
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
                        "method": method,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "best_epoch": best_epoch,
                        "parameter_count": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                        ),
                    },
                    run_dir / "best_model.pt",
                )
                test_indices = indices[2]
                frames = []
                for bits in MASK_ROWS:
                    truth, probabilities = predict(
                        model, loaders["test"], device, bits
                    )
                    frames.append(
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
                                    f"p_{activity}": probabilities[:, index]
                                    for index, activity in enumerate(
                                        ACTIVITIES
                                    )
                                },
                            }
                        )
                    )
                pd.concat(frames, ignore_index=True).to_csv(
                    predictions_path, index=False, compression="gzip"
                )
                condition_predictions(
                    method,
                    fold,
                    seed,
                    model,
                    loaders["test"],
                    arrays["subject"][test_indices],
                    device,
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
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del scaled
    summarize(args.output_dir)
    save_status(
        status_path,
        completed,
        total,
        "HuGaDB architecture baselines complete",
    )


if __name__ == "__main__":
    main()
