from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from gait_robust.cross_validate_full import subject_folds
from gait_robust.data import FoldRobustScaler
from gait_robust.external_baseline_models import (
    ADAPTAdaptation,
    CIMSleepNetAdaptation,
    CentaurAdaptation,
)
from gait_robust.models import MultimodalLiteNet
from gait_robust.robust_models import (
    ActionMAELite,
    EmbraceNetLite,
    QualityEmbraceNetLite,
    QualityRapidGait,
    RapidGait,
)
from gait_robust.sci_cross_validate import (
    MODALITIES,
    SPEEDS,
    load_arrays,
    make_loaders,
    move_inputs,
    trial_metrics,
)


SCENARIOS = (
    "noise_10db",
    "channel_dropout_30",
    "structured_dropout_30",
    "packet_loss_30",
    "shift_250ms",
    "gain_1.5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--fallback-run-root", type=Path, default=None)
    parser.add_argument("--fallback-method", default=None)
    parser.add_argument(
        "--fallback-model-type",
        choices=(
            "rapid",
            "uniform_rapid",
            "quality_rapid",
            "moddrop",
            "embracenet",
            "quality_embracenet",
            "actionmae",
            "centaur_adaptation",
            "adapt_adaptation",
            "cimsleepnet_adaptation",
        ),
        default=None,
    )
    parser.add_argument("--output-method", default=None)
    parser.add_argument(
        "--model-type",
        choices=(
            "rapid",
            "uniform_rapid",
            "quality_rapid",
            "moddrop",
            "embracenet",
            "quality_embracenet",
            "actionmae",
            "centaur_adaptation",
            "adapt_adaptation",
            "cimsleepnet_adaptation",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=None,
        help="Optional gzip CSV with trial-level probabilities and subjects.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51])
    parser.add_argument("--partition-seed", type=int, default=20260901)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _shift(signal: torch.Tensor, samples: int) -> torch.Tensor:
    shifted = torch.zeros_like(signal)
    shifted[..., samples:] = signal[..., :-samples]
    return shifted


def corrupt_inputs(
    inputs: dict[str, torch.Tensor],
    target_modalities: tuple[str, ...],
    scenario: str,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    corrupted = {key: value.clone() for key, value in inputs.items()}
    for modality in target_modalities:
        signal = corrupted[modality]
        if scenario == "noise_10db":
            rms = signal.square().mean(
                dim=(-2, -1), keepdim=True
            ).sqrt().clamp_min(0.05)
            noise = torch.randn(
                signal.shape,
                generator=generator,
                device=signal.device,
                dtype=signal.dtype,
            )
            noise_rms = noise.square().mean(
                dim=(-2, -1), keepdim=True
            ).sqrt().clamp_min(1e-6)
            corrupted[modality] = (
                signal + noise / noise_rms * rms / np.sqrt(10.0)
            )
        elif scenario == "channel_dropout_30":
            channels = signal.shape[1]
            count = max(1, round(0.30 * channels))
            selected = torch.randperm(
                channels,
                generator=generator,
                device=signal.device,
            )[:count]
            corrupted[modality][:, selected] = 0.0
        elif scenario == "structured_dropout_30":
            if modality == "imu":
                sensors = signal.shape[1] // 3
                count = max(1, round(0.30 * sensors))
                selected = torch.randperm(
                    sensors,
                    generator=generator,
                    device=signal.device,
                )[:count]
                channel_ids = (
                    selected[:, None] * 3
                    + torch.arange(3, device=signal.device)[None, :]
                ).reshape(-1)
                corrupted[modality][:, channel_ids] = 0.0
            elif modality == "fp":
                plate = int(
                    torch.randint(
                        2,
                        (),
                        generator=generator,
                        device=signal.device,
                    ).item()
                )
                corrupted[modality][
                    :, plate * 8 : (plate + 1) * 8
                ] = 0.0
            else:
                channels = signal.shape[1]
                count = max(1, round(0.30 * channels))
                selected = torch.randperm(
                    channels,
                    generator=generator,
                    device=signal.device,
                )[:count]
                corrupted[modality][:, selected] = 0.0
        elif scenario == "packet_loss_30":
            time = signal.shape[-1]
            length = round(0.30 * time)
            start = (time - length) // 2
            corrupted[modality][..., start : start + length] = 0.0
        elif scenario == "shift_250ms":
            corrupted[modality] = _shift(signal, 25)
        elif scenario == "gain_1.5":
            corrupted[modality] = 1.5 * signal
        else:
            raise ValueError(scenario)
    return corrupted


def dead_channel_route(
    inputs: dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
    min_faulty_modalities: int = 2,
) -> torch.Tensor:
    """Route only faults spanning multiple available sensor modalities.

    The specialist is trained for joint structured corruption. Requiring at
    least two affected modalities prevents a single schema/electrode failure
    from overriding the stronger FASCA baseline.
    """
    faulty_modalities = torch.zeros(
        modality_mask.shape[0],
        dtype=torch.int64,
        device=modality_mask.device,
    )
    for modality in ("eeg", "emg", "imu"):
        modality_index = MODALITIES.index(modality)
        flat_channels = (
            inputs[modality].abs().amax(dim=-1) <= 1e-8
        )
        faulty_modalities += (
            flat_channels.any(dim=1)
            & modality_mask[:, modality_index]
        ).to(dtype=torch.int64)
    return faulty_modalities >= int(min_faulty_modalities)


@torch.no_grad()
def predict_condition(
    model: nn.Module,
    loader,
    device: torch.device,
    target_modalities: tuple[str, ...],
    scenario: str | None,
    seed: int,
    fallback_model: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    if fallback_model is not None:
        fallback_model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    truth, probabilities, mean_qualities = [], [], []
    route_rates = []
    for inputs, target in loader:
        inputs = move_inputs(inputs, device)
        target = target.to(device)
        if scenario is not None:
            inputs = corrupt_inputs(
                inputs, target_modalities, scenario, generator
            )
        mask = torch.ones(
            target.shape[0],
            len(MODALITIES),
            dtype=torch.bool,
            device=device,
        )
        if fallback_model is not None:
            fallback_output = fallback_model(
                inputs, modality_mask=mask
            )
            dead_channel = dead_channel_route(inputs, mask)
            output = fallback_output
            if dead_channel.any():
                specialist_output = model(
                    inputs, modality_mask=mask
                )
                output["logits"] = torch.where(
                    dead_channel[:, None],
                    specialist_output["logits"],
                    fallback_output["logits"],
                )
            route_rates.append(dead_channel.float().mean().item())
        else:
            output = model(inputs, modality_mask=mask)
        truth.append(target.cpu().numpy())
        probabilities.append(
            output["logits"].softmax(dim=1).cpu().numpy()
        )
        if "predicted_quality" in output and target_modalities:
            indices = [
                MODALITIES.index(modality)
                for modality in target_modalities
            ]
            mean_qualities.append(
                output["predicted_quality"][:, indices]
                .mean()
                .cpu()
                .item()
            )
    if fallback_model is not None:
        quality = float(np.mean(route_rates))
    else:
        quality = (
            float(np.mean(mean_qualities))
            if mean_qualities
            else float("nan")
        )
    return (
        np.concatenate(truth),
        np.concatenate(probabilities),
        quality,
    )


def verify_split(
    split_path: Path,
    arrays: dict[str, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    names = ("train_subjects", "validation_subjects", "test_subjects")
    for name, index in zip(names, indices):
        expected = sorted(
            np.unique(arrays["subject"][index]).astype(int).tolist()
        )
        if payload.get(name) != expected:
            raise ValueError(f"Split mismatch at {split_path}: {name}")


def main() -> None:
    args = parse_args()
    arrays = load_arrays(args.data)
    folds = subject_folds(arrays["subject"], args.folds, args.partition_seed)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    channels = {
        modality: int(arrays[modality].shape[1])
        for modality in MODALITIES
    }
    model_classes = {
        "rapid": RapidGait,
        "uniform_rapid": RapidGait,
        "quality_rapid": QualityRapidGait,
        "moddrop": MultimodalLiteNet,
        "embracenet": EmbraceNetLite,
        "quality_embracenet": QualityEmbraceNetLite,
        "actionmae": ActionMAELite,
        "centaur_adaptation": CentaurAdaptation,
        "adapt_adaptation": ADAPTAdaptation,
        "cimsleepnet_adaptation": CIMSleepNetAdaptation,
    }
    model_class = model_classes[args.model_type]
    if (args.fallback_run_root is None) != (
        args.fallback_method is None
    ):
        raise ValueError(
            "fallback_run_root and fallback_method must be set together"
        )
    if args.fallback_run_root is not None:
        if args.fallback_model_type is None:
            raise ValueError(
                "fallback_model_type is required with fallback_run_root"
            )
        fallback_class = model_classes[args.fallback_model_type]
    else:
        fallback_class = None
    output_method = args.output_method or args.method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    prediction_rows = []
    pooled: dict[
        tuple[int, str, str], dict[str, list[np.ndarray] | list[float]]
    ] = {}
    for fold, indices in enumerate(folds, start=1):
        fold_dir = args.run_root / f"fold_{fold}"
        verify_split(fold_dir / "subjects.json", arrays, indices)
        train_idx, _, test_idx = indices
        scaler = FoldRobustScaler.fit(arrays, train_idx)
        for seed in args.seeds:
            checkpoint = (
                fold_dir
                / f"seed_{seed}"
                / args.method
                / "best_model.pt"
            )
            payload = torch.load(
                checkpoint, map_location=device, weights_only=True
            )
            model = model_class(
                channels=channels,
                classes=len(SPEEDS),
                **(
                    {"uniform_reliability": True}
                    if args.model_type == "uniform_rapid"
                    else {}
                ),
            ).to(device)
            model.load_state_dict(payload["state_dict"])
            fallback_model = None
            if fallback_class is not None:
                fallback_checkpoint = (
                    args.fallback_run_root
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                    / args.fallback_method
                    / "best_model.pt"
                )
                fallback_payload = torch.load(
                    fallback_checkpoint,
                    map_location=device,
                    weights_only=True,
                )
                fallback_model = fallback_class(
                    channels=channels,
                    classes=len(SPEEDS),
                ).to(device)
                fallback_model.load_state_dict(
                    fallback_payload["state_dict"]
                )
            loaders = make_loaders(
                arrays,
                indices,
                scaler,
                args.batch_size,
                device,
                int(seed * 100 + fold + 77),
            )
            trial_ids = arrays["trial_id"][test_idx]
            truth, probabilities, quality = predict_condition(
                model,
                loaders["test"],
                device,
                (),
                None,
                seed=901_000 + fold,
                fallback_model=fallback_model,
            )
            metrics, trial_predictions = trial_metrics(
                truth, probabilities, trial_ids
            )
            trial_subjects = (
                pd.DataFrame(
                    {
                        "trial_id": arrays["trial_id"][test_idx],
                        "subject": arrays["subject"][test_idx],
                    }
                )
                .drop_duplicates("trial_id")
                .set_index("trial_id")["subject"]
            )
            trial_predictions["subject"] = (
                trial_predictions["trial_id"].map(trial_subjects).astype(int)
            )
            trial_predictions = trial_predictions.assign(
                method=output_method,
                fold=fold,
                seed=seed,
                scenario="clean",
                target="none",
            )
            prediction_rows.append(trial_predictions)
            rows.append(
                {
                    "level": "fold",
                    "fold": fold,
                    "seed": seed,
                    "scenario": "clean",
                    "target": "none",
                    "trial_macro_f1": metrics["macro_f1"],
                    "predicted_target_quality": quality,
                }
            )
            clean_key = (seed, "clean", "none")
            pooled.setdefault(
                clean_key,
                {"truth": [], "probabilities": [], "trial_ids": [], "quality": []},
            )
            pooled[clean_key]["truth"].append(truth)
            pooled[clean_key]["probabilities"].append(probabilities)
            pooled[clean_key]["trial_ids"].append(trial_ids)
            pooled[clean_key]["quality"].append(quality)
            targets = [
                (modality,) for modality in MODALITIES
            ] + [tuple(MODALITIES)]
            for scenario in SCENARIOS:
                for target_modalities in targets:
                    truth, probabilities, quality = predict_condition(
                        model,
                        loaders["test"],
                        device,
                        target_modalities,
                        scenario,
                        seed=(
                            902_000
                            + 100 * fold
                            + 10 * SCENARIOS.index(scenario)
                            + len(target_modalities)
                        ),
                        fallback_model=fallback_model,
                    )
                    metrics, trial_predictions = trial_metrics(
                        truth, probabilities, trial_ids
                    )
                    trial_predictions["subject"] = (
                        trial_predictions["trial_id"]
                        .map(trial_subjects)
                        .astype(int)
                    )
                    trial_predictions = trial_predictions.assign(
                        method=output_method,
                        fold=fold,
                        seed=seed,
                        scenario=scenario,
                        target="+".join(target_modalities),
                    )
                    prediction_rows.append(trial_predictions)
                    rows.append(
                        {
                            "level": "fold",
                            "fold": fold,
                            "seed": seed,
                            "scenario": scenario,
                            "target": "+".join(target_modalities),
                            "trial_macro_f1": metrics["macro_f1"],
                            "predicted_target_quality": quality,
                        }
                    )
                    target_name = "+".join(target_modalities)
                    key = (seed, scenario, target_name)
                    pooled.setdefault(
                        key,
                        {
                            "truth": [],
                            "probabilities": [],
                            "trial_ids": [],
                            "quality": [],
                        },
                    )
                    pooled[key]["truth"].append(truth)
                    pooled[key]["probabilities"].append(probabilities)
                    pooled[key]["trial_ids"].append(trial_ids)
                    pooled[key]["quality"].append(quality)
            del model
            torch.cuda.empty_cache()
    for (seed, scenario, target_name), payload in pooled.items():
        metrics, _ = trial_metrics(
            np.concatenate(payload["truth"]),
            np.concatenate(payload["probabilities"]),
            np.concatenate(payload["trial_ids"]),
        )
        qualities = np.asarray(payload["quality"], dtype=float)
        rows.append(
            {
                "level": "pooled",
                "fold": 0,
                "seed": seed,
                "scenario": scenario,
                "target": target_name,
                "trial_macro_f1": metrics["macro_f1"],
                "predicted_target_quality": (
                    float(np.nanmean(qualities))
                    if np.isfinite(qualities).any()
                    else float("nan")
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    if args.prediction_output is not None:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            args.prediction_output,
            index=False,
            compression="gzip",
        )


if __name__ == "__main__":
    main()
