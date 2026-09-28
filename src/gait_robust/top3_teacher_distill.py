from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from gait_robust.cross_validate_full import subject_folds
from gait_robust.data import FoldRobustScaler
from gait_robust.models import MultimodalLiteNet
from gait_robust.rapid_distill import (
    MASK_ROWS,
    summarize_seed_predictions,
    train_student,
)
from gait_robust.robust_models import (
    ActionMAELite,
    CompassLite,
    EmbraceNetLite,
    RapidGait,
)
from gait_robust.sci_cross_validate import (
    MODALITIES,
    SPEEDS,
    evaluate_all_masks,
    load_arrays,
    make_loaders,
    predict,
    seed_everything,
    trial_metrics,
    write_status,
)
from gait_robust.xtinyhar import (
    XTinyHARAdaptation,
    XTinyHARTeacherAdaptation,
)


METHODS = ("top3_ensemble_aug_kd", "hybrid_top3_aug_kd")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train one RAPID-Gait student from the validation-selected top "
            "three teachers for each of the 15 non-empty modality masks."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default=METHODS[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53])
    parser.add_argument("--partition-seed", type=int, default=20260901)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs-student", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--aux-alpha", type=float, default=0.05)
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
    parser.add_argument("--augmentation-warmup-epochs", type=int, default=0)
    parser.add_argument("--quality-contrast", type=float, default=1.0)
    parser.add_argument(
        "--augmentation-profile",
        choices=("all", "drop_gain"),
        default="all",
    )
    parser.add_argument("--quality-routing-strength", type=float, default=0.5)
    parser.add_argument("--corruption-selection-weight", type=float, default=0.20)
    parser.add_argument(
        "--fasca-root",
        type=Path,
        default=root / "artifacts" / "sensor_aug_dev_v2_focus",
    )
    parser.add_argument(
        "--multimodel-root",
        type=Path,
        default=root / "artifacts" / "multimodel_confirmatory" / "main",
    )
    parser.add_argument(
        "--rapid-root",
        type=Path,
        default=root / "artifacts" / "rapid_distill_5fold_seed51",
    )
    parser.add_argument(
        "--xtiny-distill-root",
        type=Path,
        default=root
        / "artifacts"
        / "xtinyhar_confirmatory"
        / "distillation",
    )
    parser.add_argument(
        "--xtiny-baseline-root",
        type=Path,
        default=root
        / "artifacts"
        / "xtinyhar_confirmatory"
        / "baselines",
    )
    return parser.parse_args()


class TopKCombinationTeacher(nn.Module):
    uses_student_mask = True

    def __init__(
        self,
        names: list[str],
        models: list[nn.Module],
        selected_indices: torch.Tensor,
    ):
        super().__init__()
        self.names = names
        self.models = nn.ModuleList(models)
        self.register_buffer(
            "selected_indices", selected_indices.to(dtype=torch.long)
        )
        lookup = torch.full((16,), -1, dtype=torch.long)
        bit_weights = torch.tensor([8, 4, 2, 1], dtype=torch.long)
        for group, bits in enumerate(MASK_ROWS):
            code = int(
                sum(
                    int(bit) * int(weight)
                    for bit, weight in zip(bits, bit_weights.tolist())
                )
            )
            lookup[code] = group
        self.register_buffer("mask_lookup", lookup)
        self.register_buffer("bit_weights", bit_weights)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = torch.stack(
            [
                model(inputs, modality_mask=modality_mask)["logits"]
                for model in self.models
            ],
            dim=1,
        )
        codes = (
            modality_mask.to(dtype=torch.long)
            * self.bit_weights.unsqueeze(0)
        ).sum(dim=1)
        group_ids = self.mask_lookup[codes]
        if (group_ids < 0).any():
            raise ValueError("Top-K teacher received an empty modality mask")
        choices = self.selected_indices[group_ids]
        selected = torch.gather(
            logits,
            dim=1,
            index=choices.unsqueeze(-1).expand(
                -1, -1, logits.shape[-1]
            ),
        )
        return {"logits": selected.mean(dim=1)}


class HybridPrivilegedTopKTeacher(nn.Module):
    """Keep privileged full-input KD and add mask-matched Top-K KD."""

    uses_student_mask = True

    def __init__(
        self,
        privileged_teacher: nn.Module,
        combination_teacher: TopKCombinationTeacher,
    ):
        super().__init__()
        self.privileged_teacher = privileged_teacher
        self.combination_teacher = combination_teacher

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        full_mask = torch.ones_like(modality_mask)
        privileged_logits = self.privileged_teacher(
            inputs, modality_mask=full_mask
        )["logits"]
        combination_logits = self.combination_teacher(
            inputs, modality_mask=modality_mask
        )["logits"]
        return {
            "logits": privileged_logits,
            "auxiliary_logits": combination_logits,
        }


def _split_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_split(root: Path, fold: int, expected: dict[str, object]) -> None:
    actual = _split_payload(root / f"fold_{fold}" / "subjects.json")
    for key in ("train_subjects", "validation_subjects", "test_subjects"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Split mismatch at {root}: {key}")


def candidate_specs(
    args: argparse.Namespace,
    fold: int,
    seed: int,
    channels: dict[str, int],
) -> list[tuple[str, nn.Module, Path]]:
    multi = args.multimodel_root / f"fold_{fold}" / f"seed_{seed}"
    return [
        (
            "RAPID-Gait+FASCA",
            RapidGait(channels=channels, classes=len(SPEEDS)),
            args.fasca_root
            / f"fold_{fold}"
            / f"seed_{seed}"
            / "sensor_aug_kd"
            / "best_model.pt",
        ),
        (
            "EmbraceNet",
            EmbraceNetLite(channels=channels, classes=len(SPEEDS)),
            multi / "embracenet" / "best_model.pt",
        ),
        (
            "ActionMAE",
            ActionMAELite(channels=channels, classes=len(SPEEDS)),
            multi / "actionmae" / "best_model.pt",
        ),
        (
            "COMPASS",
            CompassLite(channels=channels, classes=len(SPEEDS)),
            multi / "compass" / "best_model.pt",
        ),
        (
            "ModDrop",
            MultimodalLiteNet(
                channels=channels,
                classes=len(SPEEDS),
                modality_dropout=0.3,
            ),
            multi / "dropout" / "best_model.pt",
        ),
        (
            "RAPID-Gait+KD",
            RapidGait(channels=channels, classes=len(SPEEDS)),
            args.rapid_root
            / f"fold_{fold}"
            / f"seed_{seed}"
            / "balanced_kd"
            / "best_model.pt",
        ),
        (
            "XTinyHAR+KD",
            XTinyHARAdaptation(channels=channels, classes=len(SPEEDS)),
            args.xtiny_distill_root
            / f"fold_{fold}"
            / f"seed_{seed}"
            / "original_kd"
            / "best_model.pt",
        ),
        (
            "XTinyHAR+Dropout",
            XTinyHARAdaptation(channels=channels, classes=len(SPEEDS)),
            args.xtiny_baseline_root
            / f"fold_{fold}"
            / f"seed_{seed}"
            / "xtinyhar_dropout"
            / "best_model.pt",
        ),
    ]


@torch.no_grad()
def select_top_three(
    specs: list[tuple[str, nn.Module, Path]],
    loader,
    validation_trial_ids: np.ndarray,
    device: torch.device,
) -> tuple[TopKCombinationTeacher, list[dict[str, object]]]:
    names, models = [], []
    score_rows = []
    scores = np.empty((len(specs), len(MASK_ROWS)), dtype=float)
    nll = np.empty((len(specs), len(MASK_ROWS)), dtype=float)
    for candidate_index, (name, model, checkpoint) in enumerate(specs):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(
            checkpoint, map_location=device, weights_only=True
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        names.append(name)
        models.append(model)
        for group, bits in enumerate(MASK_ROWS):
            mask = torch.tensor(bits, dtype=torch.bool).unsqueeze(0)
            truth, probabilities = predict(model, loader, device, mask)
            metrics, _ = trial_metrics(
                truth, probabilities, validation_trial_ids
            )
            scores[candidate_index, group] = metrics["macro_f1"]
            nll[candidate_index, group] = float(
                -np.log(
                    probabilities[
                        np.arange(len(truth)), truth.astype(int)
                    ].clip(1e-8, 1.0)
                ).mean()
            )
    rankings = [
        np.lexsort((nll[:, group], -scores[:, group]))
        for group in range(len(MASK_ROWS))
    ]
    top_indices = np.stack(
        [ranking[:3] for ranking in rankings], axis=0
    )
    for group, bits in enumerate(MASK_ROWS):
        ranking = rankings[group]
        score_rows.append(
            {
                "combination": "+".join(
                    modality
                    for modality, available in zip(MODALITIES, bits)
                    if available
                ),
                "selected_teachers": [
                    names[index] for index in top_indices[group]
                ],
                "selected_validation_macro_f1": [
                    float(scores[index, group])
                    for index in top_indices[group]
                ],
                "selected_validation_window_nll": [
                    float(nll[index, group])
                    for index in top_indices[group]
                ],
                "full_ranking": [
                    {
                        "teacher": names[index],
                        "validation_macro_f1": float(scores[index, group]),
                        "validation_window_nll": float(nll[index, group]),
                    }
                    for index in ranking
                ],
            }
        )
    teacher = TopKCombinationTeacher(
        names,
        models,
        torch.tensor(top_indices, device=device),
    ).to(device)
    teacher.eval()
    return teacher, score_rows


def load_privileged_teacher(
    args: argparse.Namespace,
    fold: int,
    seed: int,
    channels: dict[str, int],
    device: torch.device,
) -> nn.Module:
    checkpoint = (
        args.xtiny_distill_root
        / f"fold_{fold}"
        / f"seed_{seed}"
        / "teacher"
        / "best_model.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = XTinyHARTeacherAdaptation(
        channels=channels, classes=len(SPEEDS)
    ).to(device)
    payload = torch.load(
        checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args.data)
    folds = subject_folds(arrays["subject"], args.folds, args.partition_seed)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    channels = {
        modality: int(arrays[modality].shape[1])
        for modality in MODALITIES
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(folds) * len(args.seeds)
    completed = 0
    status_path = args.output_dir / "status.json"
    write_status(status_path, completed, total, f"Starting on {device}")
    all_metrics = []
    candidate_roots = (
        args.fasca_root,
        args.multimodel_root,
        args.rapid_root,
        args.xtiny_distill_root,
        args.xtiny_baseline_root,
    )
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
        for root in candidate_roots:
            verify_split(root, fold, split_payload)
        for seed in args.seeds:
            run_seed = int(seed * 100 + fold)
            run_dir = fold_dir / f"seed_{seed}" / args.method
            metrics_path = run_dir / "combination_metrics.csv"
            predictions_path = run_dir / "window_predictions.csv.gz"
            if metrics_path.exists() and predictions_path.exists():
                all_metrics.append(pd.read_csv(metrics_path))
                completed += 1
                write_status(
                    status_path,
                    completed,
                    total,
                    f"Reused fold {fold}, seed {seed}",
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
            combination_teacher, selection = select_top_three(
                candidate_specs(args, fold, seed, channels),
                loaders["validation"],
                arrays["trial_id"][validation_idx],
                device,
            )
            if args.method == "hybrid_top3_aug_kd":
                teacher = HybridPrivilegedTopKTeacher(
                    load_privileged_teacher(
                        args, fold, seed, channels, device
                    ),
                    combination_teacher,
                ).to(device)
            else:
                teacher = combination_teacher
            (run_dir / "teacher_selection.json").write_text(
                json.dumps(selection, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            seed_everything(run_seed)
            student = RapidGait(
                channels=channels,
                classes=len(SPEEDS),
                embedding_dim=args.embedding_dim,
            ).to(device)
            student, best_epoch, history, group_weights = train_student(
                args.method,
                student,
                teacher,
                loaders,
                arrays["trial_id"][validation_idx],
                args,
                device,
                run_seed,
            )
            metrics, predictions = evaluate_all_masks(
                f"rapid_{args.method}",
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
                parameter.numel() for parameter in student.parameters()
            )
            metrics["teacher_parameter_count"] = sum(
                parameter.numel() for parameter in teacher.parameters()
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
                    "method": args.method,
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
                f"Completed fold {fold}, seed {seed}",
            )
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": total,
                        "fold": fold,
                        "seed": seed,
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
            del teacher, student
            torch.cuda.empty_cache()
    pd.concat(all_metrics, ignore_index=True).to_csv(
        args.output_dir / "all_combination_metrics.csv", index=False
    )
    summarize_seed_predictions(args.output_dir)
    write_status(
        status_path,
        completed,
        total,
        "Top-3 combination-teacher distillation complete",
        stage="complete",
    )


if __name__ == "__main__":
    main()
