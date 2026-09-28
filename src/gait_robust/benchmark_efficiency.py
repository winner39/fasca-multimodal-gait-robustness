from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from gait_robust.models import MultimodalLiteNet
from gait_robust.robust_models import (
    ActionMAELite,
    CompassLite,
    EmbraceNetLite,
    RapidGait,
)
from gait_robust.xtinyhar import XTinyHARAdaptation


MODALITIES = ("eeg", "emg", "imu", "fp")
CHANNELS = {"eeg": 19, "emg": 12, "imu": 51, "fp": 16}
WINDOW_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-warmup", type=int, default=50)
    parser.add_argument("--cpu-iterations", type=int, default=300)
    parser.add_argument("--gpu-warmup", type=int, default=100)
    parser.add_argument("--gpu-iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def model_specs(root: Path) -> list[dict[str, object]]:
    main = root / "artifacts" / "multimodel_confirmatory" / "main"
    return [
        {
            "method": "RAPID-Gait/FASCA",
            "model": RapidGait(channels=CHANNELS),
            "checkpoint": (
                root
                / "artifacts"
                / "sensor_aug_dev_v2_focus"
                / "fold_1"
                / "seed_51"
                / "sensor_aug_kd"
                / "best_model.pt"
            ),
        },
        {
            "method": "EmbraceNet",
            "model": EmbraceNetLite(channels=CHANNELS),
            "checkpoint": (
                main
                / "fold_1"
                / "seed_51"
                / "embracenet"
                / "best_model.pt"
            ),
        },
        {
            "method": "ActionMAE",
            "model": ActionMAELite(channels=CHANNELS),
            "checkpoint": (
                main
                / "fold_1"
                / "seed_51"
                / "actionmae"
                / "best_model.pt"
            ),
        },
        {
            "method": "ModDrop",
            "model": MultimodalLiteNet(
                channels=CHANNELS, modality_dropout=0.3
            ),
            "checkpoint": (
                main
                / "fold_1"
                / "seed_51"
                / "dropout"
                / "best_model.pt"
            ),
        },
        {
            "method": "COMPASS",
            "model": CompassLite(channels=CHANNELS),
            "checkpoint": (
                main
                / "fold_1"
                / "seed_51"
                / "compass"
                / "best_model.pt"
            ),
        },
        {
            "method": "XTinyHAR",
            "model": XTinyHARAdaptation(channels=CHANNELS),
            "checkpoint": (
                root
                / "artifacts"
                / "xtinyhar_confirmatory"
                / "baselines"
                / "fold_1"
                / "seed_51"
                / "xtinyhar_dropout"
                / "best_model.pt"
            ),
        },
    ]


def make_inputs(device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260901)
    inputs = {
        modality: torch.randn(
            1,
            CHANNELS[modality],
            WINDOW_SIZE,
            generator=generator,
            device=device,
        )
        for modality in MODALITIES
    }
    mask = torch.ones(
        1, len(MODALITIES), dtype=torch.bool, device=device
    )
    return inputs, mask


def load_model(model: nn.Module, checkpoint: Path) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def count_profiled_macs(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[float, list[dict[str, float | str]]]:
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        model(inputs, modality_mask=mask)
    events = [
        {"operator": event.key, "flops": float(event.flops)}
        for event in prof.key_averages()
        if event.flops
    ]
    flops = sum(float(event["flops"]) for event in events)
    return flops / 2.0, sorted(
        events, key=lambda row: float(row["flops"]), reverse=True
    )


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def cpu_latency(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs, modality_mask=mask)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            model(inputs, modality_mask=mask)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples), percentile(samples, 95)


def gpu_metrics(
    model: nn.Module,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float, float]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = model.to(device)
    inputs, mask = make_inputs(device)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(inputs, modality_mask=mask)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    total_peak_mb = peak / (1024**2)
    incremental_peak_mb = max(0, peak - baseline) / (1024**2)

    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs, modality_mask=mask)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            model(inputs, modality_mask=mask)
            end.record()
        torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    model.to("cpu")
    del inputs, mask
    torch.cuda.empty_cache()
    return (
        statistics.median(samples),
        percentile(samples, 95),
        total_peak_mb,
        incremental_peak_mb,
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    rows: list[dict[str, float | int | str]] = []
    operator_coverage: dict[str, list[dict[str, float | str]]] = {}
    for spec in model_specs(root):
        method = str(spec["method"])
        checkpoint = Path(spec["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_model(spec["model"], checkpoint)
        inputs, mask = make_inputs(torch.device("cpu"))
        macs, events = count_profiled_macs(model, inputs, mask)
        operator_coverage[method] = events
        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        state_mb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.state_dict().values()
        ) / (1024**2)
        for repeat in range(1, args.repeats + 1):
            cpu_median, cpu_p95 = cpu_latency(
                model,
                inputs,
                mask,
                args.cpu_warmup,
                args.cpu_iterations,
            )
            if torch.cuda.is_available():
                (
                    gpu_median,
                    gpu_p95,
                    gpu_peak,
                    gpu_incremental,
                ) = gpu_metrics(
                    model, args.gpu_warmup, args.gpu_iterations
                )
            else:
                gpu_median = gpu_p95 = gpu_peak = gpu_incremental = float(
                    "nan"
                )
            rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "parameters": parameters,
                    "state_size_mb": state_mb,
                    "profiled_macs": macs,
                    "cpu_latency_median_ms": cpu_median,
                    "cpu_latency_p95_ms": cpu_p95,
                    "gpu_latency_median_ms": gpu_median,
                    "gpu_latency_p95_ms": gpu_p95,
                    "gpu_peak_allocated_mb": gpu_peak,
                    "gpu_forward_incremental_peak_mb": gpu_incremental,
                }
            )
        del model, inputs, mask

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "efficiency_repeats.csv", index=False)
    numeric = [
        column
        for column in raw.columns
        if column not in {"method", "repeat"}
    ]
    summary = raw.groupby("method", sort=False)[numeric].median().reset_index()
    summary.to_csv(args.output_dir / "efficiency_summary.csv", index=False)
    metadata = {
        "timestamp": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "batch_size": 1,
        "precision": "float32",
        "window_samples": WINDOW_SIZE,
        "channels": CHANNELS,
        "cpu_warmup": args.cpu_warmup,
        "cpu_iterations": args.cpu_iterations,
        "gpu_warmup": args.gpu_warmup,
        "gpu_iterations": args.gpu_iterations,
        "repeats": args.repeats,
        "mac_definition": (
            "Sum of FLOPs reported by torch.profiler for covered CPU "
            "operators divided by two. Functional or unsupported operators "
            "may be omitted; use for within-script model comparison."
        ),
        "latency_scope": (
            "Eager PyTorch forward pass only; inputs already resident on "
            "the measured device; excludes data loading and host-device copy."
        ),
        "peak_memory_scope": (
            "Per-process torch.cuda.max_memory_allocated during the first "
            "forward after model and batch-1 inputs were loaded."
        ),
        "profiled_operators": operator_coverage,
    }
    (args.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
