from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from gait_robust.run_gaitpdb_clinical import MaskedFootFusion


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Benchmark the shared GaitPDB inference graph."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "artifacts"
        / "gaitpdb_jmbe_extension"
        / "engineering",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    return parser.parse_args()


def parameter_bytes(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
    )


def count_macs(
    model: nn.Module, inputs: torch.Tensor, mask: torch.Tensor
) -> int:
    macs = 0
    hooks = []

    def conv_hook(
        module: nn.Conv1d,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal macs
        batch, out_channels, output_length = output.shape
        kernel_ops = (
            module.in_channels // module.groups
        ) * module.kernel_size[0]
        macs += int(batch * out_channels * output_length * kernel_ops)

    def linear_hook(
        module: nn.Linear,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal macs
        output_elements = output.numel()
        macs += int(output_elements * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(inputs, mask)
    for hook in hooks:
        hook.remove()
    return macs


def benchmark_latency(
    model: nn.Module,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    warmup: int,
    repetitions: int,
) -> np.ndarray:
    device = inputs.device
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            model(inputs, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms.append((time.perf_counter_ns() - start) / 1e6)
    return np.asarray(latency_ms, dtype=float)


def device_row(
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    model = MaskedFootFusion().to(device).eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(20260706)
    inputs = torch.randn(
        (1, 2, 8, 500),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    mask = torch.ones((1, 2), device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    latency = benchmark_latency(
        model, inputs, mask, warmup, repetitions
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else np.nan
    )
    return {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or platform.machine()
        ),
        "batch_size": 1,
        "precision": "float32",
        "warmup_iterations": warmup,
        "timed_iterations": repetitions,
        "latency_median_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.quantile(latency, 0.95)),
        "latency_mean_ms": float(latency.mean()),
        "latency_sd_ms": float(
            statistics.stdev(latency.tolist())
            if len(latency) > 1
            else 0.0
        ),
        "peak_allocated_memory_bytes": peak_memory,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = MaskedFootFusion().eval()
    cpu_inputs = torch.zeros((1, 2, 8, 500), dtype=torch.float32)
    cpu_mask = torch.ones((1, 2), dtype=torch.float32)
    trainable_parameters = int(
        sum(
            parameter.numel()
            for parameter in cpu_model.parameters()
            if parameter.requires_grad
        )
    )
    macs = count_macs(cpu_model, cpu_inputs, cpu_mask)
    buffer = io.BytesIO()
    torch.save(cpu_model.state_dict(), buffer)
    architecture = {
        "model": "MaskedFootFusion",
        "input_shape": [1, 2, 8, 500],
        "availability_mask_shape": [1, 2],
        "trainable_parameters": trainable_parameters,
        "parameter_memory_bytes_float32": parameter_bytes(cpu_model),
        "serialized_state_dict_bytes": buffer.getbuffer().nbytes,
        "macs_per_batch1_two_branch_input": macs,
        "approximate_flops_two_per_mac": 2 * macs,
        "inference_graph_note": (
            "Masked fusion, generic augmentation, IID channel corruption, "
            "and FASCA-pressure share this inference graph; all "
            "augmentations are training-only."
        ),
    }
    rows = [
        device_row(
            torch.device("cpu"), args.warmup, args.repetitions
        )
    ]
    if torch.cuda.is_available():
        rows.append(
            device_row(
                torch.device("cuda"), args.warmup, args.repetitions
            )
        )
    pd.DataFrame(rows).to_csv(
        args.output_dir / "latency_memory.csv", index=False
    )
    manifest = {
        "protocol": "docs/gaitpdb_jmbe_extension_protocol.md",
        "architecture": architecture,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    (args.output_dir / "engineering_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
