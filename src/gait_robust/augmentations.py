from __future__ import annotations

from collections.abc import Mapping

import torch


MODALITIES = ("eeg", "emg", "imu", "fp")


def _rand(
    generator: torch.Generator,
    device: torch.device,
) -> float:
    return float(
        torch.rand((), generator=generator, device=device).item()
    )


def _randint(
    high: int,
    generator: torch.Generator,
    device: torch.device,
) -> int:
    return int(
        torch.randint(
            high, (), generator=generator, device=device
        ).item()
    )


def _channel_scale(signal: torch.Tensor) -> torch.Tensor:
    return signal.std(dim=-1, keepdim=True).clamp_min(0.05)


def _drop_channels(
    signal: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    channels = signal.shape[0]
    count = max(1, min(channels - 1, round(fraction * channels)))
    selected = torch.randperm(
        channels, generator=generator, device=signal.device
    )[:count]
    signal[selected] = 0.0
    return signal


def _drop_imu_sensors(
    signal: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    sensors = signal.shape[0] // 3
    count = max(1, min(sensors - 1, round(fraction * sensors)))
    selected = torch.randperm(
        sensors, generator=generator, device=signal.device
    )[:count]
    channel_ids = (
        selected[:, None] * 3
        + torch.arange(3, device=signal.device)[None, :]
    ).reshape(-1)
    signal[channel_ids] = 0.0
    return signal


def _drop_packet(
    signal: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    time = signal.shape[-1]
    length = max(1, min(time - 1, round(fraction * time)))
    start = _randint(time - length + 1, generator, signal.device)
    signal[:, start : start + length] = 0.0
    return signal


def _time_shift(
    signal: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    if samples == 0:
        return signal
    shifted = torch.zeros_like(signal)
    if samples > 0:
        shifted[:, samples:] = signal[:, :-samples]
    else:
        shifted[:, :samples] = signal[:, -samples:]
    return shifted


def _linear_drift(
    signal: torch.Tensor,
    amplitude: float,
    sign: float,
) -> torch.Tensor:
    ramp = torch.linspace(
        -1.0,
        1.0,
        signal.shape[-1],
        device=signal.device,
        dtype=signal.dtype,
    )
    return signal + sign * amplitude * _channel_scale(signal) * ramp


def _augment_iid_drop_gain(
    signal: torch.Tensor,
    severity: float,
    kind: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, str]:
    """Matched-budget generic corruption without sensor-structure priors."""
    if kind == 0:
        fraction = 0.05 + 0.35 * severity
        return (
            _drop_channels(signal, fraction, generator),
            1.0 - fraction,
            "iid_channel_dropout",
        )
    log_gain = (2.0 * _rand(generator, signal.device) - 1.0)
    log_gain *= 0.55 * severity
    gain = float(torch.exp(torch.tensor(log_gain)).item())
    return signal * gain, 1.0 - 0.45 * severity, "global_gain"


def _augment_eeg(
    signal: torch.Tensor,
    severity: float,
    kind: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, str]:
    if kind == 0:
        scale = (0.05 + 0.30 * severity) * _channel_scale(signal)
        noise = torch.randn(
            signal.shape,
            generator=generator,
            device=signal.device,
            dtype=signal.dtype,
        )
        return signal + noise * scale, 1.0 - 0.55 * severity, "noise"
    if kind == 1:
        fraction = 0.05 + 0.35 * severity
        return (
            _drop_channels(signal, fraction, generator),
            1.0 - fraction,
            "electrode_dropout",
        )
    if kind == 2:
        fraction = 0.05 + 0.25 * severity
        return (
            _drop_packet(signal, fraction, generator),
            1.0 - fraction,
            "flat_segment",
        )
    amplitude = 0.10 + 0.45 * severity
    sign = -1.0 if _rand(generator, signal.device) < 0.5 else 1.0
    return (
        _linear_drift(signal, amplitude, sign),
        1.0 - 0.45 * severity,
        "common_drift",
    )


def _augment_emg(
    signal: torch.Tensor,
    severity: float,
    kind: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, str]:
    if kind == 0:
        log_gain = (2.0 * _rand(generator, signal.device) - 1.0)
        log_gain *= 0.55 * severity
        gain = float(torch.exp(torch.tensor(log_gain)).item())
        return signal * gain, 1.0 - 0.45 * severity, "gain"
    if kind == 1:
        fraction = 0.08 + 0.35 * severity
        return (
            _drop_channels(signal, fraction, generator),
            1.0 - fraction,
            "muscle_dropout",
        )
    if kind == 2:
        fraction = 0.05 + 0.30 * severity
        return (
            _drop_packet(signal, fraction, generator),
            1.0 - fraction,
            "burst_dropout",
        )
    amplitude = 0.08 + 0.35 * severity
    sign = -1.0 if _rand(generator, signal.device) < 0.5 else 1.0
    return (
        _linear_drift(signal, amplitude, sign),
        1.0 - 0.40 * severity,
        "baseline_drift",
    )


def _augment_imu(
    signal: torch.Tensor,
    severity: float,
    kind: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, str]:
    if kind == 0:
        scale = (0.03 + 0.20 * severity) * _channel_scale(signal)
        noise = torch.randn(
            signal.shape,
            generator=generator,
            device=signal.device,
            dtype=signal.dtype,
        )
        return signal + noise * scale, 1.0 - 0.45 * severity, "noise"
    if kind == 1:
        fraction = 0.05 + 0.30 * severity
        return (
            _drop_imu_sensors(signal, fraction, generator),
            1.0 - fraction,
            "sensor_dropout",
        )
    if kind == 2:
        amplitude = 0.08 + 0.40 * severity
        sign = -1.0 if _rand(generator, signal.device) < 0.5 else 1.0
        return (
            _linear_drift(signal, amplitude, sign),
            1.0 - 0.45 * severity,
            "bias_drift",
        )
    fraction = 0.05 + 0.25 * severity
    return (
        _drop_packet(signal, fraction, generator),
        1.0 - fraction,
        "packet_loss",
    )


def _augment_fp(
    signal: torch.Tensor,
    severity: float,
    kind: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, str]:
    if kind == 0:
        log_gain = (2.0 * _rand(generator, signal.device) - 1.0)
        log_gain *= 0.50 * severity
        gain = float(torch.exp(torch.tensor(log_gain)).item())
        return signal * gain, 1.0 - 0.40 * severity, "gain"
    if kind == 1:
        plate = _randint(2, generator, signal.device)
        signal[plate * 8 : (plate + 1) * 8] = 0.0
        quality = max(0.35, 0.65 - 0.20 * severity)
        return signal, quality, "plate_dropout"
    if kind == 2:
        center = signal.mean(dim=-1, keepdim=True)
        scale = _channel_scale(signal)
        limit = 2.5 - 1.2 * severity
        signal = torch.maximum(
            torch.minimum(signal, center + limit * scale),
            center - limit * scale,
        )
        return signal, 1.0 - 0.40 * severity, "clipping"
    fraction = 0.05 + 0.25 * severity
    return (
        _drop_packet(signal, fraction, generator),
        1.0 - fraction,
        "packet_loss",
    )


def augment_modalities(
    inputs: Mapping[str, torch.Tensor],
    modality_mask: torch.Tensor,
    generator: torch.Generator,
    probability: float = 0.55,
    severity_min: float = 0.15,
    severity_max: float = 0.75,
    desync_probability: float = 0.15,
    max_shift_samples: int = 25,
    quality_contrast: float = 1.0,
    profile: str = "all",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
    """Apply sensor-aware corruption and return continuous quality targets.

    Inputs are assumed to be fold-standardized. Missing modalities are not
    corrupted and are excluded from quality supervision by the caller.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 <= severity_min <= severity_max <= 1.0:
        raise ValueError("severity bounds must satisfy 0 <= min <= max <= 1")
    first = inputs[MODALITIES[0]]
    batch = first.shape[0]
    corrupted = {
        modality: inputs[modality].clone() for modality in MODALITIES
    }
    quality = modality_mask.to(dtype=first.dtype).clone()
    counts: dict[str, int] = {}
    augmenters = {
        "eeg": _augment_eeg,
        "emg": _augment_emg,
        "imu": _augment_imu,
        "fp": _augment_fp,
    }
    if profile == "all":
        allowed_kinds = {
            modality: (0, 1, 2, 3) for modality in MODALITIES
        }
    elif profile == "drop_gain":
        allowed_kinds = {
            "eeg": (1, 3),
            "emg": (0, 1),
            "imu": (1, 2),
            "fp": (0, 1, 2),
        }
    elif profile == "iid_drop_gain":
        allowed_kinds = {
            modality: (0, 1) for modality in MODALITIES
        }
        augmenters = {
            modality: _augment_iid_drop_gain for modality in MODALITIES
        }
    elif profile == "structured_dropout":
        allowed_kinds = {
            "eeg": (1,),
            "emg": (1,),
            "imu": (1,),
            "fp": (1,),
        }
    else:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    for row in range(batch):
        for modality_index, modality in enumerate(MODALITIES):
            if not bool(modality_mask[row, modality_index]):
                quality[row, modality_index] = 0.0
                continue
            if _rand(generator, first.device) >= probability:
                continue
            severity = severity_min + (
                severity_max - severity_min
            ) * _rand(generator, first.device)
            choices = allowed_kinds[modality]
            kind = choices[
                _randint(len(choices), generator, first.device)
            ]
            signal, target_quality, name = augmenters[modality](
                corrupted[modality][row],
                severity,
                kind,
                generator,
            )
            corrupted[modality][row] = signal
            quality[row, modality_index] = max(
                0.05,
                1.0
                - quality_contrast * (1.0 - target_quality),
            )
            key = f"{modality}:{name}"
            counts[key] = counts.get(key, 0) + 1

        available = torch.nonzero(
            modality_mask[row], as_tuple=False
        ).flatten()
        if (
            len(available) > 1
            and _rand(generator, first.device) < desync_probability
        ):
            selected_position = _randint(
                len(available), generator, first.device
            )
            modality_index = int(available[selected_position])
            modality = MODALITIES[modality_index]
            magnitude = max(
                1,
                round(
                    max_shift_samples
                    * (
                        severity_min
                        + (severity_max - severity_min)
                        * _rand(generator, first.device)
                    )
                ),
            )
            direction = (
                -1 if _rand(generator, first.device) < 0.5 else 1
            )
            corrupted[modality][row] = _time_shift(
                corrupted[modality][row], direction * magnitude
            )
            shift_fraction = magnitude / first.shape[-1]
            quality[row, modality_index] *= max(
                0.5, 1.0 - 1.5 * shift_fraction
            )
            key = f"{modality}:desynchronization"
            counts[key] = counts.get(key, 0) + 1
    return corrupted, quality.clamp(0.0, 1.0), counts
