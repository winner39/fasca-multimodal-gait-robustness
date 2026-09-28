from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("eeg", "emg", "imu", "fp")


@dataclass
class ChannelRobustStats:
    median: np.ndarray
    scale: np.ndarray


class FoldRobustScaler:
    """Channel-wise robust scaling fitted on outer-fold training subjects only."""

    def __init__(
        self,
        stats: dict[str, ChannelRobustStats],
        clip: float = 10.0,
    ):
        self.stats = stats
        self.clip = float(clip)

    @classmethod
    def fit(
        cls,
        arrays: dict[str, np.ndarray],
        train_indices: np.ndarray,
        clip: float = 10.0,
    ) -> FoldRobustScaler:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        stats = {}
        for modality in MODALITIES:
            values = arrays[modality][train_indices]
            # [window, channel, time] -> [channel, observations]
            by_channel = values.transpose(1, 0, 2).reshape(
                values.shape[1], -1
            )
            median = np.median(by_channel, axis=1).astype(np.float32)
            q25, q75 = np.percentile(
                by_channel, [25.0, 75.0], axis=1
            ).astype(np.float32)
            scale = q75 - q25
            # A channel can be nearly constant in a particular outer fold.
            # Fall back to a robust non-zero unit rather than amplifying noise.
            scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
            stats[modality] = ChannelRobustStats(
                median=median[:, None],
                scale=scale[:, None],
            )
        return cls(stats=stats, clip=clip)

    def transform(self, modality: str, values: np.ndarray) -> np.ndarray:
        stat = self.stats[modality]
        output = (np.asarray(values, dtype=np.float32) - stat.median) / stat.scale
        return np.clip(output, -self.clip, self.clip)

    def as_serializable(self) -> dict[str, dict[str, list[float]] | float]:
        payload: dict[str, dict[str, list[float]] | float] = {
            "clip": self.clip
        }
        for modality, stat in self.stats.items():
            payload[modality] = {
                "median": stat.median[:, 0].astype(float).tolist(),
                "scale": stat.scale[:, 0].astype(float).tolist(),
            }
        return payload


class ScaledWindowDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray,
        scaler: FoldRobustScaler,
    ):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, item: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        index = int(self.indices[item])
        inputs = {
            modality: torch.from_numpy(
                self.scaler.transform(modality, self.arrays[modality][index])
            )
            for modality in MODALITIES
        }
        target = torch.tensor(self.arrays["label"][index], dtype=torch.long)
        return inputs, target
