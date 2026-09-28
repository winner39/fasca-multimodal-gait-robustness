from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class EEGNetLiteEncoder(nn.Module):
    """Compact EEGNet-style spatial-temporal encoder."""

    def __init__(self, channels: int, embedding_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                8,
                16,
                kernel_size=(channels, 1),
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=16,
                bias=False,
            ),
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x).flatten(1)
        return self.projection(x)


class DepthwiseTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalLiteEncoder(nn.Module):
    """MobileNet-like 1-D temporal encoder for wearable and force signals."""

    def __init__(
        self,
        channels: int,
        embedding_dim: int = 64,
        width: int = 48,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            DepthwiseTemporalBlock(width, width, kernel_size=7, dropout=dropout),
            DepthwiseTemporalBlock(
                width, width * 2, kernel_size=5, stride=2, dropout=dropout
            ),
            DepthwiseTemporalBlock(
                width * 2, width * 2, kernel_size=5, dropout=dropout
            ),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(width * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.net(x).squeeze(-1))


class GatedFusion(nn.Module):
    def __init__(self, embedding_dim: int, modalities: int):
        super().__init__()
        self.modalities = modalities
        self.score = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )

    def forward(
        self, embeddings: torch.Tensor, modality_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # embeddings: [batch, modality, embedding]
        scores = self.score(embeddings).squeeze(-1)
        scores = scores.masked_fill(~modality_mask.bool(), -1e4)
        weights = scores.softmax(dim=1)
        fused = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        return self.output(fused), weights


class MultimodalLiteNet(nn.Module):
    modality_order = ("eeg", "emg", "imu", "fp")

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict(
            {
                "eeg": EEGNetLiteEncoder(channels["eeg"], embedding_dim),
                "emg": TemporalLiteEncoder(channels["emg"], embedding_dim),
                "imu": TemporalLiteEncoder(channels["imu"], embedding_dim),
                "fp": TemporalLiteEncoder(channels["fp"], embedding_dim),
            }
        )
        self.modality_tokens = nn.Parameter(
            torch.zeros(len(self.modality_order), embedding_dim)
        )
        nn.init.normal_(self.modality_tokens, std=0.02)
        self.fusion = GatedFusion(embedding_dim, len(self.modality_order))
        self.classifier = nn.Linear(embedding_dim, classes)

    def _training_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        mask = torch.rand(batch, len(self.modality_order), device=device)
        mask = mask >= self.modality_dropout
        empty = ~mask.any(dim=1)
        if empty.any():
            replacement = torch.randint(
                0, len(self.modality_order), (int(empty.sum()),), device=device
            )
            mask[empty] = False
            mask[empty, replacement] = True
        return mask

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embeddings = []
        for index, modality in enumerate(self.modality_order):
            encoded = self.encoders[modality](inputs[modality])
            embeddings.append(encoded + self.modality_tokens[index])
        stacked = torch.stack(embeddings, dim=1)
        if modality_mask is None:
            if self.training and self.modality_dropout > 0:
                modality_mask = self._training_mask(stacked.shape[0], stacked.device)
            else:
                modality_mask = torch.ones(
                    stacked.shape[:2], device=stacked.device, dtype=torch.bool
                )
        fused, weights = self.fusion(stacked, modality_mask)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": stacked,
            "weights": weights,
            "modality_mask": modality_mask,
        }

