from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


MODALITIES = ("eeg", "emg", "imu", "fp")


class XTinyHARAdaptation(nn.Module):
    """XTinyHAR Inertial Transformer adapted to aligned gait modalities.

    The official XTinyHAR student flattens non-overlapping temporal patches
    across all inertial channels before a two-layer Transformer. Here the same
    tokenization is applied to the synchronized EEG, EMG, IMU, and force-plate
    channels. Unavailable modalities are zeroed after fold-only scaling.

    This is an architecture adaptation, not a reproduction of the original
    skeleton-to-IMU experiment.
    """

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        window_size: int = 200,
        patch_size: int = 20,
        embedding_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if window_size % patch_size:
            raise ValueError("window_size must be divisible by patch_size")
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.channels = {key: int(channels[key]) for key in MODALITIES}
        self.window_size = int(window_size)
        self.patch_size = int(patch_size)
        self.total_channels = sum(self.channels.values())
        self.patch_projection = nn.Linear(
            self.patch_size * self.total_channels, embedding_dim
        )
        token_count = self.window_size // self.patch_size
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(
            torch.zeros(1, token_count + 1, embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def _masked_signal(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        signals = []
        for index, modality in enumerate(MODALITIES):
            available = modality_mask[:, index].to(
                dtype=inputs[modality].dtype
            )
            signals.append(inputs[modality] * available[:, None, None])
        return torch.cat(signals, dim=1)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        first = inputs[MODALITIES[0]]
        if modality_mask is None:
            modality_mask = torch.ones(
                first.shape[0],
                len(MODALITIES),
                dtype=torch.bool,
                device=first.device,
            )
        modality_mask = modality_mask.bool()
        signal = self._masked_signal(inputs, modality_mask)
        batch, _, time = signal.shape
        if time != self.window_size:
            raise ValueError(
                f"Expected {self.window_size} samples, received {time}"
            )
        patches = (
            signal.transpose(1, 2)
            .reshape(
                batch,
                self.window_size // self.patch_size,
                self.patch_size * self.total_channels,
            )
        )
        tokens = self.patch_projection(patches)
        class_token = self.class_token.expand(batch, -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        tokens = self.transformer(tokens)
        fused = self.output_norm(tokens[:, 0])
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "tokens": tokens,
            "modality_mask": modality_mask,
        }


class XTinyHARTeacherAdaptation(nn.Module):
    """Larger multimodal temporal Transformer used only during training.

    Each modality is patch-projected separately so the teacher retains the
    modality structure that the flattened XTinyHAR student deliberately
    removes for efficiency. The design follows the role of XTinyHAR's
    multimodal ST-ConvT teacher, while matching this dataset's signal types.
    """

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        window_size: int = 200,
        patch_size: int = 20,
        embedding_dim: int = 192,
        depth: int = 3,
        heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if window_size % patch_size:
            raise ValueError("window_size must be divisible by patch_size")
        self.channels = {key: int(channels[key]) for key in MODALITIES}
        self.window_size = int(window_size)
        self.patch_size = int(patch_size)
        self.patch_projections = nn.ModuleDict(
            {
                modality: nn.Linear(
                    self.patch_size * self.channels[modality],
                    embedding_dim,
                )
                for modality in MODALITIES
            }
        )
        patches_per_modality = self.window_size // self.patch_size
        token_count = len(MODALITIES) * patches_per_modality
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(
            torch.zeros(1, token_count + 1, embedding_dim)
        )
        self.modality_tokens = nn.Parameter(
            torch.zeros(1, len(MODALITIES), 1, embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.modality_tokens, std=0.02)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        first = inputs[MODALITIES[0]]
        batch = first.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones(
                batch,
                len(MODALITIES),
                dtype=torch.bool,
                device=first.device,
            )
        modality_mask = modality_mask.bool()
        modality_sequences = []
        for index, modality in enumerate(MODALITIES):
            signal = inputs[modality]
            patches = (
                signal.transpose(1, 2)
                .reshape(
                    batch,
                    self.window_size // self.patch_size,
                    self.patch_size * self.channels[modality],
                )
            )
            tokens = self.patch_projections[modality](patches)
            tokens = tokens + self.modality_tokens[:, index]
            tokens = tokens * modality_mask[:, index, None, None].to(
                tokens.dtype
            )
            modality_sequences.append(tokens)
        tokens = torch.cat(modality_sequences, dim=1)
        tokens = torch.cat(
            (self.class_token.expand(batch, -1, -1), tokens), dim=1
        )
        tokens = tokens + self.position[:, : tokens.shape[1]]
        tokens = self.transformer(tokens)
        fused = self.output_norm(tokens[:, 0])
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "tokens": tokens,
            "modality_mask": modality_mask,
        }
