from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from gait_robust.robust_models import MODALITIES, ModalityEncoderBank


def _availability_mask(
    reference: torch.Tensor,
    modality_mask: torch.Tensor | None,
) -> torch.Tensor:
    if modality_mask is None:
        return torch.ones(
            reference.shape[:2], dtype=torch.bool, device=reference.device
        )
    return modality_mask.to(device=reference.device, dtype=torch.bool)


class MaskedTokenTransformer(nn.Module):
    """Small masked Transformer shared by the task adaptations."""

    def __init__(
        self,
        embedding_dim: int,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = nn.Parameter(
            torch.zeros(1, len(MODALITIES) + 1, embedding_dim)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        modality_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat((cls, tokens), dim=1) + self.position
        padding_mask = None
        if modality_mask is not None:
            cls_available = torch.ones(
                batch, 1, dtype=torch.bool, device=tokens.device
            )
            padding_mask = ~torch.cat(
                (cls_available, modality_mask.bool()), dim=1
            )
        encoded = self.encoder(
            sequence, src_key_padding_mask=padding_mask
        )
        return self.output_norm(encoded[:, 0])


class ConvolutionalDenoiser(nn.Module):
    """Lightweight residual 1-D denoiser for one sensor modality."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=9,
                padding=4,
                groups=channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal + self.net(signal)


class CentaurAdaptation(nn.Module):
    """Centaur-style denoising plus convolution/attention task adaptation.

    The original Centaur model operates on homogeneous multi-sensor HAR data.
    Here each heterogeneous modality is denoised separately and encoded with
    the study's common modality encoders before masked self-attention fusion.
    """

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.denoisers = nn.ModuleDict(
            {
                modality: ConvolutionalDenoiser(int(channels[modality]))
                for modality in MODALITIES
            }
        )
        self.encoder_bank = ModalityEncoderBank(channels, embedding_dim)
        self.fusion = MaskedTokenTransformer(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
        reconstruction_targets: Mapping[str, torch.Tensor] | None = None,
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
        reconstructions: dict[str, torch.Tensor] = {}
        cleaned: dict[str, torch.Tensor] = {}
        reconstruction_terms = []
        for index, modality in enumerate(MODALITIES):
            present = modality_mask[:, index, None, None]
            masked_signal = torch.where(
                present, inputs[modality], torch.zeros_like(inputs[modality])
            )
            reconstructed = self.denoisers[modality](masked_signal)
            reconstructions[modality] = reconstructed
            cleaned[modality] = torch.where(
                present, reconstructed, torch.zeros_like(reconstructed)
            )
            if reconstruction_targets is not None:
                per_sample = (
                    reconstructed - reconstruction_targets[modality]
                ).square().mean(dim=(1, 2))
                available = modality_mask[:, index]
                if available.any():
                    reconstruction_terms.append(per_sample[available].mean())
        tokens = self.encoder_bank(cleaned)
        fused = self.fusion(tokens, modality_mask)
        reconstruction_loss = fused.new_zeros(())
        if reconstruction_terms:
            reconstruction_loss = torch.stack(reconstruction_terms).mean()
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": tokens,
            "modality_mask": modality_mask,
            "reconstructions": reconstructions,
            "reconstruction_loss": reconstruction_loss,
        }


class ADAPTAdaptation(nn.Module):
    """Anchor-aligned masked multimodal Transformer adaptation of ADAPT."""

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
        anchor_modality: str = "imu",
    ) -> None:
        super().__init__()
        if anchor_modality not in MODALITIES:
            raise ValueError(f"Unknown anchor modality: {anchor_modality}")
        self.anchor_index = MODALITIES.index(anchor_modality)
        self.anchor_modality = anchor_modality
        self.encoder_bank = ModalityEncoderBank(channels, embedding_dim)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                for _ in MODALITIES
            ]
        )
        self.missing_tokens = nn.Parameter(
            torch.zeros(1, len(MODALITIES), embedding_dim)
        )
        self.fusion = MaskedTokenTransformer(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)
        nn.init.normal_(self.missing_tokens, std=0.02)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        modality_mask = _availability_mask(real_tokens, modality_mask)
        projected = torch.stack(
            [
                projection(real_tokens[:, index])
                for index, projection in enumerate(self.projections)
            ],
            dim=1,
        )
        fusion_tokens = torch.where(
            modality_mask.unsqueeze(-1),
            projected,
            self.missing_tokens.expand(projected.shape[0], -1, -1),
        )
        fused = self.fusion(fusion_tokens, modality_mask)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "projected_tokens": projected,
            "modality_mask": modality_mask,
            "anchor_index": torch.tensor(
                self.anchor_index, device=fused.device, dtype=torch.long
            ),
        }


class CIMSleepNetAdaptation(nn.Module):
    """Latent modality imagination and semantic calibration adaptation.

    The sleep-specific inter-epoch sequence block is intentionally omitted.
    Missing modality tokens are generated only from available tokens and a
    learned target-modality query, preventing hidden-input leakage at inference.
    """

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder_bank = ModalityEncoderBank(channels, embedding_dim)
        self.target_queries = nn.Parameter(
            torch.zeros(1, len(MODALITIES), embedding_dim)
        )
        self.imaginers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embedding_dim * 2),
                    nn.Linear(embedding_dim * 2, embedding_dim * 2),
                    nn.GELU(),
                    nn.Linear(embedding_dim * 2, embedding_dim),
                )
                for _ in MODALITIES
            ]
        )
        self.fusion = MaskedTokenTransformer(embedding_dim)
        self.full_view_fusion = MaskedTokenTransformer(embedding_dim)
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, classes)
        nn.init.normal_(self.target_queries, std=0.02)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        modality_mask = _availability_mask(real_tokens, modality_mask)
        weights = modality_mask.to(real_tokens.dtype).unsqueeze(-1)
        context = (real_tokens * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        imagined = torch.stack(
            [
                imaginer(
                    torch.cat(
                        (
                            context,
                            self.target_queries[:, index].expand(
                                real_tokens.shape[0], -1
                            ),
                        ),
                        dim=-1,
                    )
                )
                for index, imaginer in enumerate(self.imaginers)
            ],
            dim=1,
        )
        completed = torch.where(
            modality_mask.unsqueeze(-1), real_tokens, imagined
        )
        all_available = torch.ones_like(modality_mask)
        fused = self.fusion(completed, all_available)
        full_fused = self.full_view_fusion(real_tokens, all_available)
        missing = ~modality_mask
        if missing.any():
            imagination_loss = (
                imagined[missing] - real_tokens.detach()[missing]
            ).square().mean()
        else:
            imagination_loss = fused.new_zeros(())
        contrastive_views = torch.stack(
            (
                self.contrastive_projector(fused),
                self.contrastive_projector(full_fused),
            ),
            dim=1,
        )
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "imagined_tokens": imagined,
            "completed_tokens": completed,
            "imagination_loss": imagination_loss,
            "contrastive_views": contrastive_views,
            "modality_mask": modality_mask,
        }

