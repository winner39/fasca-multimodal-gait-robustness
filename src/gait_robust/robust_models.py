from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from gait_robust.models import (
    EEGNetLiteEncoder,
    GatedFusion,
    TemporalLiteEncoder,
)


MODALITIES = ("eeg", "emg", "imu", "fp")


class ModalityEncoderBank(nn.Module):
    def __init__(
        self,
        channels: Mapping[str, int],
        embedding_dim: int,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                "eeg": EEGNetLiteEncoder(channels["eeg"], embedding_dim),
                "emg": TemporalLiteEncoder(channels["emg"], embedding_dim),
                "imu": TemporalLiteEncoder(channels["imu"], embedding_dim),
                "fp": TemporalLiteEncoder(channels["fp"], embedding_dim),
            }
        )
        self.modality_tokens = nn.Parameter(
            torch.zeros(len(MODALITIES), embedding_dim)
        )
        nn.init.normal_(self.modality_tokens, std=0.02)

    def forward(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        tokens = [
            self.encoders[modality](inputs[modality])
            + self.modality_tokens[index]
            for index, modality in enumerate(MODALITIES)
        ]
        return torch.stack(tokens, dim=1)


class EmbraceNetLite(nn.Module):
    """EmbraceNet adaptation for heterogeneous gait time-series tokens.

    Training performs coordinate-wise stochastic embracement over the
    available modality embeddings. Evaluation uses its deterministic
    expectation so repeated missing-modality tests are reproducible.
    """

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.encoder_bank = ModalityEncoderBank(channels, embedding_dim)
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
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        if modality_mask is None:
            modality_mask = torch.ones(
                real_tokens.shape[:2],
                dtype=torch.bool,
                device=real_tokens.device,
            )
        modality_mask = modality_mask.bool()
        docked = torch.stack(
            [
                layer(real_tokens[:, index])
                for index, layer in enumerate(self.docking)
            ],
            dim=1,
        )
        logits = self.selection_logits.unsqueeze(0).expand(
            real_tokens.shape[0], -1
        )
        logits = logits.masked_fill(~modality_mask, -1e4)
        probabilities = logits.softmax(dim=1)
        if self.training:
            # One modality is selected for every embedding coordinate.
            selected = torch.multinomial(
                probabilities,
                num_samples=docked.shape[-1],
                replacement=True,
            )
            by_coordinate = docked.permute(0, 2, 1)
            fused = torch.gather(
                by_coordinate, 2, selected.unsqueeze(-1)
            ).squeeze(-1)
        else:
            fused = (
                docked * probabilities.unsqueeze(-1)
            ).sum(dim=1)
        fused = self.output_norm(fused)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "docked_tokens": docked,
            "selection_probabilities": probabilities,
            "modality_mask": modality_mask,
        }


class QualityEmbraceNetLite(EmbraceNetLite):
    """EmbraceNet with sample-wise quality-adaptive embracement."""

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
        quality_floor: float = 0.05,
        quality_routing_strength: float = 0.5,
    ):
        super().__init__(
            channels=channels,
            classes=classes,
            embedding_dim=embedding_dim,
        )
        self.quality_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.quality_floor = float(quality_floor)
        self.quality_routing_strength = float(
            quality_routing_strength
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.constant_(self.quality_head[-1].bias, 2.0)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        if modality_mask is None:
            modality_mask = torch.ones(
                real_tokens.shape[:2],
                dtype=torch.bool,
                device=real_tokens.device,
            )
        modality_mask = modality_mask.bool()
        docked = torch.stack(
            [
                layer(real_tokens[:, index])
                for index, layer in enumerate(self.docking)
            ],
            dim=1,
        )
        quality_logits = self.quality_head(docked).squeeze(-1)
        predicted_quality = quality_logits.sigmoid()
        effective_quality = predicted_quality.clamp_min(
            self.quality_floor
        )
        routing_logits = self.selection_logits.unsqueeze(0).expand(
            real_tokens.shape[0], -1
        )
        routing_logits = (
            routing_logits
            + self.quality_routing_strength * effective_quality.log()
        )
        routing_logits = routing_logits.masked_fill(
            ~modality_mask, -1e4
        )
        probabilities = routing_logits.softmax(dim=1)
        if self.training:
            selected = torch.multinomial(
                probabilities,
                num_samples=docked.shape[-1],
                replacement=True,
            )
            by_coordinate = docked.permute(0, 2, 1)
            fused = torch.gather(
                by_coordinate, 2, selected.unsqueeze(-1)
            ).squeeze(-1)
        else:
            fused = (
                docked * probabilities.unsqueeze(-1)
            ).sum(dim=1)
        fused = self.output_norm(fused)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "docked_tokens": docked,
            "selection_probabilities": probabilities,
            "modality_mask": modality_mask,
            "quality_logits": quality_logits,
            "predicted_quality": predicted_quality,
        }


class ActionMAELite(nn.Module):
    """Feature-level ActionMAE adaptation for heterogeneous gait time series."""

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ):
        super().__init__()
        self.encoder_bank = ModalityEncoderBank(channels, embedding_dim)
        self.mask_tokens = nn.Parameter(
            torch.zeros(len(MODALITIES), embedding_dim)
        )
        nn.init.normal_(self.mask_tokens, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        if modality_mask is None:
            modality_mask = torch.ones(
                real_tokens.shape[:2],
                dtype=torch.bool,
                device=real_tokens.device,
            )
        modality_mask = modality_mask.bool()
        masked_tokens = torch.where(
            modality_mask.unsqueeze(-1),
            real_tokens,
            self.mask_tokens.unsqueeze(0),
        )
        completed_tokens = self.transformer(masked_tokens)
        fused = self.output_norm(completed_tokens.mean(dim=1))
        missing = ~modality_mask
        if missing.any():
            reconstruction_loss = F.smooth_l1_loss(
                completed_tokens[missing],
                real_tokens.detach()[missing],
            )
        else:
            reconstruction_loss = fused.new_zeros(())
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "completed_tokens": completed_tokens,
            "modality_mask": modality_mask,
            "reconstruction_loss": reconstruction_loss,
        }


class PairProxyGenerator(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.net(token)


class CompassLite(nn.Module):
    """Fixed-slot proxy-completion adaptation of COMPASS."""

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.encoder_bank = ModalityEncoderBank(channels, embedding_dim)
        self.generators = nn.ModuleDict(
            {
                f"{source}_to_{target}": PairProxyGenerator(embedding_dim)
                for source in MODALITIES
                for target in MODALITIES
                if source != target
            }
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def _proxy_tokens(
        self,
        real_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        proxies = []
        for target_index, target in enumerate(MODALITIES):
            candidates = []
            candidate_masks = []
            for source_index, source in enumerate(MODALITIES):
                if source == target:
                    continue
                candidates.append(
                    self.generators[f"{source}_to_{target}"](
                        real_tokens[:, source_index]
                    )
                )
                candidate_masks.append(modality_mask[:, source_index])
            stacked = torch.stack(candidates, dim=1)
            available = torch.stack(candidate_masks, dim=1).to(
                stacked.dtype
            )
            denominator = available.sum(dim=1, keepdim=True).clamp_min(1.0)
            proxy = (
                stacked * available.unsqueeze(-1)
            ).sum(dim=1) / denominator
            proxies.append(proxy)
        return torch.stack(proxies, dim=1)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        if modality_mask is None:
            modality_mask = torch.ones(
                real_tokens.shape[:2],
                dtype=torch.bool,
                device=real_tokens.device,
            )
        modality_mask = modality_mask.bool()
        proxy_tokens = self._proxy_tokens(real_tokens, modality_mask)
        completed_tokens = torch.where(
            modality_mask.unsqueeze(-1), real_tokens, proxy_tokens
        )
        fused = self.output_norm(completed_tokens.sum(dim=1))
        missing = ~modality_mask
        if missing.any():
            alignment_loss = F.mse_loss(
                proxy_tokens[missing], real_tokens.detach()[missing]
            )
        else:
            alignment_loss = fused.new_zeros(())
        proxy_logits = self.classifier(
            self.output_norm(proxy_tokens.reshape(-1, proxy_tokens.shape[-1]))
        ).reshape(
            proxy_tokens.shape[0], proxy_tokens.shape[1], -1
        )
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "completed_tokens": completed_tokens,
            "proxy_tokens": proxy_tokens,
            "proxy_logits": proxy_logits,
            "modality_mask": modality_mask,
            "alignment_loss": alignment_loss,
        }


class RapidGait(nn.Module):
    """Reliability-aware shared proxy completion for missing modalities."""

    modality_order = MODALITIES

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
        identity_dim: int = 8,
        reliability_temperature: float = 0.5,
        uniform_reliability: bool = False,
    ):
        super().__init__()
        self.encoder_bank = ModalityEncoderBank(channels, embedding_dim)
        self.source_identity = nn.Embedding(len(MODALITIES), identity_dim)
        self.target_identity = nn.Embedding(len(MODALITIES), identity_dim)
        conditional_dim = embedding_dim + 2 * identity_dim
        self.proxy_generator = nn.Sequential(
            nn.LayerNorm(conditional_dim),
            nn.Linear(conditional_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.reliability = nn.Sequential(
            nn.LayerNorm(conditional_dim),
            nn.Linear(conditional_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.fusion = GatedFusion(embedding_dim, len(MODALITIES))
        self.classifier = nn.Linear(embedding_dim, classes)
        self.reliability_temperature = float(reliability_temperature)
        self.uniform_reliability = bool(uniform_reliability)

    def _conditional_candidates(
        self, real_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = real_tokens.shape[0]
        source_ids = torch.arange(
            len(MODALITIES), device=real_tokens.device
        )
        target_ids = torch.arange(
            len(MODALITIES), device=real_tokens.device
        )
        source_token = real_tokens[:, None, :, :].expand(
            batch, len(MODALITIES), len(MODALITIES), -1
        )
        source_identity = self.source_identity(source_ids)[
            None, None, :, :
        ].expand(batch, len(MODALITIES), -1, -1)
        target_identity = self.target_identity(target_ids)[
            None, :, None, :
        ].expand(batch, -1, len(MODALITIES), -1)
        conditional = torch.cat(
            (source_token, source_identity, target_identity), dim=-1
        )
        candidates = self.proxy_generator(conditional)
        reliability_logits = self.reliability(conditional).squeeze(-1)
        return candidates, reliability_logits

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        if modality_mask is None:
            modality_mask = torch.ones(
                real_tokens.shape[:2],
                dtype=torch.bool,
                device=real_tokens.device,
            )
        modality_mask = modality_mask.bool()
        candidates, reliability_logits = self._conditional_candidates(
            real_tokens
        )
        # [batch, target, source]. For a missing target, its diagonal source is
        # also missing, so the availability mask prevents self-copy leakage.
        source_available = modality_mask[:, None, :].expand(
            -1, len(MODALITIES), -1
        )
        masked_reliability = reliability_logits.masked_fill(
            ~source_available, -1e4
        )
        if self.uniform_reliability:
            reliability_weights = source_available.to(
                real_tokens.dtype
            )
            reliability_weights = reliability_weights / (
                reliability_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            )
        else:
            reliability_weights = masked_reliability.softmax(dim=-1)
        proxy_tokens = (
            candidates * reliability_weights.unsqueeze(-1)
        ).sum(dim=2)
        completed_tokens = torch.where(
            modality_mask.unsqueeze(-1), real_tokens, proxy_tokens
        )
        complete_slot_mask = torch.ones_like(modality_mask)
        fused, fusion_weights = self.fusion(
            completed_tokens, complete_slot_mask
        )
        missing = ~modality_mask
        if missing.any():
            alignment_loss = F.smooth_l1_loss(
                proxy_tokens[missing], real_tokens.detach()[missing]
            )
            candidate_error = (
                candidates - real_tokens.detach()[:, :, None, :]
            ).pow(2).mean(dim=-1)
            ideal_logits = (
                -candidate_error / self.reliability_temperature
            ).masked_fill(~source_available, -1e4)
            ideal_weights = ideal_logits.softmax(dim=-1).detach()
            if self.uniform_reliability:
                reliability_loss = fused.new_zeros(())
            else:
                reliability_loss = -(
                    ideal_weights
                    * masked_reliability.log_softmax(dim=-1)
                ).sum(dim=-1)[missing].mean()
        else:
            alignment_loss = fused.new_zeros(())
            reliability_loss = fused.new_zeros(())
        proxy_logits = self.classifier(
            proxy_tokens.reshape(-1, proxy_tokens.shape[-1])
        ).reshape(proxy_tokens.shape[0], proxy_tokens.shape[1], -1)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "completed_tokens": completed_tokens,
            "proxy_tokens": proxy_tokens,
            "proxy_candidates": candidates,
            "proxy_logits": proxy_logits,
            "reliability_logits": reliability_logits,
            "reliability_weights": reliability_weights,
            "fusion_weights": fusion_weights,
            "modality_mask": modality_mask,
            "alignment_loss": alignment_loss,
            "reliability_loss": reliability_loss,
        }


class QualityRapidGait(RapidGait):
    """RAPID-Gait with learned quality routing for degraded inputs."""

    def __init__(
        self,
        channels: Mapping[str, int],
        classes: int = 3,
        embedding_dim: int = 64,
        identity_dim: int = 8,
        reliability_temperature: float = 0.5,
        quality_floor: float = 0.05,
        quality_routing_strength: float = 0.5,
    ):
        super().__init__(
            channels=channels,
            classes=classes,
            embedding_dim=embedding_dim,
            identity_dim=identity_dim,
            reliability_temperature=reliability_temperature,
        )
        self.quality_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.quality_floor = float(quality_floor)
        self.quality_routing_strength = float(
            quality_routing_strength
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.constant_(self.quality_head[-1].bias, 2.0)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        real_tokens = self.encoder_bank(inputs)
        if modality_mask is None:
            modality_mask = torch.ones(
                real_tokens.shape[:2],
                dtype=torch.bool,
                device=real_tokens.device,
            )
        modality_mask = modality_mask.bool()
        quality_logits = self.quality_head(real_tokens).squeeze(-1)
        predicted_quality = quality_logits.sigmoid()
        effective_quality = predicted_quality * modality_mask.to(
            predicted_quality.dtype
        )

        candidates, reliability_logits = self._conditional_candidates(
            real_tokens
        )
        source_available = modality_mask[:, None, :].expand(
            -1, len(MODALITIES), -1
        )
        quality_log_bias = effective_quality.clamp_min(
            self.quality_floor
        ).log()[:, None, :]
        routed_reliability = (
            reliability_logits
            + self.quality_routing_strength * quality_log_bias
        )
        masked_reliability = routed_reliability.masked_fill(
            ~source_available, -1e4
        )
        reliability_weights = masked_reliability.softmax(dim=-1)
        proxy_tokens = (
            candidates * reliability_weights.unsqueeze(-1)
        ).sum(dim=2)
        proxy_quality = (
            reliability_weights * effective_quality[:, None, :]
        ).sum(dim=-1)
        completed_tokens = torch.where(
            modality_mask.unsqueeze(-1), real_tokens, proxy_tokens
        )
        slot_quality = torch.where(
            modality_mask, effective_quality, proxy_quality
        ).clamp_min(self.quality_floor)

        fusion_scores = self.fusion.score(
            completed_tokens
        ).squeeze(-1)
        fusion_scores = (
            fusion_scores
            + self.quality_routing_strength * slot_quality.log()
        )
        fusion_weights = fusion_scores.softmax(dim=1)
        fused = (
            completed_tokens * fusion_weights.unsqueeze(-1)
        ).sum(dim=1)
        fused = self.fusion.output(fused)

        missing = ~modality_mask
        if missing.any():
            alignment_loss = F.smooth_l1_loss(
                proxy_tokens[missing], real_tokens.detach()[missing]
            )
            candidate_error = (
                candidates - real_tokens.detach()[:, :, None, :]
            ).pow(2).mean(dim=-1)
            ideal_logits = (
                -candidate_error / self.reliability_temperature
            ).masked_fill(~source_available, -1e4)
            ideal_weights = ideal_logits.softmax(dim=-1).detach()
            reliability_loss = -(
                ideal_weights
                * masked_reliability.log_softmax(dim=-1)
            ).sum(dim=-1)[missing].mean()
        else:
            alignment_loss = fused.new_zeros(())
            reliability_loss = fused.new_zeros(())
        proxy_logits = self.classifier(
            proxy_tokens.reshape(-1, proxy_tokens.shape[-1])
        ).reshape(proxy_tokens.shape[0], proxy_tokens.shape[1], -1)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "embeddings": real_tokens,
            "completed_tokens": completed_tokens,
            "proxy_tokens": proxy_tokens,
            "proxy_candidates": candidates,
            "proxy_logits": proxy_logits,
            "reliability_logits": reliability_logits,
            "routed_reliability_logits": routed_reliability,
            "reliability_weights": reliability_weights,
            "fusion_weights": fusion_weights,
            "modality_mask": modality_mask,
            "quality_logits": quality_logits,
            "predicted_quality": predicted_quality,
            "slot_quality": slot_quality,
            "alignment_loss": alignment_loss,
            "reliability_loss": reliability_loss,
        }
