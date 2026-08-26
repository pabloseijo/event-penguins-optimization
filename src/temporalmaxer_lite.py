"""Proposal-local dense head over ordered ATSN features (phase 4).

The head keeps the temporal sequence of a proposal instead of pooling it away and
gathers context with parameter-free max pooling, as in TemporalMaxer. It predicts
per-point actionness and quality, start and end boundary maps and boundary
distances, which is what makes boundary voting possible.

This is the proposal-local predecessor of
:mod:`src.temporalmaxer_continuous`, which consumes the whole ROI timeline
instead and became the architecture of the final system.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def temporal_aware_normalization_perturbation(
    features: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply TANP while preserving the temporal residual around a scene anchor.

    This is Eq. 18 from Feng et al. (AAAI 2026). ``features`` uses the
    detector-native (B, C, T) layout; perturbation factors are sampled per
    sample and channel, as specified in the paper.
    """
    if features.ndim != 3:
        raise ValueError(f"Expected (B, C, T) features, got {tuple(features.shape)}")
    if sigma < 0:
        raise ValueError("TANP sigma must be non-negative")
    if sigma == 0:
        return features

    temporal = features.transpose(1, 2)
    normalized = F.normalize(temporal, dim=2, eps=1e-6)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    anchor_index = similarity.sum(dim=2).argmax(dim=1)
    batch_index = torch.arange(len(temporal), device=features.device)
    anchor = temporal[batch_index, anchor_index]

    anchor_norm_sq = anchor.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
    projection_scale = (
        (temporal * anchor.unsqueeze(1)).sum(dim=2, keepdim=True)
        / anchor_norm_sq.unsqueeze(1)
    )
    static_component = projection_scale * anchor.unsqueeze(1)
    static_mean = static_component.mean(dim=1, keepdim=True)

    noise_shape = (len(temporal), 1, temporal.shape[2])
    alpha = 1.0 + sigma * torch.randn(noise_shape, device=features.device, dtype=features.dtype)
    beta = 1.0 + sigma * torch.randn(noise_shape, device=features.device, dtype=features.dtype)
    perturbed = alpha * temporal + (beta - alpha) * static_mean
    return perturbed.transpose(1, 2)


class TemporalTower(nn.Module):
    """Small task-specific temporal tower used after the shared max-pooling neck."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class TemporalMaxerLiteHead(nn.Module):
    """Dense temporal quality and boundary head for ordered ATSN features.

    The neck preserves the full temporal sequence and gathers local context with
    parameter-free max pooling at progressively coarser scales. Classification
    and localization use independent towers because their useful temporal cues
    are not the same.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        pyramid_levels: int = 3,
        dropout: float = 0.15,
        trident_bins: int = 0,
        auxiliary_dim: int = 0,
        tanp_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        if pyramid_levels < 1:
            raise ValueError("pyramid_levels must be at least one")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pyramid_levels = pyramid_levels
        self.trident_bins = int(trident_bins)
        self.auxiliary_dim = int(auxiliary_dim)
        self.tanp_sigma = float(tanp_sigma)
        if not 0 <= self.auxiliary_dim < input_dim:
            raise ValueError("auxiliary_dim must be in [0, input_dim)")
        if self.tanp_sigma < 0:
            raise ValueError("tanp_sigma must be non-negative")

        self.input_norm = nn.LayerNorm(input_dim) if self.auxiliary_dim == 0 else None
        self.base_input_norm = (
            nn.LayerNorm(input_dim - self.auxiliary_dim)
            if self.auxiliary_dim > 0
            else None
        )
        self.auxiliary_input_norm = (
            nn.LayerNorm(self.auxiliary_dim) if self.auxiliary_dim > 0 else None
        )
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pyramid_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * pyramid_levels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_tower = TemporalTower(hidden_dim, dropout)
        self.boundary_tower = TemporalTower(hidden_dim, dropout)

        self.action_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.point_quality_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.start_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.end_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.distance_head = nn.Conv1d(hidden_dim, 2, kernel_size=1)
        self.trident_offset_head = (
            nn.Conv1d(hidden_dim, 2 * (self.trident_bins + 1), kernel_size=1)
            if self.trident_bins > 0
            else None
        )
        self.trident_start_head = (
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
            if self.trident_bins > 0
            else None
        )
        self.trident_end_head = (
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
            if self.trident_bins > 0
            else None
        )
        self.quality_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.start_delta = nn.Linear(hidden_dim, 1)
        self.end_delta = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.start_delta.weight)
        nn.init.zeros_(self.start_delta.bias)
        nn.init.zeros_(self.end_delta.weight)
        nn.init.zeros_(self.end_delta.bias)

    def _temporal_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        target_length = x.shape[-1]
        levels = [x]
        current = x
        for _ in range(1, self.pyramid_levels):
            current = F.max_pool1d(current, kernel_size=3, stride=2, padding=1)
            levels.append(current)
        aligned = [
            level
            if level.shape[-1] == target_length
            else F.interpolate(level, size=target_length, mode="linear", align_corners=False)
            for level in levels
        ]
        return self.pyramid_fusion(torch.cat(aligned, dim=1))

    def encode_shared(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Encode ordered proposal features before the task-specific towers."""
        if frame_features.ndim != 3:
            raise ValueError(
                "Expected frame features with shape (B, T, D), "
                f"got {tuple(frame_features.shape)}"
            )
        if frame_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {frame_features.shape[-1]}"
            )

        if self.auxiliary_dim > 0:
            split = self.input_dim - self.auxiliary_dim
            normalized = torch.cat(
                (
                    self.base_input_norm(frame_features[:, :, :split]),
                    self.auxiliary_input_norm(frame_features[:, :, split:]),
                ),
                dim=2,
            )
        else:
            normalized = self.input_norm(frame_features)
        x = self.input_projection(normalized).transpose(1, 2)
        return self._temporal_pyramid(x)

    def forward(self, frame_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict per-point actionness, quality and boundaries for a proposal.

        Args:
            frame_features: ``[B, T, input_dim]`` ordered proposal features.

        Returns:
            Dictionary with ``action_logits``, ``point_quality_logits``, ``start_logits``,
            ``end_logits``, ``boundary_distances``, ``trident_offsets_bins``,
            ``quality_logit`` and ``boundary_deltas``.
        """
        shared = self.encode_shared(frame_features)
        if self.training and self.tanp_sigma > 0:
            shared = temporal_aware_normalization_perturbation(shared, self.tanp_sigma)
        class_features = self.class_tower(shared)
        boundary_features = self.boundary_tower(shared)

        action_logits = self.action_head(class_features).squeeze(1)
        point_quality_logits = self.point_quality_head(class_features).squeeze(1)
        start_logits = self.start_head(boundary_features).squeeze(1)
        end_logits = self.end_head(boundary_features).squeeze(1)
        boundary_distances = F.softplus(self.distance_head(boundary_features).transpose(1, 2))
        trident_offsets_bins = None
        if self.trident_offset_head is not None:
            bins = self.trident_bins + 1
            center_logits = self.trident_offset_head(boundary_features).transpose(1, 2)
            center_logits = center_logits.reshape(frame_features.shape[0], -1, 2, bins)
            start_logits_relative = self.trident_start_head(boundary_features.detach()).squeeze(1)
            end_logits_relative = self.trident_end_head(boundary_features.detach()).squeeze(1)
            start_neighbors = F.pad(
                start_logits_relative,
                (self.trident_bins, 0),
            ).unfold(1, bins, 1)
            end_neighbors = F.pad(
                end_logits_relative,
                (0, self.trident_bins),
            ).unfold(1, bins, 1)
            left_probability = torch.softmax(
                start_neighbors + center_logits[:, :, 0],
                dim=2,
            )
            right_probability = torch.softmax(
                end_neighbors + center_logits[:, :, 1],
                dim=2,
            )
            left_bins = torch.arange(
                self.trident_bins,
                -1,
                -1,
                device=frame_features.device,
                dtype=frame_features.dtype,
            )
            right_bins = torch.arange(
                bins,
                device=frame_features.device,
                dtype=frame_features.dtype,
            )
            trident_offsets_bins = torch.stack(
                (
                    (left_probability * left_bins).sum(dim=2),
                    (right_probability * right_bins).sum(dim=2),
                ),
                dim=2,
            )
        pooled = torch.cat(
            (class_features.mean(dim=2), class_features.amax(dim=2)),
            dim=1,
        )
        quality_logit = self.quality_head(pooled).squeeze(1)

        start_attention = torch.softmax(start_logits, dim=1).unsqueeze(1)
        end_attention = torch.softmax(end_logits, dim=1).unsqueeze(1)
        start_feature = (boundary_features * start_attention).sum(dim=2)
        end_feature = (boundary_features * end_attention).sum(dim=2)
        boundary_deltas = torch.cat(
            (self.start_delta(start_feature), self.end_delta(end_feature)),
            dim=1,
        )
        return {
            "action_logits": action_logits,
            "point_quality_logits": point_quality_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "boundary_distances": boundary_distances,
            "trident_offsets_bins": trident_offsets_bins,
            "quality_logit": quality_logit,
            "boundary_deltas": boundary_deltas,
        }
