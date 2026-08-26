"""Boundary-Sensitive Pretext task (Xu et al., ICCV 2021) over cached features.

The pretext synthesises four kinds of temporal boundary from ordered proposal
features and asks the shared representation to tell them apart, which pushes the
encoder to represent boundaries rather than only interior appearance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


DIFFERENT_CLASS = 0
SAME_CLASS = 1
DIFFERENT_SPEED = 2
SAME_SPEED = 3
NUM_BOUNDARY_TYPES = 4


def _resample_sequence(sequence: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Linearly sample a [T,D] sequence at fractional temporal positions."""
    last = sequence.shape[0] - 1
    positions = positions.clamp(0.0, float(last))
    lower = positions.floor().long()
    upper = (lower + 1).clamp_max(last)
    weight = (positions - lower.to(positions.dtype)).unsqueeze(1)
    return sequence[lower] * (1.0 - weight) + sequence[upper] * weight


def synthesize_bsp_sequences(
    primary: torch.Tensor,
    secondary: torch.Tensor,
    boundary_types: torch.Tensor,
    split_positions: torch.Tensor,
    speed_rates: torch.Tensor,
) -> torch.Tensor:
    """Create the four boundary types from cached ordered proposal features.

    Pair selection determines whether a splice is same-class or different-class;
    this function only performs the temporal transformation. The label order
    follows Xu et al., ICCV 2021: different class, same class, different speed,
    and coherent same speed.
    """
    if primary.ndim != 3 or secondary.shape != primary.shape:
        raise ValueError("primary and secondary must have matching [B,T,D] shapes")
    batch_size, length, _ = primary.shape
    for name, value in (
        ("boundary_types", boundary_types),
        ("split_positions", split_positions),
        ("speed_rates", speed_rates),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
    if length < 3:
        raise ValueError("BSP synthesis requires at least three temporal samples")
    if torch.any((boundary_types < 0) | (boundary_types >= NUM_BOUNDARY_TYPES)):
        raise ValueError("Unknown BSP boundary type")
    if torch.any((split_positions <= 0) | (split_positions >= length)):
        raise ValueError("split positions must be strictly inside the sequence")
    if torch.any(speed_rates <= 0):
        raise ValueError("speed rates must be positive")

    output = primary.clone()
    temporal_positions = torch.arange(length, device=primary.device, dtype=primary.dtype)
    for index in range(batch_size):
        boundary_type = int(boundary_types[index])
        split = int(split_positions[index])
        if boundary_type in (DIFFERENT_CLASS, SAME_CLASS):
            output[index, split:] = secondary[index, split:]
        elif boundary_type == DIFFERENT_SPEED:
            positions = temporal_positions.clone()
            positions[split:] = split + speed_rates[index] * (
                positions[split:] - split
            )
            output[index] = _resample_sequence(primary[index], positions)
    return output


class BoundaryTypeHead(nn.Module):
    """Four-way BSP classifier over the shared TemporalMaxer representation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_BOUNDARY_TYPES),
        )

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        if shared_features.ndim != 3:
            raise ValueError("Expected shared features with shape [B,C,T]")
        differences = shared_features[:, :, 1:] - shared_features[:, :, :-1]
        pooled = torch.cat(
            (
                shared_features.mean(dim=2),
                shared_features.amax(dim=2),
                differences.abs().mean(dim=2),
                differences.abs().amax(dim=2),
            ),
            dim=1,
        )
        return self.classifier(pooled)


def boundary_type_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy over the four synthesized boundary types.

    Args:
        logits: ``[B, 4]`` predictions from :class:`BoundaryTypeHead`.
        targets: ``[B]`` boundary type indices.

    Returns:
        Scalar loss.

    Raises:
        ValueError: if the logits do not have four columns.
    """
    if logits.ndim != 2 or logits.shape[1] != NUM_BOUNDARY_TYPES:
        raise ValueError(f"Expected BSP logits with shape [B,{NUM_BOUNDARY_TYPES}]")
    return F.cross_entropy(logits, targets)
