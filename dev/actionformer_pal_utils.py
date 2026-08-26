"""Pure helpers for the supervised PAL adaptation used with ActionFormer."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def valid_background_starts(
    segments: np.ndarray,
    sequence_length: int,
    crop_length: int,
    margin: int,
) -> np.ndarray:
    if crop_length <= 0 or crop_length > sequence_length:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(sequence_length - crop_length + 1, dtype=np.int64)
    if len(segments) == 0:
        return starts
    ends = starts + crop_length
    occupied = np.zeros(len(starts), dtype=bool)
    for segment_start, segment_end in np.asarray(segments):
        occupied |= (
            (starts < segment_end + margin)
            & (ends > segment_start - margin)
        )
    return starts[~occupied]


def pal_contrastive_loss(
    recipient_embeddings: torch.Tensor,
    donor_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if recipient_embeddings.shape != donor_embeddings.shape:
        raise ValueError("PAL embedding tensors must have the same shape")
    recipient = F.normalize(recipient_embeddings, dim=1)
    donor = F.normalize(donor_embeddings, dim=1)
    if len(recipient) == 1:
        return 1.0 - (recipient * donor).sum(dim=1).mean()
    logits = recipient @ donor.transpose(0, 1) / temperature
    labels = torch.arange(len(recipient), device=recipient.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )
