"""Frozen dual-view encoder for translation-invariant TISM maps."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class FrozenTismEncoder(nn.Module):
    """Encode T-H and T-W maps with a shared ImageNet ResNet-18."""

    feature_dim = 1024

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        weights = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.register_buffer(
            "mean",
            torch.tensor(weights.mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(weights.std).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 5 or views.shape[1:3] != (2, 3):
            raise ValueError(
                "Expected TISM views with shape (B, 2, 3, H, W), "
                f"got {tuple(views.shape)}"
            )
        batch = views.shape[0]
        images = views.reshape(-1, *views.shape[2:])
        images = (images - self.mean) / self.std
        features = self.backbone(images).reshape(batch, 2, -1)
        return features.flatten(1)
