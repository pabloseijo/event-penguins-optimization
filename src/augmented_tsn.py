"""AugmentedTSN: ResNet-18 backbone with temporal segment consensus.

Implements the TSN-style classifier used by reTAG. The model pools features
across three temporal segments (start, main, end) of an augmented proposal
window and classifies the concatenated representation.
"""

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class Consensus(nn.Module):
    """Average-pool a sequence of frame features along a temporal dimension.

    Args:
        dim: Dimension to average over (typically 1, the segment/frame axis).
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=True)


class AugmentedTsn(nn.Module):
    """TSN classifier with start/main/end augmentation stages.

    Extracts ResNet-18 features for each sampled frame, applies temporal
    consensus separately to start, main, and end segments, then classifies
    the concatenated pooled features.

    Args:
        num_classes: Number of output classes.
        num_tsn_samples: Number of temporal samples in the *main* segment.
        augment_factor: Determines the number of augmentation samples per side
            as ceil(num_tsn_samples / augment_factor).
    """

    def __init__(self, num_classes: int, num_tsn_samples: int = 3, augment_factor: int = 3) -> None:
        super().__init__()

        self.num_tsn_samples = num_tsn_samples
        self.num_augment     = int(np.ceil(num_tsn_samples / augment_factor))

        backbone = models.resnet18()
        self.backbone = create_feature_extractor(
            backbone, return_nodes={"layer4.1.relu_1": "features"}
        )

        # Dry run to infer feature map channels without coupling to ResNet internals
        with torch.no_grad():
            out = self.backbone(torch.randn(1, 3, 224, 224))["features"]
        in_channels = out.shape[1]

        self.avg_pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.consensus = Consensus(dim=1)
        self.dropout   = nn.Dropout(p=0.4)
        # Three stages concatenated → 3× the channel count
        self.fc_cls    = nn.Linear(3 * in_channels, num_classes)
        nn.init.xavier_uniform_(self.fc_cls.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, C, H, W) — batch of N-frame sequences.

        Returns:
            Class logits of shape (B, num_classes).
        """
        num_segs = x.shape[1]
        # Merge batch and segment dims for a single backbone forward pass
        x = x.reshape((-1,) + x.shape[2:])
        x = self.backbone(x)["features"]
        x = self.avg_pool(x)
        # Restore (B, N, C, 1, 1)
        x = x.reshape((-1, num_segs) + x.shape[1:])

        a = self.num_augment
        start = self.consensus(x[:, :a]).squeeze(1)
        main  = self.consensus(x[:, a:num_segs - a]).squeeze(1)
        end   = self.consensus(x[:, num_segs - a:]).squeeze(1)

        x = torch.cat((start, main, end), dim=1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc_cls(x)
