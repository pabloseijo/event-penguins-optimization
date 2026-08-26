"""Augmented Temporal Segment Network, the classifier of the reTAG pipeline.

A ResNet-18 backbone encodes every temporal sample of a proposal, the samples are
averaged inside three segments (left context, proposal, right context) and the
concatenation is classified. The optional input adapter and temporal-difference
head are project additions, off by default so the released weights load unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class Consensus(nn.Module):
    """Average pooling over the temporal-sample dimension of a segment."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=True)


class TemporalDifferenceHead(nn.Module):
    """Auxiliary head over first-order differences between consecutive samples.

    The averaged features of a segment are order-invariant, so a display and its
    time-reversed copy look identical to the main head. This head consumes the
    differences instead. It is zero-initialised, so enabling it leaves the network
    output unchanged at step zero.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_channels, hidden_channels, bias=False)
        self.temporal_conv = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.activation = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(3)
        self.output = nn.Linear(3 * hidden_channels, num_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        differences = frame_features[:, 1:] - frame_features[:, :-1]
        differences = nn.functional.layer_norm(
            differences,
            normalized_shape=(differences.shape[-1],),
        )
        x = self.projection(differences).transpose(1, 2)
        x = self.activation(self.temporal_conv(x))
        x = self.pool(x).flatten(1)
        return self.output(x)


class AugmentedTsn(nn.Module):
    """TSN classifier over event time surfaces, with optional project extensions.

    Args:
        num_classes: number of output classes (2 in the ED setting).
        num_tsn_samples: temporal samples inside the proposal.
        augment_factor: inverse of the context fraction added on each side.
        use_input_adapter: map one channel of a normalised input back to the
            repeated-gray input the released weights were trained on.
        input_adapter_source_channel: which channel the adapter reads.
        use_temporal_difference_head: add the auxiliary difference head.
        temporal_hidden_channels: hidden width of that head.
    """

    def __init__(
        self,
        num_classes: int,
        num_tsn_samples: int = 3,
        augment_factor: int = 3,
        use_input_adapter: bool = False,
        input_adapter_source_channel: int = 0,
        use_temporal_difference_head: bool = False,
        temporal_hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        self.num_tsn_samples = num_tsn_samples
        self.num_augment = int(np.ceil(num_tsn_samples / augment_factor))
        self.input_adapter = nn.Conv2d(3, 3, kernel_size=1) if use_input_adapter else nn.Identity()
        if use_input_adapter:
            self._initialize_repeated_channel_adapter(input_adapter_source_channel)

        backbone = models.resnet18()
        self.backbone = create_feature_extractor(
            backbone, return_nodes={"layer4.1.relu_1": "features"}
        )

        # dry run: read the channel count instead of hard-coding ResNet internals
        with torch.no_grad():
            out = self.backbone(torch.randn(1, 3, 224, 224))["features"]
        in_channels = out.shape[1]

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.consensus = Consensus(dim=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_cls = nn.Linear(3 * in_channels, num_classes)  # tres segmentos concatenados
        nn.init.xavier_uniform_(self.fc_cls.weight)
        self.temporal_difference_head = (
            TemporalDifferenceHead(in_channels, temporal_hidden_channels, num_classes)
            if use_temporal_difference_head
            else nn.Identity()
        )

    def _initialize_repeated_channel_adapter(self, source_channel: int) -> None:
        """Map one normalized input channel to the old repeated-gray input."""
        if source_channel not in (0, 1, 2):
            raise ValueError("input_adapter_source_channel must be 0, 1, or 2")
        means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        stds = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.bias.zero_()
            for output_channel in range(3):
                self.input_adapter.weight[output_channel, source_channel, 0, 0] = float(
                    stds[source_channel] / stds[output_channel]
                )
                self.input_adapter.bias[output_channel] = float(
                    means[source_channel] - means[output_channel]
                ) / float(stds[output_channel])

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Extract one pooled ResNet feature for every temporal sample."""
        if x.ndim != 5:
            raise ValueError(f"Expected input with shape (B, N, C, H, W), got {tuple(x.shape)}")
        num_segs = x.shape[1]
        x = x.reshape((-1,) + x.shape[2:])
        x = self.input_adapter(x)
        x = self.backbone(x)["features"]
        x = self.avg_pool(x)
        x = x.reshape((-1, num_segs) + x.shape[1:])
        return x.flatten(2)

    def classify_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Apply the original start/main/end consensus to ordered frame features."""
        if frame_features.ndim != 3:
            raise ValueError(
                "Expected frame features with shape (B, N, D), "
                f"got {tuple(frame_features.shape)}"
            )
        num_segs = frame_features.shape[1]
        a = self.num_augment
        if num_segs <= 2 * a:
            raise ValueError(
                f"Need more than {2 * a} temporal samples for ATSN consensus, got {num_segs}"
            )
        start = self.consensus(frame_features[:, :a]).squeeze(1)
        main = self.consensus(frame_features[:, a:num_segs - a]).squeeze(1)
        end = self.consensus(frame_features[:, num_segs - a:]).squeeze(1)

        x = torch.cat((start, main, end), dim=1)
        x = self.dropout(x)
        logits = self.fc_cls(x)
        if isinstance(self.temporal_difference_head, TemporalDifferenceHead):
            logits = logits + self.temporal_difference_head(frame_features)
        return logits

    def forward_with_frame_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unchanged ATSN logits together with the ordered temporal features."""
        frame_features = self.encode_frames(x)
        return self.classify_frame_features(frame_features), frame_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (B, N, C, H, W)

        Returns:
            logits de clase (B, num_classes)
        """
        logits, _ = self.forward_with_frame_features(x)
        return logits
