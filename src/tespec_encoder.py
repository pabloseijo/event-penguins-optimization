"""Frozen TESPEC recurrent event encoder.

The architecture mirrors the encoder released with TESPEC (ICCV 2025): a
20-channel Swin-T backbone followed by one ConvLSTM block per feature stage.
Only the pretrained encoder weights are loaded; the reconstruction decoder is
not needed for downstream proposal features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
from torch import nn
import torch.nn.functional as F


class DWSConvLSTM2d(nn.Module):
    """TESPEC/RVT ConvLSTM block in channel-first format."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(2 * dim, 4 * dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        previous: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous is None:
            previous = (torch.zeros_like(x), torch.zeros_like(x))
        hidden, cell = previous
        mixed = self.conv1x1(torch.cat((x, hidden), dim=1))
        gates, cell_input = torch.tensor_split(mixed, [3 * self.dim], dim=1)
        forget, update, output = torch.tensor_split(torch.sigmoid(gates), 3, dim=1)
        cell = forget * cell + update * torch.tanh(cell_input)
        hidden = output * torch.tanh(cell)
        return hidden, cell


class TespecEncoder(nn.Module):
    """Return one pooled recurrent TESPEC feature per event frame."""

    stage_dims = (96, 192, 384, 768)

    def __init__(self, image_size: int | tuple[int, int] = 224) -> None:
        super().__init__()
        self.pure_backbone = timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in1k",
            in_chans=20,
            img_size=image_size,
            features_only=True,
        )
        self.lstm_blocks = nn.ModuleList(
            DWSConvLSTM2d(dim) for dim in self.stage_dims
        )

    @staticmethod
    def _to_nchw(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected a 4D Swin feature, got {tuple(feature.shape)}")
        return feature.permute(0, 3, 1, 2).contiguous()

    def encode_frame(
        self,
        x: torch.Tensor,
        previous: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Encode one event frame and return its pooled feature and LSTM state.

        Args:
            x: ``[B, 20, H, W]`` event representation of a single step.
            previous: per-stage LSTM state from the previous step, or None to start.

        Returns:
            Tuple of the pooled feature vector and the updated per-stage state.
        """
        features = []
        states = []
        for index, layer in enumerate(self.pure_backbone.children()):
            x = layer(x)
            if index > 0:
                x_nchw = self._to_nchw(x)
                old_state = None if previous is None else previous[index - 1]
                state = self.lstm_blocks[index - 1](x_nchw, old_state)
                states.append(state)
                x = state[0].permute(0, 2, 3, 1).contiguous()
            features.append(x)
        last = self._to_nchw(features[-1])
        return F.adaptive_avg_pool2d(last, 1).flatten(1), states

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of event frames recurrently.

        Args:
            sequence: ``[B, T, 20, H, W]`` event representation.

        Returns:
            ``[B, T, D]`` per-step features.

        Raises:
            ValueError: if the input does not have 20 channels.
        """
        if sequence.ndim != 5 or sequence.shape[2] != 20:
            raise ValueError(
                "Expected TESPEC input with shape (B, T, 20, H, W), "
                f"got {tuple(sequence.shape)}"
            )
        states = None
        outputs = []
        for index in range(sequence.shape[1]):
            pooled, states = self.encode_frame(sequence[:, index], states)
            outputs.append(pooled)
        return torch.stack(outputs, dim=1)

    def load_pretrained(self, checkpoint_path: str | Path) -> None:
        """Load the released TESPEC encoder weights from a checkpoint.

        Only the ``model.encoder.`` entries are read; the reconstruction decoder is not
        needed downstream. Any missing or unexpected key is an error rather than a
        warning, so a silently half-loaded encoder cannot reach an experiment.

        Args:
            checkpoint_path: path to the released checkpoint.

        Raises:
            ValueError: if the checkpoint holds no encoder weights, or if they do not match.
        """
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        prefix = "model.encoder."
        encoder_state = {
            name[len(prefix):]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if not encoder_state:
            raise ValueError("Checkpoint does not contain TESPEC encoder weights")
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "TESPEC encoder checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
