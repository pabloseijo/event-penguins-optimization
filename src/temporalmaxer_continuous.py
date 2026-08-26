"""TemporalMaxer-style detector over complete ROI feature sequences.

Unlike the proposal-local head, this module consumes the complete ``[T, D]``
timeline, learns from every background point, and decodes one-stage temporal
detections at every level of a max-pooling pyramid.
"""

from __future__ import annotations

import math
import itertools

import torch
import torch.nn.functional as F
from torch import nn

from src.rank_sort_loss import rank_sort_loss


class ChannelLayerNorm(nn.Module):
    """Layer normalization for tensors laid out as ``[B, C, T]``."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class Scale(nn.Module):
    """Learnable per-level scalar, used to rescale regression outputs."""
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.value


def temporal_aware_normalization_perturbation(
    features: torch.Tensor,
    mask: torch.Tensor,
    std: float = 0.75,
    alpha: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply TANP while preserving temporal differences up to channel scaling."""
    if features.ndim != 3:
        raise ValueError("TANP expects [B,C,T] features")
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    if mask.shape != (features.shape[0], 1, features.shape[2]):
        raise ValueError("TANP mask must have shape [B,T] or [B,1,T]")
    if std < 0:
        raise ValueError("TANP standard deviation must be non-negative")

    valid = mask.to(features.dtype)
    normalized = F.normalize(features, dim=1, eps=1e-8) * valid
    aggregate = normalized.sum(dim=2, keepdim=True)
    affinity = (normalized * aggregate).sum(dim=1)
    affinity = affinity.masked_fill(~mask.squeeze(1), float("-inf"))
    anchor_indices = affinity.argmax(dim=1)
    anchor = features.gather(
        2,
        anchor_indices[:, None, None].expand(-1, features.shape[1], 1),
    )
    projection = (features * anchor).sum(dim=1, keepdim=True)
    projection /= anchor.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
    static = projection * anchor
    static_mean = (static * valid).sum(dim=2, keepdim=True)
    static_mean /= valid.sum(dim=2, keepdim=True).clamp_min(1.0)

    noise_shape = (features.shape[0], features.shape[1], 1)
    if alpha is None:
        alpha = torch.randn(noise_shape, device=features.device, dtype=features.dtype)
        alpha = 1.0 + std * alpha
    if beta is None:
        beta = torch.randn(noise_shape, device=features.device, dtype=features.dtype)
        beta = 1.0 + std * beta
    if alpha.shape != noise_shape or beta.shape != noise_shape:
        raise ValueError(f"TANP alpha and beta must have shape {noise_shape}")
    perturbed = alpha * features + (beta - alpha) * static_mean
    return perturbed * valid


class TemporalTower(nn.Module):
    """Stack of masked 1-D convolution blocks shared by one detection head."""
    def __init__(self, channels: int, layers: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                    ChannelLayerNorm(channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.blocks(value) * mask.to(value.dtype)


class LocalSelfAttentionBlock(nn.Module):
    """ActionFormer-style local self-attention over one pyramid level.

    Only used when the model is built with ``neck_type="attention"``. It exists so
    the paper can compare the max-pool neck against windowed attention while holding
    the features, the pyramid, the heads and the protocol fixed.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        window: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be a positive odd number of bins")
        self.window = window
        self.heads = heads
        self.norm_attention = ChannelLayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = ChannelLayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 4, channels, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def _attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Window plus padding, with the diagonal always left open.

        A query whose every key is masked makes softmax return NaN, and clamping the
        output afterwards does not stop the NaN from reaching the gradient. Keeping
        each position attending to itself removes the degenerate row entirely.
        """
        valid = mask.reshape(mask.shape[0], -1).bool()
        batch, length = valid.shape
        index = torch.arange(length, device=mask.device)
        outside_window = (index[:, None] - index[None, :]).abs() > (self.window // 2)
        blocked = outside_window[None] | ~valid[:, None, :]
        blocked = blocked & ~torch.eye(length, dtype=torch.bool, device=mask.device)
        return blocked.repeat_interleave(self.heads, dim=0)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = value
        normed = self.norm_attention(value).transpose(1, 2)
        attended, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=self._attention_mask(mask),
            need_weights=False,
        )
        attended = attended.transpose(1, 2)
        value = (residual + self.dropout(attended)) * mask.to(value.dtype)
        value = value + self.dropout(self.ffn(self.norm_ffn(value)))
        return value * mask.to(value.dtype)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss over per-point action logits.

    A continuous ROI timeline is overwhelmingly background, so an unweighted loss is
    dominated by easy negatives. Focal loss down-weights them.

    Args:
        logits: raw per-point logits.
        targets: per-point targets in ``{0, 1}``.
        alpha: weight of the positive class.
        gamma: focusing exponent.

    Returns:
        Per-point loss, left unreduced so the caller can normalise by positives.
    """
    probabilities = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * ce * (1.0 - p_t).pow(gamma)


def center_diou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distance-IoU loss between predicted and target boundary distances.

    Segments are parameterised as ``[left, right]`` distances from the point. The
    distance term keeps gradients alive when two segments do not overlap at all,
    where plain IoU is flat.

    Args:
        predicted: ``[n, 2]`` predicted distances.
        target: ``[n, 2]`` target distances.

    Returns:
        Per-sample loss.
    """
    pred_left, pred_right = predicted[:, 0], predicted[:, 1]
    gt_left, gt_right = target[:, 0], target[:, 1]
    intersection = torch.minimum(pred_left, gt_left) + torch.minimum(pred_right, gt_right)
    union = pred_left + pred_right + gt_left + gt_right - intersection
    iou = intersection / union.clamp_min(1e-6)
    enclosing = torch.maximum(pred_left, gt_left) + torch.maximum(pred_right, gt_right)
    center_distance = 0.5 * (pred_right - pred_left - gt_right + gt_left)
    return 1.0 - iou + (center_distance / enclosing.clamp_min(1e-6)).square()


def center_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Temporal IoU between predicted and target ``[left, right]`` distances.

    This is the target of the quality head: a point predicts how good its own
    regression is, which is what lets ranking use localisation quality and not only
    classification confidence.
    """
    intersection = (
        torch.minimum(predicted[:, 0], target[:, 0])
        + torch.minimum(predicted[:, 1], target[:, 1])
    )
    union = predicted.sum(dim=1) + target.sum(dim=1) - intersection
    return intersection / union.clamp_min(1e-6)


def distribution_focal_loss(
    logits: torch.Tensor, target: torch.Tensor, reg_max: int
) -> torch.Tensor:
    """Interpolate cross entropy between the two bins around each boundary."""
    target = target.clamp(min=0.0, max=float(reg_max) - 1e-4)
    lower = target.floor().long()
    upper = (lower + 1).clamp(max=reg_max)
    upper_weight = target - lower.to(target.dtype)
    lower_weight = 1.0 - upper_weight
    flattened = logits.reshape(-1, reg_max + 1)
    return (
        F.cross_entropy(flattened, lower.reshape(-1), reduction="none") * lower_weight.reshape(-1)
        + F.cross_entropy(flattened, upper.reshape(-1), reduction="none") * upper_weight.reshape(-1)
    ).reshape(target.shape)


def trident_offsets(
    center_logits: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """Decode TriDet relative boundary distributions into [left,right] offsets."""
    if num_bins < 1:
        raise ValueError("Trident requires at least one relative boundary bin")
    expected_shape = (*start_logits.shape, 2, num_bins + 1)
    if center_logits.shape != expected_shape or end_logits.shape != start_logits.shape:
        raise ValueError("Incompatible Trident center and boundary logits")
    start_neighbours = F.pad(start_logits, (num_bins, 0)).unfold(
        dimension=1,
        size=num_bins + 1,
        step=1,
    )
    end_neighbours = F.pad(end_logits, (0, num_bins)).unfold(
        dimension=1,
        size=num_bins + 1,
        step=1,
    )
    left_distribution = (
        start_neighbours + center_logits[:, :, 0, :]
    ).softmax(dim=-1)
    right_distribution = (
        end_neighbours + center_logits[:, :, 1, :]
    ).softmax(dim=-1)
    left_bins = torch.arange(
        num_bins,
        -1,
        -1,
        device=center_logits.device,
        dtype=center_logits.dtype,
    )
    right_bins = torch.arange(
        num_bins + 1,
        device=center_logits.device,
        dtype=center_logits.dtype,
    )
    return torch.stack(
        (
            (left_distribution * left_bins).sum(dim=-1),
            (right_distribution * right_bins).sum(dim=-1),
        ),
        dim=-1,
    )


class TemporalMaxerContinuous(nn.Module):
    """Dense full-sequence TAD model following TemporalMaxer's max-pool neck."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        pyramid_levels: int = 6,
        head_layers: int = 3,
        dropout: float = 0.1,
        prior_probability: float = 0.01,
        regression_ranges: tuple[tuple[float, float], ...] | None = None,
        use_quality: bool = True,
        reg_max: int = 0,
        center_sampling_radius: float = 0.0,
        use_boundary_heads: bool = False,
        boundary_refine_radius_seconds: float = 0.0,
        boundary_refine_blend: float = 0.5,
        tanp_std: float = 0.0,
        tanp_probability: float = 1.0,
        use_temporal_order: bool = False,
        temporal_order_chunks: int = 3,
        classification_input_dim: int | None = None,
        trident_bins: int = 0,
        neck_type: str = "maxpool",
        attention_heads: int = 4,
        attention_window: int = 19,
    ) -> None:
        super().__init__()
        if pyramid_levels < 1:
            raise ValueError("pyramid_levels must be positive")
        if regression_ranges is None:
            regression_ranges = (
                (0.0, 8.0),
                (4.0, 16.0),
                (8.0, 32.0),
                (16.0, 64.0),
                (32.0, 128.0),
                (64.0, float("inf")),
            )[:pyramid_levels]
        if len(regression_ranges) != pyramid_levels:
            raise ValueError("One regression range is required per pyramid level")

        self.input_dim = input_dim
        self.classification_input_dim = (
            None
            if classification_input_dim is None
            else int(classification_input_dim)
        )
        if self.classification_input_dim is not None and not (
            0 < self.classification_input_dim <= self.input_dim
        ):
            raise ValueError(
                "classification_input_dim must be in (0, input_dim]"
            )
        self.hidden_dim = hidden_dim
        self.pyramid_levels = pyramid_levels
        self.regression_ranges = tuple(regression_ranges)
        self.use_quality = use_quality
        self.reg_max = int(reg_max)
        if self.reg_max < 0:
            raise ValueError("reg_max must be non-negative")
        self.trident_bins = int(trident_bins)
        if self.trident_bins < 0:
            raise ValueError("trident_bins must be non-negative")
        if self.trident_bins > 0 and self.reg_max > 0:
            raise ValueError("Trident and DFL regression are mutually exclusive")
        if self.trident_bins > 0 and use_boundary_heads:
            raise ValueError("Trident and standalone boundary refinement are mutually exclusive")
        self.center_sampling_radius = float(center_sampling_radius)
        if self.center_sampling_radius < 0:
            raise ValueError("center_sampling_radius must be non-negative")
        self.use_boundary_heads = bool(use_boundary_heads)
        self.boundary_refine_radius_seconds = float(boundary_refine_radius_seconds)
        self.boundary_refine_blend = float(boundary_refine_blend)
        self.tanp_std = float(tanp_std)
        self.tanp_probability = float(tanp_probability)
        self.use_temporal_order = bool(use_temporal_order)
        self.temporal_order_chunks = int(temporal_order_chunks)
        if self.boundary_refine_radius_seconds < 0:
            raise ValueError("boundary_refine_radius_seconds must be non-negative")
        if not 0.0 <= self.boundary_refine_blend <= 1.0:
            raise ValueError("boundary_refine_blend must be in [0,1]")
        if self.tanp_std < 0:
            raise ValueError("tanp_std must be non-negative")
        if not 0.0 <= self.tanp_probability <= 1.0:
            raise ValueError("tanp_probability must be in [0,1]")
        if self.use_temporal_order and self.temporal_order_chunks < 2:
            raise ValueError("Temporal order learning requires at least two chunks")
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            ChannelLayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            ChannelLayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classification_input_projection = (
            nn.Sequential(
                nn.Conv1d(
                    self.classification_input_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                ChannelLayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                ChannelLayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
            if self.classification_input_dim is not None
            else None
        )
        self.classification_tower = TemporalTower(hidden_dim, head_layers - 1, dropout)
        self.regression_tower = TemporalTower(hidden_dim, head_layers - 1, dropout)
        self.classification_head = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        self.quality_head = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        distribution_bins = self.trident_bins or self.reg_max
        regression_channels = (
            2 if distribution_bins == 0 else 2 * (distribution_bins + 1)
        )
        self.regression_head = nn.Conv1d(
            hidden_dim, regression_channels, kernel_size=3, padding=1
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
        self.start_boundary_head = (
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
            if self.use_boundary_heads
            else None
        )
        self.end_boundary_head = (
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
            if self.use_boundary_heads
            else None
        )
        self.regression_scales = nn.ModuleList(Scale() for _ in range(pyramid_levels))

        if neck_type not in {"maxpool", "attention"}:
            raise ValueError("neck_type must be 'maxpool' or 'attention'")
        self.neck_type = neck_type
        if neck_type == "attention":
            self.classification_neck = nn.ModuleList(
                LocalSelfAttentionBlock(hidden_dim, attention_heads, attention_window, dropout)
                for _ in range(pyramid_levels)
            )
            self.regression_neck = nn.ModuleList(
                LocalSelfAttentionBlock(hidden_dim, attention_heads, attention_window, dropout)
                for _ in range(pyramid_levels)
            )
        else:
            self.classification_neck = None
            self.regression_neck = None
        self.temporal_order_permutations = tuple(
            itertools.permutations(range(self.temporal_order_chunks))
        )
        self.temporal_order_head = (
            nn.Sequential(
                nn.Linear(hidden_dim * self.temporal_order_chunks, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, len(self.temporal_order_permutations)),
            )
            if self.use_temporal_order
            else None
        )

        prior_bias = -math.log((1.0 - prior_probability) / prior_probability)
        nn.init.constant_(self.classification_head.bias, prior_bias)
        nn.init.constant_(self.quality_head.bias, prior_bias)
        if self.trident_start_head is not None:
            nn.init.constant_(self.trident_start_head.bias, prior_bias)
            nn.init.constant_(self.trident_end_head.bias, prior_bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        """Run the detector over a batch of ROI timelines.

        Args:
            features: ``[B, T, input_dim]`` per-bin features.
            mask: ``[B, T]`` boolean mask of valid bins; all-valid when None.

        Returns:
            Dictionary of per-level lists: ``classification_logits``, ``quality_logits``,
            ``offsets``, ``masks``, and, when enabled, ``offset_distributions`` and the
            start/end boundary logits.

        Raises:
            ValueError: if ``features`` does not match the configured input dimension.
        """
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [B, T, {self.input_dim}] features, got {tuple(features.shape)}"
            )
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        current_mask = mask.unsqueeze(1)
        regression_current = self.input_projection(
            features.transpose(1, 2)
        ) * current_mask.to(features.dtype)
        if self.classification_input_projection is None:
            classification_current = regression_current
        else:
            classification_current = self.classification_input_projection(
                features[..., : self.classification_input_dim].transpose(1, 2)
            ) * current_mask.to(features.dtype)
        if (
            self.training
            and self.tanp_std > 0
            and torch.rand((), device=regression_current.device)
            < self.tanp_probability
        ):
            regression_current = temporal_aware_normalization_perturbation(
                regression_current,
                current_mask,
                std=self.tanp_std,
            )
            if self.classification_input_projection is None:
                classification_current = regression_current
            else:
                classification_current = temporal_aware_normalization_perturbation(
                    classification_current,
                    current_mask,
                    std=self.tanp_std,
                )

        pyramid_features: list[torch.Tensor] = []
        regression_pyramid_features: list[torch.Tensor] = []
        pyramid_masks: list[torch.Tensor] = []
        for level in range(self.pyramid_levels):
            if level > 0:
                classification_current = F.max_pool1d(
                    classification_current,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                regression_current = F.max_pool1d(
                    regression_current,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                current_mask = F.max_pool1d(
                    current_mask.to(regression_current.dtype),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ).bool()
                classification_current = (
                    classification_current
                    * current_mask.to(classification_current.dtype)
                )
                regression_current = (
                    regression_current * current_mask.to(regression_current.dtype)
                )
            if self.neck_type == "attention":
                classification_current = self.classification_neck[level](
                    classification_current, current_mask
                )
                regression_current = self.regression_neck[level](
                    regression_current, current_mask
                )
            pyramid_features.append(classification_current)
            regression_pyramid_features.append(regression_current)
            pyramid_masks.append(current_mask)

        classification_logits = []
        quality_logits = []
        offsets = []
        offset_distributions = []
        start_boundary_logits = []
        end_boundary_logits = []
        trident_start_logits = []
        trident_end_logits = []
        for level, (
            classification_level_features,
            regression_level_features,
            level_mask,
        ) in enumerate(
            zip(
                pyramid_features,
                regression_pyramid_features,
                pyramid_masks,
            )
        ):
            class_features = self.classification_tower(
                classification_level_features,
                level_mask,
            )
            regression_features = self.regression_tower(
                regression_level_features,
                level_mask,
            )
            classification_logits.append(self.classification_head(class_features).squeeze(1))
            quality_logits.append(self.quality_head(regression_features).squeeze(1))
            raw_offsets = self.regression_scales[level](
                self.regression_head(regression_features)
            )
            if self.trident_bins > 0:
                center_logits = raw_offsets.reshape(
                    raw_offsets.shape[0],
                    2,
                    self.trident_bins + 1,
                    raw_offsets.shape[2],
                ).permute(0, 3, 1, 2)
                start_logits = self.trident_start_head(
                    regression_features.detach()
                ).squeeze(1)
                end_logits = self.trident_end_head(
                    regression_features.detach()
                ).squeeze(1)
                offset = trident_offsets(
                    center_logits,
                    start_logits,
                    end_logits,
                    self.trident_bins,
                )
                offset_distributions.append(center_logits)
                trident_start_logits.append(start_logits)
                trident_end_logits.append(end_logits)
            elif self.reg_max > 0:
                distribution = raw_offsets.reshape(
                    raw_offsets.shape[0], 2, self.reg_max + 1, raw_offsets.shape[2]
                ).permute(0, 3, 1, 2)
                bins = torch.arange(
                    self.reg_max + 1,
                    device=distribution.device,
                    dtype=distribution.dtype,
                )
                offset = (distribution.softmax(dim=3) * bins).sum(dim=3)
                offset_distributions.append(distribution)
            else:
                offset = F.softplus(raw_offsets).transpose(1, 2)
                offset_distributions.append(None)
            offsets.append(offset)
            if self.use_boundary_heads:
                start_boundary_logits.append(
                    self.start_boundary_head(regression_features).squeeze(1)
                )
                end_boundary_logits.append(
                    self.end_boundary_head(regression_features).squeeze(1)
                )

        return {
            "pyramid_features": pyramid_features,
            "regression_pyramid_features": regression_pyramid_features,
            "classification_logits": classification_logits,
            "quality_logits": quality_logits,
            "offsets": offsets,
            "offset_distributions": offset_distributions,
            "start_boundary_logits": start_boundary_logits,
            "end_boundary_logits": end_boundary_logits,
            "trident_start_logits": trident_start_logits,
            "trident_end_logits": trident_end_logits,
            "masks": [value.squeeze(1) for value in pyramid_masks],
        }

    def temporal_order_loss(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict permutations of pooled temporal chunks as in GLAD's TOL."""
        if self.temporal_order_head is None:
            return features.sum() * 0.0
        representations = []
        targets = []
        for sample_index in range(features.shape[0]):
            length = int(mask[sample_index].sum())
            if length < self.temporal_order_chunks:
                continue
            sequence = features[sample_index, :, :length].transpose(0, 1)
            chunks = torch.tensor_split(sequence, self.temporal_order_chunks, dim=0)
            pooled = torch.stack([chunk.mean(dim=0) for chunk in chunks])
            target = int(
                torch.randint(
                    len(self.temporal_order_permutations),
                    (),
                    device=features.device,
                )
            )
            order = self.temporal_order_permutations[target]
            representations.append(pooled[list(order)].reshape(-1))
            targets.append(target)
        if not representations:
            return features.sum() * 0.0
        logits = self.temporal_order_head(torch.stack(representations))
        return F.cross_entropy(
            logits, torch.as_tensor(targets, device=features.device, dtype=torch.long)
        )

    @staticmethod
    def level_points(length: int, stride: int, device: torch.device) -> torch.Tensor:
        """Return the bin centres of one pyramid level, in grid units."""
        return (torch.arange(length, device=device, dtype=torch.float32) + 0.5) * stride

    @torch.no_grad()
    def targets_for_sequence(
        self,
        lengths: list[int],
        gt_segments_grid: torch.Tensor,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Assign ground-truth segments to points of every pyramid level.

        A point is positive when it falls inside a segment whose extent belongs to the
        regression range of its level; ties go to the shortest segment. This is what makes
        each level specialise in a duration band instead of every level chasing every
        action.

        Args:
            lengths: number of points at each pyramid level.
            gt_segments_grid: ``[n, 2]`` ground-truth segments in grid units.
            device: device the targets are built on.

        Returns:
            Per-level classification targets and ``[left, right]`` regression targets.
        """
        class_targets: list[torch.Tensor] = []
        regression_targets: list[torch.Tensor] = []
        for level, length in enumerate(lengths):
            stride = 2**level
            points = self.level_points(length, stride, device)
            if gt_segments_grid.numel() == 0:
                class_targets.append(torch.zeros(length, device=device))
                regression_targets.append(torch.zeros(length, 2, device=device))
                continue
            left = points[:, None] - gt_segments_grid[None, :, 0]
            right = gt_segments_grid[None, :, 1] - points[:, None]
            distances = torch.stack((left, right), dim=2)
            maximum = distances.amax(dim=2)
            lower, upper = self.regression_ranges[level]
            valid = distances.amin(dim=2) >= 0
            valid &= maximum >= lower
            valid &= maximum <= upper
            if self.center_sampling_radius > 0:
                centers = 0.5 * (gt_segments_grid[:, 0] + gt_segments_grid[:, 1])
                radius = self.center_sampling_radius * stride
                center_left = torch.maximum(gt_segments_grid[:, 0], centers - radius)
                center_right = torch.minimum(gt_segments_grid[:, 1], centers + radius)
                valid &= points[:, None] >= center_left[None, :]
                valid &= points[:, None] <= center_right[None, :]
            durations = (gt_segments_grid[:, 1] - gt_segments_grid[:, 0])[None, :]
            costs = durations.expand(length, -1).masked_fill(~valid, float("inf"))
            minimum_cost, gt_indices = costs.min(dim=1)
            positive = torch.isfinite(minimum_cost)
            selected = distances[torch.arange(length, device=device), gt_indices] / stride
            selected[~positive] = 0
            class_targets.append(positive.to(torch.float32))
            regression_targets.append(selected)
        return class_targets, regression_targets

    @torch.no_grad()
    def boundary_targets_for_sequence(
        self,
        lengths: list[int],
        gt_segments_grid: torch.Tensor,
        device: torch.device,
        sigma: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build gaussian start and end target maps for the boundary heads.

        Args:
            lengths: number of points at each pyramid level.
            gt_segments_grid: ``[n, 2]`` ground-truth segments in grid units.
            device: device the targets are built on.
            sigma: width of the gaussian, in points of the level.

        Returns:
            Per-level start and end target maps in ``[0, 1]``.

        Raises:
            ValueError: if ``sigma`` is not positive.
        """
        if sigma <= 0:
            raise ValueError("Boundary target sigma must be positive")
        start_targets = []
        end_targets = []
        for level, length in enumerate(lengths):
            stride = 2**level
            points = self.level_points(length, stride, device)
            if gt_segments_grid.numel() == 0:
                start_targets.append(torch.zeros(length, device=device))
                end_targets.append(torch.zeros(length, device=device))
                continue
            start_distance = (
                points[:, None] - gt_segments_grid[None, :, 0]
            ).abs() / stride
            end_distance = (
                points[:, None] - gt_segments_grid[None, :, 1]
            ).abs() / stride
            start_targets.append(torch.exp(-0.5 * (start_distance / sigma).square()).amax(1))
            end_targets.append(torch.exp(-0.5 * (end_distance / sigma).square()).amax(1))
        return start_targets, end_targets

    def losses(
        self,
        output: dict[str, list[torch.Tensor]],
        gt_segments_seconds: list[torch.Tensor],
        grid_stride_seconds: float,
        regression_weight: float = 1.0,
        quality_weight: float = 0.5,
        distribution_weight: float = 0.25,
        empty_sequence_weight: float = 1.0,
        boundary_weight: float = 0.25,
        boundary_target_sigma: float = 1.0,
        rank_sort: bool = False,
        rank_sort_delta: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Compute the training losses of one batch.

        Args:
            output: dictionary returned by :meth:`forward`.
            gt_segments_seconds: per-sequence ground-truth segments, in seconds.
            grid_stride_seconds: seconds covered by one bin of the finest level.
            regression_weight: weight of the DIoU regression term.
            quality_weight: weight of the quality head.
            distribution_weight: weight of the distribution focal loss (DFL only).
            empty_sequence_weight: weight of sequences with no annotated action, which
                control how hard the model is pushed to stay silent on empty ROIs.
            boundary_weight: weight of the boundary heads.
            boundary_target_sigma: width of the gaussian boundary targets.
            rank_sort: replace focal classification with Rank & Sort loss.
            rank_sort_delta: smoothing of the Rank & Sort step function.

        Returns:
            Dictionary with the total ``loss``, each component, and the number of
            positive points used for normalisation.
        """
        class_logits = output["classification_logits"]
        offsets = output["offsets"]
        masks = output["masks"]
        batch_size = class_logits[0].shape[0]
        class_target_batches = [[] for _ in class_logits]
        regression_target_batches = [[] for _ in class_logits]
        start_target_batches = [[] for _ in class_logits]
        end_target_batches = [[] for _ in class_logits]
        lengths = [value.shape[1] for value in class_logits]
        for gt in gt_segments_seconds:
            class_targets, regression_targets = self.targets_for_sequence(
                lengths, gt.to(class_logits[0].device) / grid_stride_seconds, class_logits[0].device
            )
            for level in range(len(lengths)):
                class_target_batches[level].append(class_targets[level])
                regression_target_batches[level].append(regression_targets[level])
            if self.use_boundary_heads:
                start_targets, end_targets = self.boundary_targets_for_sequence(
                    lengths,
                    gt.to(class_logits[0].device) / grid_stride_seconds,
                    class_logits[0].device,
                    boundary_target_sigma,
                )
                for level in range(len(lengths)):
                    start_target_batches[level].append(start_targets[level])
                    end_target_batches[level].append(end_targets[level])

        class_targets = [torch.stack(values) for values in class_target_batches]
        regression_targets = [torch.stack(values) for values in regression_target_batches]
        positive_count = sum(
            (target.bool() & mask).sum() for target, mask in zip(class_targets, masks)
        ).clamp_min(1).to(class_logits[0].dtype)
        if empty_sequence_weight <= 0:
            raise ValueError("empty_sequence_weight must be positive")
        sample_weights = class_logits[0].new_tensor(
            [empty_sequence_weight if gt.numel() == 0 else 1.0 for gt in gt_segments_seconds]
        )
        if rank_sort:
            rank_logits = []
            rank_targets = []
            for logit, predicted, target, class_target, mask in zip(
                class_logits, offsets, regression_targets, class_targets, masks
            ):
                quality_target = torch.zeros_like(logit, dtype=torch.float32)
                positive = class_target.bool() & mask
                if positive.any():
                    quality_target[positive] = center_iou(
                        predicted[positive].detach(), target[positive]
                    ).clamp_min(1e-4)
                rank_logits.append(logit[mask].float())
                rank_targets.append(quality_target[mask].float())
            classification_loss = sum(
                rank_sort_loss(
                    torch.cat(rank_logits),
                    torch.cat(rank_targets),
                    delta=rank_sort_delta,
                )
            )
        else:
            classification_loss = sum(
                (
                    sigmoid_focal_loss(logit, target)
                    * mask.to(logit.dtype)
                    * sample_weights[:, None]
                ).sum()
                for logit, target, mask in zip(class_logits, class_targets, masks)
            ) / positive_count

        regression_loss = class_logits[0].sum() * 0.0
        distribution_loss = class_logits[0].sum() * 0.0
        quality_loss = class_logits[0].sum() * 0.0
        boundary_loss = class_logits[0].sum() * 0.0
        for level, (predicted, target, class_target, mask) in enumerate(
            zip(offsets, regression_targets, class_targets, masks)
        ):
            positive = class_target.bool() & mask
            if positive.any():
                regression_loss = regression_loss + center_diou_loss(
                    predicted[positive], target[positive]
                ).sum() / positive_count
                if self.reg_max > 0:
                    distribution_loss = distribution_loss + distribution_focal_loss(
                        output["offset_distributions"][level][positive],
                        target[positive],
                        self.reg_max,
                    ).sum() / positive_count
                if self.use_quality:
                    quality_target = center_iou(
                        predicted[positive].detach(), target[positive]
                    )
                    quality_loss = quality_loss + F.binary_cross_entropy_with_logits(
                        output["quality_logits"][level][positive], quality_target, reduction="sum"
                    ) / positive_count
        if self.use_boundary_heads:
            boundary_normalizer = max(sum(len(gt) for gt in gt_segments_seconds), 1)
            start_targets = [torch.stack(values) for values in start_target_batches]
            end_targets = [torch.stack(values) for values in end_target_batches]
            boundary_loss = sum(
                (
                    sigmoid_focal_loss(start_logit, start_target)
                    + sigmoid_focal_loss(end_logit, end_target)
                ).mul(mask.to(start_logit.dtype)).sum()
                for start_logit, end_logit, start_target, end_target, mask in zip(
                    output["start_boundary_logits"],
                    output["end_boundary_logits"],
                    start_targets,
                    end_targets,
                    masks,
                )
            ) / boundary_normalizer

        total = classification_loss + regression_weight * regression_loss
        if self.reg_max > 0:
            total = total + distribution_weight * distribution_loss
        if self.use_quality:
            total = total + quality_weight * quality_loss
        if self.use_boundary_heads:
            total = total + boundary_weight * boundary_loss
        return {
            "loss": total,
            "classification_loss": classification_loss,
            "regression_loss": regression_loss,
            "distribution_loss": distribution_loss,
            "quality_loss": quality_loss,
            "boundary_loss": boundary_loss,
            "positive_points": positive_count.detach(),
        }

    @torch.no_grad()
    def decode(
        self,
        output: dict[str, list[torch.Tensor]],
        grid_stride_seconds: float,
        durations_seconds: torch.Tensor,
        score_threshold: float = 0.01,
        pre_nms_topk: int = 2000,
        quality_power: float = 0.5,
        min_duration_seconds: float = 2.0,
    ) -> list[torch.Tensor]:
        """Turn per-level predictions into scored temporal detections.

        Scores combine classification confidence with the quality head, raised to
        ``quality_power``; boundaries are decoded from the regression offsets and snapped
        towards the boundary maps when refinement is enabled. NMS is left to the caller,
        so the same decoding feeds both proposal-level and detection-level evaluation.

        Args:
            output: dictionary returned by :meth:`forward`.
            grid_stride_seconds: seconds covered by one bin of the finest level.
            durations_seconds: per-sequence duration, used to clamp the end boundary.
            score_threshold: minimum score a point needs to produce a candidate.
            pre_nms_topk: candidates kept per level and per sequence.
            quality_power: exponent applied to the quality score before fusion.
            min_duration_seconds: candidates shorter than this are dropped.

        Returns:
            One ``[n, 3]`` tensor of ``[t_start, t_end, score]`` per sequence.
        """
        decoded: list[torch.Tensor] = []
        batch_size = output["classification_logits"][0].shape[0]
        for batch_index in range(batch_size):
            candidates = []
            for level, (logits, quality, level_offsets_tensor, mask) in enumerate(
                zip(
                    output["classification_logits"],
                    output["quality_logits"],
                    output["offsets"],
                    output["masks"],
                )
            ):
                stride = 2**level
                score = logits[batch_index].sigmoid()
                if self.use_quality:
                    score = score * quality[batch_index].sigmoid().pow(quality_power)
                valid = mask[batch_index] & (score >= score_threshold)
                if not valid.any():
                    continue
                indices = valid.nonzero(as_tuple=True)[0]
                level_scores = score[indices]
                if len(indices) > pre_nms_topk:
                    level_scores, order = level_scores.topk(pre_nms_topk)
                    indices = indices[order]
                points = (indices.to(level_offsets_tensor.dtype) + 0.5) * stride
                level_offsets = level_offsets_tensor[batch_index, indices] * stride
                segments = torch.stack(
                    (points - level_offsets[:, 0], points + level_offsets[:, 1]), dim=1
                )
                if self.use_boundary_heads and self.boundary_refine_radius_seconds > 0:
                    radius_grid = self.boundary_refine_radius_seconds / grid_stride_seconds
                    level_points = self.level_points(
                        logits.shape[1], stride, level_offsets_tensor.device
                    )
                    for boundary_index, boundary_logits in enumerate(
                        (output["start_boundary_logits"], output["end_boundary_logits"])
                    ):
                        probabilities = boundary_logits[level][batch_index].sigmoid()
                        for candidate_index in range(len(segments)):
                            distance = (level_points - segments[candidate_index, boundary_index]).abs()
                            eligible = mask[batch_index] & (distance <= radius_grid)
                            if eligible.any():
                                local = probabilities.masked_fill(~eligible, -1.0)
                                snapped = level_points[int(local.argmax())]
                                segments[candidate_index, boundary_index] = (
                                    (1.0 - self.boundary_refine_blend)
                                    * segments[candidate_index, boundary_index]
                                    + self.boundary_refine_blend * snapped
                                )
                segments = segments * grid_stride_seconds
                segments[:, 0].clamp_(min=0.0)
                segments[:, 1].clamp_(max=float(durations_seconds[batch_index]))
                keep = segments[:, 1] - segments[:, 0] >= min_duration_seconds
                if keep.any():
                    candidates.append(torch.cat((segments[keep], level_scores[keep, None]), dim=1))
            if candidates:
                all_candidates = torch.cat(candidates)
                if len(all_candidates) > pre_nms_topk:
                    order = all_candidates[:, 2].topk(pre_nms_topk).indices
                    all_candidates = all_candidates[order]
                decoded.append(all_candidates)
            else:
                decoded.append(class_logits_empty(output, batch_index))
        return decoded


def class_logits_empty(
    output: dict[str, list[torch.Tensor]], batch_index: int
) -> torch.Tensor:
    """Return an empty detection tensor with the dtype and device of the batch."""
    return output["classification_logits"][0].new_empty((0, 3))
