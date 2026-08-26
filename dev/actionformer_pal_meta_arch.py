"""ActionFormer meta-architecture with PAL region consistency."""

from __future__ import annotations

import torch

from actionformer_pal_utils import pal_contrastive_loss
from libs.modeling.meta_archs import PtTransformer
from libs.modeling.models import register_meta_arch


@register_meta_arch("LocPointTransformerPAL")
class PtTransformerPAL(PtTransformer):
    def __init__(self, *args, train_cfg, **kwargs):
        super().__init__(*args, train_cfg=train_cfg, **kwargs)
        self.pal_consistency_weight = float(
            train_cfg.get("pal_consistency_weight", 0.0)
        )
        self.pal_consistency_temperature = float(
            train_cfg.get("pal_consistency_temperature", 0.07)
        )
        if self.pal_consistency_weight < 0:
            raise ValueError("pal_consistency_weight must be non-negative")
        if self.pal_consistency_temperature <= 0:
            raise ValueError(
                "pal_consistency_temperature must be positive"
            )
        if self.fpn_strides[0] != 1:
            raise ValueError("PAL consistency requires a stride-one first FPN")

    def forward(self, video_list):
        if not self.training or self.pal_consistency_weight == 0:
            return super().forward(video_list)
        captured = {}

        def capture_neck(_module, _inputs, output):
            captured["features"] = output[0]

        hook = self.neck.register_forward_hook(capture_neck)
        try:
            losses = super().forward(video_list)
        finally:
            hook.remove()

        pair_indices = [
            index
            for index, item in enumerate(video_list)
            if item.get("pal_donor_feats") is not None
        ]
        if not pair_indices:
            zero = losses["final_loss"] * 0.0
            return {**losses, "pal_consistency_loss": zero}

        donor_videos = [
            {"feats": video_list[index]["pal_donor_feats"]}
            for index in pair_indices
        ]
        donor_inputs, donor_masks = self.preprocessing(donor_videos)
        donor_features, donor_backbone_masks = self.backbone(
            donor_inputs, donor_masks
        )
        donor_fpn, donor_fpn_masks = self.neck(
            donor_features, donor_backbone_masks
        )

        recipient_embeddings = []
        donor_embeddings = []
        recipient_level = captured["features"][0]
        donor_level = donor_fpn[0]
        donor_level_mask = donor_fpn_masks[0].squeeze(1)
        for donor_index, recipient_index in enumerate(pair_indices):
            start, end = (
                video_list[recipient_index]["pal_recipient_span"].tolist()
            )
            end = min(end, recipient_level.shape[-1])
            if end <= start:
                continue
            recipient_embeddings.append(
                recipient_level[
                    recipient_index, :, start:end
                ].mean(dim=-1)
            )
            valid = donor_level_mask[donor_index]
            donor_embeddings.append(
                donor_level[donor_index, :, valid].mean(dim=-1)
            )
        if not recipient_embeddings:
            consistency = losses["final_loss"] * 0.0
        else:
            consistency = pal_contrastive_loss(
                torch.stack(recipient_embeddings),
                torch.stack(donor_embeddings),
                self.pal_consistency_temperature,
            )
        return {
            **losses,
            "pal_consistency_loss": consistency,
            "final_loss": (
                losses["final_loss"]
                + self.pal_consistency_weight * consistency
            ),
        }
