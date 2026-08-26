"""Opt-in supervised Pseudo Action Localization dataset for ActionFormer."""

from __future__ import annotations

import os

import numpy as np
import torch

from actionformer_pal_utils import valid_background_starts
from libs.datasets.datasets import register_dataset
from libs.datasets.thumos14 import THUMOS14Dataset


@register_dataset("thumos_pal")
class THUMOS14PALDataset(THUMOS14Dataset):
    def __init__(
        self,
        *args,
        pal_probability=0.5,
        pal_margin=2,
        pal_blend_min=1.0,
        pal_blend_max=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= pal_probability <= 1.0:
            raise ValueError("pal_probability must lie in [0, 1]")
        if pal_margin < 0:
            raise ValueError("pal_margin must be non-negative")
        if not 0.0 <= pal_blend_min <= pal_blend_max <= 1.0:
            raise ValueError("invalid PAL blend interval")
        self.pal_probability = float(pal_probability)
        self.pal_margin = int(pal_margin)
        self.pal_blend_min = float(pal_blend_min)
        self.pal_blend_max = float(pal_blend_max)
        self.positive_indices = np.asarray(
            [
                index
                for index, video in enumerate(self.data_list)
                if video["segments"] is not None and len(video["segments"])
            ],
            dtype=np.int64,
        )

    def _load_donor(self, recipient_video_id):
        eligible = [
            int(index)
            for index in self.positive_indices
            if self.data_list[int(index)]["id"] != recipient_video_id
        ]
        if not eligible:
            return None
        donor = self.data_list[int(np.random.choice(eligible))]
        annotation_index = int(np.random.randint(len(donor["segments"])))
        filename = os.path.join(
            self.feat_folder,
            self.file_prefix + donor["id"] + self.file_ext,
        )
        features = np.load(filename).astype(np.float32)[
            :: self.downsample_rate
        ]
        feature_stride = self.feat_stride * self.downsample_rate
        feature_offset = 0.5 * self.num_frames / feature_stride
        segment = (
            donor["segments"][annotation_index]
            * donor["fps"]
            / feature_stride
            - feature_offset
        )
        crop_start = max(int(np.floor(segment[0])), 0)
        crop_end = min(int(np.ceil(segment[1])), len(features))
        if crop_end <= crop_start:
            return None
        crop = torch.from_numpy(
            np.ascontiguousarray(features[crop_start:crop_end].transpose())
        )
        relative_segment = np.clip(
            segment - crop_start, 0.0, crop_end - crop_start
        ).astype(np.float32)
        if relative_segment[1] <= relative_segment[0]:
            return None
        return (
            crop,
            torch.from_numpy(relative_segment),
            int(donor["labels"][annotation_index]),
            donor["id"],
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if (
            not self.is_training
            or self.pal_probability == 0
            or np.random.random() >= self.pal_probability
        ):
            return item
        donor = self._load_donor(item["video_id"])
        if donor is None:
            return item
        donor_features, relative_segment, donor_label, donor_video_id = donor
        recipient_segments = item["segments"].detach().cpu().numpy()
        valid_starts = valid_background_starts(
            recipient_segments,
            item["feats"].shape[-1],
            donor_features.shape[-1],
            self.pal_margin,
        )
        if len(valid_starts) == 0:
            return item
        recipient_start = int(np.random.choice(valid_starts))
        recipient_end = recipient_start + donor_features.shape[-1]
        beta = float(
            np.random.uniform(self.pal_blend_min, self.pal_blend_max)
        )
        features = item["feats"].clone()
        features[:, recipient_start:recipient_end] = (
            beta * donor_features
            + (1.0 - beta) * features[:, recipient_start:recipient_end]
        )
        transplanted_segment = relative_segment + recipient_start
        item["feats"] = features
        item["segments"] = torch.cat(
            (item["segments"], transplanted_segment.unsqueeze(0)), dim=0
        )
        item["labels"] = torch.cat(
            (
                item["labels"],
                torch.as_tensor([donor_label], dtype=torch.int64),
            ),
            dim=0,
        )
        item["pal_donor_feats"] = donor_features
        item["pal_recipient_span"] = torch.as_tensor(
            [recipient_start, recipient_end], dtype=torch.int64
        )
        item["pal_donor_video_id"] = donor_video_id
        return item
