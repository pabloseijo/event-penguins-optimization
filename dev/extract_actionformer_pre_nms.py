#!/usr/bin/env python3
"""Export restartable ActionFormer candidates before NMS and global truncation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionformer-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def file_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def feature_grid_to_seconds(
    values: torch.Tensor,
    *,
    feat_stride: float,
    feat_num_frames: float,
    fps: float,
) -> torch.Tensor:
    return (values * feat_stride + 0.5 * feat_num_frames) / fps


def decode_level_candidates(
    *,
    logits: torch.Tensor,
    offsets: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
    level: int,
    num_classes: int,
    pre_nms_threshold: float,
    pre_nms_topk: int,
    duration_threshold: float,
    feat_stride: float,
    feat_num_frames: float,
    fps: float,
    video_duration: float,
) -> Dict[str, torch.Tensor]:
    probabilities = (logits.sigmoid() * mask.unsqueeze(-1)).flatten()
    keep = probabilities > pre_nms_threshold
    probabilities = probabilities[keep]
    flattened_indices = keep.nonzero(as_tuple=True)[0]
    number_topk = min(pre_nms_topk, flattened_indices.numel())
    probabilities, order = probabilities.sort(descending=True)
    probabilities = probabilities[:number_topk]
    flattened_indices = flattened_indices[order[:number_topk]]

    point_indices = torch.div(
        flattened_indices, num_classes, rounding_mode="floor"
    )
    labels = torch.fmod(flattened_indices, num_classes)
    selected_offsets = offsets[point_indices]
    selected_points = points[point_indices]
    selected_logits = logits.flatten()[flattened_indices]

    left = (
        selected_points[:, 0]
        - selected_offsets[:, 0] * selected_points[:, 3]
    )
    right = (
        selected_points[:, 0]
        + selected_offsets[:, 1] * selected_points[:, 3]
    )
    keep_duration = (right - left) > duration_threshold
    left = left[keep_duration]
    right = right[keep_duration]
    segments = torch.stack((left, right), dim=-1)
    segments = feature_grid_to_seconds(
        segments,
        feat_stride=feat_stride,
        feat_num_frames=feat_num_frames,
        fps=fps,
    )
    point_times = feature_grid_to_seconds(
        selected_points[keep_duration, 0],
        feat_stride=feat_stride,
        feat_num_frames=feat_num_frames,
        fps=fps,
    ).clamp(min=0.0, max=video_duration)
    point_strides = (
        selected_points[keep_duration, 3] * feat_stride / fps
    )

    count = int(keep_duration.sum().item())
    return {
        "segments": segments,
        "scores": probabilities[keep_duration],
        "logits": selected_logits[keep_duration],
        "labels": labels[keep_duration],
        "levels": torch.full(
            (count,), level, dtype=torch.int64, device=segments.device
        ),
        "point_indices": point_indices[keep_duration],
        "point_times": point_times,
        "point_strides": point_strides,
        "offsets": selected_offsets[keep_duration],
    }


def concatenate_candidates(
    candidates: Sequence[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    keys = candidates[0].keys()
    return {
        key: torch.cat([candidate[key] for candidate in candidates], dim=0)
        for key in keys
    }


@torch.inference_mode()
def extract_video(model, video: MappingLike) -> Dict[str, np.ndarray]:
    batched_inputs, batched_masks = model.preprocessing([video])
    features, masks = model.backbone(batched_inputs, batched_masks)
    fpn_features, fpn_masks = model.neck(features, masks)
    points = model.point_generator(fpn_features)
    class_logits = model.cls_head(fpn_features, fpn_masks)
    offsets = model.reg_head(fpn_features, fpn_masks)

    class_logits = [item.permute(0, 2, 1)[0] for item in class_logits]
    offsets = [item.permute(0, 2, 1)[0] for item in offsets]
    fpn_masks = [item.squeeze(1)[0] for item in fpn_masks]

    fps = float(video["fps"])
    duration = float(video["duration"])
    feat_stride = float(video["feat_stride"])
    feat_num_frames = float(video["feat_num_frames"])
    candidates = [
        decode_level_candidates(
            logits=logits,
            offsets=level_offsets,
            points=level_points,
            mask=level_mask,
            level=level,
            num_classes=model.num_classes,
            pre_nms_threshold=model.test_pre_nms_thresh,
            pre_nms_topk=model.test_pre_nms_topk,
            duration_threshold=model.test_duration_thresh,
            feat_stride=feat_stride,
            feat_num_frames=feat_num_frames,
            fps=fps,
            video_duration=duration,
        )
        for level, (logits, level_offsets, level_points, level_mask) in enumerate(
            zip(class_logits, offsets, points, fpn_masks)
        )
    ]
    merged = concatenate_candidates(candidates)

    dense_mask = fpn_masks[0]
    dense_logits = class_logits[0][dense_mask]
    dense_points = points[0][dense_mask]
    dense_times = feature_grid_to_seconds(
        dense_points[:, 0],
        feat_stride=feat_stride,
        feat_num_frames=feat_num_frames,
        fps=fps,
    ).clamp(min=0.0, max=duration)

    output = {
        key: value.detach().cpu().numpy()
        for key, value in merged.items()
    }
    output.update(
        {
            "video_id": np.asarray(str(video["video_id"])),
            "video_duration": np.asarray(duration, dtype=np.float32),
            "fps": np.asarray(fps, dtype=np.float32),
            "feat_stride": np.asarray(feat_stride, dtype=np.float32),
            "feat_num_frames": np.asarray(
                feat_num_frames, dtype=np.float32
            ),
            "dense_logits": dense_logits.detach().cpu().numpy(),
            "dense_times": dense_times.detach().cpu().numpy(),
        }
    )
    return output


MappingLike = Dict[str, Any]


def prepare(args: argparse.Namespace):
    root = args.actionformer_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)

    from libs.core import load_config
    from libs.datasets import make_dataset
    from libs.modeling import make_meta_arch
    from libs.utils import fix_random_seed

    config = load_config(str(config_path))
    validation_splits = [
        str(split).lower() for split in config["val_split"]
    ]
    if "test" in validation_splits and not args.allow_test:
        raise ValueError(
            "Refusing to export the test split without --allow-test"
        )
    dataset = make_dataset(
        config["dataset_name"],
        False,
        config["val_split"],
        **config["dataset"],
    )
    _ = fix_random_seed(0, include_cuda=True)
    model = make_meta_arch(config["model_name"], **config["model"])
    parallel_model = torch.nn.DataParallel(
        model, device_ids=config["devices"]
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=config["devices"][0],
        weights_only=False,
    )
    parallel_model.load_state_dict(checkpoint["state_dict_ema"])
    model = parallel_model.module.to(config["devices"][0]).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        config,
        dataset,
        model,
        config_path,
        checkpoint_path,
        output_dir,
    )


def main() -> None:
    args = parse_args()
    (
        config,
        dataset,
        model,
        config_path,
        checkpoint_path,
        output_dir,
    ) = prepare(args)
    end = len(dataset) if args.end is None else args.end
    if not 0 <= args.start <= end <= len(dataset):
        raise ValueError(
            f"Invalid range [{args.start}, {end}) for {len(dataset)} videos"
        )

    exported: List[str] = []
    skipped: List[str] = []
    for index in range(args.start, end):
        video = dataset[index]
        video_id = str(video["video_id"])
        output_path = output_dir / f"{video_id}.npz"
        if output_path.exists() and not args.overwrite:
            skipped.append(video_id)
            continue
        arrays = extract_video(model, video)
        np.savez_compressed(output_path, **arrays)
        exported.append(video_id)
        print(
            f"[{index + 1}/{end}] {video_id}: "
            f"{arrays['segments'].shape[0]} candidates"
        )

    metadata = {
        "actionformer_root": str(args.actionformer_root.expanduser().resolve()),
        "config": str(config_path),
        "checkpoint": file_identity(checkpoint_path),
        "validation_split": config["val_split"],
        "dataset_length": len(dataset),
        "range": [args.start, end],
        "pre_nms_threshold": model.test_pre_nms_thresh,
        "pre_nms_topk_per_level": model.test_pre_nms_topk,
        "duration_threshold": model.test_duration_thresh,
        "test_export_explicitly_allowed": bool(args.allow_test),
        "exported": exported,
        "skipped_existing": skipped,
    }
    metadata_path = output_dir / f"metadata_{args.start:03d}_{end:03d}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
