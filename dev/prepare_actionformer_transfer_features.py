#!/usr/bin/env python3
"""Prepare domain-agnostic ActionFormer candidate features and tIoU targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


FEATURE_NAMES = (
    "raw_logit",
    "raw_score",
    "log_duration",
    "level",
    "log_point_stride",
    "offset_left",
    "offset_right",
    "offset_asymmetry",
    "inside_mean",
    "inside_std",
    "left_context_mean",
    "right_context_mean",
    "completeness",
    "start_contrast",
    "end_contrast",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-ratio", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    return parser.parse_args()


def temporal_iou(
    segments: np.ndarray, ground_truth: np.ndarray
) -> np.ndarray:
    if len(segments) == 0 or len(ground_truth) == 0:
        return np.zeros((len(segments), len(ground_truth)), dtype=np.float32)
    intersection = np.maximum(
        0.0,
        np.minimum(segments[:, None, 1], ground_truth[None, :, 1])
        - np.maximum(segments[:, None, 0], ground_truth[None, :, 0]),
    )
    segment_duration = segments[:, 1] - segments[:, 0]
    gt_duration = ground_truth[:, 1] - ground_truth[:, 0]
    union = (
        segment_duration[:, None]
        + gt_duration[None, :]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def quality_targets(
    segments: np.ndarray,
    labels: np.ndarray,
    gt_segments: np.ndarray,
    gt_labels: np.ndarray,
) -> np.ndarray:
    targets = np.zeros(len(segments), dtype=np.float32)
    for label in np.unique(labels):
        candidate_indices = np.flatnonzero(labels == label)
        class_gt = gt_segments[gt_labels == label]
        if len(class_gt) == 0:
            continue
        overlaps = temporal_iou(segments[candidate_indices], class_gt)
        targets[candidate_indices] = overlaps.max(axis=1)
    return targets


def interval_mean_std(
    times: np.ndarray,
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    fallback: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.searchsorted(times, starts, side="left")
    right = np.searchsorted(times, ends, side="left")
    counts = right - left
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    prefix_square = np.concatenate(
        ([0.0], np.cumsum(np.square(values), dtype=np.float64))
    )
    sums = prefix[right] - prefix[left]
    square_sums = prefix_square[right] - prefix_square[left]
    valid = counts > 0
    means = np.zeros(len(starts), dtype=np.float64)
    means[valid] = sums[valid] / counts[valid]
    if fallback is None:
        centers = 0.5 * (starts + ends)
        means[~valid] = np.interp(
            centers[~valid],
            times,
            values,
            left=float(values[0]),
            right=float(values[-1]),
        )
    else:
        means[~valid] = fallback[~valid]
    variances = np.zeros(len(starts), dtype=np.float64)
    variances[valid] = (
        square_sums[valid] / counts[valid] - np.square(means[valid])
    )
    standard_deviations = np.sqrt(np.maximum(variances, 0.0))
    return (
        means.astype(np.float32),
        standard_deviations.astype(np.float32),
        counts,
    )


def temporal_statistics(
    *,
    dense_times: np.ndarray,
    dense_logits: np.ndarray,
    segments: np.ndarray,
    labels: np.ndarray,
    context_ratio: float,
) -> Dict[str, np.ndarray]:
    if context_ratio <= 0:
        raise ValueError("context_ratio must be positive")
    output = {
        name: np.zeros(len(segments), dtype=np.float32)
        for name in (
            "inside_mean",
            "inside_std",
            "left_context_mean",
            "right_context_mean",
            "completeness",
            "start_contrast",
            "end_contrast",
        )
    }
    durations = np.maximum(segments[:, 1] - segments[:, 0], 1e-6)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        probabilities = 1.0 / (
            1.0 + np.exp(-dense_logits[:, int(label)].astype(np.float64))
        )
        starts = segments[indices, 0]
        ends = segments[indices, 1]
        context = durations[indices] * context_ratio
        inside_mean, inside_std, _ = interval_mean_std(
            dense_times, probabilities, starts, ends
        )
        left_mean, _, _ = interval_mean_std(
            dense_times,
            probabilities,
            starts - context,
            starts,
            fallback=inside_mean,
        )
        right_mean, _, _ = interval_mean_std(
            dense_times,
            probabilities,
            ends,
            ends + context,
            fallback=inside_mean,
        )
        edge = np.minimum(context, durations[indices])
        start_inside, _, _ = interval_mean_std(
            dense_times,
            probabilities,
            starts,
            starts + edge,
            fallback=inside_mean,
        )
        end_inside, _, _ = interval_mean_std(
            dense_times,
            probabilities,
            ends - edge,
            ends,
            fallback=inside_mean,
        )
        output["inside_mean"][indices] = inside_mean
        output["inside_std"][indices] = inside_std
        output["left_context_mean"][indices] = left_mean
        output["right_context_mean"][indices] = right_mean
        output["completeness"][indices] = (
            inside_mean - 0.5 * (left_mean + right_mean)
        )
        output["start_contrast"][indices] = start_inside - left_mean
        output["end_contrast"][indices] = end_inside - right_mean
    return output


def load_ground_truth(
    annotation_path: Path, split: str
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    ground_truth = {}
    for video_id, video in raw["database"].items():
        if str(video.get("subset", "")).lower() != split.lower():
            continue
        annotations = video.get("annotations", [])
        ground_truth[video_id] = (
            np.asarray(
                [annotation["segment"] for annotation in annotations],
                dtype=np.float32,
            ).reshape(-1, 2),
            np.asarray(
                [annotation["label_id"] for annotation in annotations],
                dtype=np.int64,
            ),
        )
    return ground_truth, raw


def load_split_video_ids(annotation_path: Path, split: str) -> set:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    return {
        video_id
        for video_id, video in raw["database"].items()
        if str(video.get("subset", "")).lower() == split.lower()
    }


def prepare_video(
    raw: Mapping[str, np.ndarray],
    gt_segments: np.ndarray,
    gt_labels: np.ndarray,
    *,
    context_ratio: float,
    include_target: bool = True,
) -> Dict[str, np.ndarray]:
    segments = raw["segments"].astype(np.float32)
    video_duration = float(raw["video_duration"])
    clipped_segments = np.clip(segments, 0.0, video_duration)
    labels = raw["labels"].astype(np.int64)
    statistics = temporal_statistics(
        dense_times=raw["dense_times"].astype(np.float32),
        dense_logits=raw["dense_logits"].astype(np.float32),
        segments=clipped_segments,
        labels=labels,
        context_ratio=context_ratio,
    )
    durations = np.maximum(
        clipped_segments[:, 1] - clipped_segments[:, 0], 1e-6
    )
    features = np.column_stack(
        (
            raw["logits"],
            raw["scores"],
            np.log(durations),
            raw["levels"],
            np.log(np.maximum(raw["point_strides"], 1e-6)),
            raw["offsets"][:, 0],
            raw["offsets"][:, 1],
            np.abs(raw["offsets"][:, 0] - raw["offsets"][:, 1]),
            statistics["inside_mean"],
            statistics["inside_std"],
            statistics["left_context_mean"],
            statistics["right_context_mean"],
            statistics["completeness"],
            statistics["start_contrast"],
            statistics["end_contrast"],
        )
    ).astype(np.float32)
    prepared = {
        "video_id": raw["video_id"],
        "video_duration": raw["video_duration"],
        "segments": segments,
        "clipped_segments": clipped_segments,
        "scores": raw["scores"].astype(np.float32),
        "logits": raw["logits"].astype(np.float32),
        "labels": labels,
        "features": features,
        "feature_names": np.asarray(FEATURE_NAMES),
    }
    if include_target:
        prepared["target_tiou"] = quality_targets(
            clipped_segments, labels, gt_segments, gt_labels
        )
    return prepared


def main() -> None:
    args = parse_args()
    if args.split.lower() == "test" and not args.inference_only:
        raise ValueError(
            "Refusing to create target_tiou from test annotations; "
            "use --inference-only"
        )
    if args.inference_only:
        expected_video_ids = load_split_video_ids(
            args.annotations, args.split
        )
        ground_truth = {}
    else:
        ground_truth, _ = load_ground_truth(args.annotations, args.split)
        expected_video_ids = set(ground_truth)
    raw_paths = sorted(args.raw_dir.glob("*.npz"))
    raw_video_ids = {path.stem for path in raw_paths}
    if raw_video_ids != expected_video_ids:
        missing = sorted(expected_video_ids - raw_video_ids)
        extra = sorted(raw_video_ids - expected_video_ids)
        raise ValueError(
            f"Raw export/split mismatch; missing={missing}, extra={extra}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for raw_path in raw_paths:
        output_path = args.output_dir / raw_path.name
        if output_path.exists() and not args.overwrite:
            with np.load(output_path) as existing:
                counts[raw_path.stem] = int(existing["segments"].shape[0])
            continue
        with np.load(raw_path) as raw_file:
            raw = {name: raw_file[name] for name in raw_file.files}
        gt_segments, gt_labels = ground_truth.get(
            raw_path.stem,
            (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            ),
        )
        prepared = prepare_video(
            raw,
            gt_segments,
            gt_labels,
            context_ratio=args.context_ratio,
            include_target=not args.inference_only,
        )
        np.savez_compressed(output_path, **prepared)
        counts[raw_path.stem] = int(prepared["segments"].shape[0])
        print(f"{raw_path.stem}: {counts[raw_path.stem]} candidates")

    report = {
        "raw_dir": str(args.raw_dir.resolve()),
        "annotations": str(args.annotations.resolve()),
        "split": args.split,
        "context_ratio": args.context_ratio,
        "feature_names": list(FEATURE_NAMES),
        "videos": len(counts),
        "candidates": sum(counts.values()),
        "inference_only": bool(args.inference_only),
        "targets_written": not args.inference_only,
        "target_uses_test": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
