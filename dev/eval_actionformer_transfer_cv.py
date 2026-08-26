#!/usr/bin/env python3
"""Evaluate domain-agnostic EventPenguins transfers on THUMOS14 OOF folds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionformer-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_one_based_rank = 0.5 * ((cursor + 1) + end)
        ranks[order[cursor:end]] = average_one_based_rank
        cursor = end
    return (ranks / len(values)).astype(np.float32)


def bounded_minimize(
    function,
    lower: float,
    upper: float,
    iterations: int = 64,
) -> float:
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = float(lower)
    right = float(upper)
    first = right - golden_ratio * (right - left)
    second = left + golden_ratio * (right - left)
    first_value = function(first)
    second_value = function(second)
    for _ in range(iterations):
        if first_value <= second_value:
            right = second
            second = first
            second_value = first_value
            first = right - golden_ratio * (right - left)
            first_value = function(first)
        else:
            left = first
            first = second
            first_value = second_value
            second = left + golden_ratio * (right - left)
            second_value = function(second)
    return 0.5 * (left + right)


def classwise_percentile_ranks(
    videos: Sequence[Mapping[str, np.ndarray]],
    values: Sequence[np.ndarray],
) -> List[np.ndarray]:
    lengths = [len(video["labels"]) for video in videos]
    all_labels = np.concatenate([video["labels"] for video in videos])
    all_values = np.concatenate(values)
    ranks = np.zeros(len(all_values), dtype=np.float32)
    for label in np.unique(all_labels):
        indices = np.flatnonzero(all_labels == label)
        ranks[indices] = percentile_rank(all_values[indices])
    boundaries = np.cumsum([0] + lengths)
    return [
        ranks[boundaries[index] : boundaries[index + 1]]
        for index in range(len(videos))
    ]


def fit_classwise_ecdf(
    videos: Sequence[Mapping[str, np.ndarray]],
    values: Sequence[np.ndarray],
    num_classes: int,
) -> List[np.ndarray]:
    labels = np.concatenate([video["labels"] for video in videos])
    merged_values = np.concatenate(values)
    return [
        np.sort(merged_values[labels == label]).astype(np.float32)
        for label in range(num_classes)
    ]


def apply_classwise_ecdf(
    videos: Sequence[Mapping[str, np.ndarray]],
    values: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
) -> List[np.ndarray]:
    transformed = []
    for video, video_values in zip(videos, values):
        ranks = np.full(len(video_values), 0.5, dtype=np.float32)
        for label in np.unique(video["labels"]):
            indices = np.flatnonzero(video["labels"] == label)
            reference = np.asarray(references[int(label)])
            if len(reference) == 0:
                continue
            left = np.searchsorted(
                reference, video_values[indices], side="left"
            )
            right = np.searchsorted(
                reference, video_values[indices], side="right"
            )
            ranks[indices] = (left + right) / (2.0 * len(reference))
        transformed.append(ranks)
    return transformed


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    ).astype(np.float32)


def load_prepared_fold(prepared_root: Path, fold: int) -> List[Dict[str, np.ndarray]]:
    fold_dir = prepared_root / f"fold_{fold}"
    paths = sorted(fold_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No prepared candidates found in {fold_dir}")
    videos = []
    for path in paths:
        with np.load(path) as data:
            videos.append({name: data[name] for name in data.files})
    return videos


def load_all_folds(
    prepared_root: Path, folds: int
) -> List[List[Dict[str, np.ndarray]]]:
    return [load_prepared_fold(prepared_root, fold) for fold in range(folds)]


def fit_temperatures(
    training_folds: Sequence[Sequence[Mapping[str, np.ndarray]]],
    num_classes: int,
) -> np.ndarray:
    temperatures = np.ones(num_classes, dtype=np.float32)
    for label in range(num_classes):
        logits = np.concatenate(
            [
                video["logits"][video["labels"] == label]
                for fold in training_folds
                for video in fold
            ]
        ).astype(np.float64)
        targets = np.concatenate(
            [
                (video["target_tiou"][video["labels"] == label] >= 0.5)
                for fold in training_folds
                for video in fold
            ]
        ).astype(np.float64)
        if len(logits) == 0 or targets.min() == targets.max():
            continue

        def nll(log_temperature: float) -> float:
            temperature = np.exp(log_temperature)
            scaled = logits / temperature
            return float(
                np.mean(np.logaddexp(0.0, scaled) - targets * scaled)
            )

        optimum = bounded_minimize(
            nll,
            np.log(0.25),
            np.log(10.0),
        )
        temperatures[label] = float(np.exp(optimum))
    return temperatures


def load_duration_prior(
    annotation_path: Path,
    split: str,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    log_durations: List[List[float]] = [[] for _ in range(num_classes)]
    for video in data["database"].values():
        if str(video.get("subset", "")).lower() != split.lower():
            continue
        for annotation in video.get("annotations", []):
            duration = (
                float(annotation["segment"][1])
                - float(annotation["segment"][0])
            )
            if duration > 0:
                log_durations[int(annotation["label_id"])].append(
                    float(np.log(duration))
                )
    upper = np.zeros(num_classes, dtype=np.float32)
    scale = np.ones(num_classes, dtype=np.float32)
    for label, values in enumerate(log_durations):
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        upper[label] = float(np.quantile(array, 0.95))
        scale[label] = max(
            float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
            0.1,
        )
    return upper, scale


def duration_penalty(
    video: Mapping[str, np.ndarray],
    upper: np.ndarray,
    scale: np.ndarray,
    gamma: float,
) -> np.ndarray:
    segments = video.get("clipped_segments", video["segments"])
    durations = np.maximum(
        segments[:, 1] - segments[:, 0], 1e-6
    )
    labels = video["labels"]
    excess = np.maximum(
        0.0,
        (np.log(durations) - upper[labels]) / scale[labels],
    )
    return np.exp(-gamma * excess).astype(np.float32)


def temperature_scores(
    videos: Sequence[Mapping[str, np.ndarray]],
    temperatures: np.ndarray,
) -> List[np.ndarray]:
    return [
        sigmoid(video["logits"] / temperatures[video["labels"]])
        for video in videos
    ]


def completeness_scores(
    training_videos: Sequence[Mapping[str, np.ndarray]],
    videos: Sequence[Mapping[str, np.ndarray]],
    weight: float,
    num_classes: int,
) -> List[np.ndarray]:
    completeness_index = list(videos[0]["feature_names"]).index("completeness")
    training_raw = [video["scores"] for video in training_videos]
    validation_raw = [video["scores"] for video in videos]
    raw_references = fit_classwise_ecdf(
        training_videos, training_raw, num_classes
    )
    original = apply_classwise_ecdf(videos, validation_raw, raw_references)
    training_completeness = [
        video["features"][:, completeness_index] for video in training_videos
    ]
    validation_completeness = [
        video["features"][:, completeness_index] for video in videos
    ]
    completeness_references = fit_classwise_ecdf(
        training_videos, training_completeness, num_classes
    )
    completeness = apply_classwise_ecdf(
        videos, validation_completeness, completeness_references
    )
    return [
        (1.0 - weight) * raw_rank + weight * completeness_rank
        for raw_rank, completeness_rank in zip(original, completeness)
    ]


def combine_duration(
    videos: Sequence[Mapping[str, np.ndarray]],
    base_scores: Sequence[np.ndarray],
    upper: np.ndarray,
    scale: np.ndarray,
    gamma: float,
) -> List[np.ndarray]:
    return [
        score * duration_penalty(video, upper, scale, gamma)
        for video, score in zip(videos, base_scores)
    ]


def classwise_segment_voting(
    selected_segments: torch.Tensor,
    selected_labels: torch.Tensor,
    all_segments: torch.Tensor,
    all_scores: torch.Tensor,
    all_labels: torch.Tensor,
    threshold: float,
    segment_voting,
) -> torch.Tensor:
    refined = selected_segments.clone()
    for label in torch.unique(selected_labels):
        selected_indices = torch.where(selected_labels == label)[0]
        all_indices = torch.where(all_labels == label)[0]
        refined[selected_indices] = segment_voting(
            selected_segments[selected_indices],
            all_segments[all_indices],
            all_scores[all_indices],
            threshold,
        )
    return refined


def postprocess_video(
    video: Mapping[str, np.ndarray],
    scores: np.ndarray,
    *,
    batched_nms,
    segment_voting,
    nms_method: str,
    sigma: float,
    voting_threshold: float,
    min_score: float = 0.001,
    max_segments: int = 200,
) -> Dict[str, Any]:
    segments = torch.from_numpy(video["segments"].astype(np.float32))
    score_tensor = torch.from_numpy(scores.astype(np.float32))
    labels = torch.from_numpy(video["labels"].astype(np.int64))
    valid = score_tensor > min_score
    segments = segments[valid]
    score_tensor = score_tensor[valid]
    labels = labels[valid]

    if nms_method == "none":
        order = score_tensor.argsort(descending=True)[:max_segments]
        selected_segments = segments[order]
        selected_scores = score_tensor[order]
        selected_labels = labels[order]
    else:
        selected_segments, selected_scores, selected_labels = batched_nms(
            segments,
            score_tensor,
            labels,
            0.1,
            min_score,
            max_segments,
            use_soft_nms=(nms_method == "soft"),
            multiclass=True,
            sigma=sigma,
            voting_thresh=0.0,
        )
    if voting_threshold > 0 and len(selected_segments):
        selected_segments = classwise_segment_voting(
            selected_segments,
            selected_labels,
            segments,
            score_tensor,
            labels,
            voting_threshold,
            segment_voting,
        )
    duration = float(video["video_duration"])
    selected_segments = selected_segments.clamp(min=0.0, max=duration)
    return {
        "video_id": str(video["video_id"]),
        "segments": selected_segments,
        "scores": selected_scores,
        "labels": selected_labels,
    }


def prediction_dict(outputs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "video-id": [
            output["video_id"]
            for output in outputs
            for _ in range(len(output["scores"]))
        ],
        "t-start": torch.cat(
            [output["segments"][:, 0] for output in outputs]
        ).numpy(),
        "t-end": torch.cat(
            [output["segments"][:, 1] for output in outputs]
        ).numpy(),
        "label": torch.cat(
            [output["labels"] for output in outputs]
        ).numpy(),
        "score": torch.cat(
            [output["scores"] for output in outputs]
        ).numpy(),
    }


def evaluate_variant(
    *,
    videos: Sequence[Mapping[str, np.ndarray]],
    scores: Sequence[np.ndarray],
    annotation_path: Path,
    split: str,
    evaluator_class,
    batched_nms,
    segment_voting,
    nms_method: str,
    sigma: float,
    voting_threshold: float,
    prediction_path: Path,
) -> Dict[str, Any]:
    outputs = [
        postprocess_video(
            video,
            video_scores,
            batched_nms=batched_nms,
            segment_voting=segment_voting,
            nms_method=nms_method,
            sigma=sigma,
            voting_threshold=voting_threshold,
        )
        for video, video_scores in zip(videos, scores)
    ]
    predictions = prediction_dict(outputs)
    evaluator = evaluator_class(
        str(annotation_path),
        split,
        tiou_thresholds=np.linspace(0.3, 0.7, 5),
    )
    map_by_tiou, average_map, _ = evaluator.evaluate(
        predictions, verbose=False
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        video_id=np.asarray(predictions["video-id"]),
        t_start=predictions["t-start"],
        t_end=predictions["t-end"],
        label=predictions["label"],
        score=predictions["score"],
    )
    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    gt_instances = sum(
        len(video.get("annotations", []))
        for video in annotation_data["database"].values()
        if str(video.get("subset", "")).lower() == split.lower()
    )
    return {
        "mAP": float(average_map),
        "AP@0.3": float(map_by_tiou[0]),
        "AP@0.4": float(map_by_tiou[1]),
        "AP@0.5": float(map_by_tiou[2]),
        "AP@0.6": float(map_by_tiou[3]),
        "AP@0.7": float(map_by_tiou[4]),
        "predictions": int(len(predictions["score"])),
        "gt_instances": int(gt_instances),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    variants = sorted({str(row["variant"]) for row in rows})
    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        weights = np.asarray(
            [row["gt_instances"] for row in selected], dtype=np.float64
        )
        output = {
            "variant": variant,
            "mean_mAP": float(np.mean([row["mAP"] for row in selected])),
            "weighted_mAP": float(
                np.average([row["mAP"] for row in selected], weights=weights)
            ),
            "worst_mAP": float(min(row["mAP"] for row in selected)),
            "mean_AP@0.3": float(
                np.mean([row["AP@0.3"] for row in selected])
            ),
            "mean_AP@0.4": float(
                np.mean([row["AP@0.4"] for row in selected])
            ),
            "mean_AP@0.5": float(
                np.mean([row["AP@0.5"] for row in selected])
            ),
            "mean_AP@0.6": float(
                np.mean([row["AP@0.6"] for row in selected])
            ),
            "mean_AP@0.7": float(
                np.mean([row["AP@0.7"] for row in selected])
            ),
            "mean_predictions": float(
                np.mean([row["predictions"] for row in selected])
            ),
        }
        summary.append(output)

    summary.sort(key=lambda row: row["mean_mAP"], reverse=True)
    baseline = next(
        row for row in summary if row["variant"] == "official_soft05"
    )
    for row in summary:
        row["delta_mean_mAP"] = row["mean_mAP"] - baseline["mean_mAP"]
        row["passes_gates"] = bool(
            row["mean_mAP"] > baseline["mean_mAP"]
            and row["weighted_mAP"] >= baseline["weighted_mAP"]
            and row["worst_mAP"] >= baseline["worst_mAP"] - 0.005
            and row["mean_AP@0.7"] >= baseline["mean_AP@0.7"]
        )
    return summary


def main() -> None:
    args = parse_args()
    actionformer_root = args.actionformer_root.expanduser().resolve()
    sys.path.insert(0, str(actionformer_root))
    os.chdir(actionformer_root)
    from libs.utils import ANETdetection, batched_nms
    from libs.utils.nms import seg_voting

    all_folds = load_all_folds(args.prepared_root, args.folds)
    num_classes = int(
        max(
            int(video["labels"].max())
            for fold in all_folds
            for video in fold
        )
        + 1
    )
    rows = []
    for fold, videos in enumerate(all_folds):
        annotation_path = (
            args.annotation_root
            / f"thumos14_10classes_transfer_fold{fold}.json"
        )
        split = f"transfer_val_{fold}"
        training_folds = [
            candidate_fold
            for index, candidate_fold in enumerate(all_folds)
            if index != fold
        ]
        training_videos = [
            video for training_fold in training_folds for video in training_fold
        ]
        temperatures = fit_temperatures(training_folds, num_classes)
        duration_upper, duration_scale = load_duration_prior(
            annotation_path,
            f"transfer_train_{fold}",
            num_classes,
        )

        raw_scores = [video["scores"] for video in videos]
        calibrated_scores = temperature_scores(videos, temperatures)
        completeness_by_weight = {
            weight: completeness_scores(
                training_videos, videos, weight, num_classes
            )
            for weight in (0.10, 0.25, 0.50)
        }
        variants = [
            ("official_soft05", raw_scores, "soft", 0.50, 0.0),
            ("nms_hard", raw_scores, "hard", 0.50, 0.0),
            ("nms_none", raw_scores, "none", 0.50, 0.0),
            ("soft025", raw_scores, "soft", 0.25, 0.0),
            ("soft075", raw_scores, "soft", 0.75, 0.0),
            ("temperature_oof", calibrated_scores, "soft", 0.50, 0.0),
        ]
        for threshold in (0.50, 0.70, 0.90):
            variants.append(
                (
                    f"raw_voting{threshold:g}",
                    raw_scores,
                    "soft",
                    0.50,
                    threshold,
                )
            )
        for gamma in (0.50, 1.00):
            variants.append(
                (
                    f"duration_g{gamma:g}",
                    combine_duration(
                        videos,
                        raw_scores,
                        duration_upper,
                        duration_scale,
                        gamma,
                    ),
                    "soft",
                    0.50,
                    0.0,
                )
            )
        for weight, scores in completeness_by_weight.items():
            variants.append(
                (
                    f"completeness_w{weight:g}",
                    scores,
                    "soft",
                    0.50,
                    0.0,
                )
            )
        for gamma in (0.50, 1.00):
            variants.append(
                (
                    f"completeness025_duration_g{gamma:g}",
                    combine_duration(
                        videos,
                        completeness_by_weight[0.25],
                        duration_upper,
                        duration_scale,
                        gamma,
                    ),
                    "soft",
                    0.50,
                    0.0,
                )
            )
        for threshold in (0.50, 0.70, 0.90):
            variants.append(
                (
                    f"completeness025_voting{threshold:g}",
                    completeness_by_weight[0.25],
                    "soft",
                    0.50,
                    threshold,
                )
            )

        for name, scores, nms_method, sigma, voting_threshold in variants:
            metrics = evaluate_variant(
                videos=videos,
                scores=scores,
                annotation_path=annotation_path,
                split=split,
                evaluator_class=ANETdetection,
                batched_nms=batched_nms,
                segment_voting=seg_voting,
                nms_method=nms_method,
                sigma=sigma,
                voting_threshold=voting_threshold,
                prediction_path=(
                    args.output_dir
                    / "predictions"
                    / f"fold_{fold}_{name}.npz"
                ),
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": name,
                    "nms_method": nms_method,
                    "sigma": sigma,
                    "voting_threshold": voting_threshold,
                    **metrics,
                }
            )
            print(
                f"fold={fold} variant={name} "
                f"mAP={metrics['mAP']:.6f} AP@0.7={metrics['AP@0.7']:.6f}"
            )

        (args.output_dir / "calibration").mkdir(
            parents=True, exist_ok=True
        )
        (args.output_dir / "calibration" / f"fold_{fold}.json").write_text(
            json.dumps(
                {
                    "temperature_by_class": temperatures.tolist(),
                    "duration_log_q95_by_class": duration_upper.tolist(),
                    "duration_log_iqr_by_class": duration_scale.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    write_csv(args.output_dir / "fold_metrics.csv", rows)
    summary = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
