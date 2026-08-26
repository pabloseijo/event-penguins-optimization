#!/usr/bin/env python3
"""Paired video-cluster bootstrap for OOF temporal detection predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


TIOU_THRESHOLDS = np.linspace(0.3, 0.7, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-variant", required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--selected-variant", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def temporal_iou(segment: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    intersection = np.maximum(
        0.0,
        np.minimum(segment[1], candidates[:, 1])
        - np.maximum(segment[0], candidates[:, 0]),
    )
    union = (
        segment[1]
        - segment[0]
        + candidates[:, 1]
        - candidates[:, 0]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def interpolated_average_precision(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    positives: float,
) -> float:
    if positives <= 0:
        return np.nan
    cumulative_tp = np.cumsum(true_positive, dtype=np.float64)
    cumulative_fp = np.cumsum(false_positive, dtype=np.float64)
    recall = cumulative_tp / positives
    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros_like(cumulative_tp),
        where=(cumulative_tp + cumulative_fp) > 0,
    )
    interpolated_precision = np.hstack(([0.0], precision, [0.0]))
    interpolated_recall = np.hstack(([0.0], recall, [1.0]))
    for index in range(len(interpolated_precision) - 2, -1, -1):
        interpolated_precision[index] = max(
            interpolated_precision[index],
            interpolated_precision[index + 1],
        )
    changes = np.where(
        interpolated_recall[1:] != interpolated_recall[:-1]
    )[0] + 1
    return float(
        np.sum(
            (
                interpolated_recall[changes]
                - interpolated_recall[changes - 1]
            )
            * interpolated_precision[changes]
        )
    )


def build_class_outcome(
    *,
    ground_truth: Mapping[str, np.ndarray],
    prediction_video_ids: np.ndarray,
    prediction_segments: np.ndarray,
    prediction_scores: np.ndarray,
    video_to_index: Mapping[str, int],
    thresholds: np.ndarray = TIOU_THRESHOLDS,
) -> Dict[str, np.ndarray]:
    order = np.argsort(prediction_scores)[::-1]
    video_ids = prediction_video_ids[order].astype(str)
    segments = prediction_segments[order]
    video_indices = np.asarray(
        [video_to_index[video_id] for video_id in video_ids],
        dtype=np.int64,
    )
    true_positive = np.zeros((len(thresholds), len(order)), dtype=np.float64)
    false_positive = np.zeros_like(true_positive)
    locks = {
        video_id: np.zeros(
            (len(thresholds), len(video_ground_truth)), dtype=bool
        )
        for video_id, video_ground_truth in ground_truth.items()
    }
    for prediction_index, (video_id, segment) in enumerate(
        zip(video_ids, segments)
    ):
        video_ground_truth = ground_truth.get(video_id)
        if video_ground_truth is None or len(video_ground_truth) == 0:
            false_positive[:, prediction_index] = 1.0
            continue
        overlap = temporal_iou(segment, video_ground_truth)
        overlap_order = np.argsort(overlap)[::-1]
        for threshold_index, threshold in enumerate(thresholds):
            for ground_truth_index in overlap_order:
                if overlap[ground_truth_index] < threshold:
                    false_positive[threshold_index, prediction_index] = 1.0
                    break
                if locks[video_id][threshold_index, ground_truth_index]:
                    continue
                true_positive[threshold_index, prediction_index] = 1.0
                locks[video_id][threshold_index, ground_truth_index] = True
                break
            if (
                true_positive[threshold_index, prediction_index] == 0
                and false_positive[threshold_index, prediction_index] == 0
            ):
                false_positive[threshold_index, prediction_index] = 1.0
    ground_truth_counts = np.zeros(len(video_to_index), dtype=np.float64)
    for video_id, segments_for_video in ground_truth.items():
        ground_truth_counts[video_to_index[video_id]] = len(segments_for_video)
    return {
        "video_indices": video_indices,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "ground_truth_counts": ground_truth_counts,
    }


def weighted_class_ap(
    outcome: Mapping[str, np.ndarray],
    video_counts: np.ndarray,
) -> np.ndarray:
    prediction_weights = video_counts[outcome["video_indices"]]
    positives = float(
        np.dot(video_counts, outcome["ground_truth_counts"])
    )
    return np.asarray(
        [
            interpolated_average_precision(
                true_positive * prediction_weights,
                false_positive * prediction_weights,
                positives,
            )
            for true_positive, false_positive in zip(
                outcome["true_positive"], outcome["false_positive"]
            )
        ],
        dtype=np.float64,
    )


def prediction_path(root: Path, fold: int, variant: str) -> Path:
    return root / f"fold_{fold}_{variant}.npz"


def load_predictions(
    root: Path, variant: str, folds: int
) -> Dict[str, np.ndarray]:
    parts: Dict[str, List[np.ndarray]] = {
        "video_id": [],
        "t_start": [],
        "t_end": [],
        "label": [],
        "score": [],
    }
    for fold in range(folds):
        with np.load(prediction_path(root, fold, variant)) as prediction:
            for key in parts:
                parts[key].append(prediction[key])
    return {key: np.concatenate(values) for key, values in parts.items()}


def load_ground_truth(
    annotation_root: Path,
    folds: int,
    num_classes: int,
) -> Tuple[List[str], List[Dict[str, np.ndarray]]]:
    by_class: List[Dict[str, np.ndarray]] = [
        {} for _ in range(num_classes)
    ]
    video_ids = []
    for fold in range(folds):
        data = json.loads(
            (
                annotation_root
                / f"thumos14_10classes_transfer_fold{fold}.json"
            ).read_text(encoding="utf-8")
        )
        split = f"transfer_val_{fold}"
        for video_id, video in data["database"].items():
            if str(video.get("subset", "")).lower() != split.lower():
                continue
            video_ids.append(video_id)
            for label in range(num_classes):
                by_class[label][video_id] = np.asarray(
                    [
                        annotation["segment"]
                        for annotation in video.get("annotations", [])
                        if int(annotation["label_id"]) == label
                    ],
                    dtype=np.float64,
                ).reshape(-1, 2)
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("OOF video IDs are not disjoint")
    return sorted(video_ids), by_class


def build_outcomes(
    prediction: Mapping[str, np.ndarray],
    ground_truth_by_class: Sequence[Mapping[str, np.ndarray]],
    video_to_index: Mapping[str, int],
) -> List[Dict[str, np.ndarray]]:
    prediction_video_ids = prediction["video_id"].astype(str)
    prediction_segments = np.column_stack(
        (prediction["t_start"], prediction["t_end"])
    ).astype(np.float64)
    labels = prediction["label"].astype(np.int64)
    return [
        build_class_outcome(
            ground_truth=ground_truth,
            prediction_video_ids=prediction_video_ids[labels == label],
            prediction_segments=prediction_segments[labels == label],
            prediction_scores=prediction["score"][labels == label],
            video_to_index=video_to_index,
        )
        for label, ground_truth in enumerate(ground_truth_by_class)
    ]


def mean_ap(
    outcomes: Sequence[Mapping[str, np.ndarray]],
    video_counts: np.ndarray,
) -> float:
    per_class = np.stack(
        [weighted_class_ap(outcome, video_counts) for outcome in outcomes]
    )
    return float(np.nanmean(per_class))


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    baseline = load_predictions(
        args.baseline_root, args.baseline_variant, args.folds
    )
    selected = load_predictions(
        args.selected_root, args.selected_variant, args.folds
    )
    num_classes = int(
        max(baseline["label"].max(), selected["label"].max()) + 1
    )
    video_ids, ground_truth_by_class = load_ground_truth(
        args.annotation_root, args.folds, num_classes
    )
    video_to_index = {
        video_id: index for index, video_id in enumerate(video_ids)
    }
    baseline_outcomes = build_outcomes(
        baseline, ground_truth_by_class, video_to_index
    )
    selected_outcomes = build_outcomes(
        selected, ground_truth_by_class, video_to_index
    )
    observed_counts = np.ones(len(video_ids), dtype=np.int64)
    observed_baseline = mean_ap(baseline_outcomes, observed_counts)
    observed_selected = mean_ap(selected_outcomes, observed_counts)

    rng = np.random.default_rng(args.seed)
    deltas = np.empty(args.samples, dtype=np.float64)
    completed = 0
    while completed < args.samples:
        counts = rng.multinomial(
            len(video_ids),
            np.full(len(video_ids), 1.0 / len(video_ids)),
        )
        if any(
            np.dot(counts, outcome["ground_truth_counts"]) == 0
            for outcome in baseline_outcomes
        ):
            continue
        deltas[completed] = (
            mean_ap(selected_outcomes, counts)
            - mean_ap(baseline_outcomes, counts)
        )
        completed += 1

    report: Dict[str, Any] = {
        "videos": len(video_ids),
        "classes": num_classes,
        "thresholds": TIOU_THRESHOLDS.tolist(),
        "samples": args.samples,
        "seed": args.seed,
        "baseline_variant": args.baseline_variant,
        "selected_variant": args.selected_variant,
        "observed_baseline_mAP": observed_baseline,
        "observed_selected_mAP": observed_selected,
        "observed_delta_mAP": observed_selected - observed_baseline,
        "bootstrap_delta_mean": float(deltas.mean()),
        "bootstrap_delta_median": float(np.median(deltas)),
        "bootstrap_delta_ci95": np.quantile(
            deltas, [0.025, 0.975]
        ).tolist(),
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
