#!/usr/bin/env python3
"""Evaluate the frozen THUMOS14 transfer recipe exactly once on test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from bootstrap_actionformer_transfer_oof import (
    TIOU_THRESHOLDS,
    build_class_outcome,
    build_outcomes,
    mean_ap,
)
from eval_actionformer_transfer_cv import postprocess_video, prediction_dict
from run_actionformer_transfer_final import load_inference_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionformer-root", type=Path, required=True)
    parser.add_argument("--prepared-inference-dir", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--selected-predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234567891)
    return parser.parse_args()


def file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def npz_prediction(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"video_id", "t_start", "t_end", "label", "score"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Prediction file lacks fields: {sorted(missing)}")
        return {key: data[key] for key in sorted(required)}


def actionformer_prediction(
    prediction: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    return {
        "video-id": prediction["video_id"].astype(str).tolist(),
        "t-start": prediction["t_start"],
        "t-end": prediction["t_end"],
        "label": prediction["label"],
        "score": prediction["score"],
    }


def save_prediction(path: Path, prediction: Mapping[str, Any]) -> None:
    np.savez_compressed(
        path,
        video_id=np.asarray(prediction["video-id"]),
        t_start=np.asarray(prediction["t-start"]),
        t_end=np.asarray(prediction["t-end"]),
        label=np.asarray(prediction["label"]),
        score=np.asarray(prediction["score"]),
    )


def load_ground_truth(
    annotation_path: Path,
    split: str,
    num_classes: int,
) -> Tuple[List[str], List[Dict[str, np.ndarray]], List[str], int]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    video_ids = sorted(
        video_id
        for video_id, video in data["database"].items()
        if str(video.get("subset", "")).lower() == split.lower()
    )
    if not video_ids:
        raise ValueError(f"No videos found for split {split!r}")

    class_names = [f"class_{label}" for label in range(num_classes)]
    by_class: List[Dict[str, np.ndarray]] = [
        {} for _ in range(num_classes)
    ]
    instances = 0
    for video_id in video_ids:
        video = data["database"][video_id]
        annotations = video.get("annotations", [])
        instances += len(annotations)
        for annotation in annotations:
            label = int(annotation["label_id"])
            if not 0 <= label < num_classes:
                raise ValueError(f"Unexpected label ID {label}")
            class_names[label] = str(annotation["label"])
        for label in range(num_classes):
            by_class[label][video_id] = np.asarray(
                [
                    annotation["segment"]
                    for annotation in annotations
                    if int(annotation["label_id"]) == label
                ],
                dtype=np.float64,
            ).reshape(-1, 2)
    return video_ids, by_class, class_names, instances


def evaluate_prediction(
    evaluator_class,
    prediction: Mapping[str, np.ndarray],
    annotation_path: Path,
    split: str,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    evaluator = evaluator_class(
        str(annotation_path),
        split,
        tiou_thresholds=TIOU_THRESHOLDS,
    )
    map_by_tiou, average_map, _ = evaluator.evaluate(
        actionformer_prediction(prediction), verbose=False
    )
    per_class = {}
    for original_label, column in evaluator.activity_index.items():
        name = class_names[int(original_label)]
        values = evaluator.ap[:, int(column)]
        per_class[name] = {
            "AP_by_tIoU": values.tolist(),
            "mAP": float(values.mean()),
        }
    return {
        "mAP": float(average_map),
        "AP_by_tIoU": map_by_tiou.tolist(),
        "per_class": per_class,
        "predictions": int(len(prediction["score"])),
    }


def calibration_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    bins: int,
) -> Dict[str, Any]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if scores.shape != targets.shape or scores.ndim != 1:
        raise ValueError("scores and targets must be aligned one-dimensional arrays")
    if len(scores) == 0:
        raise ValueError("calibration requires at least one prediction")
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    bin_indices = np.minimum((clipped * bins).astype(np.int64), bins - 1)
    ece = 0.0
    bin_report = []
    for index in range(bins):
        selected = bin_indices == index
        count = int(selected.sum())
        if count == 0:
            continue
        confidence = float(clipped[selected].mean())
        accuracy = float(targets[selected].mean())
        ece += count / len(clipped) * abs(accuracy - confidence)
        bin_report.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_confidence": confidence,
                "precision": accuracy,
            }
        )
    return {
        "threshold_tIoU": 0.5,
        "predictions": int(len(clipped)),
        "positive_rate": float(targets.mean()),
        "ECE": float(ece),
        "Brier": float(np.mean((clipped - targets) ** 2)),
        "NLL": float(
            -np.mean(
                targets * np.log(clipped)
                + (1.0 - targets) * np.log(1.0 - clipped)
            )
        ),
        "bins": bin_report,
        "note": (
            "Soft-NMS scores are ranking confidences; these Bernoulli-style "
            "metrics are descriptive rather than a training objective."
        ),
    }


def detection_calibration(
    prediction: Mapping[str, np.ndarray],
    ground_truth_by_class: Sequence[Mapping[str, np.ndarray]],
    video_to_index: Mapping[str, int],
    bins: int,
) -> Dict[str, Any]:
    prediction_video_ids = prediction["video_id"].astype(str)
    prediction_segments = np.column_stack(
        (prediction["t_start"], prediction["t_end"])
    ).astype(np.float64)
    labels = prediction["label"].astype(np.int64)
    ordered_scores = []
    ordered_targets = []
    for label, ground_truth in enumerate(ground_truth_by_class):
        selected = labels == label
        class_scores = prediction["score"][selected]
        order = np.argsort(class_scores)[::-1]
        outcome = build_class_outcome(
            ground_truth=ground_truth,
            prediction_video_ids=prediction_video_ids[selected],
            prediction_segments=prediction_segments[selected],
            prediction_scores=class_scores,
            video_to_index=video_to_index,
            thresholds=np.asarray([0.5]),
        )
        ordered_scores.append(class_scores[order])
        ordered_targets.append(outcome["true_positive"][0])
    return calibration_metrics(
        np.concatenate(ordered_scores),
        np.concatenate(ordered_targets),
        bins,
    )


def paired_video_bootstrap(
    baseline: Mapping[str, np.ndarray],
    selected: Mapping[str, np.ndarray],
    video_ids: Sequence[str],
    ground_truth_by_class: Sequence[Mapping[str, np.ndarray]],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
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

    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    completed = 0
    while completed < samples:
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
    return {
        "unit": "video",
        "samples": samples,
        "seed": seed,
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


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            f"Refusing to overwrite final evaluation: {args.output_dir}"
        )
    actionformer_root = args.actionformer_root.expanduser().resolve()
    sys.path.insert(0, str(actionformer_root))
    os.chdir(actionformer_root)
    from libs.utils import ANETdetection, batched_nms
    from libs.utils.nms import seg_voting

    videos = load_inference_videos(args.prepared_inference_dir)
    if args.baseline_predictions is None:
        baseline_outputs = [
            postprocess_video(
                video,
                video["scores"],
                batched_nms=batched_nms,
                segment_voting=seg_voting,
                nms_method="soft",
                sigma=0.5,
                voting_threshold=0.0,
            )
            for video in videos
        ]
        baseline_actionformer = prediction_dict(baseline_outputs)
        baseline = {
            "video_id": np.asarray(baseline_actionformer["video-id"]),
            "t_start": baseline_actionformer["t-start"],
            "t_end": baseline_actionformer["t-end"],
            "label": baseline_actionformer["label"],
            "score": baseline_actionformer["score"],
        }
        baseline_source = {
            "kind": "reconstructed_from_pre_nms_candidates",
            "soft_nms_sigma": 0.5,
            "voting_threshold": 0.0,
        }
    else:
        baseline = npz_prediction(args.baseline_predictions)
        baseline_source = {
            "kind": "canonical_prediction_artifact",
            "artifact": file_identity(args.baseline_predictions),
        }
    selected = npz_prediction(args.selected_predictions)
    expected_video_ids = {
        str(video["video_id"]) for video in videos
    }
    if set(selected["video_id"].astype(str)) != expected_video_ids:
        raise ValueError("Selected predictions do not cover the inference videos")
    if set(baseline["video_id"].astype(str)) != expected_video_ids:
        raise ValueError("Baseline predictions do not cover the inference videos")

    num_classes = int(
        max(baseline["label"].max(), selected["label"].max()) + 1
    )
    video_ids, ground_truth, class_names, instances = load_ground_truth(
        args.annotations, args.split, num_classes
    )
    video_to_index = {
        video_id: index for index, video_id in enumerate(video_ids)
    }
    baseline_metrics = evaluate_prediction(
        ANETdetection,
        baseline,
        args.annotations,
        args.split,
        class_names,
    )
    selected_metrics = evaluate_prediction(
        ANETdetection,
        selected,
        args.annotations,
        args.split,
        class_names,
    )
    class_delta = {
        name: (
            selected_metrics["per_class"][name]["mAP"]
            - baseline_metrics["per_class"][name]["mAP"]
        )
        for name in class_names
    }
    report = {
        "status": "frozen_final_test_evaluation",
        "split": args.split,
        "videos": len(video_ids),
        "classes": num_classes,
        "gt_instances": instances,
        "tIoU_thresholds": TIOU_THRESHOLDS.tolist(),
        "checkpoint": file_identity(args.checkpoint),
        "annotations": file_identity(args.annotations),
        "baseline_prediction_source": baseline_source,
        "selected_prediction_artifact": file_identity(
            args.selected_predictions
        ),
        "baseline": baseline_metrics,
        "selected": selected_metrics,
        "delta_mAP": (
            selected_metrics["mAP"] - baseline_metrics["mAP"]
        ),
        "class_delta_mAP": class_delta,
        "classes_improved": int(
            sum(delta > 0 for delta in class_delta.values())
        ),
        "bootstrap": paired_video_bootstrap(
            baseline,
            selected,
            video_ids,
            ground_truth,
            args.bootstrap_samples,
            args.seed,
        ),
        "calibration": {
            "baseline": detection_calibration(
                baseline, ground_truth, video_to_index, args.calibration_bins
            ),
            "selected": detection_calibration(
                selected, ground_truth, video_to_index, args.calibration_bins
            ),
        },
        "protocol": {
            "selection": "five video-disjoint OOF validation folds",
            "test_tuning": False,
            "baseline": (
                "frozen canonical artifact"
                if args.baseline_predictions is not None
                else "raw ActionFormer + Soft-NMS sigma=0.5, no voting"
            ),
            "selected": (
                "linear QFL continuous tIoU target + geometric score + "
                "Soft-NMS sigma=0.5 + classwise boundary voting=0.5"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_prediction(
        args.output_dir / "baseline_predictions.npz",
        actionformer_prediction(baseline),
    )
    (args.output_dir / "final_test_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
