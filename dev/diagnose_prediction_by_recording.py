"""Diagnose localization ceiling and ranking errors of a prediction JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation import DetectionsEvaluator, segment_iou


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def best_overlap(segment: list[float], candidates: list[list[float]]) -> float:
    if not candidates:
        return 0.0
    return float(segment_iou(np.asarray(segment), np.asarray(candidates)).max())


def main() -> None:
    args = parse_args()
    ann_path = resolve(args.ann_path)
    prediction_path = resolve(args.prediction)
    database = json.loads(ann_path.read_text(encoding="utf-8"))["database"]
    results = json.loads(prediction_path.read_text(encoding="utf-8"))["results"]
    rows = []
    thresholds = np.asarray(args.tiou, dtype=np.float64)
    for recording, roi_predictions in sorted(results.items()):
        gt_by_roi: dict[int, list[list[float]]] = {}
        for roi, annotations in database[recording]["annotations"].items():
            if roi == "null":
                continue
            gt_by_roi[int(roi)] = [
                list(map(float, annotation["segment"]))
                for annotation in annotations
                if annotation["label"] == "ed"
                and float(annotation["segment"][1])
                - float(annotation["segment"][0])
                >= 2.0
            ]
        detections = [
            (int(roi), detection)
            for roi, values in roi_predictions.items()
            for detection in values
            if detection["label"] == "ed"
        ]
        gt_overlaps = [
            best_overlap(
                segment,
                [
                    detection["segment"]
                    for detection_roi, detection in detections
                    if detection_roi == roi
                ],
            )
            for roi, segments in gt_by_roi.items()
            for segment in segments
        ]
        detection_overlaps = np.asarray(
            [
                best_overlap(detection["segment"], gt_by_roi.get(roi, []))
                for roi, detection in detections
            ],
            dtype=np.float64,
        )
        scores = np.asarray(
            [float(detection["score"]) for _, detection in detections],
            dtype=np.float64,
        )
        if not gt_overlaps:
            rows.append(
                {
                    "rec_name": recording,
                    "gt": 0,
                    "predictions": len(detections),
                    "mAP": float("nan"),
                }
            )
            continue
        evaluator = DetectionsEvaluator(
            ground_truth_filename=str(ann_path),
            prediction_filename=str(prediction_path),
            tiou_thresholds=thresholds,
            valid_labels=["ed"],
            valid_sequences=[recording],
            min_duration=2.0,
        )
        evaluator.run()
        row: dict[str, float | int | str] = {
            "rec_name": recording,
            "gt": len(gt_overlaps),
            "predictions": len(detections),
            "mAP": float(np.mean(evaluator.mAP)),
        }
        for threshold, ap in zip(thresholds, evaluator.mAP):
            suffix = f"{threshold:.1f}"
            row[f"AP@{suffix}"] = float(ap)
            row[f"recall@{suffix}"] = float(
                np.mean(np.asarray(gt_overlaps) >= threshold)
            )
        positive = detection_overlaps >= 0.5
        row["median_tp05_score"] = float(np.median(scores[positive])) if positive.any() else 0.0
        row["median_fp05_score"] = float(np.median(scores[~positive])) if (~positive).any() else 0.0
        row["fp_above_tp05_median"] = int(
            np.sum((~positive) & (scores >= row["median_tp05_score"]))
        )
        rows.append(row)
    output = pd.DataFrame(rows).sort_values("mAP")
    path = resolve(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
