#!/usr/bin/env python3
"""Evaluate 20 one-vs-rest THUMOS14-E outputs with ActionFormer's evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actionformer-root", type=Path, required=True)
    parser.add_argument("--canonical-annotations", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument(
        "--prediction-name",
        default="predictions.json",
        help="Expected path is predictions-root/<class>/<prediction-name>.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_test_protocol(
    annotation_path: Path,
) -> tuple[set[str], dict[str, int], dict[str, object]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    database = payload["database"]
    test_videos = {
        video_id
        for video_id, video in database.items()
        if str(video.get("subset", "")).lower() == "test"
    }
    label_to_id = {
        str(annotation["label"]): int(annotation["label_id"])
        for video in database.values()
        for annotation in video.get("annotations", [])
    }
    return test_videos, label_to_id, payload


def prediction_rows(
    predictions_root: Path,
    prediction_name: str,
    test_videos: set[str],
    label_to_id: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    files = {}
    excluded_videos: set[str] = set()
    for label in sorted(label_to_id, key=label_to_id.get):
        path = predictions_root / label / prediction_name
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = payload.get("target_class")
        if declared not in {None, label}:
            raise ValueError(f"{path} declares target_class={declared!r}, expected {label!r}")
        files[label] = {"path": str(path), "sha256": sha256_file(path)}
        for video_id, rois in payload.get("results", {}).items():
            if video_id not in test_videos:
                excluded_videos.add(str(video_id))
                continue
            for detections in rois.values():
                for detection in detections:
                    start, stop = map(float, detection["segment"])
                    if not 0 <= start < stop:
                        raise ValueError(f"Invalid detection in {path}: {detection}")
                    rows.append(
                        {
                            "video-id": str(video_id),
                            "t-start": start,
                            "t-end": stop,
                            "label": int(label_to_id[label]),
                            "score": float(detection["score"]),
                        }
                    )
    frame = pd.DataFrame(
        rows,
        columns=["video-id", "t-start", "t-end", "label", "score"],
    )
    audit = {
        "prediction_files": files,
        "detections": len(frame),
        "excluded_noncanonical_videos": sorted(excluded_videos),
    }
    return frame, audit


def evaluate(args: argparse.Namespace) -> None:
    actionformer_root = args.actionformer_root.expanduser().resolve()
    annotation_path = args.canonical_annotations.expanduser().resolve()
    predictions_root = args.predictions_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(actionformer_root))
    from libs.utils import ANETdetection  # type: ignore

    test_videos, label_to_id, _ = canonical_test_protocol(annotation_path)
    if len(test_videos) != 212 or len(label_to_id) != 20:
        raise ValueError(
            f"Canonical THUMOS14 universe changed: videos={len(test_videos)} "
            f"classes={len(label_to_id)}"
        )
    predictions, prediction_audit = prediction_rows(
        predictions_root,
        args.prediction_name,
        test_videos,
        label_to_id,
    )
    thresholds = np.asarray([0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
    evaluator = ANETdetection(
        str(annotation_path),
        split="test",
        tiou_thresholds=thresholds,
        num_workers=args.num_workers,
        dataset_name="THUMOS14-E canonical ActionFormer protocol",
    )
    mAP, average_mAP, recall = evaluator.evaluate(predictions, verbose=True)

    rows = []
    for label, label_id in sorted(label_to_id.items(), key=lambda item: item[1]):
        column = evaluator.activity_index[label_id]
        row = {"class": label, "label_id": label_id}
        row.update(
            {
                f"AP@{threshold:.1f}": float(evaluator.ap[index, column])
                for index, threshold in enumerate(thresholds)
            }
        )
        row["average_AP"] = float(np.mean(evaluator.ap[:, column]))
        rows.append(row)
    per_class = pd.DataFrame(rows)
    per_class.to_csv(out_dir / "per_class_ap.csv", index=False)
    predictions.to_csv(out_dir / "canonical_predictions.csv", index=False)

    summary = {
        "protocol": "THUMOS14-E-ActionFormer-canonical-eval-v1",
        "canonical_annotations": str(annotation_path),
        "canonical_annotations_sha256": sha256_file(annotation_path),
        "canonical_test_videos": len(test_videos),
        "classes": len(label_to_id),
        "ambiguous_policy": (
            "match ActionFormer canonical JSON: ambiguous-only video excluded; "
            "no extra overlap mask"
        ),
        "tiou_thresholds": thresholds.tolist(),
        "mAP": {f"{threshold:.1f}": float(value) for threshold, value in zip(thresholds, mAP)},
        "average_mAP": float(average_mAP),
        "mean_recall": np.asarray(recall).tolist(),
        "prediction_audit": prediction_audit,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    evaluate(parse_args())
