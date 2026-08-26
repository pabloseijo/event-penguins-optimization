"""Classify ranked temporal-detection errors after proposal post-processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_quality_head import ROOT, build_prediction
from src.evaluation import segment_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify temporal detection errors by recording.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scored-proposals")
    source.add_argument("--prediction-json")
    parser.add_argument("--score-col")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--match-tiou", type=float, default=0.5)
    parser.add_argument("--localization-tiou", type=float, default=0.1)
    parser.add_argument("--flap-tiou", type=float, default=0.5)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--use-boundary-refinement", action="store_true", default=True)
    parser.add_argument("--no-boundary-refinement", dest="use_boundary_refinement", action="store_false")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def annotation_index(path: Path, recordings: set[str]) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    with open(path, encoding="utf-8") as handle:
        database = json.load(handle)["database"]
    result = {}
    for recording in recordings:
        for roi, annotations in database[recording]["annotations"].items():
            if roi == "null":
                continue
            labels: dict[str, list[list[float]]] = {"ed": [], "flap": []}
            for annotation in annotations:
                start, end = map(float, annotation["segment"])
                if end - start < 2.0:
                    continue
                label = annotation["label"]
                if label == "ed":
                    labels["ed"].append([start, end])
                elif "flap" in label:
                    labels["flap"].append([start, end])
            result[(recording, int(roi))] = {
                label: np.asarray(segments, dtype=np.float64).reshape(-1, 2)
                for label, segments in labels.items()
            }
    return result


def best_overlap(segment: np.ndarray, candidates: np.ndarray) -> tuple[float, int]:
    if len(candidates) == 0:
        return 0.0, -1
    overlaps = segment_iou(segment, candidates)
    index = int(np.argmax(overlaps))
    return float(overlaps[index]), index


def main() -> None:
    args = parse_args()
    if args.prediction_json:
        with open(resolve(args.prediction_json), encoding="utf-8") as handle:
            prediction = json.load(handle)
        recordings = set(prediction["results"])
    else:
        if not args.score_col:
            raise ValueError("--score-col is required with --scored-proposals")
        proposals = pd.read_csv(resolve(args.scored_proposals))
        recordings = set(proposals["rec_name"].unique())
        prediction = build_prediction(proposals, args.score_col, args.min_score, args)
    annotations = annotation_index(resolve(args.ann_path), recordings)

    rows = []
    for recording, rois in prediction["results"].items():
        for roi, detections in rois.items():
            targets = annotations.get((recording, int(roi)), {"ed": np.empty((0, 2)), "flap": np.empty((0, 2))})
            matched: set[int] = set()
            for rank, detection in enumerate(sorted(detections, key=lambda item: item["score"], reverse=True), 1):
                segment = np.asarray(detection["segment"], dtype=np.float64)
                ed_iou, ed_index = best_overlap(segment, targets["ed"])
                flap_iou, _ = best_overlap(segment, targets["flap"])
                if ed_iou >= args.match_tiou and ed_index not in matched:
                    error_type = "true_positive"
                    matched.add(ed_index)
                elif ed_iou >= args.match_tiou:
                    error_type = "duplicate"
                elif ed_iou >= args.localization_tiou:
                    error_type = "localization"
                elif flap_iou >= args.flap_tiou:
                    error_type = "flap_confusion"
                else:
                    error_type = "background"
                rows.append(
                    {
                        "rec_name": recording,
                        "roi_id": int(roi),
                        "rank_in_roi": rank,
                        "score": float(detection["score"]),
                        "t_start": float(segment[0]),
                        "t_end": float(segment[1]),
                        "ed_tiou": ed_iou,
                        "flap_tiou": flap_iou,
                        "error_type": error_type,
                    }
                )

    details = pd.DataFrame(rows)
    details["rank_global"] = (
        details["score"].rank(method="first", ascending=False).astype(int)
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_dir / "detection_errors.csv", index=False)
    summary = (
        details.groupby(["rec_name", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["true_positive", "duplicate", "localization", "flap_confusion", "background"]:
        if column not in summary:
            summary[column] = 0
    summary["false_positives"] = summary[["duplicate", "localization", "flap_confusion", "background"]].sum(axis=1)
    summary.to_csv(out_dir / "summary.csv", index=False)
    harmful_rows = []
    for recording, group in details.groupby("rec_name", sort=True):
        true_scores = group.loc[group["error_type"] == "true_positive", "score"]
        threshold = float(true_scores.median()) if len(true_scores) else float("inf")
        harmful = group[
            (group["error_type"] != "true_positive") & (group["score"] >= threshold)
        ]
        counts = harmful["error_type"].value_counts()
        harmful_rows.append(
            {
                "rec_name": recording,
                "median_tp_score": threshold,
                "harmful_fp": len(harmful),
                "duplicate": int(counts.get("duplicate", 0)),
                "localization": int(counts.get("localization", 0)),
                "flap_confusion": int(counts.get("flap_confusion", 0)),
                "background": int(counts.get("background", 0)),
            }
        )
    harmful_summary = pd.DataFrame(harmful_rows)
    harmful_summary.to_csv(out_dir / "harmful_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nFP por riba do score mediano dos TP")
    print(harmful_summary.to_string(index=False))


if __name__ == "__main__":
    main()
