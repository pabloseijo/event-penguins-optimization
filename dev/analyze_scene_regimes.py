"""Diagnose whether SAVS-like semantic regimes explain detection errors.

Regime boundaries are extracted without annotations from frozen continuous
features. Ground truth is only joined afterwards to characterize the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def semantic_change_scores(features: np.ndarray, window: int) -> np.ndarray:
    """Return left/right window cosine distance at every valid split point."""
    if window < 1:
        raise ValueError("window must be positive")
    length = len(features)
    scores = np.full(length, np.nan, dtype=np.float64)
    if length < 2 * window + 1:
        return scores
    normalized = l2_normalize(np.asarray(features, dtype=np.float64))
    prefix = np.vstack((np.zeros((1, normalized.shape[1])), normalized.cumsum(axis=0)))
    indices = np.arange(window, length - window)
    left = (prefix[indices] - prefix[indices - window]) / window
    right = (prefix[indices + window] - prefix[indices]) / window
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    scores[indices] = 1.0 - numerator / np.maximum(denominator, 1e-8)
    return scores


def robust_zscores(values: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    median = float(np.median(values[valid]))
    mad = float(np.median(np.abs(values[valid] - median)))
    scale = max(1.4826 * mad, 1e-8)
    result[valid] = (values[valid] - median) / scale
    return result


def select_boundaries(
    scores: np.ndarray, threshold_z: float, min_distance: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select high local maxima with deterministic non-maximum suppression."""
    if min_distance < 1:
        raise ValueError("min_distance must be positive")
    zscores = robust_zscores(scores)
    finite = np.isfinite(zscores)
    previous = np.r_[-np.inf, zscores[:-1]]
    following = np.r_[zscores[1:], -np.inf]
    candidates = np.flatnonzero(
        finite & (zscores >= threshold_z) & (zscores >= previous) & (zscores > following)
    )
    selected: list[int] = []
    for index in candidates[np.argsort(zscores[candidates])[::-1]]:
        if all(abs(int(index) - kept) >= min_distance for kept in selected):
            selected.append(int(index))
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    return selected_array, zscores


def load_annotations(path: Path) -> dict[tuple[str, int], list[list[float]]]:
    database = json.loads(path.read_text(encoding="utf-8"))["database"]
    output: dict[tuple[str, int], list[list[float]]] = {}
    for recording, data in database.items():
        for roi_id, annotations in data.get("annotations", {}).items():
            if roi_id == "null":
                continue
            output[(recording, int(roi_id))] = [
                list(map(float, annotation["segment"]))
                for annotation in annotations
                if annotation["label"] == "ed"
                and float(annotation["segment"][1]) - float(annotation["segment"][0]) >= 2.0
            ]
    return output


def load_predictions(path: Path | None) -> dict[tuple[str, int], list[dict]]:
    if path is None:
        return {}
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    return {
        (recording, int(roi_id)): detections
        for recording, rois in results.items()
        for roi_id, detections in rois.items()
    }


def interval_iou(segment: list[float], candidates: list[list[float]]) -> float:
    if not candidates:
        return 0.0
    start = np.maximum(float(segment[0]), np.asarray(candidates)[:, 0])
    end = np.minimum(float(segment[1]), np.asarray(candidates)[:, 1])
    intersection = np.maximum(end - start, 0.0)
    union = (
        float(segment[1])
        - float(segment[0])
        + np.asarray(candidates)[:, 1]
        - np.asarray(candidates)[:, 0]
        - intersection
    )
    return float(np.max(intersection / np.maximum(union, 1e-8)))


def nearest_distance(time_s: float, boundaries_s: np.ndarray) -> float:
    if len(boundaries_s) == 0:
        return float("inf")
    return float(np.min(np.abs(boundaries_s - time_s)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--prediction", default=None)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-label", default="unknown")
    parser.add_argument("--window-s", type=float, default=10.0)
    parser.add_argument("--threshold-z", type=float, default=3.0)
    parser.add_argument("--min-distance-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    stride_s = float(metadata["grid_stride_s"])
    window = max(1, int(round(args.window_s / stride_s)))
    min_distance = max(1, int(round(args.min_distance_s / stride_s)))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    features = np.load(feature_dir / "frame_features.npy", mmap_mode="r")
    annotations = load_annotations(resolve(args.ann_path))
    prediction_path = resolve(args.prediction) if args.prediction else None
    predictions = load_predictions(prediction_path)

    sequence_rows: list[dict] = []
    boundary_rows: list[dict] = []
    detection_rows: list[dict] = []
    for row in sequences.itertuples(index=False):
        recording = str(row.rec_name)
        roi_id = int(row.roi_id)
        length = int(row.length)
        values = np.asarray(
            features[int(row.offset) : int(row.offset) + length], dtype=np.float32
        )
        scores = semantic_change_scores(values, window)
        boundary_indices, zscores = select_boundaries(
            scores, args.threshold_z, min_distance
        )
        boundaries_s = boundary_indices.astype(np.float64) * stride_s
        gt_segments = annotations.get((recording, roi_id), [])
        gt_edges = np.asarray(
            [edge for segment in gt_segments for edge in segment], dtype=np.float64
        )
        cuts = np.r_[0.0, boundaries_s, length * stride_s]
        segment_durations = np.diff(cuts)
        valid_scores = scores[np.isfinite(scores)]
        sequence_rows.append(
            {
                "split": args.split_label,
                "rec_name": recording,
                "roi_id": roi_id,
                "boundaries": len(boundary_indices),
                "regimes": len(boundary_indices) + 1,
                "median_regime_s": float(np.median(segment_durations)),
                "min_regime_s": float(np.min(segment_durations)),
                "median_change_score": float(np.median(valid_scores)),
                "max_change_z": float(np.nanmax(zscores)),
                "gt_instances": len(gt_segments),
            }
        )
        for index, time_s in zip(boundary_indices, boundaries_s):
            inside_gt = any(start <= time_s <= end for start, end in gt_segments)
            edge_distance = nearest_distance(float(time_s), gt_edges)
            boundary_rows.append(
                {
                    "split": args.split_label,
                    "rec_name": recording,
                    "roi_id": roi_id,
                    "time_s": float(time_s),
                    "change_score": float(scores[index]),
                    "change_z": float(zscores[index]),
                    "inside_gt": inside_gt,
                    "nearest_gt_edge_s": edge_distance,
                    "near_gt_edge_5s": edge_distance <= 5.0,
                }
            )
        for detection in predictions.get((recording, roi_id), []):
            if detection.get("label") != "ed":
                continue
            segment = list(map(float, detection["segment"]))
            center = 0.5 * (segment[0] + segment[1])
            overlap = interval_iou(segment, gt_segments)
            distance = nearest_distance(center, boundaries_s)
            detection_rows.append(
                {
                    "split": args.split_label,
                    "rec_name": recording,
                    "roi_id": roi_id,
                    "t_start": segment[0],
                    "t_end": segment[1],
                    "score": float(detection["score"]),
                    "best_tiou": overlap,
                    "is_tp05": overlap >= 0.5,
                    "nearest_regime_boundary_s": distance,
                    "near_regime_boundary_5s": distance <= 5.0,
                    "near_regime_boundary_10s": distance <= 10.0,
                }
            )

    sequence_frame = pd.DataFrame(sequence_rows)
    boundary_frame = pd.DataFrame(boundary_rows)
    detection_frame = pd.DataFrame(detection_rows)
    summary = (
        sequence_frame.groupby(["split", "rec_name"], as_index=False)
        .agg(
            rois=("roi_id", "size"),
            boundaries=("boundaries", "sum"),
            median_boundaries_per_roi=("boundaries", "median"),
            median_regime_s=("median_regime_s", "median"),
            median_max_change_z=("max_change_z", "median"),
            gt_instances=("gt_instances", "sum"),
        )
        .sort_values("rec_name")
    )
    if not boundary_frame.empty:
        boundary_summary = boundary_frame.groupby(["split", "rec_name"]).agg(
            boundary_inside_gt_fraction=("inside_gt", "mean"),
            boundary_near_gt_edge_5s_fraction=("near_gt_edge_5s", "mean"),
        )
        summary = summary.merge(boundary_summary, on=["split", "rec_name"], how="left")
    if not detection_frame.empty:
        detection_summary = detection_frame.groupby(["split", "rec_name"]).agg(
            detections=("is_tp05", "size"),
            tp05_fraction=("is_tp05", "mean"),
        )
        for kind, mask in (
            ("tp05", detection_frame["is_tp05"]),
            ("fp05", ~detection_frame["is_tp05"]),
        ):
            subset = detection_frame[mask]
            if subset.empty:
                continue
            distances = subset.groupby(["split", "rec_name"]).agg(
                **{
                    f"{kind}_median_boundary_distance_s": (
                        "nearest_regime_boundary_s",
                        "median",
                    ),
                    f"{kind}_near_boundary_10s_fraction": (
                        "near_regime_boundary_10s",
                        "mean",
                    ),
                }
            )
            detection_summary = detection_summary.join(distances, how="left")
        summary = summary.merge(
            detection_summary.reset_index(), on=["split", "rec_name"], how="left"
        )

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence_frame.to_csv(out_dir / "sequences.csv", index=False)
    boundary_frame.to_csv(out_dir / "boundaries.csv", index=False)
    detection_frame.to_csv(out_dir / "detections.csv", index=False)
    summary.to_csv(out_dir / "recording_summary.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
