"""Evaluate boundary experts strictly after reference Soft-NMS selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import DetectionsEvaluator  # noqa: E402
from src.utils.detection import temporal_iou  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-root", default="tmp/temporalmaxer_dense/boundary_quality_router_cv"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/boundary_router_post_nms_cv"
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--score-column", default="quality_score")
    parser.add_argument("--nms-boundary", default="reference_blend050")
    parser.add_argument(
        "--refined-boundaries",
        nargs="+",
        default=["reference_blend050", "tespec_blend050", "router", "oracle"],
    )
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def temporal_soft_nms_indices(
    detections: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original row indices and decayed scores of Gaussian Soft-NMS."""
    scores = detections[:, 2].astype(np.float64, copy=True)
    remaining = list(range(len(detections)))
    keep = []
    while remaining:
        best_local = int(np.argmax(scores[remaining]))
        best = remaining.pop(best_local)
        keep.append(best)
        for index in remaining:
            overlap = temporal_iou(
                detections[index, 0],
                detections[index, 1],
                detections[best, 0],
                detections[best, 1],
            )
            scores[index] *= np.exp(-(overlap**2) / sigma)
    keep_array = np.asarray(keep, dtype=np.int64)
    keep_array = keep_array[scores[keep_array] >= score_threshold]
    order = np.argsort(scores[keep_array])[::-1]
    keep_array = keep_array[order]
    return keep_array, scores[keep_array]


def build_post_nms_prediction(
    scored: pd.DataFrame,
    score_column: str,
    nms_boundary: str,
    refined_boundary: str,
    args: argparse.Namespace,
) -> dict:
    nms_start = f"{nms_boundary}_t_start"
    nms_end = f"{nms_boundary}_t_end"
    refined_start = f"{refined_boundary}_t_start"
    refined_end = f"{refined_boundary}_t_end"
    required = {score_column, nms_start, nms_end, refined_start, refined_end}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Missing post-NMS columns: {missing}")
    result = {
        recording: {int(str(roi)[1:]): [] for roi in group["roi_id"].unique()}
        for recording, group in scored.groupby("rec_name")
    }
    selected = scored[scored[score_column] >= args.min_score].copy()
    nms_duration = (
        selected[nms_end].to_numpy(dtype=np.float64)
        - selected[nms_start].to_numpy(dtype=np.float64)
    ) / 1e6
    penalty = np.exp(
        -np.maximum(0.0, nms_duration - args.duration_dmax) / args.duration_sigma
    )
    selected["final_score"] = selected[score_column].to_numpy(dtype=np.float64) * penalty
    for (recording, roi), group in selected.groupby(["rec_name", "roi_id"]):
        if args.pre_nms_topk_per_roi > 0 and len(group) > args.pre_nms_topk_per_roi:
            group = group.nlargest(args.pre_nms_topk_per_roi, "final_score")
        candidates = group[[nms_start, nms_end, "final_score"]].to_numpy(dtype=np.float64)
        keep, scores = temporal_soft_nms_indices(
            candidates,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
        )
        detections = group.iloc[keep]
        values = []
        for (_, row), score in zip(detections.iterrows(), scores):
            start = float(row[refined_start])
            end = float(row[refined_end])
            if end - start < 2.0e6:
                continue
            values.append(
                {
                    "label": "ed",
                    "segment": [start / 1e6, end / 1e6],
                    "score": float(score),
                }
            )
        result[recording][int(str(roi)[1:])] = values
    return {
        "version": f"post-nms:{score_column}:{nms_boundary}:{refined_boundary}",
        "results": result,
    }


def evaluate_post_nms(
    scored: pd.DataFrame,
    refined_boundary: str,
    fold: int,
    args: argparse.Namespace,
    prediction_dir: Path,
) -> dict[str, float | str | int]:
    prediction = build_post_nms_prediction(
        scored,
        args.score_column,
        args.nms_boundary,
        refined_boundary,
        args,
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_dir / f"fold_{fold:02d}_{refined_boundary}.json"
    path.write_text(json.dumps(prediction), encoding="utf-8")
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(resolve(args.ann_path)),
        prediction_filename=str(path),
        tiou_thresholds=np.asarray(args.tiou, dtype=np.float64),
        valid_labels="ed",
        valid_sequences=sorted(scored["rec_name"].astype(str).unique()),
        min_duration=2.0,
    )
    mean_ap = float(evaluator.run())
    count = sum(
        len(detections)
        for rois in prediction["results"].values()
        for detections in rois.values()
    )
    row: dict[str, float | str | int] = {
        "score_column": args.score_column,
        "nms_boundary": args.nms_boundary,
        "boundary_mode": refined_boundary,
        "mAP": mean_ap,
        "n_predictions": count,
        "fold": fold,
    }
    for threshold, value in zip(args.tiou, evaluator.mAP):
        row[f"AP@{threshold:.1f}"] = float(value)
    return row


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    rows = []
    for fold in range(5):
        scored = pd.read_csv(
            resolve(args.router_root) / f"fold_{fold:02d}" / "scored.csv"
        )
        for boundary in args.refined_boundaries:
            row = evaluate_post_nms(
                scored, boundary, fold, args, out_dir / "predictions"
            )
            row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = []
    for boundary, group in metrics.groupby("boundary_mode"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row = {"boundary_mode": boundary}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        summary.append(row)
    result = pd.DataFrame(summary).sort_values("mean_mAP", ascending=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
