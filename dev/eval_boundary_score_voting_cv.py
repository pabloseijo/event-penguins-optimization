"""Source-CV evaluation of score-weighted temporal boundary voting.

Soft-NMS first selects and ranks detections. Each survivor then receives votes
from overlapping pre-NMS proposals in the same ROI. Voting changes only its
boundaries, preserving the selected detections and their Soft-NMS scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation import DetectionsEvaluator
from src.utils import temporal_soft_nms
from src.utils.detection import temporal_iou


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-root",
        default="tmp/temporalmaxer_dense/salient_boundary_router_pilot",
    )
    parser.add_argument(
        "--manifest", default="tmp/cv/recording_folds_r5/manifest.csv"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/boundary_score_voting_cv"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--score-column", default="quality_score")
    parser.add_argument("--nms-boundary", default="reference_blend050")
    parser.add_argument("--vote-tiou", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--vote-blend", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--vote-topk", type=int, default=20)
    parser.add_argument("--vote-score-power", type=float, default=1.0)

    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def ensure_boundary_columns(scored: pd.DataFrame, boundary_mode: str) -> pd.DataFrame:
    """Derive fixed boundary recipes that were selected in earlier source CV."""
    start_column = f"{boundary_mode}_t_start"
    end_column = f"{boundary_mode}_t_end"
    if start_column in scored and end_column in scored:
        return scored
    if boundary_mode != "router_shrink025":
        raise ValueError(f"Unknown or missing boundary mode: {boundary_mode}")
    required = (
        "reference_blend050_t_start",
        "reference_blend050_t_end",
        "router_soft_t_start",
        "router_soft_t_end",
    )
    missing = [column for column in required if column not in scored]
    if missing:
        raise ValueError(f"Cannot derive router_shrink025; missing {missing}")
    output = scored.copy()
    output[start_column] = (
        0.75 * output["reference_blend050_t_start"]
        + 0.25 * output["router_soft_t_start"]
    )
    output[end_column] = (
        0.75 * output["reference_blend050_t_end"]
        + 0.25 * output["router_soft_t_end"]
    )
    return output


def vote_detection_boundaries(
    detections: np.ndarray,
    voter_boundaries: np.ndarray,
    voter_scores: np.ndarray,
    tiou_threshold: float,
    blend: float,
    topk: int = 20,
    score_power: float = 1.0,
    minimum_duration: float = 2.0e6,
) -> np.ndarray:
    """Refine [start,end,score] detections from overlapping proposal votes."""
    detections = np.asarray(detections, dtype=np.float64)
    voter_boundaries = np.asarray(voter_boundaries, dtype=np.float64)
    voter_scores = np.asarray(voter_scores, dtype=np.float64)
    if detections.ndim != 2 or detections.shape[1] != 3:
        raise ValueError("Detections must have shape [N,3]")
    if voter_boundaries.shape != (len(voter_scores), 2):
        raise ValueError("Voter boundaries and scores are misaligned")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Voting blend must be in [0,1]")
    if not 0.0 <= tiou_threshold <= 1.0:
        raise ValueError("Voting tIoU threshold must be in [0,1]")

    output = detections.copy()
    for index, detection in enumerate(detections):
        overlaps = temporal_iou(
            voter_boundaries[:, 0],
            voter_boundaries[:, 1],
            detection[0],
            detection[1],
        )
        eligible = np.flatnonzero(overlaps >= tiou_threshold)
        if len(eligible) == 0:
            continue
        weights = np.power(np.clip(voter_scores[eligible], 1e-12, None), score_power)
        weights *= np.clip(overlaps[eligible], 1e-12, None)
        if topk > 0 and len(eligible) > topk:
            order = np.argsort(weights, kind="stable")[-topk:]
            eligible = eligible[order]
            weights = weights[order]
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            continue
        voted = np.average(voter_boundaries[eligible], axis=0, weights=weights)
        refined = (1.0 - blend) * detection[:2] + blend * voted
        if refined[1] - refined[0] < minimum_duration:
            center = 0.5 * float(refined[0] + refined[1])
            refined = np.asarray(
                [max(0.0, center - 0.5 * minimum_duration), 0.0],
                dtype=np.float64,
            )
            refined[1] = refined[0] + minimum_duration
        output[index, :2] = refined
    return output


def consensus_rescore_detections(
    detections: np.ndarray,
    voter_boundaries: np.ndarray,
    voter_scores: np.ndarray,
    tiou_threshold: float,
    blend: float,
    topk: int = 20,
) -> np.ndarray:
    """Fuse Soft-NMS confidence with overlapping proposal confidence."""
    detections = np.asarray(detections, dtype=np.float64)
    voter_boundaries = np.asarray(voter_boundaries, dtype=np.float64)
    voter_scores = np.asarray(voter_scores, dtype=np.float64)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Consensus score blend must be in [0,1]")
    if blend == 0.0 or len(detections) == 0:
        return detections.copy()
    output = detections.copy()
    for index, detection in enumerate(detections):
        overlaps = temporal_iou(
            voter_boundaries[:, 0],
            voter_boundaries[:, 1],
            detection[0],
            detection[1],
        )
        eligible = np.flatnonzero(overlaps >= tiou_threshold)
        if len(eligible) == 0:
            continue
        ranking = overlaps[eligible] * np.clip(voter_scores[eligible], 1e-12, None)
        if topk > 0 and len(eligible) > topk:
            order = np.argsort(ranking, kind="stable")[-topk:]
            eligible = eligible[order]
        overlap_weights = np.clip(overlaps[eligible], 1e-12, None)
        cluster_score = float(
            np.average(voter_scores[eligible], weights=overlap_weights)
        )
        exact = np.flatnonzero(overlaps >= 1.0 - 1e-9)
        seed_score = (
            float(np.max(voter_scores[exact]))
            if len(exact)
            else float(voter_scores[int(np.argmax(overlaps))])
        )
        decay = np.clip(detection[2] / max(seed_score, 1e-12), 0.0, 1.0)
        consensus_score = max(cluster_score * decay, 1e-12)
        output[index, 2] = np.power(max(detection[2], 1e-12), 1.0 - blend) * np.power(
            consensus_score, blend
        )
    return output


def build_voted_prediction(
    scored: pd.DataFrame,
    args: argparse.Namespace,
    vote_tiou: float | None,
    vote_blend: float,
    consensus_score_blend: float = 0.0,
    consensus_score_stage: str = "post",
) -> dict:
    if consensus_score_stage not in {"pre", "post"}:
        raise ValueError("Consensus score stage must be 'pre' or 'post'")
    scored = ensure_boundary_columns(scored, args.nms_boundary)
    min_action_duration = float(getattr(args, "min_action_duration", 2.0))
    start_column = f"{args.nms_boundary}_t_start"
    end_column = f"{args.nms_boundary}_t_end"
    result = {
        recording: {int(str(roi)[1:]): [] for roi in group["roi_id"].unique()}
        for recording, group in scored.groupby("rec_name")
    }
    selected = scored[scored[args.score_column] >= args.min_score].copy()
    durations = (
        selected[end_column].to_numpy(dtype=np.float64)
        - selected[start_column].to_numpy(dtype=np.float64)
    ) / 1e6
    selected["final_score"] = selected[args.score_column].to_numpy(dtype=np.float64) * np.exp(
        -np.maximum(0.0, durations - args.duration_dmax) / args.duration_sigma
    )
    for (recording, roi), group in selected.groupby(["rec_name", "roi_id"]):
        if args.pre_nms_topk_per_roi > 0 and len(group) > args.pre_nms_topk_per_roi:
            group = group.nlargest(args.pre_nms_topk_per_roi, "final_score")
        candidates = group[[start_column, end_column, "final_score"]].to_numpy(
            dtype=np.float64
        )
        if vote_tiou is not None and consensus_score_blend > 0.0 and consensus_score_stage == "pre":
            candidates = consensus_rescore_detections(
                candidates,
                candidates[:, :2],
                candidates[:, 2],
                vote_tiou,
                consensus_score_blend,
                args.vote_topk,
            )
        detections = temporal_soft_nms(
            candidates,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
        )
        if vote_tiou is not None and len(detections):
            if consensus_score_stage == "post":
                detections = consensus_rescore_detections(
                    detections,
                    candidates[:, :2],
                    candidates[:, 2],
                    vote_tiou,
                    consensus_score_blend,
                    args.vote_topk,
                )
            detections = vote_detection_boundaries(
                detections,
                candidates[:, :2],
                candidates[:, 2],
                vote_tiou,
                vote_blend,
                args.vote_topk,
                args.vote_score_power,
                min_action_duration * 1e6,
            )
        result[recording][int(str(roi)[1:])] = [
            {
                "label": "ed",
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(score),
            }
            for start, end, score in detections
            if end - start >= min_action_duration * 1e6
        ]
    mode = (
        "baseline"
        if vote_tiou is None
        else f"vote_{vote_tiou:.2f}_{vote_blend:.2f}_score_{consensus_score_blend:.2f}_{consensus_score_stage}"
    )
    return {"version": f"boundary-score-voting:{mode}", "results": result}


def evaluate_prediction(
    prediction: dict,
    sequences: list[str],
    label: str,
    args: argparse.Namespace,
    path: Path,
) -> dict[str, float | str | int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prediction), encoding="utf-8")
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(resolve(args.ann_path)),
        prediction_filename=str(path),
        tiou_thresholds=np.asarray(args.tiou, dtype=np.float64),
        valid_labels="ed",
        valid_sequences=sequences,
        min_duration=float(getattr(args, "min_action_duration", 2.0)),
    )
    mean_ap = float(evaluator.run())
    count = sum(
        len(detections)
        for rois in prediction["results"].values()
        for detections in rois.values()
    )
    row: dict[str, float | str | int] = {
        "variant": label,
        "mAP": mean_ap,
        "n_predictions": count,
    }
    for threshold, value in zip(args.tiou, evaluator.mAP):
        row[f"AP@{threshold:.1f}"] = float(value)
    return row


def main() -> None:
    args = parse_args()
    if args.min_action_duration < 0:
        raise ValueError("--min-action-duration must be non-negative")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    for fold in args.folds:
        fold_out = out_dir / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        if (fold_out / "metrics.csv").exists():
            continue
        scored = pd.read_csv(
            resolve(args.scored_root) / f"fold_{fold:02d}" / "scored.csv"
        )
        sequences = sorted(scored["rec_name"].astype(str).unique())
        settings: list[tuple[str, float | None, float]] = [("baseline", None, 0.0)]
        settings.extend(
            (
                f"vote_iou{int(round(tiou * 100)):03d}_blend{int(round(blend * 100)):03d}",
                tiou,
                blend,
            )
            for tiou in args.vote_tiou
            for blend in args.vote_blend
        )
        rows = []
        for label, vote_tiou, vote_blend in settings:
            prediction = build_voted_prediction(scored, args, vote_tiou, vote_blend)
            row = evaluate_prediction(
                prediction,
                sequences,
                label,
                args,
                fold_out / "predictions" / f"{label}.json",
            )
            row["fold"] = fold
            row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
            rows.append(row)
            pd.DataFrame(rows).to_csv(fold_out / "metrics_partial.csv", index=False)
        pd.DataFrame(rows).to_csv(fold_out / "metrics.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)

    metric_paths = [out_dir / f"fold_{fold:02d}" / "metrics.csv" for fold in range(5)]
    if not all(path.exists() for path in metric_paths):
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = []
    for variant, group in metrics.groupby("variant"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row: dict[str, float | str] = {"variant": variant}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        summary.append(row)
    result = pd.DataFrame(summary).sort_values("mean_mAP", ascending=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
