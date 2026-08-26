"""Source-CV robust voting across TemporalMaxer boundary experts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.eval_boundary_score_voting_cv import (
    build_voted_prediction,
    consensus_rescore_detections,
    evaluate_prediction,
    resolve,
)
from src.utils import temporal_soft_nms
from src.utils.detection import temporal_iou


EXPERT_MODES = (
    "raw",
    "reference_blend050",
    "reference_delta",
    "reference_distribution",
    "reference_point",
)


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
        "--out-dir", default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_cv"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--score-column", default="quality_score")
    parser.add_argument("--nms-boundary", default="reference_blend050")
    parser.add_argument("--vote-tiou", type=float, default=0.5)
    parser.add_argument("--multi-vote-topk", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.set_defaults(vote_topk=20, vote_score_power=1.0)
    return parser.parse_args()


def expert_boundary_tensor(
    scored: pd.DataFrame, modes: tuple[str, ...] = EXPERT_MODES
) -> np.ndarray:
    values = []
    for mode in modes:
        start = "t_start" if mode == "raw" else f"{mode}_t_start"
        end = "t_end" if mode == "raw" else f"{mode}_t_end"
        if start not in scored or end not in scored:
            raise ValueError(f"Missing boundary expert {mode!r}")
        values.append(scored[[start, end]].to_numpy(dtype=np.float64))
    return np.stack(values, axis=1)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def multi_expert_vote_boundaries(
    detections: np.ndarray,
    voter_boundaries: np.ndarray,
    voter_scores: np.ndarray,
    tiou_threshold: float,
    blend: float,
    estimator: str,
    topk: int = 100,
    minimum_duration: float = 2.0e6,
) -> np.ndarray:
    """Vote from [proposal,expert,start/end] candidates around each survivor."""
    detections = np.asarray(detections, dtype=np.float64)
    voter_boundaries = np.asarray(voter_boundaries, dtype=np.float64)
    voter_scores = np.asarray(voter_scores, dtype=np.float64)
    if voter_boundaries.ndim != 3 or voter_boundaries.shape[2] != 2:
        raise ValueError("Multi-expert boundaries must have shape [N,K,2]")
    if len(voter_boundaries) != len(voter_scores):
        raise ValueError("Multi-expert boundaries and scores are misaligned")
    if estimator not in {"mean", "median"}:
        raise ValueError("Estimator must be 'mean' or 'median'")
    flat = voter_boundaries.reshape(-1, 2)
    flat_scores = np.repeat(voter_scores, voter_boundaries.shape[1])
    output = detections.copy()
    for index, detection in enumerate(detections):
        overlaps = temporal_iou(
            flat[:, 0], flat[:, 1], detection[0], detection[1]
        )
        eligible = np.flatnonzero(overlaps >= tiou_threshold)
        if len(eligible) == 0:
            continue
        weights = np.clip(flat_scores[eligible], 1e-12, None) * np.clip(
            overlaps[eligible], 1e-12, None
        )
        if topk > 0 and len(eligible) > topk:
            chosen = np.argsort(weights, kind="stable")[-topk:]
            eligible = eligible[chosen]
            weights = weights[chosen]
        if estimator == "mean":
            voted = np.average(flat[eligible], axis=0, weights=weights)
        else:
            voted = np.asarray(
                [
                    weighted_median(flat[eligible, coordinate], weights)
                    for coordinate in range(2)
                ]
            )
        refined = (1.0 - blend) * detection[:2] + blend * voted
        if refined[1] - refined[0] < minimum_duration:
            center = 0.5 * float(refined.sum())
            refined[0] = max(0.0, center - 0.5 * minimum_duration)
            refined[1] = refined[0] + minimum_duration
        output[index, :2] = refined
    return output


def build_multi_expert_prediction(
    scored: pd.DataFrame,
    args: argparse.Namespace,
    blend: float,
    estimator: str,
) -> dict:
    min_action_duration = float(getattr(args, "min_action_duration", 2.0))
    start_column = f"{args.nms_boundary}_t_start"
    end_column = f"{args.nms_boundary}_t_end"
    all_experts = expert_boundary_tensor(scored)
    result = {
        recording: {int(str(roi)[1:]): [] for roi in group["roi_id"].unique()}
        for recording, group in scored.groupby("rec_name")
    }
    selected = scored[scored[args.score_column] >= args.min_score].copy()
    selected["source_index"] = selected.index.to_numpy(dtype=np.int64)
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
        detections = temporal_soft_nms(
            candidates,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
        )
        if len(detections):
            detections = consensus_rescore_detections(
                detections,
                candidates[:, :2],
                candidates[:, 2],
                args.vote_tiou,
                blend=0.25,
                topk=20,
            )
            source_indices = group["source_index"].to_numpy(dtype=np.int64)
            detections = multi_expert_vote_boundaries(
                detections,
                all_experts[source_indices],
                candidates[:, 2],
                args.vote_tiou,
                blend,
                estimator,
                args.multi_vote_topk,
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
    return {
        "version": f"multi-expert-vote:{estimator}:{blend:.2f}",
        "results": result,
    }


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
        settings = [("single_expert_control", None, 0.5)]
        settings.extend(
            (f"multi_{estimator}_blend{int(blend * 100):03d}", estimator, blend)
            for estimator in ("mean", "median")
            for blend in (0.25, 0.5)
        )
        rows = []
        for label, estimator, blend in settings:
            if estimator is None:
                prediction = build_voted_prediction(
                    scored,
                    args,
                    vote_tiou=0.5,
                    vote_blend=0.5,
                    consensus_score_blend=0.25,
                )
            else:
                prediction = build_multi_expert_prediction(
                    scored, args, blend, estimator
                )
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

    paths = [out_dir / f"fold_{fold:02d}" / "metrics.csv" for fold in range(5)]
    if not all(path.exists() for path in paths):
        return
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    rows = []
    for variant, group in metrics.groupby("variant"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row: dict[str, float | str] = {"variant": variant}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
