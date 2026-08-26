"""Cross-fitted second-stage quality calibration for detector fusion.

The calibrator only sees out-of-fold source detections. It predicts temporal
quality from each detector's confidence, cross-model agreement, duration and
local rank, then a fixed Soft-NMS produces the final detections.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.evaluation import DetectionsEvaluator
from src.utils import temporal_soft_nms
from src.utils.detection import temporal_iou


FEATURE_COLUMNS = [
    "own_log_score",
    "other_log_score",
    "cross_iou",
    "agreement_score",
    "log_duration",
    "local_rank",
    "log_candidate_count",
    "boundary_disagreement",
    "model_continuous",
    "continuous_log_score",
    "proposal_log_score",
]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", default="tmp/temporalmaxer_continuous/oof_quality_v1")
    parser.add_argument("--per-model-topk", type=int, default=100)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--nms-sigmas", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--targets", nargs="+", choices=["binary", "soft"], default=["binary", "soft"])
    parser.add_argument("--l2", type=float, default=1e-3)
    return parser.parse_args()


def segment_overlaps(segment: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if len(candidates) == 0:
        return np.empty(0, dtype=np.float64)
    return temporal_iou(candidates[:, 0], candidates[:, 1], segment[0], segment[1])


def load_gt(path: Path) -> dict[tuple[str, int], np.ndarray]:
    database = json.loads(path.read_text(encoding="utf-8"))["database"]
    result = {}
    for recording, value in database.items():
        for roi, annotations in value.get("annotations", {}).items():
            if roi == "null":
                continue
            segments = [
                item["segment"]
                for item in annotations
                if item["label"] == "ed"
                and float(item["segment"][1]) - float(item["segment"][0]) >= 2.0
            ]
            result[(recording, int(roi))] = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    return result


def flatten_prediction(prediction: dict) -> dict[tuple[str, int], list[dict]]:
    return {
        (recording, int(roi)): list(detections)
        for recording, rois in prediction["results"].items()
        for roi, detections in rois.items()
    }


def top_detections(values: list[dict], topk: int) -> list[dict]:
    return sorted(values, key=lambda item: float(item["score"]), reverse=True)[:topk]


def candidate_rows(
    continuous_prediction: dict,
    proposal_prediction: dict,
    fold: int,
    gt: dict[tuple[str, int], np.ndarray] | None,
    topk: int,
) -> list[dict]:
    continuous = flatten_prediction(continuous_prediction)
    proposal = flatten_prediction(proposal_prediction)
    rows = []
    for key in sorted(set(continuous) | set(proposal)):
        recording, roi_id = key
        by_model = {
            "continuous": top_detections(continuous.get(key, []), topk),
            "proposal": top_detections(proposal.get(key, []), topk),
        }
        arrays = {
            model: np.asarray([item["segment"] for item in detections], dtype=np.float64).reshape(-1, 2)
            for model, detections in by_model.items()
        }
        gt_segments = np.empty((0, 2), dtype=np.float64) if gt is None else gt.get(key, np.empty((0, 2)))
        total_count = sum(len(values) for values in by_model.values())
        for model, detections in by_model.items():
            other_model = "proposal" if model == "continuous" else "continuous"
            other_detections = by_model[other_model]
            other_segments = arrays[other_model]
            for rank, detection in enumerate(detections):
                segment = np.asarray(detection["segment"], dtype=np.float64)
                own_score = float(np.clip(detection["score"], 1e-8, 1.0))
                cross_overlaps = segment_overlaps(segment, other_segments)
                if len(cross_overlaps):
                    counterpart_index = int(np.argmax(cross_overlaps))
                    cross_iou = float(cross_overlaps[counterpart_index])
                    other_score = float(
                        np.clip(other_detections[counterpart_index]["score"], 1e-8, 1.0)
                    )
                    other_segment = other_segments[counterpart_index]
                    disagreement = float(
                        np.abs(segment - other_segment).sum()
                        / max(segment[1] - segment[0], 1e-6)
                    )
                else:
                    cross_iou = 0.0
                    other_score = 1e-8
                    disagreement = 2.0
                own_log = math.log(own_score)
                other_log = math.log(other_score)
                target_iou = (
                    float(segment_overlaps(segment, gt_segments).max())
                    if len(gt_segments)
                    else 0.0
                )
                rows.append(
                    {
                        "fold": fold,
                        "rec_name": recording,
                        "roi_id": roi_id,
                        "t_start": float(segment[0]),
                        "t_end": float(segment[1]),
                        "model": model,
                        "target_iou": target_iou,
                        "own_log_score": own_log,
                        "other_log_score": other_log,
                        "cross_iou": cross_iou,
                        "agreement_score": cross_iou * other_score,
                        "log_duration": math.log(max(segment[1] - segment[0], 1e-6)),
                        "local_rank": 1.0 - rank / max(len(detections), 1),
                        "log_candidate_count": math.log1p(total_count),
                        "boundary_disagreement": min(disagreement, 10.0),
                        "model_continuous": float(model == "continuous"),
                        "continuous_log_score": own_log if model == "continuous" else other_log,
                        "proposal_log_score": own_log if model == "proposal" else other_log,
                    }
                )
    return rows


def load_source_candidates(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = resolve(args.out_dir)
    cache_path = out_dir / "source_candidates.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    gt = load_gt(resolve(args.ann_path))
    rows = []
    for fold in range(5):
        continuous_metrics = json.loads(
            (resolve(args.continuous_root) / f"fold_{fold:02d}" / "metrics_best.json").read_text()
        )
        epoch = int(continuous_metrics["epoch"])
        continuous = json.loads(
            (
                resolve(args.continuous_root)
                / f"fold_{fold:02d}"
                / "predictions"
                / f"epoch_{epoch:03d}.json"
            ).read_text()
        )
        proposal = json.loads(
            (
                resolve(args.proposal_root)
                / f"fold_{fold:02d}"
                / "predictions"
                / f"{args.proposal_variant}.json"
            ).read_text()
        )
        rows.extend(candidate_rows(continuous, proposal, fold, gt, args.per_model_topk))
    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def fit_linear_quality(
    frame: pd.DataFrame,
    target_mode: str,
    l2: float,
) -> dict[str, np.ndarray | float | str]:
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = torch.from_numpy(((features - mean) / scale).astype(np.float32))
    target_iou = frame["target_iou"].to_numpy(dtype=np.float32)
    if target_mode == "binary":
        targets = torch.from_numpy((target_iou >= 0.5).astype(np.float32))
        positive = max(float(targets.sum()), 1.0)
        positive_weight = min(float((len(targets) - targets.sum()) / positive), 20.0)
        sample_weights = torch.where(targets > 0, positive_weight, 1.0)
    else:
        targets = torch.from_numpy(np.clip(target_iou, 0.0, 1.0))
        sample_weights = 1.0 + 4.0 * targets
    weights = torch.zeros(normalized.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias], max_iter=100, tolerance_grad=1e-8, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = normalized @ weights + bias
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            * sample_weights
        ).mean() + l2 * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "target_mode": target_mode,
        "mean": mean,
        "scale": scale,
        "weights": weights.detach().numpy().astype(np.float64),
        "bias": float(bias.detach()),
    }


def quality_scores(frame: pd.DataFrame, model: dict) -> np.ndarray:
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    normalized = (features - model["mean"]) / model["scale"]
    logits = normalized @ model["weights"] + float(model["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def build_prediction(
    frame: pd.DataFrame,
    scores: np.ndarray,
    sigma: float,
    max_predictions: int,
) -> dict:
    scored = frame.copy()
    scored["quality_score"] = scores
    results = {
        recording: {str(int(roi)): [] for roi in group["roi_id"].unique()}
        for recording, group in scored.groupby("rec_name")
    }
    for (recording, roi), group in scored.groupby(["rec_name", "roi_id"]):
        candidates = group[["t_start", "t_end", "quality_score"]].to_numpy(dtype=np.float64)
        detections = temporal_soft_nms(candidates, sigma=sigma, score_threshold=1e-5)
        results[recording][str(int(roi))] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(score),
            }
            for start, end, score in detections[:max_predictions]
            if end - start >= 2.0
        ]
    return {"version": "oof-linear-fusion-quality-v1", "results": results}


def evaluate_prediction(
    prediction: dict,
    recordings: list[str],
    args: argparse.Namespace,
    path: Path,
) -> dict[str, float | int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prediction), encoding="utf-8")
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(resolve(args.ann_path)),
        prediction_filename=str(path),
        tiou_thresholds=np.asarray([0.1, 0.3, 0.5, 0.7]),
        valid_sequences=recordings,
        valid_labels=["ed"],
        min_duration=2.0,
    )
    row: dict[str, float | int] = {
        "mAP": float(evaluator.run()),
        "n_predictions": sum(
            len(values) for rois in prediction["results"].values() for values in rois.values()
        ),
    }
    for threshold, value in zip((0.1, 0.3, 0.5, 0.7), evaluator.mAP):
        row[f"AP@{threshold:.1f}"] = float(value)
    return row


def save_model(model: dict, path: Path, sigma: float) -> None:
    np.savez(
        path,
        target_mode=model["target_mode"],
        feature_columns=np.asarray(FEATURE_COLUMNS),
        mean=model["mean"],
        scale=model["scale"],
        weights=model["weights"],
        bias=model["bias"],
        nms_sigma=sigma,
    )


def main() -> None:
    args = parse_args()
    frame = load_source_candidates(args)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    rows = []
    for target_mode in args.targets:
        for fold in range(5):
            train = frame[frame["fold"] != fold]
            validation = frame[frame["fold"] == fold]
            model = fit_linear_quality(train, target_mode, args.l2)
            scores = quality_scores(validation, model)
            recordings = str(manifest.loc[fold, "val_record_names"]).split()
            for sigma in args.nms_sigmas:
                label = f"{target_mode}_sigma{sigma:g}_fold{fold:02d}"
                prediction = build_prediction(validation, scores, sigma, args.max_predictions)
                rows.append(
                    {
                        "target_mode": target_mode,
                        "nms_sigma": sigma,
                        "fold": fold,
                        "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                        **evaluate_prediction(
                            prediction, recordings, args, out_dir / "predictions" / f"{label}.json"
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for (target_mode, sigma), group in metrics.groupby(["target_mode", "nms_sigma"]):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "target_mode": target_mode,
                "nms_sigma": sigma,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    best = summary.iloc[0]
    final_model = fit_linear_quality(frame, str(best["target_mode"]), args.l2)
    save_model(final_model, out_dir / "model.npz", float(best["nms_sigma"]))
    (out_dir / "recipe.json").write_text(
        json.dumps(best.to_dict(), indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
