"""Cross-fit a regularized final-boundary regressor on source recordings.

The model follows the relative boundary parameterization used by temporal
detectors: start/end residuals are normalized by candidate duration. A QFL
quality estimate gates corrections so low-quality background detections move
less than likely action detections.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.diagnose_atsn_linear_separability import load_annotations  # noqa: E402
from dev.eval_actionness_profile_quality_head_cv import (  # noqa: E402
    PROFILE_COLUMNS,
    SHAPE_COLUMNS,
    add_shape_profiles,
)
from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    fit_linear_qfl,
    recording_weights,
    weighted_mean_std,
)
from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_final_boundary_gradient_cv import frame_prediction  # noqa: E402
from src.evaluation import segment_iou  # noqa: E402


SCORE_COLUMNS = (
    "log_duration",
    "score_global_rank",
    "score_recording_rank",
    "score_roi_rank",
)
BOUNDARY_FEATURE_COLUMNS = SCORE_COLUMNS + SHAPE_COLUMNS
QUALITY_FEATURE_COLUMNS = SCORE_COLUMNS + PROFILE_COLUMNS + SHAPE_COLUMNS


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        default=(
            "tmp/temporalmaxer_continuous/final_boundary_gradient_cv_v1/"
            "final_detection_profiles.csv"
        ),
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/final_boundary_ridge_cv_v1",
    )
    parser.add_argument("--positive-tiou", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument(
        "--refinement-blends",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
    )
    parser.add_argument("--max-relative-offset", type=float, default=0.5)
    parser.add_argument("--quality-steps", type=int, default=500)
    parser.add_argument("--quality-learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def add_score_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    durations = np.maximum(
        output["t_end"].to_numpy(np.float64)
        - output["t_start"].to_numpy(np.float64),
        1e-6,
    )
    output["log_duration"] = np.log1p(durations)
    output["score_global_rank"] = output["score"].rank(
        method="average",
        pct=True,
    )
    output["score_recording_rank"] = output.groupby("rec_name")["score"].rank(
        method="average",
        pct=True,
    )
    output["score_roi_rank"] = output.groupby(["rec_name", "roi_id"])["score"].rank(
        method="average",
        pct=True,
    )
    return output


def add_boundary_targets(
    frame: pd.DataFrame,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
) -> pd.DataFrame:
    target_tiou = []
    start_delta = []
    end_delta = []
    for row in frame.itertuples(index=False):
        targets = annotations.get((str(row.rec_name), int(row.roi_id)), [])
        duration = max(float(row.t_end) - float(row.t_start), 1e-6)
        if not targets:
            target_tiou.append(0.0)
            start_delta.append(0.0)
            end_delta.append(0.0)
            continue
        overlaps = segment_iou(
            np.asarray([row.t_start, row.t_end], dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
        )
        index = int(np.argmax(overlaps))
        gt_start, gt_end = targets[index]
        target_tiou.append(float(overlaps[index]))
        start_delta.append((float(gt_start) - float(row.t_start)) / duration)
        end_delta.append((float(gt_end) - float(row.t_end)) / duration)
    output = frame.copy()
    output["target_tiou"] = target_tiou
    output["target_start_delta"] = start_delta
    output["target_end_delta"] = end_delta
    return output


def fit_weighted_ridge(
    frame: pd.DataFrame,
    alpha: float,
    positive_tiou: float,
    max_relative_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive = frame[frame["target_tiou"] >= positive_tiou].copy()
    if positive.empty:
        raise ValueError("No positive candidates available for boundary regression")
    features = positive[list(BOUNDARY_FEATURE_COLUMNS)].to_numpy(np.float64)
    weights = recording_weights(positive).astype(np.float64)
    weights *= np.square(positive["target_tiou"].to_numpy(np.float64))
    mean, std = weighted_mean_std(features, weights)
    standardized = (features - mean) / std
    targets = positive[
        ["target_start_delta", "target_end_delta"]
    ].to_numpy(np.float64)
    targets = np.clip(targets, -max_relative_offset, max_relative_offset)
    normalized_weights = weights / weights.sum()
    target_mean = np.sum(targets * normalized_weights[:, None], axis=0)
    centered_targets = targets - target_mean
    weighted_features = standardized * np.sqrt(normalized_weights[:, None])
    weighted_targets = centered_targets * np.sqrt(normalized_weights[:, None])
    gram = weighted_features.T @ weighted_features
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
        weighted_features.T @ weighted_targets,
    )
    return mean, std, coefficients, target_mean


def predict_ridge(
    frame: pd.DataFrame,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, std, coefficients, target_mean = model
    features = frame[list(BOUNDARY_FEATURE_COLUMNS)].to_numpy(np.float64)
    return ((features - mean) / std) @ coefficients + target_mean


def predict_qfl_quality(
    frame: pd.DataFrame,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    mean, std, weights, bias = model
    features = frame[list(QUALITY_FEATURE_COLUMNS)].to_numpy(np.float32)
    logits = ((features - mean) / std) @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def apply_boundary_regression(
    frame: pd.DataFrame,
    offsets: np.ndarray,
    quality: np.ndarray,
    blend: float,
    max_relative_offset: float,
    minimum_duration_s: float = 2.0,
) -> pd.DataFrame:
    starts = frame["t_start"].to_numpy(np.float64)
    ends = frame["t_end"].to_numpy(np.float64)
    durations = np.maximum(ends - starts, minimum_duration_s)
    clipped = np.clip(offsets, -max_relative_offset, max_relative_offset)
    gate = np.clip(quality, 0.0, 1.0) * blend
    refined_starts = starts + durations * clipped[:, 0] * gate
    refined_ends = ends + durations * clipped[:, 1] * gate
    valid = refined_ends - refined_starts >= minimum_duration_s
    output = frame.copy()
    output["t_start"] = np.where(valid, refined_starts, starts)
    output["t_end"] = np.where(valid, refined_ends, ends)
    output["boundary_quality"] = quality
    output["predicted_start_delta"] = clipped[:, 0]
    output["predicted_end_delta"] = clipped[:, 1]
    return output


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    annotations = load_annotations(resolve(args.ann_path), min_duration_s=2.0)
    profiles = pd.read_csv(resolve(args.profiles))
    profiles = add_boundary_targets(
        add_score_features(add_shape_profiles(profiles)),
        annotations,
    )
    profiles.to_csv(out_dir / "labeled_profiles.csv", index=False)
    device = torch.device(args.device)

    rows = []
    for fold in range(5):
        train = profiles[profiles["fold"] != fold].copy()
        validation = profiles[profiles["fold"] == fold].copy()
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        quality_model = fit_linear_qfl(
            train,
            device,
            args.quality_steps,
            args.quality_learning_rate,
            feature_columns=QUALITY_FEATURE_COLUMNS,
        )
        quality = predict_qfl_quality(validation, quality_model)
        variants: list[tuple[str, pd.DataFrame]] = [("control", validation)]
        for positive_tiou in args.positive_tiou:
            for alpha in args.ridge_alphas:
                model = fit_weighted_ridge(
                    train,
                    alpha,
                    positive_tiou,
                    args.max_relative_offset,
                )
                offsets = predict_ridge(validation, model)
                for blend in args.refinement_blends:
                    variant = (
                        f"ridge_p{int(round(100 * positive_tiou)):02d}"
                        f"_a{alpha:g}_w{int(round(100 * blend)):03d}"
                    )
                    variants.append(
                        (
                            variant,
                            apply_boundary_regression(
                                validation,
                                offsets,
                                quality,
                                blend,
                                args.max_relative_offset,
                            ),
                        )
                    )
        for variant, frame in variants:
            prediction = frame_prediction(
                frame,
                f"source-final-boundary-{variant}",
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_{variant}.json",
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
