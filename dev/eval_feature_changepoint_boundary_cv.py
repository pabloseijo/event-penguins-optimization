"""Refine candidate boundaries with local ATSN feature change points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_final_boundary_gradient_cv import (  # noqa: E402
    frame_prediction,
    prediction_frame,
)
from dev.diagnose_final_prediction_oracles import source_prediction_path  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument(
        "--feature-dir",
        default="tmp/temporalmaxer_continuous/source_features_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/feature_changepoint_boundary_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--window-points", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--radii-seconds", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--confidence-quantiles", type=float, nargs="+", default=[0.8, 0.9])
    return parser.parse_args()


def feature_changepoint_saliency(
    features: np.ndarray,
    window_points: int,
) -> np.ndarray:
    if window_points < 1:
        raise ValueError("window_points must be positive")
    values = np.asarray(features, dtype=np.float64)
    saliency = np.zeros(len(values), dtype=np.float64)
    for boundary in range(1, len(values)):
        before = values[max(0, boundary - window_points) : boundary].mean(axis=0)
        after = values[
            boundary : min(len(values), boundary + window_points)
        ].mean(axis=0)
        denominator = max(
            float(np.linalg.norm(before) * np.linalg.norm(after)),
            1e-8,
        )
        saliency[boundary] = 1.0 - np.clip(
            float(before @ after) / denominator,
            -1.0,
            1.0,
        )
    return saliency


def local_saliency_peak(
    saliency: np.ndarray,
    boundary_seconds: float,
    stride_seconds: float,
    radius_seconds: float,
) -> tuple[float, float]:
    center = boundary_seconds / stride_seconds
    radius = radius_seconds / stride_seconds
    low = max(1, int(np.ceil(center - radius)))
    high = min(len(saliency), int(np.floor(center + radius)) + 1)
    if high <= low:
        index = min(len(saliency) - 1, max(1, int(round(center))))
    else:
        local = saliency[low:high]
        maximum = float(local.max())
        tied = np.flatnonzero(np.isclose(local, maximum))
        index = low + int(tied[np.argmin(np.abs((low + tied) - center))])
    return index * stride_seconds, float(saliency[index])


def add_changepoint_saliency(
    features: np.ndarray,
    sequences: pd.DataFrame,
    window_points: int,
) -> dict[tuple[str, int], tuple[np.ndarray, float]]:
    result = {}
    for row in sequences.itertuples(index=False):
        values = np.asarray(
            features[int(row.offset) : int(row.offset) + int(row.length)],
            dtype=np.float32,
        )
        saliency = feature_changepoint_saliency(values, window_points)
        result[(str(row.rec_name), int(row.roi_id))] = (
            saliency,
            float(np.quantile(saliency[1:], 0.8)) if len(saliency) > 1 else 0.0,
        )
    return result


def snap_changepoint_boundaries(
    frame: pd.DataFrame,
    saliency_maps: dict[tuple[str, int], np.ndarray],
    stride_seconds: float,
    radius_seconds: float,
    blend: float,
    confidence_quantile: float,
    minimum_duration_seconds: float = 2.0,
) -> pd.DataFrame:
    starts = frame["t_start"].to_numpy(np.float64).copy()
    ends = frame["t_end"].to_numpy(np.float64).copy()
    for index, row in enumerate(frame.itertuples(index=False)):
        saliency = saliency_maps[(str(row.rec_name), int(row.roi_id))]
        threshold = (
            float(np.quantile(saliency[1:], confidence_quantile))
            if len(saliency) > 1
            else float("inf")
        )
        snapped_start, start_score = local_saliency_peak(
            saliency,
            starts[index],
            stride_seconds,
            radius_seconds,
        )
        snapped_end, end_score = local_saliency_peak(
            saliency,
            ends[index],
            stride_seconds,
            radius_seconds,
        )
        refined_start = (
            (1.0 - blend) * starts[index] + blend * snapped_start
            if start_score >= threshold
            else starts[index]
        )
        refined_end = (
            (1.0 - blend) * ends[index] + blend * snapped_end
            if end_score >= threshold
            else ends[index]
        )
        if refined_end - refined_start >= minimum_duration_seconds:
            starts[index] = refined_start
            ends[index] = refined_end
    output = frame.copy()
    output["t_start"] = starts
    output["t_end"] = ends
    return output


def main() -> None:
    args = parse_args()
    feature_dir = resolve(args.feature_dir)
    source_root = resolve(args.source_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    features = np.load(feature_dir / "frame_features.npy", mmap_mode="r")
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    rows = []

    for fold in args.folds:
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        selected_sequences = sequences[
            sequences["rec_name"].isin(recordings)
        ].copy()
        control = json.loads(
            source_prediction_path(source_root, fold).read_text(encoding="utf-8")
        )
        control_frame = prediction_frame(control, fold)
        variants = [("control", control_frame)]
        for window_points in args.window_points:
            saliency_maps = {
                key: value[0]
                for key, value in add_changepoint_saliency(
                    features,
                    selected_sequences,
                    window_points,
                ).items()
            }
            for radius in args.radii_seconds:
                for blend in args.blends:
                    for confidence_quantile in args.confidence_quantiles:
                        variant = (
                            f"change_w{window_points}_r{radius:g}"
                            f"_b{int(round(100 * blend)):03d}"
                            f"_q{int(round(100 * confidence_quantile)):02d}"
                        )
                        variants.append(
                            (
                                variant,
                                snap_changepoint_boundaries(
                                    control_frame,
                                    saliency_maps,
                                    float(metadata["grid_stride_s"]),
                                    radius,
                                    blend,
                                    confidence_quantile,
                                ),
                            )
                        )
        for variant, frame in variants:
            prediction = frame_prediction(frame, f"source-changepoint-{variant}")
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
        pd.DataFrame(rows).to_csv(out_dir / "fold_metrics_partial.csv", index=False)
        print(f"Completed fold {fold}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "folds": len(group),
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
