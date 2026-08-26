"""Refine final detection boundaries from local actionness transitions.

The refinement is label-free at inference time. Source annotations are used
only to evaluate recording-disjoint folds and are never consulted by the
transition detector.
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

from dev.eval_actionness_profile_quality_head_cv import (  # noqa: E402
    PROFILE_COLUMNS,
    add_profiles,
)
from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import extract_actionness  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402


START_GRADIENT_INDEX = 7
END_GRADIENT_INDEX = 23
PROFILE_INTERVALS_PER_DURATION = 16.0


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
    parser.add_argument(
        "--continuous-root",
        default="tmp/temporalmaxer_continuous/cv_recipe_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/final_boundary_gradient_cv_v1",
    )
    parser.add_argument("--context-ratio", type=float, default=0.5)
    parser.add_argument("--search-radii", type=int, nargs="+", default=[2, 4])
    parser.add_argument(
        "--refinement-blends",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def source_prediction_path(root: Path, fold: int) -> Path:
    return root / "predictions" / f"fold{fold:02d}_cw0.2_ew0.4_pw0.4.json"


def prediction_frame(prediction: dict, fold: int) -> pd.DataFrame:
    rows = []
    for recording, rois in prediction["results"].items():
        for roi_id, detections in rois.items():
            for detection in detections:
                rows.append(
                    {
                        "fold": fold,
                        "rec_name": str(recording),
                        "roi_id": int(roi_id),
                        "t_start": float(detection["segment"][0]),
                        "t_end": float(detection["segment"][1]),
                        "score": float(detection["score"]),
                    }
                )
    return pd.DataFrame(rows)


def frame_prediction(frame: pd.DataFrame, version: str) -> dict:
    recordings = sorted(frame["rec_name"].unique())
    results: dict[str, dict[str, list[dict]]] = {
        str(recording): {} for recording in recordings
    }
    for (recording, roi_id), group in frame.groupby(["rec_name", "roi_id"]):
        ordered = group.sort_values("score", ascending=False)
        results[str(recording)][str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(row.t_start), float(row.t_end)],
                "score": float(row.score),
            }
            for row in ordered.itertuples(index=False)
        ]
    return {"version": version, "results": results}


def smooth_profiles(profiles: np.ndarray) -> np.ndarray:
    padded = np.pad(profiles, ((0, 0), (1, 1)), mode="edge")
    return (
        0.25 * padded[:, :-2]
        + 0.5 * padded[:, 1:-1]
        + 0.25 * padded[:, 2:]
    )


def local_transition_indices(
    profiles: np.ndarray,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if search_radius < 1:
        raise ValueError("search_radius must be positive")
    gradients = np.diff(smooth_profiles(profiles), axis=1)
    start_low = START_GRADIENT_INDEX - search_radius
    start_high = START_GRADIENT_INDEX + search_radius + 1
    end_low = END_GRADIENT_INDEX - search_radius
    end_high = END_GRADIENT_INDEX + search_radius + 1
    start_region = gradients[:, start_low:start_high]
    end_region = gradients[:, end_low:end_high]
    start_local = np.argmax(start_region, axis=1)
    end_local = np.argmin(end_region, axis=1)
    rows = np.arange(len(profiles))
    start_indices = start_low + start_local
    end_indices = end_low + end_local
    return (
        start_indices,
        end_indices,
        start_region[rows, start_local],
        end_region[rows, end_local],
    )


def refine_boundaries(
    frame: pd.DataFrame,
    blend: float,
    search_radius: int,
    minimum_duration_s: float = 2.0,
) -> pd.DataFrame:
    profiles = frame[list(PROFILE_COLUMNS)].to_numpy(np.float64)
    start_indices, end_indices, start_strength, end_strength = (
        local_transition_indices(profiles, search_radius)
    )
    starts = frame["t_start"].to_numpy(np.float64)
    ends = frame["t_end"].to_numpy(np.float64)
    durations = np.maximum(ends - starts, minimum_duration_s)
    start_offsets = (
        (start_indices - START_GRADIENT_INDEX)
        / PROFILE_INTERVALS_PER_DURATION
        * durations
    )
    end_offsets = (
        (end_indices - END_GRADIENT_INDEX)
        / PROFILE_INTERVALS_PER_DURATION
        * durations
    )
    refined_starts = starts + blend * start_offsets * (start_strength > 0.0)
    refined_ends = ends + blend * end_offsets * (end_strength < 0.0)
    valid = refined_ends - refined_starts >= minimum_duration_s
    output = frame.copy()
    output["t_start"] = np.where(valid, refined_starts, starts)
    output["t_end"] = np.where(valid, refined_ends, ends)
    output["start_transition"] = start_strength
    output["end_transition"] = end_strength
    return output


def main() -> None:
    args = parse_args()
    source_root = resolve(args.source_root)
    feature_dir = resolve(args.feature_dir)
    continuous_root = resolve(args.continuous_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_cache = out_dir / "final_detection_profiles.csv"
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")

    if profile_cache.exists():
        profiled = pd.read_csv(profile_cache)
    else:
        metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
        sequences = pd.read_csv(feature_dir / "sequences.csv")
        device = torch.device(args.device)
        parts = []
        for fold in range(5):
            recordings = str(manifest.loc[fold, "val_record_names"]).split()
            selected_sequences = sequences[
                sequences["rec_name"].isin(recordings)
            ].copy()
            models = load_models(
                continuous_root,
                int(metadata["feature_dim"]),
                device,
                [continuous_root / f"fold_{fold:02d}" / "best.pt"],
            )
            actionness = extract_actionness(
                models,
                make_loader(
                    feature_dir,
                    selected_sequences,
                    args.batch_size,
                    args.num_workers,
                    device,
                ),
                device,
            )
            prediction = json.loads(
                source_prediction_path(source_root, fold).read_text(encoding="utf-8")
            )
            frame = prediction_frame(prediction, fold)
            parts.append(
                add_profiles(
                    frame,
                    actionness,
                    float(metadata["grid_stride_s"]),
                    args.context_ratio,
                )
            )
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
        profiled = pd.concat(parts, ignore_index=True)
        profiled.to_csv(profile_cache, index=False)

    rows = []
    for fold in range(5):
        validation = profiled[profiled["fold"] == fold].copy()
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        variants: list[tuple[str, pd.DataFrame]] = [("control", validation)]
        for radius in args.search_radii:
            for blend in args.refinement_blends:
                label = f"gradient_r{radius}_w{int(round(100 * blend)):03d}"
                variants.append(
                    (label, refine_boundaries(validation, blend, radius))
                )
        for variant, frame in variants:
            prediction = frame_prediction(
                frame,
                f"source-boundary-gradient-{variant}",
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
