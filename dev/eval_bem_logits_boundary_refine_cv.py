"""Refine canonical detections with explicit BEM start/end probabilities.

Unlike actionness-gradient snapping, this experiment consumes heads supervised
directly with Gaussian start/end targets. Candidate scores and ordering remain
unchanged; only their temporal boundaries can move.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_final_boundary_gradient_cv import (  # noqa: E402
    frame_prediction,
    prediction_frame,
)
from dev.eval_temporalmaxer_continuous_test import (  # noqa: E402
    average_outputs,
    load_models,
)


BoundaryMaps = Dict[Tuple[str, int], Dict[str, np.ndarray]]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument(
        "--feature-dir",
        default="tmp/temporalmaxer_continuous/source_features_v1",
    )
    parser.add_argument(
        "--bem-root",
        default="tmp/temporalmaxer_continuous/cv_bem_base_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/bem_logits_boundary_refine_cv_v1",
    )
    parser.add_argument("--map-modes", nargs="+", default=["level0", "pyramid_mean"])
    parser.add_argument("--radii-seconds", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def source_prediction_path(root: Path, fold: int) -> Path:
    return root / "predictions" / f"fold{fold:02d}_cw0.2_ew0.4_pw0.4.json"


def resample_probabilities(
    logits: list[torch.Tensor],
    masks: list[torch.Tensor],
    sample_index: int,
    base_length: int,
) -> np.ndarray:
    values = []
    for level_logits, level_mask in zip(logits, masks):
        valid_length = int(level_mask[sample_index].sum())
        cropped = level_logits[sample_index, :valid_length].float()
        if valid_length == base_length:
            resized = cropped
        else:
            resized = F.interpolate(
                cropped[None, None, :],
                size=base_length,
                mode="linear",
                align_corners=False,
            )[0, 0]
        values.append(resized.sigmoid())
    return torch.stack(values).mean(0).cpu().numpy()


@torch.no_grad()
def extract_boundary_maps(models, loader, device: torch.device) -> BoundaryMaps:
    result: BoundaryMaps = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = average_outputs([model(features, mask) for model in models])
        for index, (recording, roi_id) in enumerate(
            zip(batch["rec_name"], batch["roi_id"])
        ):
            base_length = int(output["masks"][0][index].sum())
            result[(str(recording), int(roi_id))] = {
                "start_level0": output["start_boundary_logits"][0][
                    index, :base_length
                ].float().sigmoid().cpu().numpy(),
                "end_level0": output["end_boundary_logits"][0][
                    index, :base_length
                ].float().sigmoid().cpu().numpy(),
                "start_pyramid_mean": resample_probabilities(
                    output["start_boundary_logits"],
                    output["masks"],
                    index,
                    base_length,
                ),
                "end_pyramid_mean": resample_probabilities(
                    output["end_boundary_logits"],
                    output["masks"],
                    index,
                    base_length,
                ),
            }
    return result


def closest_boundary_peak(
    probabilities: np.ndarray,
    boundary_seconds: float,
    stride_seconds: float,
    radius_seconds: float,
) -> tuple[float, float]:
    center = boundary_seconds / stride_seconds - 0.5
    radius = radius_seconds / stride_seconds
    low = max(0, int(np.ceil(center - radius)))
    high = min(len(probabilities), int(np.floor(center + radius)) + 1)
    if high <= low:
        index = min(len(probabilities) - 1, max(0, int(round(center))))
    else:
        local = probabilities[low:high]
        maximum = float(local.max())
        tied = np.flatnonzero(np.isclose(local, maximum))
        index = low + int(tied[np.argmin(np.abs((low + tied) - center))])
    return (index + 0.5) * stride_seconds, float(probabilities[index])


def snap_bem_boundaries(
    frame: pd.DataFrame,
    boundary_maps: BoundaryMaps,
    stride_seconds: float,
    map_mode: str,
    radius_seconds: float,
    blend: float,
    confidence_threshold: float,
    minimum_duration_seconds: float = 2.0,
) -> pd.DataFrame:
    starts = frame["t_start"].to_numpy(np.float64).copy()
    ends = frame["t_end"].to_numpy(np.float64).copy()
    start_confidence = np.zeros(len(frame), dtype=np.float64)
    end_confidence = np.zeros(len(frame), dtype=np.float64)
    for index, row in enumerate(frame.itertuples(index=False)):
        maps = boundary_maps[(str(row.rec_name), int(row.roi_id))]
        snapped_start, start_score = closest_boundary_peak(
            maps[f"start_{map_mode}"],
            starts[index],
            stride_seconds,
            radius_seconds,
        )
        snapped_end, end_score = closest_boundary_peak(
            maps[f"end_{map_mode}"],
            ends[index],
            stride_seconds,
            radius_seconds,
        )
        start_confidence[index] = start_score
        end_confidence[index] = end_score
        refined_start = (
            (1.0 - blend) * starts[index] + blend * snapped_start
            if start_score >= confidence_threshold
            else starts[index]
        )
        refined_end = (
            (1.0 - blend) * ends[index] + blend * snapped_end
            if end_score >= confidence_threshold
            else ends[index]
        )
        if refined_end - refined_start >= minimum_duration_seconds:
            starts[index] = refined_start
            ends[index] = refined_end
    output = frame.copy()
    output["t_start"] = starts
    output["t_end"] = ends
    output["bem_start_confidence"] = start_confidence
    output["bem_end_confidence"] = end_confidence
    return output


def main() -> None:
    args = parse_args()
    feature_dir = resolve(args.feature_dir)
    bem_root = resolve(args.bem_root)
    seed_root = resolve(args.seed_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    rows = []

    for fold in args.folds:
        if fold not in range(5):
            raise ValueError(f"Fold outside [0, 4]: {fold}")
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        selected_sequences = sequences[
            sequences["rec_name"].isin(recordings)
        ].copy()
        models = load_models(
            bem_root,
            int(metadata["feature_dim"]),
            device,
            [bem_root / f"fold_{fold:02d}" / "best.pt"],
        )
        boundary_maps = extract_boundary_maps(
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
        seed = json.loads(
            source_prediction_path(seed_root, fold).read_text(encoding="utf-8")
        )
        seed_frame = prediction_frame(seed, fold)
        variants: list[tuple[str, pd.DataFrame]] = [("control", seed_frame)]
        for map_mode in args.map_modes:
            for radius in args.radii_seconds:
                for blend in args.blends:
                    for threshold in args.confidence_thresholds:
                        variant = (
                            f"{map_mode}_r{radius:g}_w{int(round(100 * blend)):03d}"
                            f"_c{int(round(100 * threshold)):02d}"
                        )
                        variants.append(
                            (
                                variant,
                                snap_bem_boundaries(
                                    seed_frame,
                                    boundary_maps,
                                    float(metadata["grid_stride_s"]),
                                    map_mode,
                                    radius,
                                    blend,
                                    threshold,
                                ),
                            )
                        )
        for variant, frame in variants:
            prediction = frame_prediction(frame, f"source-bem-logits-{variant}")
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
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
