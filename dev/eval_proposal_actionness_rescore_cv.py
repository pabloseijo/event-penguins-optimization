"""Rescore high-recall proposal segments with source-OOF actionness."""

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

from dev.diagnose_continuous_point_scores import pyramid_point_score  # noqa: E402
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import (  # noqa: E402
    average_outputs,
    load_models,
)
from src.utils import temporal_soft_nms  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


@torch.no_grad()
def extract_actionness(models, loader, device: torch.device) -> dict[tuple[str, int], np.ndarray]:
    result = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = average_outputs([model(features, mask) for model in models])
            scores = pyramid_point_score(
                output, include_quality=False
            ).float().cpu().numpy()
        for index, (recording, roi_id) in enumerate(
            zip(batch["rec_name"], batch["roi_id"])
        ):
            valid_length = int(batch["mask"][index].sum())
            result[(str(recording), int(roi_id))] = scores[index, :valid_length].copy()
    return result


def segment_actionness(
    frame: pd.DataFrame,
    actionness: dict[tuple[str, int], np.ndarray],
    stride_s: float,
    mode: str,
    context_ratio: float,
) -> np.ndarray:
    if mode not in {"mean", "completeness"}:
        raise ValueError(f"Unknown actionness score mode: {mode}")
    scores = np.zeros(len(frame), dtype=np.float64)
    for index, row in enumerate(frame.itertuples(index=False)):
        sequence = actionness[(str(row.rec_name), int(row.roi_id))]
        start = max(0, int(np.floor(float(row.t_start) / stride_s)))
        end = min(len(sequence), int(np.ceil(float(row.t_end) / stride_s)))
        if end <= start:
            center = min(
                len(sequence) - 1,
                max(0, int(round(0.5 * (float(row.t_start) + float(row.t_end)) / stride_s))),
            )
            scores[index] = float(sequence[center])
        else:
            inside_mean = float(sequence[start:end].mean())
            if mode == "mean":
                scores[index] = inside_mean
                continue
            context_length = max(1, int(round((end - start) * context_ratio)))
            context_parts = []
            if start > 0:
                context_parts.append(sequence[max(0, start - context_length) : start])
            if end < len(sequence):
                context_parts.append(sequence[end : min(len(sequence), end + context_length)])
            outside_mean = (
                float(np.concatenate(context_parts).mean())
                if context_parts
                else inside_mean
            )
            scores[index] = inside_mean - outside_mean
    return scores


def local_boundary_contrast(
    sequence: np.ndarray,
    boundary: int,
    window: int,
    boundary_type: str,
) -> float:
    before = sequence[max(0, boundary - window) : boundary]
    after = sequence[boundary : min(len(sequence), boundary + window)]
    if len(before) == 0 or len(after) == 0:
        return float("-inf")
    if boundary_type == "start":
        return float(after.mean() - before.mean())
    if boundary_type == "end":
        return float(before.mean() - after.mean())
    raise ValueError(f"Unknown boundary type: {boundary_type}")


def snap_actionness_boundaries(
    frame: pd.DataFrame,
    actionness: dict[tuple[str, int], np.ndarray],
    stride_s: float,
    blend: float,
    radius_s: float,
    window_s: float,
    min_duration_s: float = 2.0,
) -> pd.DataFrame:
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Boundary blend must be in [0, 1]")
    if blend == 0.0:
        return frame.copy()
    radius = max(1, int(round(radius_s / stride_s)))
    window = max(1, int(round(window_s / stride_s)))
    output = frame.copy()
    starts = output["t_start"].to_numpy(dtype=np.float64).copy()
    ends = output["t_end"].to_numpy(dtype=np.float64).copy()
    for index, row in enumerate(output.itertuples(index=False)):
        sequence = actionness[(str(row.rec_name), int(row.roi_id))]
        snapped = []
        for value, boundary_type in (
            (starts[index], "start"),
            (ends[index], "end"),
        ):
            center = min(len(sequence) - 1, max(1, int(round(value / stride_s))))
            candidates = range(
                max(1, center - radius),
                min(len(sequence) - 1, center + radius) + 1,
            )
            best = max(
                candidates,
                key=lambda candidate: local_boundary_contrast(
                    sequence, candidate, window, boundary_type
                ),
            )
            snapped.append(best * stride_s)
        refined_start = (1.0 - blend) * starts[index] + blend * snapped[0]
        refined_end = (1.0 - blend) * ends[index] + blend * snapped[1]
        if refined_end - refined_start >= min_duration_s:
            starts[index] = refined_start
            ends[index] = refined_end
    output["t_start"] = starts
    output["t_end"] = ends
    return output


def blended_score_frame(
    proposal: dict,
    actionness: dict[tuple[str, int], np.ndarray],
    stride_s: float,
    actionness_weight: float,
    score_mode: str,
    context_ratio: float,
    boundary_blend: float = 0.0,
    boundary_radius_s: float = 2.0,
    boundary_window_s: float = 2.0,
) -> pd.DataFrame:
    frame = prediction_rows(proposal, "proposal").drop(columns=["rank_score", "model"])
    frame = snap_actionness_boundaries(
        frame,
        actionness,
        stride_s,
        boundary_blend,
        boundary_radius_s,
        boundary_window_s,
    )
    frame["actionness_score"] = segment_actionness(
        frame, actionness, stride_s, score_mode, context_ratio
    )
    proposal_rank = frame["raw_score"].rank(method="average", pct=True).to_numpy(np.float64)
    actionness_rank = frame["actionness_score"].rank(
        method="average", pct=True
    ).to_numpy(np.float64)
    frame["raw_score"] = (
        (1.0 - actionness_weight) * proposal_rank
        + actionness_weight * actionness_rank
    )
    frame["rank_score"] = frame["raw_score"].rank(method="average", pct=True)
    frame["model"] = "proposal"
    return frame


def frame_prediction(
    frame: pd.DataFrame,
    sigma: float,
    max_predictions: int,
) -> dict:
    results: dict[str, dict[str, list]] = {}
    for (recording, roi_id), values in frame.groupby(["rec_name", "roi_id"], sort=False):
        detections = temporal_soft_nms(
            values[["t_start", "t_end", "raw_score"]].to_numpy(np.float64),
            sigma=sigma,
            score_threshold=1e-5,
        )[:max_predictions]
        results.setdefault(str(recording), {})[str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(score),
            }
            for start, end, score in detections
        ]
    return {"version": "proposal-actionness-rank-rescore-v1", "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1")
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/proposal_actionness_rescore_cv_v1"
    )
    parser.add_argument(
        "--actionness-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--score-modes", choices=("mean", "completeness"), nargs="+", default=["completeness"]
    )
    parser.add_argument("--context-ratio", type=float, default=0.5)
    parser.add_argument("--boundary-blends", type=float, nargs="+", default=[0.0])
    parser.add_argument("--boundary-radius", type=float, default=2.0)
    parser.add_argument("--boundary-window", type=float, default=2.0)
    parser.add_argument("--proposal-nms-sigma", type=float, default=0.5)
    parser.add_argument("--fusion-nms-sigma", type=float, default=0.5)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    feature_dir = resolve(args.feature_dir)
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    proposal_root = resolve(args.proposal_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    input_dim = int(metadata["feature_dim"])
    stride_s = float(metadata["grid_stride_s"])
    device = torch.device(args.device)
    rows = []

    for fold in range(5):
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        selected = sequences[sequences["rec_name"].isin(recordings)].copy()
        models = load_models(
            continuous_root,
            input_dim,
            device,
            [continuous_root / f"fold_{fold:02d}" / "best.pt"],
        )
        actionness = extract_actionness(
            models,
            make_loader(feature_dir, selected, args.batch_size, args.num_workers, device),
            device,
        )
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

        proposal_path = (
            proposal_root
            / f"fold_{fold:02d}"
            / "predictions"
            / f"{args.proposal_variant}.json"
        )
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        continuous_frame = prediction_rows(
            best_prediction(continuous_root, fold), "continuous"
        )
        event_frame = prediction_rows(best_prediction(event_root, fold), "event")
        settings = (
            (score_mode, boundary_blend, weight)
            for score_mode in args.score_modes
            for boundary_blend in args.boundary_blends
            for weight in args.actionness_weights
        )
        for score_mode, boundary_blend, weight in settings:
                proposal_frame = blended_score_frame(
                    proposal,
                    actionness,
                    stride_s,
                    weight,
                    score_mode,
                    args.context_ratio,
                    boundary_blend,
                    args.boundary_radius,
                    args.boundary_window,
                )
                suffix = (
                    f"{score_mode}_w{int(round(weight * 100)):03d}"
                    f"_b{int(round(boundary_blend * 100)):03d}"
                )
                proposal_prediction = frame_prediction(
                    proposal_frame, args.proposal_nms_sigma, args.max_predictions
                )
                proposal_out = (
                    out_dir
                    / "predictions"
                    / f"fold_{fold:02d}_proposal_{suffix}.json"
                )
                rows.append(
                    {
                        "fold": fold,
                        "variant": f"proposal_{suffix}",
                        "actionness_weight": weight,
                        "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                        **evaluate(
                            proposal_prediction,
                            recordings,
                            resolve(args.ann_path),
                            proposal_out,
                        ),
                    }
                )
                fusion = build_prediction(
                    [continuous_frame, event_frame, proposal_frame],
                    {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
                    sigma=args.fusion_nms_sigma,
                    per_model_topk=100,
                    max_predictions=args.max_predictions,
                )
                fusion_out = (
                    out_dir
                    / "predictions"
                    / f"fold_{fold:02d}_fusion_{suffix}.json"
                )
                rows.append(
                    {
                        "fold": fold,
                        "variant": f"fusion_{suffix}",
                        "actionness_weight": weight,
                        "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                        **evaluate(
                            fusion,
                            recordings,
                            resolve(args.ann_path),
                            fusion_out,
                        ),
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, values in metrics.groupby("variant", sort=False):
        instance_weights = values["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "mean_mAP": float(values["mAP"].mean()),
                "weighted_mAP": float(
                    np.average(values["mAP"], weights=instance_weights)
                ),
                "worst_mAP": float(values["mAP"].min()),
                "mean_AP@0.1": float(values["AP@0.1"].mean()),
                "mean_AP@0.3": float(values["AP@0.3"].mean()),
                "mean_AP@0.5": float(values["AP@0.5"].mean()),
                "mean_AP@0.7": float(values["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
