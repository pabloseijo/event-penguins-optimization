"""Source-validation calibration of continuous TemporalMaxer scores and boundaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_boundary_score_voting_cv import vote_detection_boundaries
from dev.train_temporalmaxer_continuous import (
    ContinuousSequenceDataset,
    collate_sequences,
    load_annotations,
)
from src.evaluation import DetectionsEvaluator
from src.temporalmaxer_continuous import TemporalMaxerContinuous
from src.utils import temporal_soft_nms


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1")
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--candidate-topk", type=int, default=1000)
    parser.add_argument("--pre-nms-topk", type=int, default=200)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--quality-powers", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--nms-sigmas", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--vote-tious", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--vote-blends", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def extract_candidates(
    model: TemporalMaxerContinuous,
    loader: DataLoader,
    grid_stride_s: float,
    topk: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    for batch in tqdm(loader, desc="candidates"):
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(features, mask)
        for batch_index, (recording, roi_id, duration_s) in enumerate(
            zip(batch["rec_name"], batch["roi_id"], batch["duration_s"])
        ):
            per_level = []
            for level, (logits, quality, offsets, level_mask) in enumerate(
                zip(
                    output["classification_logits"],
                    output["quality_logits"],
                    output["offsets"],
                    output["masks"],
                )
            ):
                valid = level_mask[batch_index]
                indices = valid.nonzero(as_tuple=True)[0]
                stride = 2**level
                points = (indices.to(offsets.dtype) + 0.5) * stride
                distances = offsets[batch_index, indices] * stride
                segments = torch.stack(
                    (points - distances[:, 0], points + distances[:, 1]), dim=1
                ) * grid_stride_s
                segments[:, 0].clamp_(min=0.0)
                segments[:, 1].clamp_(max=float(duration_s))
                action = logits[batch_index, indices].sigmoid()
                location_quality = quality[batch_index, indices].sigmoid()
                keep = segments[:, 1] - segments[:, 0] >= 2.0
                if keep.any():
                    per_level.append(
                        torch.cat(
                            (segments[keep], action[keep, None], location_quality[keep, None]),
                            dim=1,
                        )
                    )
            candidates = torch.cat(per_level)
            if len(candidates) > topk:
                candidates = candidates[candidates[:, 2].topk(topk).indices]
            for start, end, action, quality in candidates.float().cpu().tolist():
                rows.append(
                    {
                        "rec_name": recording,
                        "roi_id": int(roi_id),
                        "t_start": start,
                        "t_end": end,
                        "action_score": action,
                        "quality_score": quality,
                    }
                )
    return pd.DataFrame(rows)


def build_prediction(
    candidates: pd.DataFrame,
    quality_power: float,
    nms_sigma: float,
    pre_nms_topk: int,
    max_predictions: int,
    vote_tiou: float | None = None,
    vote_blend: float = 0.0,
) -> dict:
    results = {
        recording: {str(int(roi)): [] for roi in group["roi_id"].unique()}
        for recording, group in candidates.groupby("rec_name")
    }
    for (recording, roi_id), group in candidates.groupby(["rec_name", "roi_id"]):
        score = group["action_score"].to_numpy() * np.power(
            group["quality_score"].to_numpy(), quality_power
        )
        order = np.argsort(score, kind="stable")[::-1][:pre_nms_topk]
        voters = np.column_stack(
            (
                group["t_start"].to_numpy()[order],
                group["t_end"].to_numpy()[order],
                score[order],
            )
        )
        detections = temporal_soft_nms(voters, sigma=nms_sigma, score_threshold=0.001)
        if vote_tiou is not None and vote_blend > 0.0:
            detections = vote_detection_boundaries(
                detections,
                voters[:, :2],
                voters[:, 2],
                tiou_threshold=vote_tiou,
                blend=vote_blend,
                topk=20,
                score_power=1.0,
                minimum_duration=2.0,
            )
        results[recording][str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(detection_score),
            }
            for start, end, detection_score in detections[:max_predictions]
            if end - start >= 2.0
        ]
    mode = f"q{quality_power:g}:sigma{nms_sigma:g}:vote{vote_tiou}:{vote_blend:g}"
    return {"version": f"temporalmaxer-continuous-sweep:{mode}", "results": results}


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
    metrics: dict[str, float | int] = {
        "mAP": float(evaluator.run()),
        "n_predictions": sum(
            len(detections)
            for rois in prediction["results"].values()
            for detections in rois.values()
        ),
    }
    for threshold, value in zip((0.1, 0.3, 0.5, 0.7), evaluator.mAP):
        metrics[f"AP@{threshold:.1f}"] = float(value)
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    manifest = pd.read_csv(resolve(args.fold_manifest)).set_index("fold")
    recordings = str(manifest.loc[args.fold, "val_record_names"]).split()
    val_sequences = sequences[sequences["rec_name"].isin(recordings)].copy()
    annotations = load_annotations(resolve(args.ann_path))
    dataset = ContinuousSequenceDataset(
        feature_dir / "frame_features.npy", val_sequences, annotations
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )
    checkpoint = torch.load(resolve(args.checkpoint), map_location=device, weights_only=False)
    checkpoint_args = checkpoint["args"]
    model = TemporalMaxerContinuous(
        input_dim=int(metadata["feature_dim"]),
        hidden_dim=int(checkpoint_args["hidden_dim"]),
        pyramid_levels=int(checkpoint_args["pyramid_levels"]),
        head_layers=int(checkpoint_args["head_layers"]),
        dropout=float(checkpoint_args["dropout"]),
        use_quality=not bool(checkpoint_args["disable_quality"]),
        reg_max=int(checkpoint_args.get("reg_max", 0)),
        center_sampling_radius=float(checkpoint_args.get("center_sampling_radius", 0.0)),
        use_boundary_heads=bool(checkpoint_args.get("use_boundary_heads", False)),
        boundary_refine_radius_seconds=float(
            checkpoint_args.get("boundary_refine_radius", 0.0)
            if checkpoint_args.get("use_boundary_heads", False)
            else 0.0
        ),
        boundary_refine_blend=float(checkpoint_args.get("boundary_refine_blend", 0.5)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "candidates.csv"
    if cache_path.exists():
        candidates = pd.read_csv(cache_path)
    else:
        candidates = extract_candidates(
            model, loader, float(metadata["grid_stride_s"]), args.candidate_topk, device
        )
        candidates.to_csv(cache_path, index=False)
    partial_path = out_dir / "metrics_partial.csv"
    rows = (
        pd.read_csv(partial_path).to_dict("records")
        if partial_path.exists()
        else []
    )
    completed = {str(row["variant"]) for row in rows}
    for quality_power in args.quality_powers:
        for sigma in args.nms_sigmas:
            label = f"quality_{quality_power:g}_sigma_{sigma:g}"
            if label in completed:
                continue
            prediction = build_prediction(
                candidates,
                quality_power,
                sigma,
                args.pre_nms_topk,
                args.max_predictions,
            )
            rows.append(
                {
                    "variant": label,
                    "quality_power": quality_power,
                    "nms_sigma": sigma,
                    "vote_tiou": np.nan,
                    "vote_blend": 0.0,
                    **evaluate_prediction(
                        prediction, recordings, args, out_dir / "predictions" / f"{label}.json"
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            completed.add(label)
    base_labels = {
        f"quality_{quality_power:g}_sigma_{sigma:g}"
        for quality_power in args.quality_powers
        for sigma in args.nms_sigmas
    }
    base_rows = [row for row in rows if str(row["variant"]) in base_labels]
    if len(base_rows) != len(base_labels):
        raise RuntimeError("Base sweep is incomplete")
    base = max(base_rows, key=lambda row: float(row["mAP"]))
    for vote_tiou in args.vote_tious:
        for vote_blend in args.vote_blends:
            label = f"{base['variant']}_vote_{vote_tiou:g}_{vote_blend:g}"
            if label in completed:
                continue
            prediction = build_prediction(
                candidates,
                float(base["quality_power"]),
                float(base["nms_sigma"]),
                args.pre_nms_topk,
                args.max_predictions,
                vote_tiou,
                vote_blend,
            )
            rows.append(
                {
                    "variant": label,
                    "quality_power": base["quality_power"],
                    "nms_sigma": base["nms_sigma"],
                    "vote_tiou": vote_tiou,
                    "vote_blend": vote_blend,
                    **evaluate_prediction(
                        prediction, recordings, args, out_dir / "predictions" / f"{label}.json"
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            completed.add(label)
    frame = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    frame.to_csv(out_dir / "metrics.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
