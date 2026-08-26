"""Evaluate TAG proposals from source-OOF continuous actionness maps.

TAG converts snippet actionness into variable-length proposals by grouping
high-actionness runs. This experiment keeps the trained detector frozen and
selects any fusion only with recording-disjoint source CV.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


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
from dev.eval_temporalmaxer_continuous_test import (  # noqa: E402
    average_outputs,
    load_models,
)
from dev.train_temporalmaxer_continuous import (  # noqa: E402
    ContinuousSequenceDataset,
    collate_sequences,
)
from src.utils import temporal_soft_nms  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def active_runs(mask: np.ndarray) -> np.ndarray:
    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return np.column_stack((np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def merge_runs(runs: np.ndarray, grouping_threshold: float) -> list[tuple[int, int]]:
    if len(runs) == 0:
        return []
    merged = []
    start, end = map(int, runs[0])
    active_length = end - start
    for next_start, next_end in runs[1:]:
        next_start, next_end = int(next_start), int(next_end)
        candidate_active = active_length + next_end - next_start
        candidate_span = next_end - start
        if candidate_active / candidate_span >= grouping_threshold:
            end = next_end
            active_length = candidate_active
        else:
            merged.append((start, end))
            start, end = next_start, next_end
            active_length = end - start
    merged.append((start, end))
    return merged


def tag_candidates(
    actionness: np.ndarray,
    stride_s: float,
    duration_s: float,
    thresholds: list[float],
    grouping_thresholds: list[float],
    min_duration_s: float,
    max_predictions: int,
    nms_sigma: float,
) -> np.ndarray:
    candidates: dict[tuple[int, int], float] = {}
    for threshold in thresholds:
        runs = active_runs(actionness >= threshold)
        for grouping_threshold in grouping_thresholds:
            for start, end in merge_runs(runs, grouping_threshold):
                segment_start = max(0.0, start * stride_s)
                segment_end = min(float(duration_s), end * stride_s)
                if segment_end - segment_start < min_duration_s:
                    continue
                score = float(actionness[start:end].mean())
                key = (start, end)
                candidates[key] = max(candidates.get(key, 0.0), score)
    if not candidates:
        return np.empty((0, 3), dtype=np.float64)
    values = np.asarray(
        [
            [start * stride_s, min(end * stride_s, duration_s), score]
            for (start, end), score in candidates.items()
        ],
        dtype=np.float64,
    )
    return temporal_soft_nms(values, sigma=nms_sigma, score_threshold=1e-5)[
        :max_predictions
    ]


@torch.no_grad()
def tag_prediction(
    models,
    loader: DataLoader,
    sequences: pd.DataFrame,
    grid_stride_s: float,
    device: torch.device,
    thresholds: list[float],
    grouping_thresholds: list[float],
    min_duration_s: float,
    max_predictions: int,
    nms_sigma: float,
) -> dict:
    results = {
        recording: {str(int(roi)): [] for roi in group["roi_id"].unique()}
        for recording, group in sequences.groupby("rec_name")
    }
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = average_outputs([model(features, mask) for model in models])
            actionness = pyramid_point_score(
                output, include_quality=False
            ).float().cpu().numpy()
        for index, (recording, roi_id, duration_s) in enumerate(
            zip(batch["rec_name"], batch["roi_id"], batch["duration_s"])
        ):
            valid_length = int(batch["mask"][index].sum())
            detections = tag_candidates(
                actionness[index, :valid_length],
                grid_stride_s,
                float(duration_s),
                thresholds,
                grouping_thresholds,
                min_duration_s,
                max_predictions,
                nms_sigma,
            )
            results[str(recording)][str(int(roi_id))] = [
                {
                    "label": "ed",
                    "segment": [float(start), float(end)],
                    "score": float(score),
                }
                for start, end, score in detections
            ]
    return {"version": "temporal-actionness-grouping-source-oof-v1", "results": results}


def make_loader(
    feature_dir: Path,
    sequences: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = ContinuousSequenceDataset(
        feature_dir / "frame_features.npy",
        sequences.reset_index(drop=True),
        annotations={},
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )


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
        "--out-dir", default="tmp/temporalmaxer_continuous/tag_cv_v1"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    parser.add_argument(
        "--grouping-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7]
    )
    parser.add_argument("--tag-weights", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--tag-nms-sigma", type=float, default=0.5)
    parser.add_argument("--fusion-nms-sigma", type=float, default=0.5)
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
        tag_path = out_dir / "predictions" / f"fold_{fold:02d}_tag.json"
        tag_path.parent.mkdir(parents=True, exist_ok=True)
        if tag_path.exists():
            tag = json.loads(tag_path.read_text(encoding="utf-8"))
        else:
            models = load_models(
                continuous_root,
                input_dim,
                device,
                [continuous_root / f"fold_{fold:02d}" / "best.pt"],
            )
            tag = tag_prediction(
                models,
                make_loader(feature_dir, selected, args.batch_size, args.num_workers, device),
                selected,
                stride_s,
                device,
                args.thresholds,
                args.grouping_thresholds,
                args.min_duration,
                args.max_predictions,
                args.tag_nms_sigma,
            )
            tag_path.write_text(json.dumps(tag), encoding="utf-8")
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(
            {
                "fold": fold,
                "variant": "tag_standalone",
                "tag_weight": 1.0,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **evaluate(tag, recordings, resolve(args.ann_path), tag_path),
            }
        )
        base_frames = [
            prediction_rows(best_prediction(continuous_root, fold), "continuous"),
            prediction_rows(best_prediction(event_root, fold), "event"),
            prediction_rows(
                json.loads(
                    (
                        proposal_root
                        / f"fold_{fold:02d}"
                        / "predictions"
                        / f"{args.proposal_variant}.json"
                    ).read_text()
                ),
                "proposal",
            ),
        ]
        tag_frame = prediction_rows(tag, "tag")
        for tag_weight in args.tag_weights:
            old_weight = 1.0 - tag_weight
            weights = {
                "continuous": 0.2 * old_weight,
                "event": 0.4 * old_weight,
                "proposal": 0.4 * old_weight,
                "tag": tag_weight,
            }
            prediction = build_prediction(
                [*base_frames, tag_frame],
                weights,
                args.fusion_nms_sigma,
                per_model_topk=100,
                max_predictions=args.max_predictions,
            )
            variant = f"fusion_tag_w{int(round(tag_weight * 100)):02d}"
            prediction_path = out_dir / "predictions" / f"fold_{fold:02d}_{variant}.json"
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "tag_weight": tag_weight,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction, recordings, resolve(args.ann_path), prediction_path
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, values in metrics.groupby("variant", sort=False):
        weights = values["val_ed_instances"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "mean_mAP": float(values["mAP"].mean()),
                "weighted_mAP": float(np.average(values["mAP"], weights=weights)),
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
