"""Evaluate capacity limits in the final PAL-consistency fusion funnel."""

from __future__ import annotations

import argparse
import itertools
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
from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    fit_linear_qfl,
    score_quality_head,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
)
from src.evaluation import segment_iou  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def detection_recall(
    prediction: dict,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
    recordings: list[str],
    threshold: float,
) -> float:
    hits = []
    for (recording, roi_id), targets in annotations.items():
        if recording not in recordings:
            continue
        detections = prediction["results"].get(recording, {}).get(
            str(int(roi_id)), []
        )
        segments = np.asarray(
            [item["segment"] for item in detections], dtype=np.float64
        ).reshape(-1, 2)
        for target in targets:
            overlap = (
                float(segment_iou(np.asarray(target), segments).max())
                if len(segments)
                else 0.0
            )
            hits.append(overlap >= threshold)
    return float(np.mean(hits)) if hits else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features",
        default=(
            "tmp/temporalmaxer_continuous/actionness_qfl_cv_v1/"
            "candidate_features.csv"
        ),
    )
    parser.add_argument(
        "--base-root",
        default="tmp/temporalmaxer_continuous/cv_recipe_v1",
    )
    parser.add_argument(
        "--pal-root",
        default="tmp/temporalmaxer_continuous/cv_pal_consistency_pilot_v1",
    )
    parser.add_argument(
        "--event-root",
        default="tmp/temporalmaxer_continuous/cv_eventstats_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_pal_postprocess_cv_v1"
        ),
    )
    parser.add_argument(
        "--per-model-topk", type=int, nargs="+", default=(100, 200, 400)
    )
    parser.add_argument(
        "--max-predictions", type=int, nargs="+", default=(200, 400, 800)
    )
    parser.add_argument("--nms-sigma", type=float, default=0.5)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(resolve(args.candidate_features))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    annotations = load_annotations(resolve(args.ann_path), min_duration_s=2.0)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    settings = list(
        itertools.product(args.per_model_topk, args.max_predictions)
    )
    rows = []
    for fold in range(5):
        train = candidates[candidates["fold"] != fold].copy()
        validation = candidates[candidates["fold"] == fold].copy()
        model = fit_linear_qfl(
            train, device, args.steps, args.learning_rate
        )
        frames = [
            prediction_rows(
                best_prediction(resolve(args.base_root), fold), "base"
            ),
            prediction_rows(
                best_prediction(resolve(args.pal_root), fold), "pal_consistency"
            ),
            prediction_rows(
                best_prediction(resolve(args.event_root), fold), "event"
            ),
            score_quality_head(validation, model, args.score_blend),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for topk, maximum in settings:
            prediction = build_prediction(
                frames,
                {
                    "base": 0.1,
                    "pal_consistency": 0.1,
                    "event": 0.4,
                    "proposal": 0.4,
                },
                sigma=args.nms_sigma,
                per_model_topk=topk,
                max_predictions=maximum,
            )
            rows.append(
                {
                    "fold": fold,
                    "per_model_topk": topk,
                    "max_predictions": maximum,
                    "nms_sigma": args.nms_sigma,
                    "val_ed_instances": int(
                        manifest.loc[fold, "val_ed_instances"]
                    ),
                    "recall@0.5": detection_recall(
                        prediction, annotations, recordings, 0.5
                    ),
                    "recall@0.7": detection_recall(
                        prediction, annotations, recordings, 0.7
                    ),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir
                        / "predictions"
                        / f"fold{fold:02d}_top{topk}_max{maximum}.json",
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for (topk, maximum), group in metrics.groupby(
        ["per_model_topk", "max_predictions"]
    ):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "per_model_topk": int(topk),
                "max_predictions": int(maximum),
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(
                    np.average(group["mAP"], weights=weights)
                ),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
                "weighted_recall@0.5": float(
                    np.average(group["recall@0.5"], weights=weights)
                ),
                "weighted_recall@0.7": float(
                    np.average(group["recall@0.7"], weights=weights)
                ),
                "mean_predictions": float(group["n_predictions"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        "mean_mAP", ascending=False
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
