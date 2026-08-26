"""OOF fusion of ATSN, two event representations, and proposal-local TAD."""

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

from dev.eval_continuous_multi_rep_fusion_cv import (
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
    resolve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-v1-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument("--event-v2-root", default="tmp/temporalmaxer_continuous/cv_eventv2_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/four_rep_fusion_cv_v1"
    )
    parser.add_argument("--event-v2-weights", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument("--nms-sigma", type=float, default=0.5)
    parser.add_argument("--per-model-topk", type=int, default=100)
    parser.add_argument("--max-predictions", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    continuous_root = resolve(args.continuous_root)
    event_v1_root = resolve(args.event_v1_root)
    event_v2_root = resolve(args.event_v2_root)
    proposal_root = resolve(args.proposal_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    rows = []
    for fold in range(5):
        proposal_path = (
            proposal_root / f"fold_{fold:02d}" / "predictions" / f"{args.proposal_variant}.json"
        )
        frames = [
            prediction_rows(best_prediction(continuous_root, fold), "continuous"),
            prediction_rows(best_prediction(event_v1_root, fold), "event_v1"),
            prediction_rows(best_prediction(event_v2_root, fold), "event_v2"),
            prediction_rows(json.loads(proposal_path.read_text()), "proposal"),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for event_v2_weight in args.event_v2_weights:
            event_v1_weight = 0.4
            if event_v2_weight <= 0.0:
                raise ValueError("Event-v2 weight must be positive")
            label = f"fold{fold:02d}_ev1{event_v1_weight:g}_ev2{event_v2_weight:g}"
            prediction = build_prediction(
                frames,
                {
                    "continuous": 0.2,
                    "event_v1": event_v1_weight,
                    "event_v2": event_v2_weight,
                    "proposal": 0.4,
                },
                args.nms_sigma,
                args.per_model_topk,
                args.max_predictions,
            )
            rows.append(
                {
                    "fold": fold,
                    "continuous_weight": 0.2,
                    "event_v1_weight": event_v1_weight,
                    "event_v2_weight": event_v2_weight,
                    "proposal_weight": 0.4,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"{label}.json",
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for event_v2_weight, group in metrics.groupby("event_v2_weight"):
        instance_weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "event_v1_weight": 0.4,
                "event_v2_weight": event_v2_weight,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=instance_weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
