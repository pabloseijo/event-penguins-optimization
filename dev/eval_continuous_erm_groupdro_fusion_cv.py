"""OOF fusion of complementary ERM and GroupDRO continuous detectors."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erm-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--groupdro-root", default="tmp/temporalmaxer_continuous/cv_groupdro_continuous_v1"
    )
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/erm_groupdro_fusion_cv_v1"
    )
    parser.add_argument(
        "--groupdro-weights", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2]
    )
    args = parser.parse_args()
    if any(not 0.0 <= value <= 0.2 for value in args.groupdro_weights):
        raise ValueError("GroupDRO weights must be in [0,0.2]")
    roots = {
        "erm": resolve(args.erm_root),
        "groupdro": resolve(args.groupdro_root),
        "event": resolve(args.event_root),
    }
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(5):
        proposal_path = (
            resolve(args.proposal_root)
            / f"fold_{fold:02d}"
            / "predictions"
            / f"{args.proposal_variant}.json"
        )
        frames = [
            prediction_rows(best_prediction(roots["erm"], fold), "erm"),
            prediction_rows(best_prediction(roots["groupdro"], fold), "groupdro"),
            prediction_rows(best_prediction(roots["event"], fold), "event"),
            prediction_rows(json.loads(proposal_path.read_text()), "proposal"),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for groupdro_weight in args.groupdro_weights:
            erm_weight = 0.2 - groupdro_weight
            prediction = build_prediction(
                frames,
                {
                    "erm": erm_weight,
                    "groupdro": groupdro_weight,
                    "event": 0.4,
                    "proposal": 0.4,
                },
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            rows.append(
                {
                    "fold": fold,
                    "erm_weight": erm_weight,
                    "groupdro_weight": groupdro_weight,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir
                        / "predictions"
                        / f"fold{fold:02d}_erm{erm_weight:g}_dro{groupdro_weight:g}.json",
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for groupdro_weight, group in metrics.groupby("groupdro_weight"):
        instance_weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "erm_weight": 0.2 - groupdro_weight,
                "groupdro_weight": groupdro_weight,
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
