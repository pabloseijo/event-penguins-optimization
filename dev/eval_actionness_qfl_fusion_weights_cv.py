"""Reselect three-expert fusion weights after replacing the proposal scorer."""

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

from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    fit_linear_qfl,
    score_quality_head,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
    simplex_weights,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features",
        default="tmp/temporalmaxer_continuous/actionness_qfl_cv_v1/candidate_features.csv",
    )
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1"
    )
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    candidates = pd.read_csv(resolve(args.candidate_features))
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_grid = simplex_weights(args.weight_step)
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {
        (int(row["fold"]), float(row["continuous_weight"]), float(row["event_weight"]))
        for row in rows
    }
    device = torch.device(args.device)

    for fold in range(5):
        train = candidates[candidates["fold"] != fold].copy()
        validation = candidates[candidates["fold"] == fold].copy()
        model = fit_linear_qfl(train, device, args.steps, args.learning_rate)
        proposal_frame = score_quality_head(validation, model, args.score_blend)
        continuous_frame = prediction_rows(
            best_prediction(continuous_root, fold), "continuous"
        )
        event_frame = prediction_rows(best_prediction(event_root, fold), "event")
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for continuous_weight, event_weight, proposal_weight in weights_grid:
            key = (fold, continuous_weight, event_weight)
            if key in completed:
                continue
            prediction = build_prediction(
                [continuous_frame, event_frame, proposal_frame],
                {
                    "continuous": continuous_weight,
                    "event": event_weight,
                    "proposal": proposal_weight,
                },
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            label = (
                f"fold{fold:02d}_cw{continuous_weight:g}_ew{event_weight:g}"
                f"_pw{proposal_weight:g}"
            )
            rows.append(
                {
                    "fold": fold,
                    "continuous_weight": continuous_weight,
                    "event_weight": event_weight,
                    "proposal_weight": proposal_weight,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"{label}.json",
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            completed.add(key)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for key, values in metrics.groupby(
        ["continuous_weight", "event_weight", "proposal_weight"]
    ):
        instance_weights = values["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "continuous_weight": key[0],
                "event_weight": key[1],
                "proposal_weight": key[2],
                "mean_mAP": float(values["mAP"].mean()),
                "weighted_mAP": float(np.average(values["mAP"], weights=instance_weights)),
                "worst_mAP": float(values["mAP"].min()),
                "mean_AP@0.1": float(values["AP@0.1"].mean()),
                "mean_AP@0.3": float(values["AP@0.3"].mean()),
                "mean_AP@0.5": float(values["AP@0.5"].mean()),
                "mean_AP@0.7": float(values["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
