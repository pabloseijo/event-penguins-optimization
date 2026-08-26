"""Test source-OOF complementarity between shared and decoupled event experts."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--old-event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--cltdr-event-root",
        default="tmp/temporalmaxer_continuous/cv_cltdr_eventv2_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/actionness_qfl_event_expert_blend_cv_v1",
    )
    parser.add_argument(
        "--cltdr-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.4),
    )
    parser.add_argument("--continuous-weight", type=float, default=0.2)
    parser.add_argument("--event-budget", type=float, default=0.4)
    parser.add_argument("--proposal-weight", type=float, default=0.4)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not np.isclose(
        args.continuous_weight + args.event_budget + args.proposal_weight, 1.0
    ):
        raise ValueError("The continuous, event, and proposal budgets must sum to one")
    if any(weight < 0.0 or weight > args.event_budget for weight in args.cltdr_weights):
        raise ValueError("Every CLTDR weight must lie inside the fixed event budget")

    candidates = pd.read_csv(resolve(args.candidate_features))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {
        (int(row["fold"]), float(row["cltdr_event_weight"])) for row in rows
    }
    device = torch.device(args.device)

    for fold in range(5):
        train = candidates[candidates["fold"] != fold].copy()
        validation = candidates[candidates["fold"] == fold].copy()
        model = fit_linear_qfl(train, device, args.steps, args.learning_rate)
        proposal_frame = score_quality_head(validation, model, args.score_blend)
        frames = [
            prediction_rows(
                best_prediction(resolve(args.continuous_root), fold), "continuous"
            ),
            prediction_rows(
                best_prediction(resolve(args.old_event_root), fold), "old_event"
            ),
            prediction_rows(
                best_prediction(resolve(args.cltdr_event_root), fold), "cltdr_event"
            ),
            proposal_frame,
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for cltdr_weight in args.cltdr_weights:
            cltdr_weight = float(cltdr_weight)
            key = (fold, cltdr_weight)
            if key in completed:
                continue
            old_event_weight = args.event_budget - cltdr_weight
            prediction = build_prediction(
                frames,
                {
                    "continuous": args.continuous_weight,
                    "old_event": old_event_weight,
                    "cltdr_event": cltdr_weight,
                    "proposal": args.proposal_weight,
                },
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            rows.append(
                {
                    "fold": fold,
                    "continuous_weight": args.continuous_weight,
                    "old_event_weight": old_event_weight,
                    "cltdr_event_weight": cltdr_weight,
                    "proposal_weight": args.proposal_weight,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir
                        / "predictions"
                        / f"fold{fold:02d}_cltdr{cltdr_weight:g}.json",
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            completed.add(key)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for cltdr_weight, values in metrics.groupby("cltdr_event_weight"):
        instance_weights = values["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "continuous_weight": args.continuous_weight,
                "old_event_weight": args.event_budget - float(cltdr_weight),
                "cltdr_event_weight": float(cltdr_weight),
                "proposal_weight": args.proposal_weight,
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
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
