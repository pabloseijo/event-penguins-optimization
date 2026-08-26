"""Source-CV post-processing ablation for the actionness QFL fusion."""

from __future__ import annotations

import argparse
import itertools
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
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/actionness_qfl_postprocess_cv_v1"
    )
    parser.add_argument("--per-model-topk", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--max-predictions", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--nms-sigma", type=float, nargs="+", default=[0.25, 0.5])
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
    device = torch.device(args.device)
    settings = list(
        itertools.product(args.per_model_topk, args.max_predictions, args.nms_sigma)
    )
    rows = []
    for fold in range(5):
        train = candidates[candidates["fold"] != fold].copy()
        validation = candidates[candidates["fold"] == fold].copy()
        model = fit_linear_qfl(train, device, args.steps, args.learning_rate)
        frames = [
            prediction_rows(best_prediction(continuous_root, fold), "continuous"),
            prediction_rows(best_prediction(event_root, fold), "event"),
            score_quality_head(validation, model, args.score_blend),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for per_model_topk, max_predictions, sigma in settings:
            prediction = build_prediction(
                frames,
                {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
                sigma=sigma,
                per_model_topk=per_model_topk,
                max_predictions=max_predictions,
            )
            label = (
                f"fold{fold:02d}_top{per_model_topk}_max{max_predictions}"
                f"_sigma{sigma:g}"
            )
            rows.append(
                {
                    "fold": fold,
                    "per_model_topk": per_model_topk,
                    "max_predictions": max_predictions,
                    "nms_sigma": sigma,
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
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for key, values in metrics.groupby(
        ["per_model_topk", "max_predictions", "nms_sigma"]
    ):
        instance_weights = values["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "per_model_topk": key[0],
                "max_predictions": key[1],
                "nms_sigma": key[2],
                "mean_mAP": float(values["mAP"].mean()),
                "weighted_mAP": float(np.average(values["mAP"], weights=instance_weights)),
                "worst_mAP": float(values["mAP"].min()),
                "mean_AP@0.1": float(values["AP@0.1"].mean()),
                "mean_AP@0.3": float(values["AP@0.3"].mean()),
                "mean_AP@0.5": float(values["AP@0.5"].mean()),
                "mean_AP@0.7": float(values["AP@0.7"].mean()),
                "mean_predictions": float(values["n_predictions"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
