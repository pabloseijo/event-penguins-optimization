"""Evaluate source-OOF complementarity between two continuous experts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

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
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--secondary-root", required=True)
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument(
        "--secondary-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--nms-sigma", type=float, default=0.25)
    parser.add_argument("--per-model-topk", type=int, default=200)
    parser.add_argument("--max-predictions", type=int, default=200)
    args = parser.parse_args()

    if any(weight < 0.0 or weight > 1.0 for weight in args.secondary_weights):
        raise ValueError("Every secondary weight must lie in [0, 1]")

    primary_root = resolve(args.primary_root)
    secondary_root = resolve(args.secondary_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in args.folds:
        primary = prediction_rows(best_prediction(primary_root, fold), "primary")
        secondary = prediction_rows(best_prediction(secondary_root, fold), "secondary")
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for secondary_weight in args.secondary_weights:
            secondary_weight = float(secondary_weight)
            prediction = build_prediction(
                [primary, secondary],
                {
                    "primary": 1.0 - secondary_weight,
                    "secondary": secondary_weight,
                },
                sigma=args.nms_sigma,
                per_model_topk=args.per_model_topk,
                max_predictions=args.max_predictions,
            )
            rows.append(
                {
                    "fold": fold,
                    "primary_weight": 1.0 - secondary_weight,
                    "secondary_weight": secondary_weight,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir
                        / "predictions"
                        / f"fold{fold:02d}_secondary{secondary_weight:g}.json",
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for secondary_weight, group in metrics.groupby("secondary_weight"):
        instance_weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "primary_weight": 1.0 - float(secondary_weight),
                "secondary_weight": float(secondary_weight),
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(
                    np.average(group["mAP"], weights=instance_weights)
                ),
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


if __name__ == "__main__":
    main()
