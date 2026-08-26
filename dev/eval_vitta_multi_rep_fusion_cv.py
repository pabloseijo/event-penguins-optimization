"""Evaluate the fixed three-expert fusion with ViTTA continuous predictions."""

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
    parser.add_argument(
        "--vitta-root", default="tmp/temporalmaxer_continuous/cv_vitta_gradient_v1"
    )
    parser.add_argument(
        "--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1"
    )
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/vitta_multi_rep_fusion_cv_v1"
    )
    args = parser.parse_args()

    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    oof_results = {}
    for fold in range(5):
        frames = [
            prediction_rows(
                json.loads(
                    (
                        resolve(args.vitta_root)
                        / "predictions"
                        / f"fold_{fold:02d}.json"
                    ).read_text()
                ),
                "continuous",
            ),
            prediction_rows(best_prediction(resolve(args.event_root), fold), "event"),
            prediction_rows(
                json.loads(
                    (
                        resolve(args.proposal_root)
                        / f"fold_{fold:02d}"
                        / "predictions"
                        / f"{args.proposal_variant}.json"
                    ).read_text()
                ),
                "proposal",
            ),
        ]
        prediction = build_prediction(
            frames,
            {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
            sigma=0.5,
            per_model_topk=100,
            max_predictions=200,
        )
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        metrics = evaluate(
            prediction,
            recordings,
            resolve(args.ann_path),
            out_dir / "predictions" / f"fold_{fold:02d}.json",
        )
        rows.append(
            {
                "fold": fold,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **metrics,
            }
        )
        oof_results.update(prediction["results"])

    frame = pd.DataFrame(rows).sort_values("fold")
    frame.to_csv(out_dir / "metrics.csv", index=False)
    weights = frame["val_ed_instances"].to_numpy(np.float64)
    summary = {
        "mean_mAP": float(frame["mAP"].mean()),
        "weighted_mAP": float(np.average(frame["mAP"], weights=weights)),
        "worst_mAP": float(frame["mAP"].min()),
        "mean_AP@0.7": float(frame["AP@0.7"].mean()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "oof_predictions.json").write_text(
        json.dumps({"version": "vitta-multi-representation-oof-v1", "results": oof_results}),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
