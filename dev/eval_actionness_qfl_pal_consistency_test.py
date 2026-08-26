"""Frozen test evaluation of the source-approved PAL-consistency QFL blend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    build_prediction,
    evaluate,
    prediction_rows,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_pal_consistency_blend_cv_v1/summary.csv"
        ),
    )
    parser.add_argument(
        "--base-continuous",
        default=(
            "tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/"
            "continuous_ensemble.json"
        ),
    )
    parser.add_argument(
        "--pal-consistency-continuous",
        default=(
            "tmp/temporalmaxer_continuous/pal_consistency_test_v1/"
            "continuous_ensemble.json"
        ),
    )
    parser.add_argument(
        "--event-prediction",
        default=(
            "tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/"
            "event_ensemble.json"
        ),
    )
    parser.add_argument(
        "--proposal-frame",
        default=(
            "tmp/temporalmaxer_continuous/actionness_qfl_test_v1/"
            "proposal_frame.csv"
        ),
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_pal_consistency_test_v1"
        ),
    )
    return parser.parse_args()


def subset_prediction(prediction: dict, recording: str) -> dict:
    return {
        "version": prediction["version"],
        "results": {recording: prediction["results"][recording]},
    }


def main() -> None:
    args = parse_args()
    source = pd.read_csv(resolve(args.source_summary))
    source_row = source[
        source["old_event_weight"].round(8).eq(0.1)
        & source["cltdr_event_weight"].round(8).eq(0.1)
    ]
    if len(source_row) != 1:
        raise ValueError("Expected one source-approved 0.1/0.1 PAL blend")
    source_row = source_row.iloc[0]

    base = json.loads(resolve(args.base_continuous).read_text(encoding="utf-8"))
    pal = json.loads(
        resolve(args.pal_consistency_continuous).read_text(encoding="utf-8")
    )
    event = json.loads(resolve(args.event_prediction).read_text(encoding="utf-8"))
    if set(base["results"]) != set(pal["results"]) or set(base["results"]) != set(
        event["results"]
    ):
        raise ValueError("Test experts cover different recordings")
    proposal = pd.read_csv(resolve(args.proposal_frame)).copy()
    proposal["model"] = "proposal"

    base_frame = prediction_rows(base, "base")
    pal_frame = prediction_rows(pal, "pal_consistency")
    event_frame = prediction_rows(event, "event")
    control = build_prediction(
        [base_frame, event_frame, proposal],
        {"base": 0.2, "event": 0.4, "proposal": 0.4},
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction = build_prediction(
        [base_frame, pal_frame, event_frame, proposal],
        {
            "base": 0.1,
            "pal_consistency": 0.1,
            "event": 0.4,
            "proposal": 0.4,
        },
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction["version"] = "source-approved-pal-consistency-qfl-test-v1"

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = sorted(base["results"])
    control_metrics = evaluate(
        control,
        recordings,
        resolve(args.ann_path),
        out_dir / "control_predictions.json",
    )
    metrics = evaluate(
        prediction,
        recordings,
        resolve(args.ann_path),
        out_dir / "predictions.json",
    )
    per_recording = []
    for recording in recordings:
        per_recording.append(
            {
                "rec_name": recording,
                **evaluate(
                    subset_prediction(prediction, recording),
                    [recording],
                    resolve(args.ann_path),
                    out_dir / "per_recording" / f"{recording}.json",
                ),
            }
        )
    pd.DataFrame(per_recording).to_csv(
        out_dir / "per_recording_metrics.csv",
        index=False,
    )
    result = {
        "source_cv": {
            "base_weight": 0.1,
            "pal_consistency_weight": 0.1,
            "event_weight": 0.4,
            "proposal_weight": 0.4,
            "mean_mAP": float(source_row["mean_mAP"]),
            "weighted_mAP": float(source_row["weighted_mAP"]),
            "worst_mAP": float(source_row["worst_mAP"]),
            "mean_AP@0.7": float(source_row["mean_AP@0.7"]),
        },
        "test_control": control_metrics,
        "test_pal_consistency": metrics,
        "test_delta": {
            key: float(metrics[key]) - float(control_metrics[key])
            for key in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7")
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
