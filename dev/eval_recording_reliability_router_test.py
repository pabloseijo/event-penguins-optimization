"""Frozen historical-test evaluation of the source-selected reliability router."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.diagnose_recording_expert_reliability import (  # noqa: E402
    CANONICAL_WEIGHTS,
    prediction_rows,
    recording_features,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    build_prediction,
    evaluate,
)
from dev.eval_recording_reliability_router_cv import (  # noqa: E402
    FEATURE,
    routed_prediction,
    select_rule,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-diagnostic-root",
        default="tmp/temporalmaxer_continuous/recording_expert_reliability_v1",
    )
    parser.add_argument(
        "--continuous-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/continuous_ensemble.json",
    )
    parser.add_argument(
        "--event-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/event_ensemble.json",
    )
    parser.add_argument(
        "--proposal-frame",
        default="tmp/temporalmaxer_continuous/actionness_qfl_test_v1/proposal_frame.csv",
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/recording_reliability_router_test_v1",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6],
    )
    parser.add_argument("--min-train-routed", type=int, default=2)
    parser.add_argument("--min-proposal-weight", type=float, default=0.5)
    return parser.parse_args()


def subset_prediction(prediction: dict, recording: str) -> dict:
    return {
        "version": prediction["version"],
        "results": {recording: prediction["results"][recording]},
    }


def main() -> None:
    args = parse_args()
    source_root = resolve(args.source_diagnostic_root)
    source_features = pd.read_csv(source_root / "recording_features.csv")
    source_metrics = pd.read_csv(source_root / "recording_weight_metrics.csv")
    rule = select_rule(
        source_features,
        source_metrics,
        outer_fold=-1,
        thresholds=args.thresholds,
        min_train_routed=args.min_train_routed,
        min_proposal_weight=args.min_proposal_weight,
    )
    alternative_weights = (
        float(rule["continuous_weight"]),
        float(rule["event_weight"]),
        float(rule["proposal_weight"]),
    )

    continuous = json.loads(
        resolve(args.continuous_prediction).read_text(encoding="utf-8")
    )
    event = json.loads(resolve(args.event_prediction).read_text(encoding="utf-8"))
    proposal_frame = pd.read_csv(resolve(args.proposal_frame))
    frames = [
        prediction_rows(continuous, "continuous"),
        prediction_rows(event, "event"),
        proposal_frame,
    ]
    recordings = sorted(continuous["results"])
    feature_rows = [
        recording_features(frames, recording, topk=25)
        for recording in recordings
    ]
    target_features = pd.DataFrame(feature_rows)
    feature_values = dict(
        zip(
            target_features["rec_name"].astype(str),
            target_features[FEATURE].astype(float),
        )
    )

    canonical = build_prediction(
        frames,
        dict(zip(("continuous", "event", "proposal"), CANONICAL_WEIGHTS)),
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    alternative = build_prediction(
        frames,
        dict(zip(("continuous", "event", "proposal"), alternative_weights)),
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction, routed = routed_prediction(
        canonical,
        alternative,
        feature_values,
        float(rule["threshold"]),
        alternative_weights,
    )
    prediction["version"] = "source-selected-recording-reliability-router-test-v1"

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    control_metrics = evaluate(
        canonical,
        recordings,
        resolve(args.ann_path),
        out_dir / "control_predictions.json",
    )
    router_metrics = evaluate(
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
                "feature": feature_values[recording],
                "routed": recording in routed,
                **evaluate(
                    subset_prediction(prediction, recording),
                    [recording],
                    resolve(args.ann_path),
                    out_dir / "per_recording" / f"{recording}.json",
                ),
            }
        )
    target_features["threshold"] = float(rule["threshold"])
    target_features["routed"] = target_features["rec_name"].isin(routed)
    target_features.to_csv(out_dir / "recording_features.csv", index=False)
    pd.DataFrame(per_recording).to_csv(
        out_dir / "per_recording_metrics.csv",
        index=False,
    )
    result = {
        "source_rule": rule,
        "control_metrics": control_metrics,
        "router_metrics": router_metrics,
        "delta_mAP": router_metrics["mAP"] - control_metrics["mAP"],
        "routed_recordings": routed,
        "test_labels_used_for_routing": False,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(per_recording).to_string(index=False))


if __name__ == "__main__":
    main()
