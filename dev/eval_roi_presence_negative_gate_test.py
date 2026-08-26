"""Frozen test evaluation of the source-approved negative ROI gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_roi_presence_gate_cv import (  # noqa: E402
    FEATURE_COLUMNS,
    fit_presence_model,
    presence_probabilities,
    roi_bag_features,
)
from dev.eval_roi_presence_negative_gate_cv import (  # noqa: E402
    apply_negative_gate,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bags",
        default=(
            "tmp/temporalmaxer_continuous/roi_presence_gate_cv_v1/"
            "oof_presence_probabilities.csv"
        ),
    )
    parser.add_argument(
        "--source-configuration",
        default=(
            "tmp/temporalmaxer_continuous/"
            "roi_presence_nested_negative_gate_cv_v1/"
            "final_configuration.json"
        ),
    )
    parser.add_argument(
        "--test-prediction",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_pal_consistency_test_v1/predictions.json"
        ),
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        default=(
            "tmp/temporalmaxer_continuous/"
            "roi_presence_negative_gate_test_v1"
        ),
    )
    return parser.parse_args()


def unlabeled_prediction_bags(prediction: dict) -> pd.DataFrame:
    """Build test bag features without reading annotations."""
    rows = []
    for recording, rois in prediction["results"].items():
        for roi, detections in rois.items():
            rows.append(
                {
                    "rec_name": recording,
                    "roi_id": int(roi),
                    **roi_bag_features(detections),
                }
            )
    return pd.DataFrame(rows)


def subset_prediction(prediction: dict, recording: str) -> dict:
    return {
        "version": prediction["version"],
        "results": {recording: prediction["results"][recording]},
    }


def main() -> None:
    args = parse_args()
    source_bags = pd.read_csv(resolve(args.source_bags))
    missing = set(FEATURE_COLUMNS + ["target_present"]) - set(source_bags)
    if missing:
        raise ValueError(f"Source bag table lacks columns: {sorted(missing)}")
    configuration = json.loads(
        resolve(args.source_configuration).read_text(encoding="utf-8")
    )
    threshold = float(configuration["threshold"])
    factor = float(configuration["suppression_factor"])
    prediction = json.loads(
        resolve(args.test_prediction).read_text(encoding="utf-8")
    )

    model = fit_presence_model(source_bags, l2=args.l2)
    test_bags = unlabeled_prediction_bags(prediction)
    test_bags["presence_probability"] = presence_probabilities(test_bags, model)
    gated = apply_negative_gate(prediction, test_bags, threshold, factor)
    gated["version"] = "source-approved-negative-roi-gate-test-v1"

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_bags.to_csv(out_dir / "test_presence_probabilities.csv", index=False)
    (out_dir / "fitted_presence_model.json").write_text(
        json.dumps(
            {
                "l2": args.l2,
                "feature_columns": FEATURE_COLUMNS,
                "mean": model["mean"].tolist(),
                "scale": model["scale"].tolist(),
                "weights": model["weights"].tolist(),
                "bias": model["bias"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "predictions.json").write_text(
        json.dumps(gated), encoding="utf-8"
    )

    recordings = sorted(prediction["results"])
    control_metrics = evaluate(
        prediction,
        recordings,
        resolve(args.ann_path),
        out_dir / "control_predictions.json",
    )
    gated_metrics = evaluate(
        gated,
        recordings,
        resolve(args.ann_path),
        out_dir / "predictions.json",
    )
    per_recording = []
    for recording in recordings:
        control_recording = evaluate(
            subset_prediction(prediction, recording),
            [recording],
            resolve(args.ann_path),
            out_dir / "per_recording" / f"{recording}_control.json",
        )
        gated_recording = evaluate(
            subset_prediction(gated, recording),
            [recording],
            resolve(args.ann_path),
            out_dir / "per_recording" / f"{recording}_gated.json",
        )
        per_recording.append(
            {
                "rec_name": recording,
                "control_mAP": control_recording["mAP"],
                "gated_mAP": gated_recording["mAP"],
                "delta_mAP": (
                    gated_recording["mAP"] - control_recording["mAP"]
                ),
            }
        )
    pd.DataFrame(per_recording).to_csv(
        out_dir / "per_recording_metrics.csv", index=False
    )
    result = {
        "source_configuration": configuration,
        "test_flagged_rois": int(
            (test_bags["presence_probability"] < threshold).sum()
        ),
        "test_control": control_metrics,
        "test_negative_gate": gated_metrics,
        "test_delta": {
            key: float(gated_metrics[key]) - float(control_metrics[key])
            for key in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7")
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
