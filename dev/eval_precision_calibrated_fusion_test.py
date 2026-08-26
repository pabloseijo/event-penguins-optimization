"""Apply the broad-pool precision calibrator to continuous+event test predictions, then fuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import os
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import build_prediction, evaluate, prediction_rows


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def load_calibrator(path: Path) -> tuple[np.ndarray, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["weights"], dtype=np.float64), float(data["bias"])


def calibrate_prediction(pred: dict, weights: np.ndarray, bias: float) -> dict:
    calibrated = {"version": pred.get("version", "v"), "results": {}}
    for rec, rois in pred["results"].items():
        calibrated["results"][rec] = {}
        for roi, dets in rois.items():
            new_dets = []
            for det in dets:
                duration = float(det["segment"][1]) - float(det["segment"][0])
                feats = np.asarray([float(det["score"]), np.log1p(duration)])
                logit = float(feats @ weights + bias)
                new_score = 1.0 / (1.0 + np.exp(-logit))
                new_dets.append({**det, "score": new_score})
            calibrated["results"][rec][roi] = new_dets
    return calibrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion-dir", default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1"
    )
    parser.add_argument(
        "--calibrator-dir", default="tmp/temporalmaxer_continuous/precision_calibrator_v1"
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_calibrated_v1"
    )
    parser.add_argument("--weights", type=float, nargs=3, default=(0.2, 0.4, 0.4))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fusion_dir = resolve(args.fusion_dir)
    calibrator_dir = resolve(args.calibrator_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    continuous = json.loads((fusion_dir / "continuous_ensemble.json").read_text())
    event = json.loads((fusion_dir / "event_ensemble.json").read_text())
    proposal = json.loads(
        resolve(
            "tmp/temporalmaxer_dense/multi_expert_boundary_voting_test/"
            "predictions/multi_median_blend050.json"
        ).read_text()
    )

    cont_w, cont_b = load_calibrator(calibrator_dir / "continuous_calibrator.json")
    event_w, event_b = load_calibrator(calibrator_dir / "event_calibrator.json")

    continuous_cal = calibrate_prediction(continuous, cont_w, cont_b)
    event_cal = calibrate_prediction(event, event_w, event_b)

    fused = build_prediction(
        [
            prediction_rows(continuous_cal, "continuous"),
            prediction_rows(event_cal, "event"),
            prediction_rows(proposal, "proposal"),
        ],
        {"continuous": args.weights[0], "event": args.weights[1], "proposal": args.weights[2]},
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    fused["version"] = "precision-calibrated-fusion-v1"
    metrics = evaluate(
        fused,
        sorted(set(continuous["results"]) | set(event["results"]) | set(proposal["results"])),
        resolve(args.ann_path),
        out_dir / "predictions.json",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
