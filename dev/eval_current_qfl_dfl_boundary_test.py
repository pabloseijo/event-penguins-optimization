"""Frozen test of source-selected DFL boundaries on the canonical QFL seed."""

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
from dev.eval_distributional_boundary_transfer_cv import rows, transfer  # noqa: E402


SOURCE_VARIANT = "event_dfl_tiou0.5_blend0.5"
SOURCE_CONTROL_MAP = 0.8421706544882641


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-summary",
        default=(
            "tmp/temporalmaxer_continuous/current_qfl_boundary_transfer_cv_v1/"
            "summary.csv"
        ),
    )
    parser.add_argument(
        "--seed-prediction",
        default="tmp/temporalmaxer_continuous/actionness_qfl_test_v1/predictions.json",
    )
    parser.add_argument(
        "--dfl-voter",
        default=(
            "tmp/temporalmaxer_continuous/dfl_boundary_transfer_test_v1/"
            "event_dfl_ensemble.json"
        ),
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/current_qfl_dfl_boundary_test_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_summary = pd.read_csv(resolve(args.source_summary))
    selected = source_summary[source_summary["variant"] == SOURCE_VARIANT]
    if len(selected) != 1:
        raise ValueError(f"Missing unique source variant {SOURCE_VARIANT}")
    source_row = selected.iloc[0]
    if float(source_row["mean_mAP"]) <= SOURCE_CONTROL_MAP:
        raise ValueError("Frozen DFL transfer does not improve source control")

    seed = json.loads(resolve(args.seed_prediction).read_text(encoding="utf-8"))
    voter = json.loads(resolve(args.dfl_voter).read_text(encoding="utf-8"))
    prediction = transfer(
        seed,
        rows(voter, "event_dfl"),
        tiou=0.5,
        blend=0.5,
        topk=20,
    )
    prediction["version"] = "source-selected-current-qfl-dfl-boundary-test-v1"
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = sorted(seed["results"])
    control_metrics = evaluate(
        seed,
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
    result = {
        "source_cv": {
            "variant": SOURCE_VARIANT,
            "mean_mAP": float(source_row["mean_mAP"]),
            "weighted_mAP": float(source_row["weighted_mAP"]),
            "worst_mAP": float(source_row["worst_mAP"]),
            "mean_AP@0.7": float(source_row["mean_AP@0.7"]),
        },
        "control_metrics": control_metrics,
        "metrics": metrics,
        "delta_mAP": metrics["mAP"] - control_metrics["mAP"],
        "frozen_parameters": {
            "vote_tiou": 0.5,
            "vote_blend": 0.5,
            "vote_topk": 20,
            "scores_changed": False,
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
