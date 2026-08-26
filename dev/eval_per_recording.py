"""Break down a scored proposal file by recording and acquisition condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dev.train_quality_head import ROOT, evaluate_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proposal scores recording by recording.")
    parser.add_argument("--scored-proposals", required=True)
    parser.add_argument("--score-col", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--recording-info", default="config/annotations/recording_info.csv")
    parser.add_argument("--min-score", type=float, nargs="+", default=[0.1])
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--use-boundary-refinement", action="store_true", default=True)
    parser.add_argument("--no-boundary-refinement", dest="use_boundary_refinement", action="store_false")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(resolve(args.scored_proposals)).reset_index(drop=True)
    info = pd.read_csv(resolve(args.recording_info)).set_index("timestamp")
    with open(resolve(args.ann_path), encoding="utf-8") as handle:
        database = json.load(handle)["database"]

    recordings_with_gt = set()
    for recording, value in database.items():
        if any(
            annotation["label"] == "ed"
            and float(annotation["segment"][1]) - float(annotation["segment"][0]) >= 2.0
            for roi, annotations in value.get("annotations", {}).items()
            if roi != "null"
            for annotation in annotations
        ):
            recordings_with_gt.add(recording)

    rows = []
    for recording, group in df.groupby("rec_name", sort=True):
        if recording not in recordings_with_gt:
            row = {
                "variant": str(recording),
                "score_col": args.score_col,
                "min_score": float(args.min_score[0]),
                "n_pred": 0,
                "mAP": float("nan"),
                "rec_name": recording,
                "without_gt": True,
            }
            if recording in info.index:
                for column in ["split", "precipitation", "night", "ed_cnt", "event_count"]:
                    row[column] = info.loc[recording, column]
            rows.append(row)
            continue
        metrics = evaluate_score(
            group.reset_index(drop=True),
            args.score_col,
            str(recording),
            args,
            pred_dir,
            "per_recording",
        )
        for row in metrics:
            row["rec_name"] = recording
            row["without_gt"] = False
            if recording in info.index:
                for column in ["split", "precipitation", "night", "ed_cnt", "event_count"]:
                    row[column] = info.loc[recording, column]
            rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mAP")
    summary.to_csv(out_dir / "summary.csv", index=False)
    columns = [
        "rec_name", "precipitation", "night", "ed_cnt", "mAP",
        "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7", "n_pred",
    ]
    print(summary[[column for column in columns if column in summary]].to_string(index=False))


if __name__ == "__main__":
    main()
