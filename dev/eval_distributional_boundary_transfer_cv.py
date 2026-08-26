"""Transfer distributional-head boundaries to a fixed, source-validated ranking."""

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

from dev.eval_boundary_score_voting_cv import vote_detection_boundaries
from dev.eval_continuous_multi_rep_fusion_cv import evaluate


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-prediction", required=True)
    parser.add_argument("--voter", action="append", required=True)
    parser.add_argument("--voter-name", action="append", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vote-tious", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--vote-blends", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--vote-topk", type=int, default=20)
    return parser.parse_args()


def rows(prediction: dict, name: str) -> pd.DataFrame:
    values = []
    for recording, rois in prediction["results"].items():
        for roi_id, detections in rois.items():
            for detection in detections:
                values.append(
                    {
                        "rec_name": str(recording),
                        "roi_id": int(roi_id),
                        "t_start": float(detection["segment"][0]),
                        "t_end": float(detection["segment"][1]),
                        "score": float(detection["score"]),
                        "model": name,
                    }
                )
    frame = pd.DataFrame(values)
    frame["voter_score"] = frame["score"].rank(method="average", pct=True)
    return frame


def transfer(seed: dict, voters: pd.DataFrame, tiou: float, blend: float, topk: int) -> dict:
    output = json.loads(json.dumps(seed))
    for recording, rois in output["results"].items():
        for roi_id, detections in rois.items():
            if not detections:
                continue
            selected = voters[
                (voters["rec_name"] == str(recording))
                & (voters["roi_id"] == int(roi_id))
            ]
            if selected.empty:
                continue
            detection_values = np.asarray(
                [
                    [item["segment"][0], item["segment"][1], item["score"]]
                    for item in detections
                ],
                dtype=np.float64,
            )
            refined = vote_detection_boundaries(
                detection_values,
                selected[["t_start", "t_end"]].to_numpy(np.float64),
                selected["voter_score"].to_numpy(np.float64),
                tiou_threshold=tiou,
                blend=blend,
                topk=topk,
                score_power=1.0,
                minimum_duration=2.0,
            )
            for item, value in zip(detections, refined):
                item["segment"] = [float(value[0]), float(value[1])]
    output["version"] = f"distributional-boundary-transfer:t{tiou:g}:b{blend:g}"
    return output


def main() -> None:
    args = parse_args()
    if len(args.voter) != len(args.voter_name):
        raise ValueError("Each voter requires one voter-name")
    seed = json.loads(resolve(args.seed_prediction).read_text(encoding="utf-8"))
    voter_frames = [
        rows(json.loads(resolve(path).read_text(encoding="utf-8")), name)
        for path, name in zip(args.voter, args.voter_name)
    ]
    voter_sets = [(name, frame) for name, frame in zip(args.voter_name, voter_frames)]
    if len(voter_frames) > 1:
        voter_sets.append(("all", pd.concat(voter_frames, ignore_index=True)))
    recordings = str(
        pd.read_csv(resolve(args.manifest)).set_index("fold").loc[args.fold, "val_record_names"]
    ).split()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for name, frame in voter_sets:
        for tiou in args.vote_tious:
            for blend in args.vote_blends:
                label = f"{name}_tiou{tiou:g}_blend{blend:g}"
                prediction = transfer(seed, frame, tiou, blend, args.vote_topk)
                rows_out.append(
                    {
                        "variant": label,
                        "voter": name,
                        "vote_tiou": tiou,
                        "vote_blend": blend,
                        **evaluate(
                            prediction,
                            recordings,
                            resolve(args.ann_path),
                            out_dir / "predictions" / f"{label}.json",
                        ),
                    }
                )
    metrics = pd.DataFrame(rows_out).sort_values("mAP", ascending=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
