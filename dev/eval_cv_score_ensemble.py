"""Evaluate an equal-weight ensemble of aligned cross-validation predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_quality_head import ROOT, evaluate_score


KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average aligned CV proposal scores and boundaries.")
    parser.add_argument("--scored-proposals", required=True, nargs="+")
    parser.add_argument("--score-col", default="quality_score")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
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

    frames = [pd.read_csv(resolve(path)).reset_index(drop=True) for path in args.scored_proposals]
    if len(frames) < 2:
        raise ValueError("At least two scored proposal files are required")
    base = frames[0].copy()
    for index, frame in enumerate(frames):
        if len(frame) != len(base) or not base[KEY_COLUMNS].equals(frame[KEY_COLUMNS]):
            raise ValueError(f"Scored proposal file {index} is not aligned")
        if args.score_col not in frame:
            raise ValueError(f"Missing score column in file {index}: {args.score_col}")

    base["cv_mean_score"] = np.mean(
        [frame[args.score_col].to_numpy(dtype=np.float64) for frame in frames], axis=0
    )
    boundary_columns = {"refined_t_start", "refined_t_end"}
    if args.use_boundary_refinement:
        missing = [index for index, frame in enumerate(frames) if not boundary_columns <= set(frame.columns)]
        if missing:
            raise ValueError(f"Missing refined boundaries in files: {missing}")
        base["refined_t_start"] = np.mean(
            [frame["refined_t_start"].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )
        base["refined_t_end"] = np.mean(
            [frame["refined_t_end"].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )

    rows = evaluate_score(base, "cv_mean_score", "cv_equal_ensemble", args, pred_dir, "cv_mean")
    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    base.to_csv(out_dir / "scored_cv_ensemble.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
