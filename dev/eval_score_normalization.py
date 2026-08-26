"""Evaluate label-free group normalization of temporal proposal scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_quality_head import ROOT, evaluate_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize scored proposals within acquisition groups.")
    parser.add_argument("--scored-proposals", required=True)
    parser.add_argument("--base-score-col", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--only-score-col", default=None)
    parser.add_argument("--score-cols", nargs="+", default=None)
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
    scored = pd.read_csv(resolve(args.scored_proposals)).reset_index(drop=True)
    if args.base_score_col not in scored:
        raise ValueError(f"Missing score column: {args.base_score_col}")

    base = scored[args.base_score_col].to_numpy(dtype=np.float64)
    groupings = {
        "recording_rank": ["rec_name"],
        "roi_rank": ["rec_name", "roi_id"],
    }
    if "source" in scored:
        groupings["recording_source_rank"] = ["rec_name", "source"]

    normalized_columns = []
    for name, group_columns in groupings.items():
        scored[name] = scored.groupby(group_columns)[args.base_score_col].rank(method="average", pct=True)
        normalized_columns.append(name)

    score_columns = [args.base_score_col]
    for normalized_column in normalized_columns:
        normalized = scored[normalized_column].to_numpy(dtype=np.float64)
        score_columns.append(normalized_column)
        for weight in args.weights:
            if not 0.0 <= weight <= 1.0:
                raise ValueError("Normalization weights must be in [0, 1]")
            column = f"fusion_{normalized_column}_w{weight:g}"
            scored[column] = (1.0 - weight) * base + weight * normalized
            score_columns.append(column)

    if args.only_score_col:
        if args.only_score_col not in score_columns:
            raise ValueError(f"Unknown score column: {args.only_score_col}")
        score_columns = [args.only_score_col]
    if args.score_cols:
        unknown = sorted(set(args.score_cols) - set(score_columns))
        if unknown:
            raise ValueError(f"Unknown score columns: {unknown}")
        score_columns = args.score_cols

    scored.to_csv(out_dir / "scored_normalized.csv", index=False)
    rows = []
    for score_column in score_columns:
        rows.extend(evaluate_score(scored, score_column, score_column, args, pred_dir, "normalized"))
    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
