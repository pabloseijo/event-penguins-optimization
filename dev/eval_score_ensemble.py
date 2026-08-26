"""Evaluate a fixed weighted ensemble of aligned proposal-score files."""

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
    parser = argparse.ArgumentParser(description="Fuse aligned central and auxiliary proposal scores.")
    parser.add_argument("--base-scored", required=True)
    parser.add_argument("--aux-scored", required=True)
    parser.add_argument("--base-score-col", default="quality_avg_cnn")
    parser.add_argument("--aux-score-col", default="quality_score")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--modes", nargs="+", choices=["linear", "geometric"], default=["linear", "geometric"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-score", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=500)
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

    base_df = pd.read_csv(resolve(args.base_scored)).reset_index(drop=True)
    aux_df = pd.read_csv(resolve(args.aux_scored)).reset_index(drop=True)
    if len(base_df) != len(aux_df) or not base_df[KEY_COLUMNS].equals(aux_df[KEY_COLUMNS]):
        raise ValueError("Score files are not aligned on proposal identity and order")
    for frame, column, label in (
        (base_df, args.base_score_col, "base"),
        (aux_df, args.aux_score_col, "auxiliary"),
    ):
        if column not in frame:
            raise ValueError(f"Missing {label} score column: {column}")

    base = np.clip(base_df[args.base_score_col].to_numpy(dtype=np.float64), 1e-8, 1.0)
    aux = np.clip(aux_df[args.aux_score_col].to_numpy(dtype=np.float64), 1e-8, 1.0)
    score_columns = []
    for weight in args.weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Ensemble weights must be in [0, 1]")
        if "linear" in args.modes:
            column = f"ensemble_linear_w{weight:g}"
            base_df[column] = (1.0 - weight) * base + weight * aux
            score_columns.append(column)
        if "geometric" in args.modes:
            column = f"ensemble_geometric_w{weight:g}"
            base_df[column] = np.power(base, 1.0 - weight) * np.power(aux, weight)
            score_columns.append(column)

    rows = []
    for column in score_columns:
        rows.extend(evaluate_score(base_df, column, column, args, pred_dir, suffix="score_ensemble"))
    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    base_df.to_csv(out_dir / "scored_ensembles.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
