"""Evaluate label-free surrounding-context penalties on proposal quality scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_quality_head import ROOT, evaluate_score, softmax_ed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse proposal scores with adjacent ATSN context.")
    parser.add_argument("--scored-proposals", required=True)
    parser.add_argument("--previous-logits", required=True)
    parser.add_argument("--next-logits", required=True)
    parser.add_argument("--base-score-col", default="quality_avg_cnn")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--outer-betas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--gate-weights", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--rank-weights", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--min-score", type=float, nargs="+", default=[0.01, 0.05, 0.1])
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


def load_logits(path: str | Path, expected_rows: int) -> np.ndarray:
    logits = np.load(resolve(path), allow_pickle=False)["logits"].astype(np.float64)
    if len(logits) != expected_rows:
        raise ValueError(f"{path} has {len(logits)} rows, expected {expected_rows}")
    return logits


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(resolve(args.scored_proposals)).reset_index(drop=True)
    if args.base_score_col not in df:
        raise ValueError(f"Missing base score column: {args.base_score_col}")
    previous_logits = load_logits(args.previous_logits, len(df))
    next_logits = load_logits(args.next_logits, len(df))
    previous_score = softmax_ed(previous_logits, args.temperature)
    next_score = softmax_ed(next_logits, args.temperature)
    outer_score = np.maximum(previous_score, next_score)
    outer_margin = np.maximum(
        previous_logits[:, 1] - previous_logits[:, 0],
        next_logits[:, 1] - next_logits[:, 0],
    )
    inside_margin = df["cnn_margin"].to_numpy(dtype=np.float64)
    margin_contrast = inside_margin - outer_margin
    context_gate = sigmoid(margin_contrast)
    df["context_margin_contrast"] = margin_contrast
    context_rank = (
        df.groupby(["rec_name", "roi_id"])["context_margin_contrast"]
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )
    base = np.clip(df[args.base_score_col].to_numpy(dtype=np.float64), 1e-8, 1.0)

    score_columns = []
    for beta in args.outer_betas:
        col = f"context_outer_exp_b{beta:g}"
        df[col] = base * np.exp(-beta * outer_score)
        score_columns.append(col)
    for weight in args.gate_weights:
        col = f"context_margin_geom_w{weight:g}"
        df[col] = np.power(base, 1.0 - weight) * np.power(np.clip(context_gate, 1e-8, 1.0), weight)
        score_columns.append(col)
    for weight in args.rank_weights:
        col = f"context_margin_rank_w{weight:g}"
        df[col] = base * np.power(np.clip(context_rank, 1e-8, 1.0), weight)
        score_columns.append(col)

    rows = []
    for col in score_columns:
        rows.extend(evaluate_score(df, col, col, args, pred_dir, suffix="context_fusion"))
    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    df.to_csv(out_dir / "scored_context_fusions.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
