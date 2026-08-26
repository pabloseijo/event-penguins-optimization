"""Evaluate an aligned auxiliary ATSN temporal view and fixed score fusions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_quality_head import ROOT, evaluate_score, softmax_ed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse proposal scores with aligned ATSN logits.")
    parser.add_argument("--scored-proposals", required=True)
    parser.add_argument("--aux-logits", required=True)
    parser.add_argument("--base-score-col", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--skip-scored-output",
        action="store_true",
        help="Do not write the large proposal-level fusion CSV.",
    )
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--score-cols", nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-score", type=float, nargs="+", default=[0.02])
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=0)
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
        raise ValueError(f"Missing base score column: {args.base_score_col}")
    logits = np.load(resolve(args.aux_logits), allow_pickle=False)["logits"]
    if len(logits) != len(scored):
        raise ValueError(f"Auxiliary logits rows={len(logits)}, proposals={len(scored)}")

    base = np.clip(scored[args.base_score_col].to_numpy(dtype=np.float64), 1e-8, 1.0)
    auxiliary = np.clip(softmax_ed(logits, args.temperature), 1e-8, 1.0)
    scored["aux_score"] = auxiliary
    score_columns = [args.base_score_col, "aux_score"]
    for weight in args.weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Fusion weights must be in [0, 1]")
        linear = f"aux_linear_w{weight:g}"
        geometric = f"aux_geometric_w{weight:g}"
        rescue = f"aux_rescue_w{weight:g}"
        scored[linear] = (1.0 - weight) * base + weight * auxiliary
        scored[geometric] = np.power(base, 1.0 - weight) * np.power(auxiliary, weight)
        scored[rescue] = np.maximum(base, weight * auxiliary)
        score_columns.extend([linear, geometric, rescue])

    if args.score_cols:
        unknown = sorted(set(args.score_cols) - set(score_columns))
        if unknown:
            raise ValueError(f"Unknown score columns: {unknown}")
        score_columns = args.score_cols

    if not args.skip_scored_output:
        scored.to_csv(out_dir / "scored_aux_fusions.csv", index=False)
    rows = []
    for score_column in score_columns:
        rows.extend(evaluate_score(scored, score_column, score_column, args, pred_dir, "aux_fusion"))
    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
