"""Test the source-selected proposal-consensus confidence fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dev.eval_boundary_score_voting_cv import (
    build_voted_prediction,
    evaluate_prediction,
    resolve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-proposals",
        default="tmp/temporalmaxer_dense/salient_boundary_router_test/scored.csv",
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/boundary_consensus_score_test"
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--score-column", default="quality_score")
    parser.add_argument("--nms-boundary", default="reference_blend050")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.set_defaults(vote_topk=20, vote_score_power=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists():
        print(pd.read_csv(metrics_path).to_string(index=False))
        return
    scored = pd.read_csv(resolve(args.scored_proposals))
    sequences = sorted(scored["rec_name"].astype(str).unique())
    rows = []
    for label, score_blend in (
        ("boundary_vote_control", 0.0),
        ("boundary_vote_consensus_score025", 0.25),
    ):
        prediction = build_voted_prediction(
            scored,
            args,
            vote_tiou=0.5,
            vote_blend=0.5,
            consensus_score_blend=score_blend,
        )
        row = evaluate_prediction(
            prediction,
            sequences,
            label,
            args,
            out_dir / "predictions" / f"{label}.json",
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "metrics_partial.csv", index=False)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    (out_dir / "recipe.json").write_text(
        json.dumps(
            {
                "source_cv": "boundary_consensus_score_pilot",
                "score_column": args.score_column,
                "nms_boundary": args.nms_boundary,
                "vote_tiou": 0.5,
                "vote_blend": 0.5,
                "vote_topk": args.vote_topk,
                "consensus_score_blend": 0.25,
                "consensus_confidence": "overlap-weighted mean voter score with Soft-NMS decay",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
