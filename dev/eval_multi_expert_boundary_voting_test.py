"""Test the source-selected robust fusion of TemporalMaxer boundary experts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.eval_boundary_score_voting_cv import (
    build_voted_prediction,
    evaluate_prediction,
    resolve,
)
from dev.eval_multi_expert_boundary_voting_cv import build_multi_expert_prediction
from dev.train_temporalmaxer_dense import stable_proposal_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-proposals",
        default="tmp/temporalmaxer_dense/salient_boundary_router_test/scored.csv",
    )
    parser.add_argument(
        "--temporal-root", default="tmp/temporalmaxer_dense/hybrid_test_eval"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_test"
    )
    parser.add_argument("--score-column", default="quality_score")
    parser.add_argument("--nms-boundary", default="reference_blend050")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.set_defaults(
        vote_tiou=0.5,
        vote_topk=20,
        multi_vote_topk=100,
        vote_score_power=1.0,
    )
    return parser.parse_args()


def add_test_expert_boundaries(
    scored: pd.DataFrame, temporal_frames: list[pd.DataFrame]
) -> pd.DataFrame:
    output = scored.copy()
    expected = stable_proposal_index(output)
    for fold, frame in enumerate(temporal_frames):
        if len(frame) != len(output) or not stable_proposal_index(frame).equals(expected):
            raise ValueError(f"Temporal test fold {fold} is not aligned")
    for target, source in (
        ("reference_delta", "delta"),
        ("reference_distribution", "distribution"),
        ("reference_point", "point"),
    ):
        for suffix in ("t_start", "t_end"):
            column = f"{source}_{suffix}"
            output[f"{target}_{suffix}"] = np.mean(
                [frame[column].to_numpy(dtype=np.float64) for frame in temporal_frames],
                axis=0,
            )
    return output


def main() -> None:
    args = parse_args()
    if args.min_action_duration < 0:
        raise ValueError("--min-action-duration must be non-negative")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists():
        print(pd.read_csv(metrics_path).to_string(index=False))
        return
    scored = pd.read_csv(resolve(args.scored_proposals)).reset_index(drop=True)
    frames = [
        pd.read_csv(
            resolve(args.temporal_root) / f"temporal_scored_fold_{fold:02d}.csv"
        ).reset_index(drop=True)
        for fold in range(5)
    ]
    scored = add_test_expert_boundaries(scored, frames)
    scored.to_csv(out_dir / "scored.csv", index=False)
    sequences = sorted(scored["rec_name"].astype(str).unique())
    rows = []
    settings = (
        (
            "single_expert_control",
            build_voted_prediction(
                scored, args, 0.5, 0.5, consensus_score_blend=0.25
            ),
        ),
        (
            "multi_median_blend050",
            build_multi_expert_prediction(scored, args, 0.5, "median"),
        ),
    )
    for label, prediction in settings:
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
                "source_cv": "multi_expert_boundary_voting_pilot_v2",
                "experts": [
                    "raw",
                    "reference_blend050",
                    "reference_delta",
                    "reference_distribution",
                    "reference_point",
                ],
                "estimator": "weighted_median",
                "boundary_blend": 0.5,
                "vote_tiou": 0.5,
                "multi_vote_topk": 100,
                "consensus_score_blend": 0.25,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
