"""Compare source-selected consensus quality before versus after Soft-NMS."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-root",
        default="tmp/temporalmaxer_dense/salient_boundary_router_pilot",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/pre_nms_consensus_cv"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--score-column", default="quality_score")
    parser.add_argument("--nms-boundary", default="reference_blend050")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
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
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    for fold in args.folds:
        fold_out = out_dir / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        if (fold_out / "metrics.csv").exists():
            continue
        scored = pd.read_csv(
            resolve(args.scored_root) / f"fold_{fold:02d}" / "scored.csv"
        )
        sequences = sorted(scored["rec_name"].astype(str).unique())
        rows = []
        for stage in ("post", "pre"):
            label = f"consensus_score025_{stage}_nms"
            prediction = build_voted_prediction(
                scored,
                args,
                vote_tiou=0.5,
                vote_blend=0.5,
                consensus_score_blend=0.25,
                consensus_score_stage=stage,
            )
            row = evaluate_prediction(
                prediction,
                sequences,
                label,
                args,
                fold_out / "predictions" / f"{label}.json",
            )
            row["fold"] = fold
            row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
            rows.append(row)
            pd.DataFrame(rows).to_csv(fold_out / "metrics_partial.csv", index=False)
        pd.DataFrame(rows).to_csv(fold_out / "metrics.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)

    paths = [out_dir / f"fold_{fold:02d}" / "metrics.csv" for fold in range(5)]
    if not all(path.exists() for path in paths):
        return
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    rows = []
    for variant, group in metrics.groupby("variant"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row: dict[str, float | str] = {"variant": variant}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
