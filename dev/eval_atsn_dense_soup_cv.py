"""Evaluate a fixed 50/50 base/adapted boundary soup in source CV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_temporalmaxer_dense import evaluate_variant


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-root", default="tmp/temporalmaxer_dense/atsn_dense_firstblock_cv"
    )
    parser.add_argument(
        "--fold-zero-root",
        default="tmp/temporalmaxer_dense/atsn_dense_firstblock_pilot/fold_00",
    )
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--manifest", default="tmp/cv/recording_folds_r5/manifest.csv"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/atsn_dense_firstblock_soup_cv"
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def soup_boundaries(scored: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    output = scored.copy()
    for suffix in ("start", "end"):
        output[f"soup050_t_{suffix}"] = (
            (1.0 - alpha) * output[f"reference_t_{suffix}"]
            + alpha * output[f"blend050_t_{suffix}"]
        )
    return output


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_args = SimpleNamespace(
        min_score=0.1,
        pre_nms_topk_per_roi=1000,
        soft_nms_sigma=0.25,
        soft_nms_score_threshold=0.001,
        duration_dmax=60.0,
        duration_sigma=20.0,
        ann_path=args.ann_path,
        tiou=[0.1, 0.3, 0.5, 0.7],
    )
    partial_path = out_dir / "metrics_partial.csv"
    rows = (
        pd.read_csv(partial_path).to_dict("records")
        if partial_path.exists()
        else []
    )
    completed = {
        (int(row["fold"]), str(row["score_column"]), str(row["boundary_mode"]))
        for row in rows
    }
    for fold in args.folds:
        fold_dir = (
            resolve(args.fold_zero_root)
            if fold == 0
            else resolve(args.cv_root) / f"fold_{fold:02d}"
        )
        scored_path = fold_dir / f"scored_epoch_{args.epoch:02d}.csv"
        if not scored_path.exists():
            raise FileNotFoundError(scored_path)
        scored = soup_boundaries(pd.read_csv(scored_path))
        for score, boundary in (
            ("quality_score", "soup050"),
            ("qhead_brem_score", "blend050"),
            ("qhead_brem_score", "soup050"),
            ("qhead_brem_w020_score", "blend050"),
            ("qhead_brem_w020_score", "soup050"),
        ):
            if (fold, score, boundary) in completed:
                continue
            row = evaluate_variant(
                scored,
                score,
                boundary,
                f"firstblock_{score}_{boundary}_fold_{fold:02d}",
                eval_args,
                out_dir / "predictions",
            )
            row["fold"] = fold
            rows.append(row)
            completed.add((fold, score, boundary))
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    metrics = pd.DataFrame(rows)
    metrics = metrics[metrics["fold"].astype(int).isin(args.folds)].copy()
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    metrics["val_ed_instances"] = metrics["fold"].map(
        manifest["val_ed_instances"].to_dict()
    )
    metrics.to_csv(out_dir / "all_metrics.csv", index=False)
    summaries = []
    for (score, boundary), group in metrics.groupby(
        ["score_column", "boundary_mode"]
    ):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        summary = {
            "score_column": score,
            "boundary_mode": boundary,
            "mean_mAP": float(group["mAP"].mean()),
            "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
            "worst_mAP": float(group["mAP"].min()),
        }
        for column in ("AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            summary[f"mean_{column}"] = float(group[column].mean())
            summary[f"weighted_{column}"] = float(
                np.average(group[column], weights=weights)
            )
            summary[f"worst_{column}"] = float(group[column].min())
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries).sort_values(
        "weighted_mAP", ascending=False
    )
    summary_frame.to_csv(out_dir / "summary.csv", index=False)
    print(metrics.to_string(index=False))
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
