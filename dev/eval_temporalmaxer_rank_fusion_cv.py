"""Select a conservative GroupDRO/TemporalMaxer score fusion with source CV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.train_temporalmaxer_dense import evaluate_variant, map_to_master  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--temporal-eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--boundary-blend", type=float, default=0.50)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    args = parse_args()
    fold_dir = resolve(args.fold_dir)
    temporal_dir = resolve(args.temporal_eval_dir)
    groupdro_root = resolve(args.groupdro_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [("quality_score", "groupdro")]
    for dense_column in ("dense_quality", "dense_score"):
        for beta in args.betas:
            variants.append(
                (f"fusion_{dense_column}_{int(round(beta * 100)):02d}", dense_column)
            )

    for fold in args.folds:
        metrics_path = out_dir / f"metrics_fold_{fold:02d}.csv"
        if metrics_path.exists():
            print(f"[FOLD {fold:02d}] reutilizado")
            continue
        proposals = pd.read_csv(fold_dir / f"fold_{fold:02d}" / "val_proposals.csv")
        scored = pd.read_csv(temporal_dir / f"scored_fold_{fold:02d}.csv")
        positions = map_to_master(scored, proposals)
        scored = scored.iloc[positions].reset_index(drop=True)
        groupdro = pd.read_csv(
            groupdro_root
            / f"fold_{fold:02d}"
            / "cache"
            / "val_scores_qhead_qfl_only.csv"
        )
        positions = map_to_master(groupdro, proposals)
        selected = groupdro.iloc[positions].reset_index(drop=True)
        scored["quality_score"] = selected["quality_score"].to_numpy(dtype=np.float64)
        alpha = float(args.boundary_blend)
        scored["hybrid_t_start"] = (
            (1.0 - alpha) * scored["t_start"] + alpha * scored["delta_t_start"]
        )
        scored["hybrid_t_end"] = (
            (1.0 - alpha) * scored["t_end"] + alpha * scored["delta_t_end"]
        )
        for score_column, dense_column in variants[1:]:
            beta = float(score_column.rsplit("_", 1)[-1]) / 100.0
            scored[score_column] = (
                (1.0 - beta) * scored["quality_score"]
                + beta * scored[dense_column]
            )

        fold_rows = []
        for score_column, _ in variants:
            fold_rows.append(
                evaluate_variant(
                    scored,
                    score_column,
                    "hybrid",
                    f"rank_fusion_fold_{fold:02d}",
                    args,
                    out_dir / "predictions",
                )
            )
        frame = pd.DataFrame(fold_rows)
        frame.insert(0, "fold", fold)
        frame.to_csv(metrics_path, index=False)
        print(f"[FOLD {fold:02d}] completo")

    metric_paths = [out_dir / f"metrics_fold_{fold:02d}.csv" for fold in range(5)]
    if not all(path.exists() for path in metric_paths):
        print("[INFO] Agardando polos folds restantes; non se agrega ainda.")
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    metrics["val_ed_instances"] = metrics["fold"].map(
        manifest["val_ed_instances"].astype(int)
    )
    metrics.to_csv(out_dir / "all_metrics.csv", index=False)
    rows = []
    for score_column, group in metrics.groupby("score_column"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row = {"score_column": score_column}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
