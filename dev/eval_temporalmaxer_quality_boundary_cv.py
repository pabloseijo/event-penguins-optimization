"""Pair quality-focused TESPEC scores with fixed full-model TESPEC boundaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.eval_temporalmaxer_groupdro_cv import (  # noqa: E402
    RANKING_COLUMNS,
    add_ranking_fusions,
)
from dev.train_temporalmaxer_dense import (  # noqa: E402
    evaluate_variant,
    load_cache,
    make_model,
    map_to_master,
    score_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source")
    parser.add_argument(
        "--master-proposals",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument(
        "--cache-dir", default="tmp/temporalmaxer_dense/hybrid_source_cache"
    )
    parser.add_argument(
        "--event-feature-cache-dir",
        default="tmp/temporalmaxer_dense/tespec_hybrid_source",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="tmp/temporalmaxer_dense/screened_cv_tespec_quality_groupdro",
    )
    parser.add_argument(
        "--quality-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_tespec_quality_groupdro_eval",
    )
    parser.add_argument(
        "--boundary-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_tespec_combined_eval",
    )
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_dense/hybrid_cv_tespec_quality_fixed_boundary",
    )
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
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


def aligned(frame: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    positions = map_to_master(frame, proposals)
    return frame.iloc[positions].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, logits, metadata = load_cache(resolve(args.cache_dir))
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {
        (int(row["fold"]), str(row["score_column"])) for row in rows
    }
    for fold in range(5):
        proposals = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        scored_path = out_dir / f"scored_fold_{fold:02d}.csv"
        quality_path = resolve(args.quality_root) / f"scored_fold_{fold:02d}.csv"
        if scored_path.exists():
            scored = aligned(pd.read_csv(scored_path), proposals)
        elif quality_path.exists():
            scored = aligned(pd.read_csv(quality_path), proposals)
        else:
            checkpoint = torch.load(
                resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt",
                map_location=device,
                weights_only=False,
            )
            for name in (
                "hidden_dim",
                "pyramid_levels",
                "dropout",
                "trident_bins",
                "event_features_only",
            ):
                if name in checkpoint["args"]:
                    setattr(args, name, checkpoint["args"][name])
            model = make_model(metadata, args).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            scored = score_model(
                model,
                proposals,
                map_to_master(master, proposals),
                resolve(args.cache_dir) / "frame_features.npy",
                logits,
                args,
                device,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        boundary = aligned(
            pd.read_csv(resolve(args.boundary_root) / f"scored_fold_{fold:02d}.csv"),
            proposals,
        )
        groupdro = aligned(
            pd.read_csv(
                resolve(args.groupdro_root)
                / f"fold_{fold:02d}"
                / "cache"
                / "val_scores_qhead_qfl_only.csv"
            ),
            proposals,
        )
        scored["quality_score"] = groupdro["quality_score"].to_numpy(dtype=np.float64)
        scored = add_ranking_fusions(scored)
        scored["blend050_t_start"] = 0.5 * (
            scored["t_start"].to_numpy(dtype=np.float64)
            + boundary["delta_t_start"].to_numpy(dtype=np.float64)
        )
        scored["blend050_t_end"] = 0.5 * (
            scored["t_end"].to_numpy(dtype=np.float64)
            + boundary["delta_t_end"].to_numpy(dtype=np.float64)
        )
        scored.to_csv(scored_path, index=False)
        for score_column in RANKING_COLUMNS:
            if (fold, score_column) in completed:
                continue
            row = evaluate_variant(
                scored,
                score_column,
                "blend050",
                f"quality_boundary_fold_{fold:02d}",
                args,
                out_dir / "predictions",
            )
            row["fold"] = fold
            row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
            rows.append(row)
            completed.add((fold, score_column))
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for score_column, group in metrics.groupby("score_column"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        summary = {"score_column": score_column, "boundary_mode": "blend050"}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            summary[f"mean_{column}"] = float(values.mean())
            summary[f"weighted_{column}"] = float(np.average(values, weights=weights))
            summary[f"worst_{column}"] = float(values.min())
        summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
