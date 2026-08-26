"""Evaluate one BREM-style TESPEC quality-aware boundary gate on source CV."""

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
    parser.add_argument("--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source")
    parser.add_argument(
        "--tespec-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_tespec_combined_eval",
    )
    parser.add_argument(
        "--reference-root", default="tmp/temporalmaxer_dense/hybrid_cv_eval"
    )
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/hybrid_cv_tespec_boundary_gate"
    )
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


def aligned(frame: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    positions = map_to_master(frame, proposals)
    return frame.iloc[positions].reset_index(drop=True)


def quality_aware_boundaries(
    starts: np.ndarray,
    ends: np.ndarray,
    reference_delta_start: np.ndarray,
    reference_delta_end: np.ndarray,
    tespec_delta_start: np.ndarray,
    tespec_delta_end: np.ndarray,
    tespec_quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_start = 0.5 * (starts + reference_delta_start)
    reference_end = 0.5 * (ends + reference_delta_end)
    tespec_start = 0.5 * (starts + tespec_delta_start)
    tespec_end = 0.5 * (ends + tespec_delta_end)
    gate = np.clip(tespec_quality, 0.0, 1.0)
    return (
        reference_start + gate * (tespec_start - reference_start),
        reference_end + gate * (tespec_end - reference_end),
    )


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    rows = []
    for fold in range(5):
        proposals = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        tespec = aligned(
            pd.read_csv(resolve(args.tespec_root) / f"scored_fold_{fold:02d}.csv"),
            proposals,
        )
        reference = aligned(
            pd.read_csv(resolve(args.reference_root) / f"scored_fold_{fold:02d}.csv"),
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
        scored = tespec.copy()
        scored["quality_score"] = groupdro["quality_score"].to_numpy(dtype=np.float64)
        gated_start, gated_end = quality_aware_boundaries(
            scored["t_start"].to_numpy(dtype=np.float64),
            scored["t_end"].to_numpy(dtype=np.float64),
            reference["delta_t_start"].to_numpy(dtype=np.float64),
            reference["delta_t_end"].to_numpy(dtype=np.float64),
            scored["delta_t_start"].to_numpy(dtype=np.float64),
            scored["delta_t_end"].to_numpy(dtype=np.float64),
            scored["dense_quality"].to_numpy(dtype=np.float64),
        )
        scored["quality_gate_t_start"] = gated_start
        scored["quality_gate_t_end"] = gated_end
        scored.to_csv(out_dir / f"scored_fold_{fold:02d}.csv", index=False)
        row = evaluate_variant(
            scored,
            "quality_score",
            "quality_gate",
            f"tespec_boundary_gate_fold_{fold:02d}",
            args,
            out_dir / "predictions",
        )
        row["fold"] = fold
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    weights = metrics["val_ed_instances"].to_numpy(dtype=np.float64)
    summary = {"score_column": "quality_score", "boundary_mode": "quality_gate"}
    for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
        values = metrics[column].to_numpy(dtype=np.float64)
        summary[f"mean_{column}"] = float(values.mean())
        summary[f"weighted_{column}"] = float(np.average(values, weights=weights))
        summary[f"worst_{column}"] = float(values.min())
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
