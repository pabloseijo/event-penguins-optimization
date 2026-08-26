"""Evaluate one fixed TESPEC/TISM/GroupDRO fusion on external source CV.

The candidate is intentionally fixed: an equal geometric mean of GroupDRO,
TESPEC BREM, and TISM BREM scores, with the source-approved TESPEC blend050
boundaries. No weights or boundary modes are swept.
"""

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
    parser.add_argument(
        "--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source"
    )
    parser.add_argument(
        "--tespec-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_tespec_combined_eval",
    )
    parser.add_argument(
        "--tism-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_tism_combined_eval",
    )
    parser.add_argument(
        "--groupdro-root", default="tmp/quality_head/family_groupdro_cv"
    )
    parser.add_argument(
        "--manifest", default="tmp/cv/recording_folds_r5/manifest.csv"
    )
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_dense/hybrid_cv_tespec_tism_fixed",
    )
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def geometric_trimodal_score(
    groupdro: np.ndarray,
    tespec_brem: np.ndarray,
    tism_brem: np.ndarray,
) -> np.ndarray:
    values = np.stack((groupdro, tespec_brem, tism_brem), axis=0)
    return np.prod(np.clip(values, 0.0, 1.0), axis=0) ** (1.0 / 3.0)


def aligned(frame: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    positions = map_to_master(frame, proposals)
    return frame.iloc[positions].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    metrics = []
    for fold in range(5):
        proposals = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        tespec = aligned(
            pd.read_csv(resolve(args.tespec_root) / f"scored_fold_{fold:02d}.csv"),
            proposals,
        )
        tism = aligned(
            pd.read_csv(resolve(args.tism_root) / f"scored_fold_{fold:02d}.csv"),
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
        scored["quality_score"] = groupdro["quality_score"].to_numpy(
            dtype=np.float64
        )
        scored["tism_brem_score"] = tism["brem_score"].to_numpy(dtype=np.float64)
        scored["trimodal_score"] = geometric_trimodal_score(
            scored["quality_score"].to_numpy(dtype=np.float64),
            scored["brem_score"].to_numpy(dtype=np.float64),
            scored["tism_brem_score"].to_numpy(dtype=np.float64),
        )
        scored["blend050_t_start"] = 0.5 * (
            scored["t_start"] + scored["delta_t_start"]
        )
        scored["blend050_t_end"] = 0.5 * (
            scored["t_end"] + scored["delta_t_end"]
        )
        scored.to_csv(out_dir / f"scored_fold_{fold:02d}.csv", index=False)
        row = evaluate_variant(
            scored,
            "trimodal_score",
            "blend050",
            f"cross_representation_fold_{fold:02d}",
            args,
            out_dir / "predictions",
        )
        row["fold"] = fold
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        metrics.append(row)
        pd.DataFrame(metrics).to_csv(out_dir / "metrics_partial.csv", index=False)

    frame = pd.DataFrame(metrics)
    frame.to_csv(out_dir / "metrics.csv", index=False)
    weights = frame["val_ed_instances"].to_numpy(dtype=np.float64)
    summary = {"score_column": "trimodal_score", "boundary_mode": "blend050"}
    for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
        values = frame[column].to_numpy(dtype=np.float64)
        summary[f"mean_{column}"] = float(values.mean())
        summary[f"weighted_{column}"] = float(np.average(values, weights=weights))
        summary[f"worst_{column}"] = float(values.min())
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
