"""Measure the source-CV ceiling of routing between learned boundary experts.

The oracle uses ground truth only as a diagnostic upper bound.  Its output must
never be reported as an operational result or used directly on test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.train_temporalmaxer_dense import (  # noqa: E402
    best_match,
    evaluate_variant,
    load_annotation_index,
    map_to_master,
    roi_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source")
    parser.add_argument(
        "--reference-root", default="tmp/temporalmaxer_dense/hybrid_cv_eval"
    )
    parser.add_argument(
        "--tespec-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_tespec_combined_eval",
    )
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/boundary_router_oracle_cv"
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def aligned(frame: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[map_to_master(frame, proposals)].reset_index(drop=True)


def stable_boundaries(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = np.maximum(np.asarray(start, dtype=np.float64), 0.0)
    end = np.asarray(end, dtype=np.float64).copy()
    center = 0.5 * (start + end)
    short = end - start < 2.0e6
    start[short] = np.maximum(0.0, center[short] - 1.0e6)
    end[short] = start[short] + 2.0e6
    return start, end


def boundary_candidates(
    proposals: pd.DataFrame,
    reference: pd.DataFrame,
    tespec: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    raw_start = proposals["t_start"].to_numpy(dtype=np.float64)
    raw_end = proposals["t_end"].to_numpy(dtype=np.float64)
    values: list[tuple[str, np.ndarray, np.ndarray]] = [("raw", raw_start, raw_end)]
    for prefix, frame in (("reference", reference), ("tespec", tespec)):
        delta_start = frame["delta_t_start"].to_numpy(dtype=np.float64)
        delta_end = frame["delta_t_end"].to_numpy(dtype=np.float64)
        values.extend(
            [
                (
                    f"{prefix}_blend050",
                    0.5 * (raw_start + delta_start),
                    0.5 * (raw_end + delta_end),
                ),
                (f"{prefix}_delta", delta_start, delta_end),
                (
                    f"{prefix}_distribution",
                    frame["distribution_t_start"].to_numpy(dtype=np.float64),
                    frame["distribution_t_end"].to_numpy(dtype=np.float64),
                ),
                (
                    f"{prefix}_point",
                    frame["point_t_start"].to_numpy(dtype=np.float64),
                    frame["point_t_end"].to_numpy(dtype=np.float64),
                ),
            ]
        )
    reference_blend_start = values[1][1]
    reference_blend_end = values[1][2]
    tespec_blend_start = values[5][1]
    tespec_blend_end = values[5][2]
    values.append(
        (
            "mean_blend050",
            0.5 * (reference_blend_start + tespec_blend_start),
            0.5 * (reference_blend_end + tespec_blend_end),
        )
    )
    names = [name for name, _, _ in values]
    stabilized = [stable_boundaries(start, end) for _, start, end in values]
    starts = np.stack([start for start, _ in stabilized], axis=1)
    ends = np.stack([end for _, end in stabilized], axis=1)
    return names, starts, ends


def oracle_choice(
    proposals: pd.DataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
    annotations: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    choices = np.zeros(len(proposals), dtype=np.int64)
    oracle_tiou = np.zeros(len(proposals), dtype=np.float64)
    for index, row in proposals.reset_index(drop=True).iterrows():
        segments = annotations.get(
            (str(row["rec_name"]), roi_key(row["roi_id"])),
            np.empty((0, 2), dtype=np.float64),
        )
        quality = np.asarray(
            [best_match(starts[index, j], ends[index, j], segments)[0] for j in range(starts.shape[1])]
        )
        choices[index] = int(np.argmax(quality))
        oracle_tiou[index] = float(quality[choices[index]])
    return choices, oracle_tiou


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotation_index(resolve(args.ann_path))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    rows = []
    choices_all = []
    for fold in range(5):
        proposals = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        reference = aligned(
            pd.read_csv(resolve(args.reference_root) / f"scored_fold_{fold:02d}.csv"),
            proposals,
        )
        tespec = aligned(
            pd.read_csv(resolve(args.tespec_root) / f"scored_fold_{fold:02d}.csv"),
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
        names, starts, ends = boundary_candidates(proposals, reference, tespec)
        choice, oracle_tiou = oracle_choice(proposals, starts, ends, annotations)
        scored = proposals.copy()
        scored["quality_score"] = groupdro["quality_score"].to_numpy(dtype=np.float64)
        for candidate, name in enumerate(names):
            scored[f"{name}_t_start"] = starts[:, candidate]
            scored[f"{name}_t_end"] = ends[:, candidate]
        scored["oracle_t_start"] = starts[np.arange(len(scored)), choice]
        scored["oracle_t_end"] = ends[np.arange(len(scored)), choice]
        scored["oracle_tiou"] = oracle_tiou
        scored["oracle_candidate"] = np.asarray(names, dtype=object)[choice]
        scored.to_csv(out_dir / f"scored_fold_{fold:02d}.csv", index=False)
        choices_all.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "candidate": scored["oracle_candidate"],
                    "oracle_tiou": oracle_tiou,
                }
            )
        )
        for mode in ("reference_blend050", "tespec_blend050", "mean_blend050", "oracle"):
            row = evaluate_variant(
                scored,
                "quality_score",
                mode,
                f"boundary_oracle_fold_{fold:02d}",
                args,
                out_dir / "predictions",
            )
            row["fold"] = fold
            row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    choices = pd.concat(choices_all, ignore_index=True)
    choices.groupby(["fold", "candidate"]).size().rename("count").reset_index().to_csv(
        out_dir / "choice_counts.csv", index=False
    )
    summary = []
    for mode, group in metrics.groupby("boundary_mode"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row = {"boundary_mode": mode}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        summary.append(row)
    result = pd.DataFrame(summary).sort_values("mean_mAP", ascending=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
