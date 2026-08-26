"""Diagnose the proposal ceiling for a target mAP.

The script measures, for each GT instance, the best temporal IoU available in a
proposal file. This is an optimistic upper bound for any downstream CNN/ranker:
if proposal recall is below 0.90 at a threshold, no classifier can reach 90 AP
at that threshold without generating/refining better segments.

Run from event_penguins/:
    python dev/diagnose_proposal_ceiling.py \
        --proposals tmp/deep_diagnosis/fixed_r5_single_remote/proposals.csv \
        --split test --out-dir tmp/ceiling/fixed_r5_test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute proposal recall ceiling by GT instance.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--recording-info", default="config/annotations/recording_info.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default="tmp/proposal_ceiling")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_split_records(path: Path, split: str) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["timestamp"] for row in csv.DictReader(f) if row["split"] == split}


def load_gt(ann_path: Path, valid_records: set[str], min_duration_s: float) -> pd.DataFrame:
    with open(ann_path, encoding="utf-8") as f:
        db = json.load(f)["database"]
    rows = []
    for rec, value in db.items():
        if rec not in valid_records:
            continue
        for roi, anns in value.get("annotations", {}).items():
            if roi == "null":
                continue
            for ann in anns:
                if ann["label"] != "ed":
                    continue
                start, end = map(float, ann["segment"])
                if end - start < min_duration_s:
                    continue
                rows.append(
                    {
                        "rec_name": rec,
                        "roi_id": f"N{int(roi):02d}",
                        "roi_int": int(roi),
                        "gt_start_s": start,
                        "gt_end_s": end,
                        "gt_duration_s": end - start,
                    }
                )
    return pd.DataFrame(rows)


def segment_iou(target: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if candidates.size == 0:
        return np.zeros(0, dtype=np.float64)
    inter = np.maximum(0.0, np.minimum(target[1], candidates[:, 1]) - np.maximum(target[0], candidates[:, 0]))
    union = (target[1] - target[0]) + (candidates[:, 1] - candidates[:, 0]) - inter
    out = np.zeros(len(candidates), dtype=np.float64)
    valid = union > 0
    out[valid] = inter[valid] / union[valid]
    return out


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_split_records(resolve(args.recording_info), args.split)
    gt = load_gt(resolve(args.ann_path), records, args.min_duration_s)
    proposals = pd.read_csv(resolve(args.proposals))
    proposals = proposals[proposals["rec_name"].isin(records)].copy()
    proposals["t_start_s"] = proposals["t_start"].astype(float) / 1e6
    proposals["t_end_s"] = proposals["t_end"].astype(float) / 1e6

    grouped = {
        key: grp[["t_start_s", "t_end_s", "score"]].to_numpy(dtype=np.float64)
        for key, grp in proposals.groupby(["rec_name", "roi_id"])
    }

    rows = []
    for idx, row in gt.iterrows():
        candidates = grouped.get((row["rec_name"], row["roi_id"]), np.empty((0, 3), dtype=np.float64))
        iou = segment_iou(np.asarray([row["gt_start_s"], row["gt_end_s"]]), candidates[:, :2])
        if len(iou):
            best_idx = int(np.argmax(iou))
            best_iou = float(iou[best_idx])
            best_start = float(candidates[best_idx, 0])
            best_end = float(candidates[best_idx, 1])
            best_score = float(candidates[best_idx, 2])
        else:
            best_iou = 0.0
            best_start = float("nan")
            best_end = float("nan")
            best_score = float("nan")
        rows.append(
            {
                "gt_index": idx,
                **row.to_dict(),
                "best_tiou": best_iou,
                "best_prop_start_s": best_start,
                "best_prop_end_s": best_end,
                "best_prop_score": best_score,
            }
        )

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "gt_ceiling_detail.csv", index=False)

    summary_rows = []
    for thr in args.tiou:
        covered = int((detail["best_tiou"] >= thr).sum()) if not detail.empty else 0
        total = int(len(detail))
        summary_rows.append(
            {
                "tiou": thr,
                "covered_gt": covered,
                "total_gt": total,
                "recall_ceiling": covered / total if total else float("nan"),
                "missed_gt": total - covered,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[INFO] Detail written to {out_dir / 'gt_ceiling_detail.csv'}")


if __name__ == "__main__":
    main()
