"""Evaluate event-rate boundary refinement on existing detections.

The script refines predicted temporal segments without rerunning the CNN. It
uses only the event-rate signal inside each detection window, so it is a
general boundary post-processing diagnostic rather than a case-specific fix.

Run from event_penguins/:
    python dev/eval_boundary_refinement.py \
        --predictions tmp/scoring_variants/val/predictions/cnn.json \
        --split val --out-dir tmp/boundary_refinement/val --sweep
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd

from src.evaluation import DetectionsEvaluator, segment_iou
from src.utils import temporal_soft_nms


ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "config/annotations/annotations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate boundary refinement variants.")
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default=str(ANN))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--out-dir", default="tmp/boundary_refinement")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--mode", default="trim", choices=["trim", "split_replace", "split_add_low"])
    parser.add_argument("--bin-width-s", type=float, default=0.033)
    parser.add_argument("--smooth-s", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--pad-s", type=float, default=0.50)
    parser.add_argument("--min-run-s", type=float, default=0.50)
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--add-score-scale", type=float, default=0.05)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def load_split_sequences(data_path: str, split: str | None) -> list[str] | None:
    if split is None:
        return None
    with h5py.File(data_path, "r") as hf:
        return sorted(rec for rec in hf.keys() if hf[rec].attrs.get("split") == split)


def prediction_to_rows(prediction: dict, valid_sequences: list[str] | None) -> pd.DataFrame:
    rows = []
    for rec, rois in prediction["results"].items():
        if valid_sequences is not None and rec not in valid_sequences:
            continue
        for roi, detections in rois.items():
            for idx, det in enumerate(detections):
                start, end = det["segment"]
                if end - start < 2:
                    continue
                rows.append(
                    {
                        "rec_name": rec,
                        "roi_id": f"N{int(roi):02d}",
                        "roi_int": int(roi),
                        "source_idx": idx,
                        "t_start": float(start),
                        "t_end": float(end),
                        "score": float(det["score"]),
                    }
                )
    return pd.DataFrame(rows)


def smooth_signal(rate: np.ndarray, bin_width_s: float, smooth_s: float) -> np.ndarray:
    window = max(1, int(round(smooth_s / bin_width_s)))
    if window <= 1:
        return rate.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(rate.astype(np.float64), kernel, mode="same")


def active_runs(mask: np.ndarray, min_bins: int) -> list[tuple[int, int]]:
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_bins]


def ensure_min_duration(
    start: float,
    end: float,
    min_duration: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    if end - start >= min_duration:
        return start, end
    center = 0.5 * (start + end)
    start = center - 0.5 * min_duration
    end = center + 0.5 * min_duration
    if start < lower:
        end += lower - start
        start = lower
    if end > upper:
        start -= end - upper
        end = upper
    return max(lower, start), min(upper, end)


def refine_one_detection(
    events: np.ndarray,
    row: pd.Series,
    args: argparse.Namespace,
) -> list[dict]:
    start_s = float(row["t_start"])
    end_s = float(row["t_end"])
    duration = end_s - start_s
    if duration < args.min_duration_s:
        return [row.to_dict()]

    start_us = start_s * 1e6
    end_us = end_s * 1e6
    left = int(np.searchsorted(events[:, 2], start_us, side="left"))
    right = int(np.searchsorted(events[:, 2], end_us, side="right"))
    subset = events[left:right]
    n_bins = max(2, int(np.ceil(duration / args.bin_width_s)))
    edges_s = start_s + np.arange(n_bins + 1, dtype=np.float64) * args.bin_width_s
    edges_s[-1] = end_s

    if len(subset) == 0 or np.any(np.diff(edges_s) <= 0):
        return [row.to_dict()]

    counts = np.histogram(subset[:, 2].astype(np.float64) / 1e6, bins=edges_s)[0]
    smoothed = smooth_signal(counts, args.bin_width_s, args.smooth_s)
    max_val = float(smoothed.max())
    if max_val <= 1e-12:
        return [row.to_dict()]

    mask = smoothed >= args.threshold * max_val
    runs = active_runs(mask, max(1, int(round(args.min_run_s / args.bin_width_s))))
    if not runs:
        return [row.to_dict()]

    base = row.to_dict()
    if args.mode == "trim":
        first = min(s for s, _ in runs)
        last = max(e for _, e in runs)
        new_start = max(start_s, float(edges_s[first]) - args.pad_s)
        new_end = min(end_s, float(edges_s[last]) + args.pad_s)
        new_start, new_end = ensure_min_duration(
            new_start,
            new_end,
            args.min_duration_s,
            start_s,
            end_s,
        )
        item = dict(base)
        item["t_start"] = new_start
        item["t_end"] = new_end
        return [item]

    scored_runs = []
    for s, e in runs:
        peak = float(smoothed[s:e].max())
        scored_runs.append((peak, s, e))
    scored_runs.sort(reverse=True)
    scored_runs = scored_runs[: args.max_runs]
    scored_runs.sort(key=lambda item: item[1])

    refined = []
    if args.mode == "split_add_low":
        refined.append(dict(base))

    for peak, s, e in scored_runs:
        new_start = max(start_s, float(edges_s[s]) - args.pad_s)
        new_end = min(end_s, float(edges_s[e]) + args.pad_s)
        new_start, new_end = ensure_min_duration(
            new_start,
            new_end,
            args.min_duration_s,
            start_s,
            end_s,
        )
        item = dict(base)
        item["t_start"] = new_start
        item["t_end"] = new_end
        if args.mode == "split_add_low":
            item["score"] = float(row["score"]) * args.add_score_scale
        else:
            item["score"] = float(row["score"]) * (peak / max_val)
        refined.append(item)
    return refined


def rows_to_prediction(rows: pd.DataFrame, original_prediction: dict, args: argparse.Namespace) -> dict:
    result = {rec: {} for rec in original_prediction["results"].keys()}
    if rows.empty:
        return {"version": "VERSION 0.0", "results": result}

    for (rec, roi_id), grp in rows.groupby(["rec_name", "roi_id"]):
        arr = (grp[["t_start", "t_end", "score"]].to_numpy(dtype=np.float64)).copy()
        arr[:, :2] *= 1e6
        processed = temporal_soft_nms(
            arr,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
        )
        result.setdefault(rec, {})
        result[rec][int(roi_id[1:])] = [
            {
                "label": "ed",
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(score),
            }
            for start, end, score in processed
            if (float(end) - float(start)) / 1e6 >= args.min_duration_s
        ]
    return {"version": "VERSION 0.0", "results": result}


def refine_prediction(prediction: dict, valid_sequences: list[str] | None, args: argparse.Namespace) -> dict:
    rows = prediction_to_rows(prediction, valid_sequences)
    refined_rows = []
    if rows.empty:
        return copy.deepcopy(prediction)

    with h5py.File(args.data_path, "r") as hf:
        for (rec, roi_id), grp in rows.groupby(["rec_name", "roi_id"], sort=False):
            if rec not in hf or roi_id not in hf[rec]:
                refined_rows.extend(grp.to_dict("records"))
                continue
            events = np.asarray(hf[rec][roi_id]["events"])
            for _, row in grp.iterrows():
                refined_rows.extend(refine_one_detection(events, row, args))

    return rows_to_prediction(pd.DataFrame(refined_rows), prediction, args)


def evaluate(prediction: dict, valid_sequences: list[str] | None, args: argparse.Namespace, out_path: Path) -> dict:
    out_path.write_text(json.dumps(prediction), encoding="utf-8")
    evaluator = DetectionsEvaluator(
        ground_truth_filename=args.ann_path,
        prediction_filename=str(out_path),
        tiou_thresholds=np.array(args.tiou),
        valid_labels="ed",
        valid_sequences=valid_sequences,
        min_duration=args.min_duration_s,
    )
    mean_ap = evaluator.run()
    return {
        "n_pred": int(sum(len(v) for rois in prediction["results"].values() for v in rois.values())),
        "mAP": float(mean_ap),
        "AP@0.1": float(evaluator.mAP[0]),
        "AP@0.3": float(evaluator.mAP[1]),
        "AP@0.5": float(evaluator.mAP[2]),
        "AP@0.7": float(evaluator.mAP[3]),
    }


def apply_params(args: argparse.Namespace, params: dict) -> argparse.Namespace:
    updated = copy.copy(args)
    for key, value in params.items():
        setattr(updated, key, value)
    return updated


def sweep_params() -> list[dict]:
    configs = []
    for mode in ["trim", "split_replace", "split_add_low"]:
        for threshold in [0.15, 0.25, 0.35, 0.50]:
            for pad_s in [0.25, 0.50, 1.00]:
                for min_run_s in [0.33, 0.66, 1.00]:
                    configs.append(
                        {
                            "mode": mode,
                            "threshold": threshold,
                            "pad_s": pad_s,
                            "min_run_s": min_run_s,
                        }
                    )
    return configs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    prediction = json.loads(Path(args.predictions).read_text())
    valid_sequences = load_split_sequences(args.data_path, args.split)

    configs = sweep_params() if args.sweep else [
        {
            "mode": args.mode,
            "threshold": args.threshold,
            "pad_s": args.pad_s,
            "min_run_s": args.min_run_s,
        }
    ]

    rows = []
    baseline = evaluate(prediction, valid_sequences, args, pred_dir / "baseline.json")
    rows.append({"variant": "baseline", **baseline})
    print(f"baseline mAP={baseline['mAP']:.4f} AP05={baseline['AP@0.5']:.4f} AP07={baseline['AP@0.7']:.4f}")

    for i, params in enumerate(configs):
        cfg = apply_params(args, params)
        refined = refine_prediction(prediction, valid_sequences, cfg)
        name = (
            f"{cfg.mode}_thr{cfg.threshold:g}_pad{cfg.pad_s:g}_run{cfg.min_run_s:g}"
            .replace(".", "p")
        )
        metrics = evaluate(refined, valid_sequences, cfg, pred_dir / f"{name}.json")
        rows.append({"variant": name, **params, **metrics})
        print(
            f"{name:38s} mAP={metrics['mAP']:.4f} "
            f"AP05={metrics['AP@0.5']:.4f} AP07={metrics['AP@0.7']:.4f} "
            f"n={metrics['n_pred']}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[INFO] Summary: {out_dir / 'summary.csv'}")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
