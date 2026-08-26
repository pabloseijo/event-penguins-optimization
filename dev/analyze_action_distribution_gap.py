"""Measure action-density and duration gaps without fitting to the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return the duration covered by possibly overlapping intervals."""
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def recording_statistics(
    recording: str,
    sequence_rows: pd.DataFrame,
    annotation_entry: dict,
    min_duration: float,
) -> dict[str, float | int | str]:
    observed = {
        int(row.roi_id): float(row.duration_s)
        for row in sequence_rows.itertuples(index=False)
    }
    segments_by_roi: dict[int, list[tuple[float, float]]] = {
        roi_id: [] for roi_id in observed
    }
    for roi_key, annotations in annotation_entry.get("annotations", {}).items():
        if roi_key == "null" or int(roi_key) not in observed:
            continue
        roi_id = int(roi_key)
        for annotation in annotations:
            if annotation["label"] != "ed":
                continue
            start, end = map(float, annotation["segment"])
            start = max(0.0, start)
            end = min(observed[roi_id], end)
            if end - start >= min_duration:
                segments_by_roi[roi_id].append((start, end))

    segments = [
        segment for roi_segments in segments_by_roi.values() for segment in roi_segments
    ]
    durations = np.asarray([end - start for start, end in segments], dtype=np.float64)
    gaps = np.asarray(
        [
            max(0.0, right[0] - left[1])
            for roi_segments in segments_by_roi.values()
            for left, right in zip(sorted(roi_segments), sorted(roi_segments)[1:])
        ],
        dtype=np.float64,
    )
    observed_seconds = float(sum(observed.values()))
    action_seconds = float(
        sum(union_duration(roi_segments) for roi_segments in segments_by_roi.values())
    )
    active_rois = sum(bool(roi_segments) for roi_segments in segments_by_roi.values())
    return {
        "rec_name": recording,
        "rois": len(observed),
        "observed_roi_hours": observed_seconds / 3600.0,
        "gt_instances": len(segments),
        "instances_per_roi_hour": (
            len(segments) / (observed_seconds / 3600.0) if observed_seconds else np.nan
        ),
        "active_roi_fraction": active_rois / len(observed) if observed else np.nan,
        "action_fraction": action_seconds / observed_seconds if observed_seconds else np.nan,
        "median_duration_s": float(np.median(durations)) if len(durations) else np.nan,
        "p90_duration_s": (
            float(np.quantile(durations, 0.9)) if len(durations) else np.nan
        ),
        "duration_iqr_s": (
            float(np.quantile(durations, 0.75) - np.quantile(durations, 0.25))
            if len(durations)
            else np.nan
        ),
        "median_gap_s": float(np.median(gaps)) if len(gaps) else np.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-feature-dir", required=True)
    parser.add_argument("--test-feature-dir", required=True)
    parser.add_argument("--source-metrics", default=None)
    parser.add_argument("--test-metrics", default=None)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-duration", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = json.loads(resolve(args.ann_path).read_text(encoding="utf-8"))["database"]
    frames = []
    for split, feature_dir in (
        ("source", resolve(args.source_feature_dir)),
        ("test", resolve(args.test_feature_dir)),
    ):
        sequences = pd.read_csv(feature_dir / "sequences.csv")
        for recording, rows in sequences.groupby("rec_name", sort=True):
            stats = recording_statistics(
                str(recording), rows, database[str(recording)], args.min_duration
            )
            stats["split"] = split
            frames.append(stats)
    output = pd.DataFrame(frames)

    metric_frames = []
    for split, metrics_path in (
        ("source", args.source_metrics),
        ("test", args.test_metrics),
    ):
        if not metrics_path:
            continue
        metrics = pd.read_csv(resolve(metrics_path))
        metrics = metrics.rename(
            columns={
                "mAP": "detection_mAP",
                "AP@0.7": "detection_AP@0.7",
            }
        )
        keep = [
            column
            for column in ("rec_name", "detection_mAP", "detection_AP@0.7")
            if column in metrics
        ]
        metric_frames.append(metrics[keep].assign(split=split))
    if metric_frames:
        output = output.merge(
            pd.concat(metric_frames, ignore_index=True),
            on=["split", "rec_name"],
            how="left",
        )

    features = [
        "gt_instances",
        "instances_per_roi_hour",
        "active_roi_fraction",
        "action_fraction",
        "median_duration_s",
        "p90_duration_s",
        "duration_iqr_s",
        "median_gap_s",
    ]
    source = output[(output["split"] == "source") & (output["gt_instances"] > 0)]
    source_ranges = pd.DataFrame(
        [
            {
                "feature": feature,
                "source_min": source[feature].min(),
                "source_median": source[feature].median(),
                "source_max": source[feature].max(),
            }
            for feature in features
        ]
    )
    range_lookup = source_ranges.set_index("feature")
    for feature in features:
        output[f"{feature}_outside_source"] = (
            (output[feature] < range_lookup.loc[feature, "source_min"])
            | (output[feature] > range_lookup.loc[feature, "source_max"])
        )

    correlations = []
    for target in ("detection_mAP", "detection_AP@0.7"):
        if target not in source:
            continue
        for feature in features:
            valid = source[[feature, target]].dropna()
            correlations.append(
                {
                    "feature": feature,
                    "target": target,
                    "source_recordings": len(valid),
                    "spearman": valid[feature].corr(valid[target], method="spearman"),
                }
            )

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "recording_statistics.csv", index=False)
    source_ranges.to_csv(out_dir / "source_ranges.csv", index=False)
    pd.DataFrame(correlations).to_csv(out_dir / "source_correlations.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output.to_string(index=False))
    if correlations:
        print("\nSource-only Spearman correlations:")
        print(pd.DataFrame(correlations).to_string(index=False))


if __name__ == "__main__":
    main()
