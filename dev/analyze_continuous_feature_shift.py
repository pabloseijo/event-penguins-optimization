"""Diagnose recording-level shifts in compact continuous event features."""

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


def load_ed_annotations(path: Path) -> dict[tuple[str, int], list[tuple[float, float]]]:
    database = json.loads(path.read_text(encoding="utf-8"))["database"]
    result = {}
    for recording, recording_data in database.items():
        for roi_id, annotations in recording_data.get("annotations", {}).items():
            if roi_id == "null":
                continue
            result[(recording, int(roi_id))] = [
                tuple(map(float, annotation["segment"]))
                for annotation in annotations
                if annotation["label"] == "ed"
                and float(annotation["segment"][1])
                - float(annotation["segment"][0])
                >= 2.0
            ]
    return result


def action_mask(
    length: int, stride_s: float, segments: list[tuple[float, float]]
) -> np.ndarray:
    centers = (np.arange(length, dtype=np.float64) + 0.5) * stride_s
    selected = np.zeros(length, dtype=bool)
    for start, end in segments:
        selected |= (centers >= start) & (centers <= end)
    return selected


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    values = np.concatenate((positive, negative))
    ranks = pd.Series(values).rank(method="average").to_numpy(np.float64)
    positive_rank_sum = ranks[: len(positive)].sum()
    return float(
        (positive_rank_sum - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def analyze_split(
    split: str,
    feature_dir: Path,
    event_dir: Path,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
) -> tuple[list[dict], list[dict]]:
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    feature_metadata = json.loads((feature_dir / "metadata.json").read_text())
    event_metadata = json.loads((event_dir / "metadata.json").read_text())
    names = list(event_metadata["feature_names"])
    matrix = np.load(event_dir / "event_stats.npy", mmap_mode="r")
    stride_s = float(feature_metadata["grid_stride_s"])
    summary_rows = []
    separation_rows = []
    for recording, recording_sequences in sequences.groupby("rec_name", sort=True):
        values_parts = []
        action_parts = []
        for row in recording_sequences.itertuples(index=False):
            start = int(row.offset)
            length = int(row.length)
            values_parts.append(np.asarray(matrix[start : start + length], dtype=np.float64))
            action_parts.append(
                action_mask(
                    length,
                    stride_s,
                    annotations.get((str(recording), int(row.roi_id)), []),
                )
            )
        values = np.concatenate(values_parts)
        is_action = np.concatenate(action_parts)
        row = {
            "split": split,
            "rec_name": str(recording),
            "points": len(values),
            "action_points": int(is_action.sum()),
            "action_fraction": float(is_action.mean()),
        }
        for index, name in enumerate(names):
            row[f"{name}_mean"] = float(values[:, index].mean())
            row[f"{name}_std"] = float(values[:, index].std())
            row[f"{name}_q90"] = float(np.quantile(values[:, index], 0.9))
            auc = binary_auc(values[is_action, index], values[~is_action, index])
            separation_rows.append(
                {
                    "split": split,
                    "rec_name": str(recording),
                    "feature": name,
                    "auc_action_high": auc,
                    "auc_best_direction": max(auc, 1.0 - auc) if np.isfinite(auc) else auc,
                    "action_mean": (
                        float(values[is_action, index].mean()) if is_action.any() else np.nan
                    ),
                    "background_mean": float(values[~is_action, index].mean()),
                }
            )
        summary_rows.append(row)
    return summary_rows, separation_rows


def nearest_source_recordings(summary: pd.DataFrame) -> pd.DataFrame:
    source = summary[summary["split"] == "source"].copy()
    target = summary[summary["split"] == "target"].copy()
    feature_columns = [
        column
        for column in summary.columns
        if column.endswith(("_mean", "_std", "_q90"))
    ]
    source_values = source[feature_columns].to_numpy(np.float64)
    target_values = target[feature_columns].to_numpy(np.float64)
    mean = source_values.mean(axis=0)
    std = np.maximum(source_values.std(axis=0), 1e-6)
    source_values = (source_values - mean) / std
    target_values = (target_values - mean) / std
    rows = []
    for target_index, target_row in enumerate(target.itertuples(index=False)):
        distances = np.sqrt(np.square(source_values - target_values[target_index]).mean(axis=1))
        order = np.argsort(distances)[:3]
        for rank, source_index in enumerate(order, 1):
            rows.append(
                {
                    "target_recording": target_row.rec_name,
                    "rank": rank,
                    "source_recording": source.iloc[source_index]["rec_name"],
                    "standardized_distance": float(distances[source_index]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1"
    )
    parser.add_argument(
        "--source-event-dir", default="tmp/temporalmaxer_continuous/source_event_stats_v1"
    )
    parser.add_argument(
        "--target-feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1"
    )
    parser.add_argument(
        "--target-event-dir", default="tmp/temporalmaxer_continuous/test_event_stats_v1"
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/feature_shift_analysis_v1"
    )
    args = parser.parse_args()
    annotations = load_ed_annotations(resolve(args.ann_path))
    summary_rows = []
    separation_rows = []
    for split, feature_dir, event_dir in (
        ("source", args.source_feature_dir, args.source_event_dir),
        ("target", args.target_feature_dir, args.target_event_dir),
    ):
        split_summary, split_separation = analyze_split(
            split, resolve(feature_dir), resolve(event_dir), annotations
        )
        summary_rows.extend(split_summary)
        separation_rows.extend(split_separation)
    summary = pd.DataFrame(summary_rows)
    separation = pd.DataFrame(separation_rows)
    nearest = nearest_source_recordings(summary)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "recording_summary.csv", index=False)
    separation.to_csv(out_dir / "feature_separation.csv", index=False)
    nearest.to_csv(out_dir / "nearest_source_recordings.csv", index=False)
    print(
        summary[
            [
                "split",
                "rec_name",
                "action_fraction",
                "log_event_count_mean",
                "polarity_balance_mean",
                "spectral_energy_ratio_mean",
                "dominant_frequency_mean",
            ]
        ].to_string(index=False)
    )
    print("\nNearest source recordings")
    print(nearest.to_string(index=False))


if __name__ == "__main__":
    main()
