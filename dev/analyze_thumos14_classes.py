#!/usr/bin/env python3
"""Select a reproducible THUMOS14 class subset from annotation structure."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--select", type=int, default=10)
    parser.add_argument("--filtered-json", type=Path)
    return parser.parse_args()


def percentile_scores(
    values: Mapping[str, float], *, higher_is_better: bool
) -> Dict[str, float]:
    """Return average-rank percentiles in [0, 1], preserving ties."""
    if len(values) <= 1:
        return {key: 1.0 for key in values}

    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: Dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + end - 1) / 2
        score = average_rank / (len(ordered) - 1)
        if not higher_is_better:
            score = 1.0 - score
        for index in range(cursor, end):
            ranks[ordered[index][0]] = score
        cursor = end
    return ranks


def interquartile_range(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return quantile(0.75) - quantile(0.25)


def segments_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return min(float(left[1]), float(right[1])) > max(
        float(left[0]), float(right[0])
    )


def analyze_classes(
    database: Mapping[str, Mapping[str, Any]], split: str
) -> List[Dict[str, Any]]:
    split = split.lower()
    durations: Dict[str, List[float]] = defaultdict(list)
    videos: Dict[str, set[str]] = defaultdict(set)
    overlapping: Dict[str, int] = defaultdict(int)
    other_labels: Dict[str, List[int]] = defaultdict(list)
    label_ids: Dict[str, int] = {}

    for video_id, video in database.items():
        if str(video.get("subset", "")).lower() != split:
            continue
        annotations = video.get("annotations", [])
        labels_in_video = {str(item["label"]) for item in annotations}
        for index, annotation in enumerate(annotations):
            label = str(annotation["label"])
            label_ids[label] = int(annotation["label_id"])
            segment = annotation["segment"]
            durations[label].append(float(segment[1]) - float(segment[0]))
            videos[label].add(video_id)
            has_interclass_overlap = any(
                str(other["label"]) != label
                and segments_overlap(segment, other["segment"])
                for other_index, other in enumerate(annotations)
                if other_index != index
            )
            overlapping[label] += int(has_interclass_overlap)

        for label in labels_in_video:
            other_labels[label].append(len(labels_in_video - {label}))

    if not durations:
        raise ValueError(f"No annotations found for split {split!r}")

    rows: List[Dict[str, Any]] = []
    for label in sorted(durations):
        class_durations = durations[label]
        median_duration = median(class_durations)
        duration_iqr_ratio = (
            interquartile_range(class_durations) / median_duration
            if median_duration > 0
            else float("inf")
        )
        rows.append(
            {
                "label": label,
                "original_label_id": label_ids[label],
                "instances": len(class_durations),
                "positive_videos": len(videos[label]),
                "median_duration": median_duration,
                "duration_iqr_ratio": duration_iqr_ratio,
                "interclass_overlap_rate": overlapping[label]
                / len(class_durations),
                "mean_other_labels_per_video": sum(other_labels[label])
                / len(other_labels[label]),
            }
        )

    by_label = {row["label"]: row for row in rows}
    instance_score = percentile_scores(
        {label: row["instances"] for label, row in by_label.items()},
        higher_is_better=True,
    )
    video_score = percentile_scores(
        {label: row["positive_videos"] for label, row in by_label.items()},
        higher_is_better=True,
    )
    duration_score = percentile_scores(
        {label: row["duration_iqr_ratio"] for label, row in by_label.items()},
        higher_is_better=False,
    )
    overlap_score = percentile_scores(
        {
            label: row["interclass_overlap_rate"]
            for label, row in by_label.items()
        },
        higher_is_better=False,
    )
    cooccurrence_score = percentile_scores(
        {
            label: row["mean_other_labels_per_video"]
            for label, row in by_label.items()
        },
        higher_is_better=False,
    )

    for row in rows:
        label = row["label"]
        row["annotation_support_score"] = (
            instance_score[label] + video_score[label]
        ) / 2
        row["structural_simplicity_score"] = (
            duration_score[label]
            + overlap_score[label]
            + cooccurrence_score[label]
        ) / 3
        row["selection_score"] = (
            row["annotation_support_score"]
            + row["structural_simplicity_score"]
        ) / 2

    rows.sort(key=lambda row: (-row["selection_score"], row["label"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def filter_annotations(
    annotation_data: Mapping[str, Any], selected_labels: Iterable[str]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    selected = set(selected_labels)
    original_ids: Dict[str, int] = {}
    for video in annotation_data["database"].values():
        for annotation in video.get("annotations", []):
            label = str(annotation["label"])
            if label in selected:
                original_ids[label] = int(annotation["label_id"])

    missing = selected - original_ids.keys()
    if missing:
        raise ValueError(f"Selected labels missing from database: {sorted(missing)}")

    ordered_labels = sorted(selected, key=lambda label: (original_ids[label], label))
    label_map = {label: index for index, label in enumerate(ordered_labels)}
    filtered = copy.deepcopy(annotation_data)
    for video in filtered["database"].values():
        kept_annotations = []
        for annotation in video.get("annotations", []):
            label = str(annotation["label"])
            if label not in selected:
                continue
            annotation["label_id"] = label_map[label]
            kept_annotations.append(annotation)
        video["annotations"] = kept_annotations
    return filtered, label_map


def jackknife_stability(
    database: Mapping[str, Mapping[str, Any]],
    split: str,
    labels: Sequence[str],
    number_selected: int,
) -> List[Dict[str, Any]]:
    split_video_ids = [
        video_id
        for video_id, video in database.items()
        if str(video.get("subset", "")).lower() == split.lower()
    ]
    selected_counts = {label: 0 for label in labels}
    ranks = {label: [] for label in labels}
    missing_rank = len(labels) + 1

    for omitted_video_id in split_video_ids:
        reduced_database = {
            video_id: video
            for video_id, video in database.items()
            if video_id != omitted_video_id
        }
        reduced_rows = analyze_classes(reduced_database, split)
        rank_by_label = {row["label"]: row["rank"] for row in reduced_rows}
        for label in labels:
            rank = rank_by_label.get(label, missing_rank)
            ranks[label].append(rank)
            selected_counts[label] += int(rank <= number_selected)

    return [
        {
            "label": label,
            "selection_frequency": selected_counts[label] / len(split_video_ids),
            "mean_rank": sum(ranks[label]) / len(ranks[label]),
            "best_rank": min(ranks[label]),
            "worst_rank": max(ranks[label]),
        }
        for label in labels
    ]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raw_bytes = args.annotations.read_bytes()
    annotation_data = json.loads(raw_bytes)
    rows = analyze_classes(annotation_data["database"], args.split)
    if not 0 < args.select <= len(rows):
        raise ValueError("--select must be between 1 and the number of classes")

    selected_labels = [row["label"] for row in rows[: args.select]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "class_ranking.csv", rows)

    report = {
        "source_annotations": str(args.annotations),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "selection_split": args.split,
        "test_annotations_consulted_for_selection": False,
        "number_selected": args.select,
        "selection_rule": {
            "annotation_support_weight": 0.5,
            "structural_simplicity_weight": 0.5,
            "annotation_support_metrics": [
                "instances",
                "positive_videos",
            ],
            "structural_simplicity_metrics": [
                "duration_iqr_ratio",
                "interclass_overlap_rate",
                "mean_other_labels_per_video",
            ],
            "aggregation": "average rank percentile within each family",
        },
        "selected_labels_by_rank": selected_labels,
        "classes": rows,
        "jackknife_leave_one_validation_video_out": jackknife_stability(
            annotation_data["database"],
            args.split,
            [row["label"] for row in rows],
            args.select,
        ),
    }

    if args.filtered_json is not None:
        filtered, label_map = filter_annotations(
            annotation_data, selected_labels
        )
        args.filtered_json.parent.mkdir(parents=True, exist_ok=True)
        args.filtered_json.write_text(
            json.dumps(filtered, indent=2) + "\n", encoding="utf-8"
        )
        report["filtered_annotations"] = str(args.filtered_json)
        report["contiguous_label_map"] = label_map

    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
