#!/usr/bin/env python3
"""Build deterministic video-disjoint THUMOS14 folds for transfer studies."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--annotation-config-prefix",
        default="./data/thumos/annotations",
    )
    return parser.parse_args()


def collect_labels(
    database: Mapping[str, Mapping[str, Any]], split: str
) -> List[str]:
    return sorted(
        {
            str(annotation["label"])
            for video in database.values()
            if str(video.get("subset", "")).lower() == split.lower()
            for annotation in video.get("annotations", [])
        }
    )


def video_vector(
    video: Mapping[str, Any], labels: Sequence[str]
) -> Tuple[List[int], List[int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    instance_counts = [0] * len(labels)
    for annotation in video.get("annotations", []):
        label = str(annotation["label"])
        if label in label_to_index:
            instance_counts[label_to_index[label]] += 1
    presence = [int(count > 0) for count in instance_counts]
    presence.append(int(not any(presence)))
    return presence, instance_counts


def squared_normalized_error(
    observed: Sequence[float], target: Sequence[float]
) -> float:
    return math.fsum(
        ((value - expected) / max(abs(expected), 1.0)) ** 2
        for value, expected in zip(observed, target)
    )


def assign_folds(
    database: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    labels: Sequence[str],
    folds: int,
    seed: int,
) -> Dict[str, int]:
    if folds < 2:
        raise ValueError("At least two folds are required")

    records = []
    for video_id, video in database.items():
        if str(video.get("subset", "")).lower() != split.lower():
            continue
        presence, instances = video_vector(video, labels)
        records.append(
            {
                "video_id": video_id,
                "presence": presence,
                "instances": instances,
                "duration": float(video.get("duration", 0.0)),
            }
        )
    if len(records) < folds:
        raise ValueError("The split contains fewer videos than folds")

    support = [
        sum(record["presence"][index] for record in records)
        for index in range(len(labels) + 1)
    ]
    stable_ties = {
        record["video_id"]: hashlib.sha256(
            f"{seed}:{record['video_id']}".encode("utf-8")
        ).digest()
        for record in records
    }

    def rarity(record: Mapping[str, Any]) -> float:
        return sum(
            value / max(support[index], 1)
            for index, value in enumerate(record["presence"])
        )

    records.sort(
        key=lambda record: (
            -rarity(record),
            -sum(record["instances"]),
            stable_ties[record["video_id"]],
            record["video_id"],
        )
    )

    total_presence = [
        sum(record["presence"][index] for record in records)
        for index in range(len(labels) + 1)
    ]
    total_instances = [
        sum(record["instances"][index] for record in records)
        for index in range(len(labels))
    ]
    target_presence = [value / folds for value in total_presence]
    target_instances = [value / folds for value in total_instances]
    target_videos = len(records) / folds
    target_duration = math.fsum(record["duration"] for record in records) / folds

    fold_presence = [[0] * (len(labels) + 1) for _ in range(folds)]
    fold_instances = [[0] * len(labels) for _ in range(folds)]
    fold_sizes = [0] * folds
    fold_durations = [0.0] * folds
    assignments: Dict[str, int] = {}

    for record in records:
        candidate_costs = []
        for fold_index in range(folds):
            next_presence = [
                current + added
                for current, added in zip(
                    fold_presence[fold_index], record["presence"]
                )
            ]
            next_instances = [
                current + added
                for current, added in zip(
                    fold_instances[fold_index], record["instances"]
                )
            ]
            cost = 3.0 * (
                squared_normalized_error(next_presence, target_presence)
                - squared_normalized_error(
                    fold_presence[fold_index], target_presence
                )
            )
            cost += (
                squared_normalized_error(next_instances, target_instances)
                - squared_normalized_error(
                    fold_instances[fold_index], target_instances
                )
            )
            cost += 0.5 * (
                (fold_sizes[fold_index] + 1 - target_videos)
                / max(target_videos, 1.0)
            ) ** 2 - 0.5 * (
                (fold_sizes[fold_index] - target_videos)
                / max(target_videos, 1.0)
            ) ** 2
            cost += 0.1 * (
                (
                    fold_durations[fold_index]
                    + record["duration"]
                    - target_duration
                )
                / max(target_duration, 1.0)
            ) ** 2 - 0.1 * (
                (fold_durations[fold_index] - target_duration)
                / max(target_duration, 1.0)
            ) ** 2
            candidate_costs.append(
                (cost, fold_sizes[fold_index], fold_index)
            )

        _, _, chosen_fold = min(candidate_costs)
        assignments[record["video_id"]] = chosen_fold
        fold_presence[chosen_fold] = [
            current + added
            for current, added in zip(
                fold_presence[chosen_fold], record["presence"]
            )
        ]
        fold_instances[chosen_fold] = [
            current + added
            for current, added in zip(
                fold_instances[chosen_fold], record["instances"]
            )
        ]
        fold_sizes[chosen_fold] += 1
        fold_durations[chosen_fold] += record["duration"]

    missing = [
        (fold_index, labels[label_index])
        for fold_index in range(folds)
        for label_index in range(len(labels))
        if fold_presence[fold_index][label_index] == 0
    ]
    if missing:
        raise ValueError(f"At least one class is absent from a fold: {missing}")
    return assignments


def build_fold_annotations(
    annotation_data: Mapping[str, Any],
    assignments: Mapping[str, int],
    *,
    fold_index: int,
    split: str,
) -> Dict[str, Any]:
    output = copy.deepcopy(annotation_data)
    train_subset = f"transfer_train_{fold_index}"
    validation_subset = f"transfer_val_{fold_index}"
    for video_id, video in output["database"].items():
        if str(video.get("subset", "")).lower() != split.lower():
            continue
        video["subset"] = (
            validation_subset
            if assignments[video_id] == fold_index
            else train_subset
        )
    return output


def summarize_folds(
    database: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, int],
    labels: Sequence[str],
    folds: int,
) -> List[Dict[str, Any]]:
    rows = []
    for fold_index in range(folds):
        videos = [
            (video_id, database[video_id])
            for video_id, assigned_fold in assignments.items()
            if assigned_fold == fold_index
        ]
        instance_counts = Counter(
            str(annotation["label"])
            for _, video in videos
            for annotation in video.get("annotations", [])
        )
        positive_video_counts = Counter(
            label
            for _, video in videos
            for label in {
                str(annotation["label"])
                for annotation in video.get("annotations", [])
            }
        )
        rows.append(
            {
                "fold": fold_index,
                "videos": len(videos),
                "negative_videos": sum(
                    not video.get("annotations", []) for _, video in videos
                ),
                "duration": sum(
                    float(video.get("duration", 0.0)) for _, video in videos
                ),
                "instances": sum(instance_counts.values()),
                "instances_by_class": {
                    label: instance_counts[label] for label in labels
                },
                "positive_videos_by_class": {
                    label: positive_video_counts[label] for label in labels
                },
            }
        )
    return rows


def validate_partition(
    database: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, int],
    *,
    split: str,
    folds: int,
) -> None:
    expected = {
        video_id
        for video_id, video in database.items()
        if str(video.get("subset", "")).lower() == split.lower()
    }
    if set(assignments) != expected:
        raise ValueError("Assignments do not cover the requested split exactly")
    if set(assignments.values()) != set(range(folds)):
        raise ValueError("At least one fold is empty")


def main() -> None:
    args = parse_args()
    annotation_bytes = args.annotations.read_bytes()
    annotation_data = json.loads(annotation_bytes)
    config = yaml.safe_load(args.config_template.read_text(encoding="utf-8"))
    labels = collect_labels(annotation_data["database"], args.split)
    assignments = assign_folds(
        annotation_data["database"],
        split=args.split,
        labels=labels,
        folds=args.folds,
        seed=args.seed,
    )
    validate_partition(
        annotation_data["database"],
        assignments,
        split=args.split,
        folds=args.folds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_folds(
        annotation_data["database"], assignments, labels, args.folds
    )
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "fold"])
        writer.writeheader()
        writer.writerows(
            {"video_id": video_id, "fold": fold}
            for video_id, fold in sorted(assignments.items())
        )

    for fold_index in range(args.folds):
        annotation_name = f"thumos14_10classes_transfer_fold{fold_index}.json"
        fold_annotations = build_fold_annotations(
            annotation_data,
            assignments,
            fold_index=fold_index,
            split=args.split,
        )
        (args.output_dir / annotation_name).write_text(
            json.dumps(fold_annotations, indent=2) + "\n",
            encoding="utf-8",
        )

        fold_config = copy.deepcopy(config)
        fold_config["train_split"] = [f"transfer_train_{fold_index}"]
        fold_config["val_split"] = [f"transfer_val_{fold_index}"]
        fold_config["dataset"]["json_file"] = (
            f"{args.annotation_config_prefix.rstrip('/')}/{annotation_name}"
        )
        (args.output_dir / f"thumos14_transfer_fold{fold_index}.yaml").write_text(
            yaml.safe_dump(fold_config, sort_keys=False),
            encoding="utf-8",
        )

    report = {
        "source_annotations": str(args.annotations),
        "source_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "split": args.split,
        "folds": args.folds,
        "seed": args.seed,
        "method": (
            "deterministic rarity-first multilabel balancing over positive "
            "videos, instance counts, negatives, video count, and duration"
        ),
        "labels": labels,
        "test_used_for_assignment": False,
        "fold_summary": rows,
    }
    (args.output_dir / "fold_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
