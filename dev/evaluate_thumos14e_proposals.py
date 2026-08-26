#!/usr/bin/env python3
"""Evaluate frozen reTAG/EventPenguins proposals for all THUMOS14 classes.

This is the class-agnostic localization control. It reproduces reTAG's
AR@20/30/50 protocol at tIoU {0.1, 0.3, 0.5, 0.7}, writes one immutable result
per class, and reports a separately named IoU-ranked oracle ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dev.prepare_thumos14_event_corpus import THUMOS_CLASSES, sha256_file
from dev.run_thumos14_event_generalization import (
    detection_ap,
    load_generic_ground_truth,
    normalize_roi,
    proposal_oracle_detections,
    proposal_recall,
)


RETAG_TIOU = (0.1, 0.3, 0.5, 0.7)
THUMOS_TIOU = (0.3, 0.4, 0.5, 0.6, 0.7)
BUDGETS = (20, 30, 50)
PROTOCOL_DECLARATION = {
    "scientific_scope": {
        "corpora": 1,
        "corpus": "THUMOS14 temporal localization subset converted once to events",
        "class_tasks": 20,
        "source_recordings_mixed_into_target": False,
        "source_domain_use": "hashed pretrained weights only",
    },
    "published_rules": {
        "thumos14_split_and_per_class_ap": "ActionFormer ECCV 2022 / THUMOS14 evaluation",
        "proposal_recall_budgets_and_tiou": "reTAG CVPR 2024",
        "rgb_to_synthetic_events": "Vid2E CVPR 2020 and v2e CVPRW 2021",
        "untrimmed_inherited_annotations_precedent": "Event ActivityNet 2026",
    },
    "predeclared_adaptations": {
        "twenty_one_vs_rest_binary_tasks": (
            "reTAG is binary; one-vs-rest preserves its output structure while reporting "
            "all twenty THUMOS14 classes"
        ),
        "same_converted_corpus_for_both_methods": "required paired experimental control",
        "iou_ranked_oracle_ap": (
            "transparent ceiling because reTAG does not release Perfect Classifier tie ordering"
        ),
    },
}


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--retag-proposals", type=Path, required=True)
    parser.add_argument("--ours-proposals", type=Path, required=True)
    parser.add_argument(
        "--retag-time-unit", choices=("microseconds", "seconds"), default="microseconds"
    )
    parser.add_argument(
        "--ours-time-unit", choices=("microseconds", "seconds"), default="microseconds"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_test_recordings(manifest_path: Path) -> list[str]:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {"video_id", "official_subset", "evaluation_included"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest lacks columns {sorted(missing)}")
    included = manifest["evaluation_included"].astype(str).str.lower().isin(
        {"1", "true", "yes"}
    )
    recordings = sorted(
        manifest.loc[
            (manifest["official_subset"].astype(str).str.lower() == "test") & included,
            "video_id",
        ].astype(str)
    )
    if len(recordings) != 212 or len(set(recordings)) != 212:
        raise ValueError(f"Expected 212 canonical test recordings, got {len(recordings)}")
    return recordings


def load_proposals(path: Path, recordings: list[str], time_unit: str) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    required = {"rec_name", "t_start", "t_end", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    if "roi_id" not in frame:
        frame["roi_id"] = 1
    frame["rec_name"] = frame["rec_name"].astype(str)
    frame["roi_id"] = frame["roi_id"].map(normalize_roi)
    for column in ("t_start", "t_end", "score"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if time_unit == "microseconds":
        frame[["t_start", "t_end"]] = frame[["t_start", "t_end"]] / 1e6
    values = frame[["t_start", "t_end", "score"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite proposal values")
    if (frame["t_end"] <= frame["t_start"]).any():
        raise ValueError(f"{path} contains non-positive proposal durations")
    selected = frame[frame["rec_name"].isin(set(recordings))].copy()
    return selected[["rec_name", "roi_id", "t_start", "t_end", "score"]]


def method_result(proposals: pd.DataFrame, ground_truth: pd.DataFrame) -> dict[str, object]:
    recall = proposal_recall(
        proposals,
        ground_truth,
        thresholds=RETAG_TIOU,
        budgets=BUDGETS,
    )
    oracle = proposal_oracle_detections(proposals, ground_truth)
    return {
        "proposal_count": int(len(proposals)),
        "retag_proposal_recall": recall,
        "iou_ranked_oracle_ap": detection_ap(
            oracle, ground_truth, thresholds=THUMOS_TIOU
        ),
    }


def flatten_result(label: str, method: str, gt_count: int, result: dict) -> dict:
    row = {"class": label, "method": method, "ground_truth_instances": gt_count}
    row["proposal_count"] = result["proposal_count"]
    for budget, metrics in result["retag_proposal_recall"].items():
        row[f"AR@{budget}"] = metrics["mean_AR"]
        for threshold in RETAG_TIOU:
            row[f"R@{budget}_tIoU{threshold:.1f}"] = metrics[f"AR@{threshold:.1f}"]
    for name, value in result["iou_ranked_oracle_ap"].items():
        row[f"oracle_{name}"] = value
    return row


def main() -> None:
    args = parse_args()
    work_dir = resolve(args.work_dir)
    out_dir = resolve(args.out_dir)
    per_class_dir = out_dir / "per_class"
    per_class_dir.mkdir(parents=True, exist_ok=True)
    recordings = canonical_test_recordings(work_dir / "manifest.csv")
    methods = {
        "retag": load_proposals(
            resolve(args.retag_proposals), recordings, args.retag_time_unit
        ),
        "eventpenguins_full": load_proposals(
            resolve(args.ours_proposals), recordings, args.ours_time_unit
        ),
    }
    rows = []
    for label in THUMOS_CLASSES:
        annotation_path = work_dir / "config" / "by_class" / label / "annotations.json"
        ground_truth = load_generic_ground_truth(annotation_path, recordings)
        class_result = {
            "protocol": "THUMOS14-E-frozen-proposals-v1",
            "protocol_declaration": PROTOCOL_DECLARATION,
            "class": label,
            "canonical_test_recordings": len(recordings),
            "ground_truth_instances": int(len(ground_truth)),
            "retag_tiou_thresholds": RETAG_TIOU,
            "proposal_budgets": BUDGETS,
            "methods": {},
            "oracle_note": (
                "IoU-ranked oracle is a transparent proposal ceiling. It is not named "
                "reTAG Perfect Classifier because the paper does not release its binary-score "
                "tie ordering before NMS."
            ),
        }
        for method, proposals in methods.items():
            result = method_result(proposals, ground_truth)
            class_result["methods"][method] = result
            rows.append(flatten_result(label, method, len(ground_truth), result))
        (per_class_dir / f"{label}.json").write_text(
            json.dumps(class_result, indent=2), encoding="utf-8"
        )

    per_class = pd.DataFrame(rows).sort_values(["class", "method"])
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    numeric = [
        column
        for column in per_class.columns
        if column not in {"class", "method", "ground_truth_instances", "proposal_count"}
    ]
    macro = per_class.groupby("method", sort=True)[numeric].mean().reset_index()
    macro.to_csv(out_dir / "macro_metrics.csv", index=False)
    report = {
        "protocol": "THUMOS14-E-frozen-proposals-v1",
        "protocol_declaration": PROTOCOL_DECLARATION,
        "classes": len(THUMOS_CLASSES),
        "canonical_test_recordings": len(recordings),
        "manifest_sha256": sha256_file(work_dir / "manifest.csv"),
        "proposal_inputs": {
            "retag": {
                "path": str(resolve(args.retag_proposals)),
                "sha256": sha256_file(resolve(args.retag_proposals)),
                "time_unit": args.retag_time_unit,
            },
            "eventpenguins_full": {
                "path": str(resolve(args.ours_proposals)),
                "sha256": sha256_file(resolve(args.ours_proposals)),
                "time_unit": args.ours_time_unit,
            },
        },
        "per_class_csv": str(out_dir / "per_class_metrics.csv"),
        "macro_csv": str(out_dir / "macro_metrics.csv"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
