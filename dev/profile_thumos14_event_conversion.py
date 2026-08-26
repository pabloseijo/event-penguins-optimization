#!/usr/bin/env python3
"""Profile the THUMOS14-E conversion recipe on one action window per class.

This is a resource audit, not an evaluation subset. Samples come only from the
official THUMOS14 validation split and never enter training or reported metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from dev.prepare_thumos14_event_corpus import (
    THUMOS_CLASSES,
    atomic_write_json,
    conversion_recipe,
    corpus_paths,
    implementation_provenance,
    read_manifest,
    recipe_id,
    resolve,
    sha256_file,
    v2e_command,
)


DEFAULT_OUTPUT = Path("data/thumos14_events/profiles/fixed3ms_action_center_10s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--work-dir", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    plan.add_argument("--window-seconds", type=float, default=10.0)
    plan.add_argument("--seed", type=int, default=1234567891)

    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--v2e-entry", type=Path, required=True)
    run.add_argument("--v2e-python", required=True)
    run.add_argument("--v2e-pythonpath", type=Path, required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--force", action="store_true")

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def stable_key(seed: int, label: str, candidate: Mapping[str, object]) -> str:
    payload = (
        f"{seed}|{label}|{candidate['video_id']}|"
        f"{candidate['annotation_start_s']}|{candidate['annotation_stop_s']}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def action_window(
    annotation_start: float,
    annotation_stop: float,
    duration: float,
    window_seconds: float,
) -> tuple[float, float]:
    span = min(window_seconds, duration)
    midpoint = (annotation_start + annotation_stop) / 2.0
    start = min(max(0.0, midpoint - span / 2.0), max(0.0, duration - span))
    return start, min(duration, start + span)


def build_plan(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    candidates: dict[str, list[dict[str, object]]] = {
        label: [] for label in THUMOS_CLASSES
    }
    for row in read_manifest(paths["manifest"]):
        if row["official_subset"] != "validation":
            continue
        metadata_path = paths["source_metadata"] / f"{row['video_id']}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        duration = float(metadata["duration_s"])
        for annotation in json.loads(row["annotations_json"]):
            label = str(annotation["label"])
            annotation_start, annotation_stop = map(float, annotation["segment"])
            if annotation_start >= duration:
                continue
            start, stop = action_window(
                annotation_start,
                min(annotation_stop, duration),
                duration,
                args.window_seconds,
            )
            candidates[label].append(
                {
                    "class": label,
                    "video_id": row["video_id"],
                    "source_path": row["source_path"],
                    "source_sha256": metadata["source_sha256"],
                    "source_duration_s": duration,
                    "annotation_start_s": annotation_start,
                    "annotation_stop_s": annotation_stop,
                    "start_time_s": start,
                    "stop_time_s": stop,
                }
            )

    samples = []
    for label in THUMOS_CLASSES:
        if not candidates[label]:
            raise RuntimeError(f"No reachable validation action for {label}")
        sample = min(
            candidates[label], key=lambda item: stable_key(args.seed, label, item)
        )
        sample["sample_id"] = f"{label.lower()}-{sample['video_id']}"
        samples.append(sample)

    output_root = resolve(args.output_root)
    payload = {
        "purpose": "resource_audit_only_not_training_or_evaluation",
        "selection": "one deterministic hash-selected reachable validation instance per class",
        "seed": args.seed,
        "window_seconds": args.window_seconds,
        "classes": len(THUMOS_CLASSES),
        "samples": samples,
        "output_root": str(output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "plan.json"
    atomic_write_json(plan_path, payload)
    print(f"[OK] profile plan classes={len(samples)} path={plan_path}")


def conversion_args(args: argparse.Namespace, sample: Mapping[str, object]) -> Namespace:
    return Namespace(
        v2e_python=args.v2e_python,
        v2e_entry=args.v2e_entry,
        v2e_pythonpath=args.v2e_pythonpath,
        output_width=346,
        output_height=260,
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.03,
        cutoff_hz=300.0,
        leak_rate_hz=0.01,
        shot_noise_rate_hz=0.001,
        refractory_period=0.0005,
        seed=1234567891,
        batch_size=args.batch_size,
        disable_slomo=False,
        fixed_timestamp_resolution=True,
        timestamp_resolution=0.003,
        dvs_profile="clean",
        start_time=float(sample["start_time_s"]),
        stop_time=float(sample["stop_time_s"]),
    )


def event_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(len(handle["events"]))


def run_profile(args: argparse.Namespace) -> None:
    plan_path = resolve(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output_root = Path(plan["output_root"])
    environment = os.environ.copy()
    pythonpath = str(resolve(args.v2e_pythonpath))
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    base_args = conversion_args(args, plan["samples"][0])
    implementation = implementation_provenance(base_args, environment)

    samples = sorted(plan["samples"], key=lambda item: str(item["sample_id"]))
    samples = [
        sample
        for index, sample in enumerate(samples)
        if index % args.num_shards == args.shard_index
    ]
    for sample in samples:
        output_dir = output_root / "samples" / str(sample["sample_id"])
        status_path = output_dir / "profile.json"
        if status_path.exists() and not args.force:
            print(f"[SKIP] {sample['sample_id']}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        current_args = conversion_args(args, sample)
        recipe = conversion_recipe(current_args)
        protocol = recipe_id(recipe, implementation)
        command = v2e_command(current_args, sample, output_dir)
        log_path = output_dir / "v2e.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                check=True,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.monotonic() - started
        events_path = output_dir / "events.h5"
        count = event_count(events_path)
        duration = float(sample["stop_time_s"]) - float(sample["start_time_s"])
        status = {
            **sample,
            "purpose": plan["purpose"],
            "protocol_id": protocol,
            "recipe": recipe,
            "implementation": implementation,
            "command": command,
            "wall_time_s": elapsed,
            "events": count,
            "event_rate_hz": count / duration,
            "output_bytes": events_path.stat().st_size,
            "output_sha256": sha256_file(events_path),
            "log_path": str(log_path),
        }
        atomic_write_json(status_path, status)
        print(
            f"[OK] {sample['class']} events={count} "
            f"rate={count / duration:.1f}/s wall={elapsed:.1f}s"
        )


def summarize(args: argparse.Namespace) -> None:
    plan = json.loads(resolve(args.plan).read_text(encoding="utf-8"))
    output_root = Path(plan["output_root"])
    records = []
    missing = []
    for sample in plan["samples"]:
        path = output_root / "samples" / str(sample["sample_id"]) / "profile.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(str(sample["sample_id"]))
    rates = np.asarray([float(item["event_rate_hz"]) for item in records])
    wall_per_source_second = np.asarray(
        [
            float(item["wall_time_s"])
            / (float(item["stop_time_s"]) - float(item["start_time_s"]))
            for item in records
        ]
    )
    bytes_per_source_second = np.asarray(
        [
            float(item["output_bytes"])
            / (float(item["stop_time_s"]) - float(item["start_time_s"]))
            for item in records
        ]
    )
    summary = {
        "purpose": plan["purpose"],
        "complete": not missing,
        "completed_samples": len(records),
        "missing_samples": missing,
        "event_rate_hz": quantiles(rates),
        "wall_seconds_per_source_second": quantiles(wall_per_source_second),
        "bytes_per_source_second": quantiles(bytes_per_source_second),
        "per_class": {
            str(item["class"]): {
                "video_id": item["video_id"],
                "event_rate_hz": item["event_rate_hz"],
                "wall_time_s": item["wall_time_s"],
                "output_bytes": item["output_bytes"],
            }
            for item in sorted(records, key=lambda value: str(value["class"]))
        },
    }
    path = output_root / "summary.json"
    atomic_write_json(path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def quantiles(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        build_plan(args)
    elif args.command == "run":
        run_profile(args)
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
