#!/usr/bin/env python3
"""Prepare the complete, auditable THUMOS14-E corpus for reTAG comparisons.

The corpus preserves the official 20 action labels. ``annotations.json`` is a
binary action/background view for the original single-class pipeline, while
``annotations_multiclass.json`` keeps the labels used by per-class evaluation.

Conversion uses the official v2e ``clean`` preset and the documented 3 ms
fixed-resolution SuperSloMo recipe by default. Adaptive timing, the official
``noisy`` preset and ``--disable-slomo`` are predeclared sensitivity ablations;
custom sensor parameters must be requested explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.build_thumos14_transfer_folds import assign_folds  # noqa: E402
from dev.prepare_thumos14_event_pilot import (  # noqa: E402
    ANNOTATION_URLS,
    VIDEO_URLS,
    atomic_download,
    converted_duration,
    copy_v2e_events,
    parse_class_annotations,
    probe_video,
    sha256_file,
)


DEFAULT_WORK_DIR = ROOT / "data" / "thumos14_events" / "thumos14e_v1"
THUMOS_CLASSES = (
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
)
CLASS_TO_ID = {label: index for index, label in enumerate(THUMOS_CLASSES)}
DVS_PRESETS = {
    "clean": {
        "pos_thres": 0.2,
        "neg_thres": 0.2,
        "sigma_thres": 0.02,
        "cutoff_hz": 0.0,
        "leak_rate_hz": 0.0,
        "shot_noise_rate_hz": 0.0,
        "refractory_period_s": 0.0,
        "leak_jitter_fraction": 0.0,
        "noise_rate_cov_decades": 0.0,
        "photoreceptor_noise": False,
    },
    "noisy": {
        "pos_thres": 0.2,
        "neg_thres": 0.2,
        "sigma_thres": 0.05,
        "cutoff_hz": 30.0,
        "leak_rate_hz": 0.1,
        "shot_noise_rate_hz": 5.0,
        "refractory_period_s": 0.0,
        "leak_jitter_fraction": 0.1,
        "noise_rate_cov_decades": 0.1,
        "photoreceptor_noise": False,
    },
}
MANIFEST_FIELDS = (
    "video_id",
    "official_subset",
    "split",
    "cv_fold",
    "evaluation_included",
    "source_url",
    "source_path",
    "labels_json",
    "annotations_json",
    "ambiguous_json",
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def corpus_paths(work_dir: str | Path) -> dict[str, Path]:
    root = resolve(work_dir)
    return {
        "root": root,
        "official": root / "official_annotations",
        "videos": root / "videos",
        "source_metadata": root / "source_metadata",
        "source_audit": root / "source_audit.json",
        "variant_parent": root / "variant_parent.json",
        "v2e": root / "v2e",
        "canonical": root / "canonical",
        "config": root / "config" / "annotations",
        "manifest": root / "manifest.csv",
        "split_audit": root / "split_audit.json",
        "data": root / "preprocessed.h5",
        "corpus_manifest": root / "corpus_manifest.json",
        "validation_report": root / "validation_report.json",
    }


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--video-ids", nargs="+", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = common_parser()

    prepare_parser = commands.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--validation-annotations", type=Path, default=None)
    prepare_parser.add_argument("--test-annotations", type=Path, default=None)
    prepare_parser.add_argument("--folds", type=int, default=5)
    prepare_parser.add_argument("--validation-fold", type=int, default=0)
    prepare_parser.add_argument("--seed", type=int, default=1234567891)

    download_parser = commands.add_parser("download", parents=[common])
    download_parser.add_argument("--retries", type=int, default=3)

    probe_parser = commands.add_parser("probe", parents=[common])
    probe_parser.add_argument("--ffprobe", default="ffprobe")

    audit_parser = commands.add_parser("audit", parents=[common])
    audit_parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Trust hashes recorded during download instead of recomputing every source hash.",
    )

    fork_parser = commands.add_parser("fork")
    fork_parser.add_argument("--work-dir", type=Path, required=True)
    fork_parser.add_argument("--source-work-dir", type=Path, required=True)
    fork_parser.add_argument("--purpose", required=True)

    convert_parser = commands.add_parser("convert", parents=[common])
    convert_parser.add_argument("--v2e-entry", type=Path, required=True)
    convert_parser.add_argument("--v2e-python", default=sys.executable)
    convert_parser.add_argument(
        "--v2e-pythonpath",
        type=Path,
        default=ROOT / "dev" / "v2e_headless_stubs",
    )
    timing_group = convert_parser.add_mutually_exclusive_group()
    timing_group.add_argument(
        "--fixed-timestamp-resolution",
        dest="fixed_timestamp_resolution",
        action="store_true",
        help="Use the fixed --timestamp-resolution recipe (the default).",
    )
    timing_group.add_argument(
        "--adaptive-timestamp-resolution",
        dest="fixed_timestamp_resolution",
        action="store_false",
        help="Sensitivity ablation using v2e optical-flow adaptive timing.",
    )
    convert_parser.set_defaults(fixed_timestamp_resolution=True)
    convert_parser.add_argument("--timestamp-resolution", type=float, default=0.003)
    convert_parser.add_argument(
        "--disable-slomo",
        action="store_true",
        help="Original-rate sensitivity ablation; not the primary corpus recipe.",
    )
    convert_parser.add_argument(
        "--dvs-profile",
        choices=("clean", "noisy", "custom"),
        default="clean",
        help="Official v2e clean/noisy preset, or explicit custom sensor parameters.",
    )
    convert_parser.add_argument("--pos-thres", type=float, default=0.2)
    convert_parser.add_argument("--neg-thres", type=float, default=0.2)
    convert_parser.add_argument("--sigma-thres", type=float, default=0.03)
    convert_parser.add_argument("--cutoff-hz", type=float, default=300.0)
    convert_parser.add_argument("--leak-rate-hz", type=float, default=0.01)
    convert_parser.add_argument("--shot-noise-rate-hz", type=float, default=0.001)
    convert_parser.add_argument("--refractory-period", type=float, default=0.0005)
    convert_parser.add_argument("--output-width", type=int, default=346)
    convert_parser.add_argument("--output-height", type=int, default=260)
    convert_parser.add_argument("--batch-size", type=int, default=8)
    convert_parser.add_argument("--seed", type=int, default=1234567891)
    convert_parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Absolute source-video start time for resource profiling only.",
    )
    convert_parser.add_argument("--stop-time", type=float, default=None)
    convert_parser.add_argument("--force", action="store_true")

    assemble_parser = commands.add_parser("assemble", parents=[common])
    assemble_parser.add_argument("--output", type=Path, default=None)
    assemble_parser.add_argument("--chunk-size", type=int, default=1_000_000)
    assemble_parser.add_argument("--force", action="store_true")

    validate_parser = commands.add_parser("validate", parents=[common])
    validate_parser.add_argument("--data-path", type=Path, default=None)
    validate_parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest {path}; run prepare first")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    os.replace(temporary, path)


def fork(args: argparse.Namespace) -> None:
    source = corpus_paths(args.source_work_dir)
    target = corpus_paths(args.work_dir)
    if source["root"] == target["root"]:
        raise ValueError("Variant work directory must differ from its source")
    required_files = ("manifest", "split_audit", "source_audit")
    required_dirs = ("official", "videos", "source_metadata")
    for name in required_files + required_dirs:
        if not source[name].exists():
            raise FileNotFoundError(source[name])
    target["root"].mkdir(parents=True, exist_ok=True)
    existing = [path for path in target["root"].iterdir()]
    if existing:
        raise FileExistsError(
            f"Variant directory must be empty, found {[path.name for path in existing]}"
        )
    for name in required_files:
        shutil.copy2(source[name], target[name])
    for name in required_dirs:
        target[name].symlink_to(source[name], target_is_directory=True)
    payload = {
        "protocol": "THUMOS14-E-source-fork-v1",
        "purpose": args.purpose,
        "source_root": str(source["root"]),
        "shared_read_only_directories": {
            name: str(source[name]) for name in required_dirs
        },
        "copied_manifests": {
            name: {
                "source": str(source[name]),
                "sha256": sha256_file(source[name]),
            }
            for name in required_files
        },
    }
    atomic_write_json(target["variant_parent"], payload)
    print(f"[OK] variant source fork={target['root']} parent={source['root']}")


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def selected_rows(
    rows: list[dict[str, str]],
    video_ids: list[str] | None,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, str]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= --shard-index < --num-shards")
    ordered = sorted(rows, key=lambda row: row["video_id"])
    if video_ids is not None:
        requested = set(video_ids)
        ordered = [row for row in ordered if row["video_id"] in requested]
        missing = requested - {row["video_id"] for row in ordered}
        if missing:
            raise ValueError(f"Videos absent from manifest: {sorted(missing)}")
    return [row for index, row in enumerate(ordered) if index % num_shards == shard_index]


def parse_official_annotations(
    archive_path: Path,
    subset: str,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[list[float]]]]:
    actions: dict[str, list[dict[str, object]]] = {}
    for label in THUMOS_CLASSES:
        for video_id, segments in parse_class_annotations(archive_path, label, subset).items():
            actions.setdefault(video_id, []).extend(
                {"label": label, "segment": [float(start), float(end)]}
                for start, end in segments
            )
    ambiguous = parse_class_annotations(archive_path, "Ambiguous", subset)
    return (
        {
            video_id: sorted(
                annotations,
                key=lambda item: (float(item["segment"][0]), str(item["label"])),
            )
            for video_id, annotations in sorted(actions.items())
        },
        ambiguous,
    )


def validation_fold_assignments(
    actions: Mapping[str, list[dict[str, object]]],
    folds: int,
    seed: int,
) -> dict[str, int]:
    database = {
        video_id: {
            "subset": "validation",
            "duration": 0.0,
            "annotations": [
                {
                    "label": item["label"],
                    "label_id": CLASS_TO_ID[str(item["label"])],
                    "segment": item["segment"],
                }
                for item in annotations
            ],
        }
        for video_id, annotations in actions.items()
    }
    return assign_folds(
        database,
        split="validation",
        labels=THUMOS_CLASSES,
        folds=folds,
        seed=seed,
    )


def class_summary(rows: Iterable[Mapping[str, str]]) -> dict[str, object]:
    split_instances: dict[str, Counter[str]] = {}
    split_videos: dict[str, Counter[str]] = {}
    split_counts: Counter[str] = Counter()
    for row in rows:
        split = str(row["split"])
        annotations = json.loads(str(row["annotations_json"]))
        labels = [str(item["label"]) for item in annotations]
        split_counts[split] += 1
        split_instances.setdefault(split, Counter()).update(labels)
        split_videos.setdefault(split, Counter()).update(set(labels))
    return {
        "videos_by_split": dict(sorted(split_counts.items())),
        "instances_by_split_and_class": {
            split: {label: counts[label] for label in THUMOS_CLASSES}
            for split, counts in sorted(split_instances.items())
        },
        "positive_videos_by_split_and_class": {
            split: {label: counts[label] for label in THUMOS_CLASSES}
            for split, counts in sorted(split_videos.items())
        },
    }


def prepare(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    paths["official"].mkdir(parents=True, exist_ok=True)
    archives = {
        "validation": resolve(args.validation_annotations)
        if args.validation_annotations
        else paths["official"] / "TH14_Temporal_annotations_validation.zip",
        "test": resolve(args.test_annotations)
        if args.test_annotations
        else paths["official"] / "TH14_Temporal_annotations_test.zip",
    }
    for subset, archive in archives.items():
        if not archive.exists():
            atomic_download(ANNOTATION_URLS[subset], archive)

    parsed = {
        subset: parse_official_annotations(archive, subset)
        for subset, archive in archives.items()
    }
    validation_actions, validation_ambiguous = parsed["validation"]
    test_actions, test_ambiguous = parsed["test"]
    validation_ids = set(validation_actions) | set(validation_ambiguous)
    test_ids = set(test_actions) | set(test_ambiguous)
    if len(validation_ids) != 200 or len(test_ids) != 213 or len(test_actions) != 212:
        raise ValueError(
            "Official THUMOS14 temporal universe changed or archives are invalid: "
            f"validation={len(validation_ids)} test={len(test_ids)} "
            f"test_action_videos={len(test_actions)}"
        )
    assignments = validation_fold_assignments(validation_actions, args.folds, args.seed)
    if not 0 <= args.validation_fold < args.folds:
        raise ValueError("--validation-fold must identify one of --folds")

    rows: list[dict[str, object]] = []
    for subset, video_ids, actions, ambiguous in (
        ("validation", validation_ids, validation_actions, validation_ambiguous),
        ("test", test_ids, test_actions, test_ambiguous),
    ):
        for video_id in sorted(video_ids):
            annotations = actions.get(video_id, [])
            labels = sorted({str(item["label"]) for item in annotations})
            cv_fold = assignments[video_id] if subset == "validation" else ""
            split = (
                "test"
                if subset == "test"
                else ("val" if cv_fold == args.validation_fold else "train")
            )
            rows.append(
                {
                    "video_id": video_id,
                    "official_subset": subset,
                    "split": split,
                    "cv_fold": cv_fold,
                    "evaluation_included": int(bool(annotations)),
                    "source_url": VIDEO_URLS[subset].format(video_id=video_id),
                    "source_path": str(paths["videos"] / subset / f"{video_id}.mp4"),
                    "labels_json": json.dumps(labels, separators=(",", ":")),
                    "annotations_json": json.dumps(annotations, separators=(",", ":")),
                    "ambiguous_json": json.dumps(ambiguous.get(video_id, []), separators=(",", ":")),
                }
            )
    write_manifest(paths["manifest"], rows)
    audit = {
        "protocol": "THUMOS14-E-v1",
        "classes": list(THUMOS_CLASSES),
        "folds": args.folds,
        "validation_fold": args.validation_fold,
        "seed": args.seed,
        "annotation_archives": {
            subset: {"path": str(path), "sha256": sha256_file(path)}
            for subset, path in archives.items()
        },
        "canonical_evaluation": {
            "validation_videos": 200,
            "test_files": 213,
            "test_action_videos": 212,
            "excluded_ambiguous_only": ["video_test_0001292"],
        },
        **class_summary(rows),
    }
    atomic_write_json(paths["split_audit"], audit)
    print(f"[OK] manifest={paths['manifest']} rows={len(rows)}")
    print(f"[OK] splits={audit['videos_by_split']}")


def metadata_path(paths: Mapping[str, Path], video_id: str) -> Path:
    return paths["source_metadata"] / f"{video_id}.json"


def load_metadata(paths: Mapping[str, Path], row: Mapping[str, str]) -> dict[str, object]:
    path = metadata_path(paths, row["video_id"])
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run download and probe first")
    return json.loads(path.read_text(encoding="utf-8"))


def download(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    rows = selected_rows(
        read_manifest(paths["manifest"]), args.video_ids, args.shard_index, args.num_shards
    )
    for row in rows:
        destination = Path(row["source_path"])
        print(f"[DOWNLOAD] {row['video_id']}")
        atomic_download(row["source_url"], destination, args.retries)
        status = {
            "video_id": row["video_id"],
            "source_path": str(destination),
            "source_url": row["source_url"],
            "source_bytes": destination.stat().st_size,
            "source_sha256": sha256_file(destination),
        }
        atomic_write_json(metadata_path(paths, row["video_id"]), status)


def probe(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    rows = selected_rows(
        read_manifest(paths["manifest"]), args.video_ids, args.shard_index, args.num_shards
    )
    for row in rows:
        source = Path(row["source_path"])
        if not source.exists():
            raise FileNotFoundError(f"Missing {source}; run download first")
        path = metadata_path(paths, row["video_id"])
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        status.update(
            {
                "video_id": row["video_id"],
                "source_path": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": status.get("source_sha256") or sha256_file(source),
                **probe_video(source, args.ffprobe),
            }
        )
        atomic_write_json(path, status)
        print(
            f"[PROBE] {row['video_id']} duration={float(status['duration_s']):.3f}s "
            f"fps={float(status['fps']):.3f}"
        )


def annotation_reachability(
    row: Mapping[str, str], duration: float
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    sources = (
        (
            "action",
            [
                {
                    "label": str(item["label"]),
                    "segment": item["segment"],
                }
                for item in json.loads(row["annotations_json"])
            ],
        ),
        (
            "ambiguous",
            [
                {"label": "Ambiguous", "segment": segment}
                for segment in json.loads(row.get("ambiguous_json", "[]"))
            ],
        ),
    )
    for kind, annotations in sources:
        for item in annotations:
            start, end = map(float, item["segment"])
            if not 0 <= start < end:
                raise ValueError(
                    f"{row['video_id']} has malformed {kind} segment {[start, end]}"
                )
            if end > duration + 1e-3:
                issues.append(
                    {
                        "video_id": row["video_id"],
                        "official_subset": row["official_subset"],
                        "kind": kind,
                        "label": item["label"],
                        "segment": [start, end],
                        "duration_s": duration,
                        "starts_after_video": start >= duration,
                        "overflow_s": end - duration,
                    }
                )
    return issues


def audit(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    rows = selected_rows(
        read_manifest(paths["manifest"]), args.video_ids, args.shard_index, args.num_shards
    )
    if args.num_shards != 1:
        raise ValueError("audit writes one source report and cannot run in shards")
    recordings = []
    issues = []
    total_seconds = 0.0
    total_bytes = 0
    for row in rows:
        metadata = load_metadata(paths, row)
        source = Path(row["source_path"])
        if not source.exists():
            raise FileNotFoundError(source)
        source_bytes = source.stat().st_size
        if source_bytes != int(metadata["source_bytes"]):
            raise ValueError(f"Source size changed for {row['video_id']}")
        if not args.skip_hash_verification:
            actual_sha = sha256_file(source)
            if actual_sha != metadata["source_sha256"]:
                raise ValueError(f"Source hash changed for {row['video_id']}")
        duration = float(metadata["duration_s"])
        fps = float(metadata["fps"])
        width = int(metadata["width"])
        height = int(metadata["height"])
        if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid probe metadata for {row['video_id']}: {metadata}")
        row_issues = annotation_reachability(row, duration)
        issues.extend(row_issues)
        total_seconds += duration
        total_bytes += source_bytes
        recordings.append(
            {
                "video_id": row["video_id"],
                "official_subset": row["official_subset"],
                "duration_s": duration,
                "fps": fps,
                "width": width,
                "height": height,
                "source_bytes": source_bytes,
                "source_sha256": metadata["source_sha256"],
                "unreachable_annotations": len(row_issues),
            }
        )
    issue_counts = Counter(
        (str(item["official_subset"]), str(item["label"])) for item in issues
    )
    payload = {
        "status": "ok_with_canonical_unreachable_annotations" if issues else "ok",
        "videos": len(recordings),
        "total_duration_s": total_seconds,
        "total_source_bytes": total_bytes,
        "hashes_verified": not args.skip_hash_verification,
        "unreachable_annotation_count": len(issues),
        "unreachable_video_count": len({str(item["video_id"]) for item in issues}),
        "unreachable_by_split_and_class": {
            f"{split}:{label}": count
            for (split, label), count in sorted(issue_counts.items())
        },
        "unreachable_annotations": issues,
        "recordings": recordings,
    }
    report_path = (
        paths["source_audit"]
        if args.video_ids is None
        else paths["root"] / "source_audit.partial.json"
    )
    atomic_write_json(report_path, payload)
    print(
        f"[OK] source audit videos={len(recordings)} "
        f"unreachable_gt={len(issues)} report={report_path}"
    )


def effective_dvs_parameters(args: argparse.Namespace) -> dict[str, object]:
    if args.dvs_profile in DVS_PRESETS:
        return dict(DVS_PRESETS[args.dvs_profile])
    return {
        "pos_thres": args.pos_thres,
        "neg_thres": args.neg_thres,
        "sigma_thres": args.sigma_thres,
        "cutoff_hz": args.cutoff_hz,
        "leak_rate_hz": args.leak_rate_hz,
        "shot_noise_rate_hz": args.shot_noise_rate_hz,
        "refractory_period_s": args.refractory_period,
        "leak_jitter_fraction": 0.1,
        "noise_rate_cov_decades": 0.1,
        "photoreceptor_noise": False,
    }


def conversion_recipe(args: argparse.Namespace) -> dict[str, object]:
    timing = "original_rate_ablation" if args.disable_slomo else (
        "fixed_interpolated" if args.fixed_timestamp_resolution else "adaptive_interpolated"
    )
    return {
        "protocol": "THUMOS14-E-v2e-v1",
        "timing": timing,
        "auto_timestamp_resolution": (
            None if args.disable_slomo else not args.fixed_timestamp_resolution
        ),
        "timestamp_resolution_s": (
            None if args.disable_slomo else args.timestamp_resolution
        ),
        "disable_slomo": bool(args.disable_slomo),
        "dvs_profile": args.dvs_profile,
        "effective_dvs_parameters": effective_dvs_parameters(args),
        "output_width": args.output_width,
        "output_height": args.output_height,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "start_time_s": args.start_time,
        "stop_time_s": args.stop_time,
    }


def source_inventory(root: Path) -> dict[str, object]:
    selected = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".ckpt", ".pt", ".pth"}
    )
    digest = hashlib.sha256()
    files = []
    for path in selected:
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        if path.suffix in {".ckpt", ".pt", ".pth"}:
            files.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": file_hash}
            )
    return {
        "root": str(root),
        "source_and_weight_files": len(selected),
        "aggregate_sha256": digest.hexdigest(),
        "weights": files,
    }


def git_provenance(root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short")
        diff = subprocess.run(
            ("git", "-C", str(root), "diff", "HEAD", "--binary", "--", "."),
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": None, "diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def python_runtime(python: str, environment: Mapping[str, str]) -> dict[str, object]:
    program = """
import json, platform, sys
import cv2, numpy, torch
payload = {
    'python': sys.version,
    'platform': platform.platform(),
    'numpy': numpy.__version__,
    'opencv': cv2.__version__,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
}
if torch.cuda.is_available():
    payload['cuda_device'] = torch.cuda.get_device_name(0)
    payload['cuda_capability'] = list(torch.cuda.get_device_capability(0))
print(json.dumps(payload, sort_keys=True))
"""
    output = subprocess.run(
        (python, "-c", program),
        check=True,
        capture_output=True,
        text=True,
        env=dict(environment),
    ).stdout
    return json.loads(output)


def implementation_provenance(
    args: argparse.Namespace, environment: Mapping[str, str]
) -> dict[str, object]:
    v2e_entry = resolve(args.v2e_entry)
    v2e_root = v2e_entry.parent
    stubs_root = resolve(args.v2e_pythonpath)
    return {
        "v2e_entry": str(v2e_entry),
        "v2e_entry_sha256": sha256_file(v2e_entry),
        "v2e_source_inventory": source_inventory(v2e_root),
        "v2e_git": git_provenance(v2e_root),
        "headless_stubs_inventory": source_inventory(stubs_root),
        "runtime": python_runtime(args.v2e_python, environment),
    }


def recipe_id(
    recipe: Mapping[str, object], implementation: Mapping[str, object]
) -> str:
    fingerprint = {
        "v2e_entry_sha256": implementation["v2e_entry_sha256"],
        "v2e_source_inventory_sha256": implementation["v2e_source_inventory"][
            "aggregate_sha256"
        ],
        "v2e_git_commit": implementation["v2e_git"]["commit"],
        "v2e_git_diff_sha256": implementation["v2e_git"]["diff_sha256"],
        "headless_stubs_sha256": implementation["headless_stubs_inventory"][
            "aggregate_sha256"
        ],
        "runtime": implementation["runtime"],
    }
    payload = json.dumps(
        {"recipe": recipe, "implementation": fingerprint},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def v2e_command(
    args: argparse.Namespace,
    row: Mapping[str, str],
    output_dir: Path,
) -> list[str]:
    command = [
        args.v2e_python,
        str(resolve(args.v2e_entry)),
        "-i",
        row["source_path"],
        "--output_folder",
        str(output_dir),
        "--overwrite",
        "--unique_output_folder=False",
        "--skip_video_output",
        "--no_preview",
        "--dvs_h5",
        "events.h5",
        "--output_width",
        str(args.output_width),
        "--output_height",
        str(args.output_height),
        "--dvs_emulator_seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
    ]
    if args.dvs_profile in {"clean", "noisy"}:
        command.append(f"--dvs_params={args.dvs_profile}")
    else:
        command.extend(
            (
                "--pos_thres",
                str(args.pos_thres),
                "--neg_thres",
                str(args.neg_thres),
                "--sigma_thres",
                str(args.sigma_thres),
                "--cutoff_hz",
                str(args.cutoff_hz),
                "--leak_rate_hz",
                str(args.leak_rate_hz),
                "--shot_noise_rate_hz",
                str(args.shot_noise_rate_hz),
                "--refractory_period",
                str(args.refractory_period),
            )
        )
    if args.disable_slomo:
        command.append("--disable_slomo")
    elif args.fixed_timestamp_resolution:
        if args.timestamp_resolution is None:
            raise ValueError("--fixed-timestamp-resolution requires --timestamp-resolution")
        command.extend(
            (
                "--auto_timestamp_resolution=False",
                f"--timestamp_resolution={args.timestamp_resolution}",
            )
        )
    else:
        command.append("--auto_timestamp_resolution=True")
        if args.timestamp_resolution is not None:
            command.append(f"--timestamp_resolution={args.timestamp_resolution}")
    if args.start_time is not None:
        command.extend(("--start_time", str(args.start_time)))
    if args.stop_time is not None:
        command.extend(("--stop_time", str(args.stop_time)))
    return command


def convert(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    rows = selected_rows(
        read_manifest(paths["manifest"]), args.video_ids, args.shard_index, args.num_shards
    )
    v2e_entry = resolve(args.v2e_entry)
    if args.start_time is not None and args.start_time < 0:
        raise ValueError("--start-time cannot be negative")
    if args.stop_time is not None and args.stop_time <= (args.start_time or 0.0):
        raise ValueError("--stop-time must be greater than --start-time")
    cutoff_hz = float(effective_dvs_parameters(args)["cutoff_hz"])
    if cutoff_hz > 0 and not args.disable_slomo:
        maximum_step = 0.3 / (2.0 * math.pi * cutoff_hz)
        if args.timestamp_resolution is None or args.timestamp_resolution > maximum_step:
            raise ValueError(
                f"{args.dvs_profile} cutoff_hz={cutoff_hz:g} requires "
                f"--timestamp-resolution <= {maximum_step:.9f}s for v2e's IIR condition"
            )
    environment = os.environ.copy()
    pythonpath = str(resolve(args.v2e_pythonpath))
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    recipe = conversion_recipe(args)
    implementation = implementation_provenance(args, environment)
    protocol_id = recipe_id(recipe, implementation)

    for row in rows:
        metadata = load_metadata(paths, row)
        if "duration_s" not in metadata:
            raise ValueError(f"Missing probe metadata for {row['video_id']}")
        output_dir = paths["v2e"] / row["video_id"]
        output_h5 = output_dir / "events.h5"
        status_path = output_dir / "conversion.json"
        if output_h5.exists() and status_path.exists() and not args.force:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("protocol_id") != protocol_id:
                raise RuntimeError(
                    f"{row['video_id']} was converted with another recipe; use --force explicitly"
                )
            print(f"[SKIP] {row['video_id']} already converted")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = v2e_command(args, row, output_dir)
        print(f"[V2E] {row['video_id']} protocol={protocol_id[:12]}")
        subprocess.run(command, check=True, env=environment)
        if not output_h5.exists():
            raise FileNotFoundError(f"v2e did not create {output_h5}")
        status = {
            "video_id": row["video_id"],
            "protocol_id": protocol_id,
            "recipe": recipe,
            "source_sha256": metadata["source_sha256"],
            "implementation": implementation,
            "command": command,
            "output_path": str(output_h5),
            "output_bytes": output_h5.stat().st_size,
            "output_sha256": sha256_file(output_h5),
        }
        atomic_write_json(status_path, status)


def valid_annotations(
    row: Mapping[str, str],
    duration: float,
    trim_to_duration: bool = False,
    allow_unreachable: bool = False,
) -> list[dict[str, object]]:
    annotations = json.loads(row["annotations_json"])
    if trim_to_duration:
        annotations = [
            {
                **item,
                "segment": [
                    max(0.0, float(item["segment"][0])),
                    min(duration, float(item["segment"][1])),
                ],
            }
            for item in annotations
            if float(item["segment"][0]) < duration
            and float(item["segment"][1]) > 0.0
        ]
    invalid = [
        item
        for item in annotations
        if not 0 <= float(item["segment"][0]) < float(item["segment"][1])
        or (
            not allow_unreachable
            and float(item["segment"][1]) > duration + 1e-3
        )
    ]
    if invalid:
        raise ValueError(f"{row['video_id']} has annotations outside duration: {invalid[:3]}")
    return annotations


def reachable_annotations(
    annotations: Iterable[Mapping[str, object]], duration: float
) -> list[dict[str, object]]:
    return [
        dict(item)
        for item in annotations
        if 0 <= float(item["segment"][0])
        < float(item["segment"][1])
        <= duration + 1e-3
    ]


def write_annotation_views(
    config_dir: Path,
    rows: list[dict[str, str]],
    durations: Mapping[str, float],
    event_counts: Mapping[str, int],
    partial_video_ids: set[str] | None = None,
) -> None:
    binary_database: dict[str, object] = {}
    binary_trainable_database: dict[str, object] = {}
    multiclass_database: dict[str, object] = {}
    multiclass_trainable_database: dict[str, object] = {}
    class_databases: dict[str, dict[str, object]] = {
        label: {} for label in THUMOS_CLASSES
    }
    class_trainable_databases: dict[str, dict[str, object]] = {
        label: {} for label in THUMOS_CLASSES
    }
    class_info_rows: dict[str, list[dict[str, object]]] = {
        label: [] for label in THUMOS_CLASSES
    }
    info_rows = []
    for recording_id, row in enumerate(rows):
        video_id = row["video_id"]
        trim = video_id in (partial_video_ids or set())
        annotations = valid_annotations(
            row,
            durations[video_id],
            trim_to_duration=trim,
            allow_unreachable=not trim,
        )
        trainable_annotations = reachable_annotations(
            annotations, durations[video_id]
        )
        ambiguous_segments = json.loads(row.get("ambiguous_json", "[]"))
        if trim:
            ambiguous_segments = [
                [max(0.0, float(start)), min(durations[video_id], float(end))]
                for start, end in ambiguous_segments
                if float(start) < durations[video_id] and float(end) > 0.0
            ]
        ambiguous = [
            {"label": "ambiguous", "segment": segment}
            for segment in ambiguous_segments
        ]
        binary = [
            {
                "label": "ed",
                "source_label": item["label"],
                "segment": item["segment"],
            }
            for item in annotations
        ] + ambiguous
        binary_trainable = [
            {
                "label": "ed",
                "source_label": item["label"],
                "segment": item["segment"],
            }
            for item in trainable_annotations
        ] + ambiguous
        multiclass = [
            {
                "label": item["label"],
                "label_id": CLASS_TO_ID[str(item["label"])],
                "segment": item["segment"],
            }
            for item in annotations
        ] + ambiguous
        multiclass_trainable = [
            {
                "label": item["label"],
                "label_id": CLASS_TO_ID[str(item["label"])],
                "segment": item["segment"],
            }
            for item in trainable_annotations
        ] + ambiguous
        binary_database[video_id] = {"annotations": {"1": binary}}
        binary_trainable_database[video_id] = {
            "annotations": {"1": binary_trainable}
        }
        multiclass_database[video_id] = {"annotations": {"1": multiclass}}
        multiclass_trainable_database[video_id] = {
            "annotations": {"1": multiclass_trainable}
        }
        class_counts = Counter(str(item["label"]) for item in annotations)
        info_rows.append(
            {
                "timestamp": video_id,
                "recording_id": recording_id,
                "roi_group_id": 1,
                "split": row["split"],
                "cv_fold": row["cv_fold"],
                "official_subset": row["official_subset"],
                "evaluation_included": row["evaluation_included"],
                "ed_cnt": len(annotations),
                "trainable_ed_cnt": len(trainable_annotations),
                "unreachable_gt_cnt": len(annotations) - len(trainable_annotations),
                "event_count": event_counts[video_id],
                "duration_s": f"{durations[video_id]:.9f}",
                "class_counts_json": json.dumps(class_counts, sort_keys=True, separators=(",", ":")),
            }
        )
        for target_label in THUMOS_CLASSES:
            target_annotations = [
                {
                    "label": "ed" if item["label"] == target_label else "other_action",
                    "source_label": item["label"],
                    "segment": item["segment"],
                }
                for item in annotations
            ] + ambiguous
            target_trainable_annotations = [
                {
                    "label": "ed" if item["label"] == target_label else "other_action",
                    "source_label": item["label"],
                    "segment": item["segment"],
                }
                for item in trainable_annotations
            ] + ambiguous
            target_count = sum(item["label"] == target_label for item in annotations)
            target_trainable_count = sum(
                item["label"] == target_label for item in trainable_annotations
            )
            class_databases[target_label][video_id] = {
                "annotations": {"1": target_annotations}
            }
            class_trainable_databases[target_label][video_id] = {
                "annotations": {"1": target_trainable_annotations}
            }
            class_info_rows[target_label].append(
                {
                    "timestamp": video_id,
                    "recording_id": recording_id,
                    "roi_group_id": 1,
                    "split": row["split"],
                    "cv_fold": row["cv_fold"],
                    "official_subset": row["official_subset"],
                    "evaluation_included": row["evaluation_included"],
                    "ed_cnt": target_count,
                    "trainable_ed_cnt": target_trainable_count,
                    "unreachable_target_gt_cnt": target_count - target_trainable_count,
                    "other_action_cnt": len(annotations) - target_count,
                    "ambiguous_cnt": len(ambiguous),
                    "event_count": event_counts[video_id],
                    "duration_s": f"{durations[video_id]:.9f}",
                }
            )
    config_dir.mkdir(parents=True, exist_ok=True)
    binary_payload = {"version": "THUMOS14-E-binary-v1", "database": binary_database}
    atomic_write_json(config_dir / "annotations.json", binary_payload)
    atomic_write_json(config_dir / "annotations_action.json", binary_payload)
    atomic_write_json(
        config_dir / "annotations_trainable.json",
        {
            "version": "THUMOS14-E-binary-trainable-v1",
            "database": binary_trainable_database,
        },
    )
    atomic_write_json(
        config_dir / "annotations_multiclass.json",
        {"version": "THUMOS14-E-multiclass-v1", "database": multiclass_database},
    )
    atomic_write_json(
        config_dir / "annotations_multiclass_trainable.json",
        {
            "version": "THUMOS14-E-multiclass-trainable-v1",
            "database": multiclass_trainable_database,
        },
    )
    atomic_write_json(
        config_dir / "class_map.json",
        {"classes": list(THUMOS_CLASSES), "class_to_id": CLASS_TO_ID},
    )
    info_path = config_dir / "recording_info.csv"
    temporary = info_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(info_rows[0]))
        writer.writeheader()
        writer.writerows(info_rows)
    os.replace(temporary, info_path)
    validation_rows = [
        row
        for row in rows
        if row["official_subset"] == "validation" and str(row["cv_fold"]) != ""
    ]
    fold_rows = []
    for fold in sorted({int(row["cv_fold"]) for row in validation_rows}):
        validation_recordings = sorted(
            row["video_id"] for row in validation_rows if int(row["cv_fold"]) == fold
        )
        training_recordings = sorted(
            row["video_id"] for row in validation_rows if int(row["cv_fold"]) != fold
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_record_names": " ".join(training_recordings),
                "val_record_names": " ".join(validation_recordings),
                "train_videos": len(training_recordings),
                "val_videos": len(validation_recordings),
            }
        )
    if fold_rows:
        folds_path = config_dir / "fold_manifest.csv"
        folds_temporary = folds_path.with_suffix(".csv.tmp")
        with folds_temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
            writer.writeheader()
            writer.writerows(fold_rows)
        os.replace(folds_temporary, folds_path)
    for target_label in THUMOS_CLASSES:
        target_dir = config_dir / "by_class" / target_label
        atomic_write_json(
            target_dir / "annotations.json",
            {
                "version": "THUMOS14-E-one-vs-rest-v1",
                "target_class": target_label,
                "database": class_databases[target_label],
            },
        )
        atomic_write_json(
            target_dir / "annotations_trainable.json",
            {
                "version": "THUMOS14-E-one-vs-rest-trainable-v1",
                "target_class": target_label,
                "database": class_trainable_databases[target_label],
            },
        )
        target_info_path = target_dir / "recording_info.csv"
        target_temporary = target_info_path.with_suffix(".csv.tmp")
        with target_temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(class_info_rows[target_label][0]),
            )
            writer.writeheader()
            writer.writerows(class_info_rows[target_label])
        os.replace(target_temporary, target_info_path)
        if fold_rows:
            target_counts = {
                str(row["timestamp"]): int(row["ed_cnt"])
                for row in class_info_rows[target_label]
            }
            target_fold_rows = [
                {
                    **row,
                    "val_ed_instances": sum(
                        target_counts[name]
                        for name in str(row["val_record_names"]).split()
                    ),
                }
                for row in fold_rows
            ]
            target_fold_path = target_dir / "fold_manifest.csv"
            target_fold_temporary = target_fold_path.with_suffix(".csv.tmp")
            with target_fold_temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(target_fold_rows[0]),
                )
                writer.writeheader()
                writer.writerows(target_fold_rows)
            os.replace(target_fold_temporary, target_fold_path)


def assemble(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    all_rows = read_manifest(paths["manifest"])
    rows = selected_rows(all_rows, args.video_ids, args.shard_index, args.num_shards)
    if args.num_shards != 1:
        raise ValueError("assemble builds one canonical index and cannot run in shards")
    output = resolve(args.output) if args.output else paths["data"]
    if args.video_ids is not None and args.output is None:
        raise ValueError("A partial assembly requires an explicit --output")
    partial = args.video_ids is not None or len(rows) != len(all_rows)
    artifact_root = output.parent if partial else paths["root"]
    config_path = artifact_root / "config" / "annotations"
    corpus_manifest_path = artifact_root / "corpus_manifest.json"

    durations: dict[str, float] = {}
    event_counts: dict[str, int] = {}
    partial_video_ids: set[str] = set()
    conversion_protocols: dict[str, dict[str, object]] = {}
    recording_entries = []
    for row in rows:
        metadata = load_metadata(paths, row)
        source_h5 = paths["v2e"] / row["video_id"] / "events.h5"
        conversion_path = paths["v2e"] / row["video_id"] / "conversion.json"
        if not source_h5.exists() or not conversion_path.exists():
            raise FileNotFoundError(f"Missing v2e output for {row['video_id']}")
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        protocol_id = str(conversion["protocol_id"])
        protocol_record = {
            "recipe": conversion["recipe"],
            "implementation": conversion["implementation"],
        }
        existing_protocol = conversion_protocols.setdefault(protocol_id, protocol_record)
        if existing_protocol != protocol_record:
            raise RuntimeError(
                f"Conversion provenance changed within protocol {protocol_id}"
            )
        if conversion["recipe"].get("start_time_s") is not None:
            raise ValueError(
                f"Profiling fragment {row['video_id']} with start_time cannot be assembled"
            )
        if conversion["recipe"].get("stop_time_s") is not None:
            partial_video_ids.add(row["video_id"])
        canonical_path = paths["canonical"] / f"{row['video_id']}.h5"
        if canonical_path.exists() and not args.force:
            with h5py.File(canonical_path, "r") as existing:
                if str(existing["recording"].attrs.get("protocol_id", "")) != str(
                    conversion["protocol_id"]
                ):
                    raise RuntimeError(f"Stale canonical file {canonical_path}")
                durations[row["video_id"]] = float(existing["recording"].attrs["duration_s"])
                event_counts[row["video_id"]] = int(len(existing["recording/N01/events"]))
        else:
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = canonical_path.with_suffix(".h5.building")
            if temporary.exists():
                temporary.unlink()
            with h5py.File(source_h5, "r") as source, h5py.File(temporary, "w") as target:
                source_events = source["events"]
                duration = converted_duration(
                    {"duration_s": str(metadata["duration_s"])}, conversion["recipe"], source_events
                )
                recording = target.create_group("recording")
                recording.attrs["split"] = row["split"]
                recording.attrs["official_subset"] = row["official_subset"]
                recording.attrs["cv_fold"] = row["cv_fold"]
                recording.attrs["duration_s"] = duration
                recording.attrs["protocol_id"] = conversion["protocol_id"]
                recording.attrs["source_sha256"] = metadata["source_sha256"]
                recording.attrs["labels_json"] = row["labels_json"]
                recording.attrs["partial_conversion"] = row["video_id"] in partial_video_ids
                roi = recording.create_group("N01")
                roi.attrs["width"] = int(conversion["recipe"]["output_width"])
                roi.attrs["height"] = int(conversion["recipe"]["output_height"])
                roi.attrs["duration_s"] = duration
                copy_v2e_events(source_events, roi, args.chunk_size)
                durations[row["video_id"]] = duration
                event_counts[row["video_id"]] = int(len(source_events))
            os.replace(temporary, canonical_path)
        recording_entries.append(
            {
                "video_id": row["video_id"],
                "canonical_path": str(canonical_path),
                "canonical_sha256": sha256_file(canonical_path),
                "events": event_counts[row["video_id"]],
                "duration_s": durations[row["video_id"]],
                "protocol_id": conversion["protocol_id"],
            }
        )
        print(f"[ASSEMBLE] {row['video_id']} events={event_counts[row['video_id']]:,}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = output.with_suffix(output.suffix + ".building")
    if temporary_index.exists():
        temporary_index.unlink()
    with h5py.File(temporary_index, "w") as index:
        index.attrs["format"] = "event-penguins-external-[x,y,t_us,p]-v1"
        index.attrs["source"] = "THUMOS14 converted with v2e"
        for entry in recording_entries:
            relative = os.path.relpath(entry["canonical_path"], output.parent)
            index[entry["video_id"]] = h5py.ExternalLink(relative, "/recording")
    os.replace(temporary_index, output)
    write_annotation_views(
        config_path,
        rows,
        durations,
        event_counts,
        partial_video_ids=partial_video_ids,
    )
    atomic_write_json(
        corpus_manifest_path,
        {
            "protocol": "THUMOS14-E-v1",
                "partial": partial,
                "index_path": str(output),
                "index_sha256": sha256_file(output),
                "conversion_protocols": conversion_protocols,
                "recordings": recording_entries,
            },
        )
    print(f"[OK] data index={output} recordings={len(recording_entries)}")


def validate_event_dataset(events: h5py.Dataset, width: int, height: int, chunk_size: int) -> dict:
    if events.ndim != 2 or events.shape[1] != 4 or len(events) == 0:
        raise ValueError(f"Invalid event dataset shape {events.shape}")
    previous = -1
    x_min = y_min = 2**32 - 1
    x_max = y_max = 0
    polarities: set[int] = set()
    for start in range(0, len(events), chunk_size):
        values = np.asarray(events[start : start + chunk_size])
        timestamps = values[:, 2].astype(np.int64)
        if timestamps[0] < previous or np.any(np.diff(timestamps) < 0):
            raise ValueError("Non-monotonic event timestamps")
        previous = int(timestamps[-1])
        x_min = min(x_min, int(values[:, 0].min()))
        x_max = max(x_max, int(values[:, 0].max()))
        y_min = min(y_min, int(values[:, 1].min()))
        y_max = max(y_max, int(values[:, 1].max()))
        polarities.update(map(int, np.unique(values[:, 3])))
    if not (0 <= x_min <= x_max < width and 0 <= y_min <= y_max < height):
        raise ValueError("Event coordinates outside the declared sensor")
    if not polarities.issubset({0, 1}):
        raise ValueError(f"Invalid polarities {sorted(polarities)}")
    return {
        "events": int(len(events)),
        "first_timestamp_s": float(events[0, 2]) / 1e6,
        "last_timestamp_s": float(events[-1, 2]) / 1e6,
        "x_range": [x_min, x_max],
        "y_range": [y_min, y_max],
        "polarities": sorted(polarities),
    }


def validate(args: argparse.Namespace) -> None:
    paths = corpus_paths(args.work_dir)
    all_rows = read_manifest(paths["manifest"])
    rows = selected_rows(all_rows, args.video_ids, args.shard_index, args.num_shards)
    if args.num_shards != 1:
        raise ValueError("validate checks one canonical index and cannot run in shards")
    data_path = resolve(args.data_path) if args.data_path else paths["data"]
    if args.video_ids is not None and args.data_path is None:
        raise ValueError("Partial validation requires an explicit --data-path")
    expected = {row["video_id"] for row in rows}
    reports = []
    validated_rows = []
    with h5py.File(data_path, "r") as handle:
        actual = set(handle.keys())
        if actual != expected:
            raise ValueError(f"Index mismatch missing={expected-actual} extra={actual-expected}")
        for row in rows:
            recording = handle[row["video_id"]]
            roi = recording["N01"]
            duration = float(recording.attrs["duration_s"])
            annotations = valid_annotations(
                row,
                duration,
                trim_to_duration=bool(recording.attrs.get("partial_conversion", False)),
                allow_unreachable=not bool(
                    recording.attrs.get("partial_conversion", False)
                ),
            )
            validated_row = dict(row)
            validated_row["annotations_json"] = json.dumps(
                annotations, separators=(",", ":")
            )
            validated_rows.append(validated_row)
            event_report = validate_event_dataset(
                roi["events"], int(roi.attrs["width"]), int(roi.attrs["height"]), args.chunk_size
            )
            reports.append(
                {
                    "video_id": row["video_id"],
                    "split": row["split"],
                    "duration_s": duration,
                    "labels": sorted({str(item["label"]) for item in annotations}),
                    "unreachable_annotations": len(
                        annotation_reachability(row, duration)
                    ),
                    **event_report,
                }
            )
            print(f"[VALID] {row['video_id']} events={event_report['events']:,}")
    payload = {
        "status": "ok",
        "data_path": str(data_path),
        "index_sha256": sha256_file(data_path),
        "recordings": reports,
        "class_summary": class_summary(validated_rows),
    }
    report_path = (
        data_path.parent / "validation_report.json"
        if args.video_ids is not None or len(rows) != len(all_rows)
        else paths["validation_report"]
    )
    atomic_write_json(report_path, payload)
    print(f"[OK] validation report={report_path}")


def main() -> None:
    args = parse_args()
    args.work_dir = resolve(args.work_dir)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
