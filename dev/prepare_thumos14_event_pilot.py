#!/usr/bin/env python3
"""Prepare and validate a one-class THUMOS14 synthetic-event pilot.

The script keeps one canonical event HDF5 for both reTAG and the complete
EventPenguins pipeline. v2e writes rows as ``[t_us, x, y, p]``; the canonical
project format is ``[x, y, t_us, p]``.

Typical use from the repository root::

    python dev/prepare_thumos14_event_pilot.py prepare
    python dev/prepare_thumos14_event_pilot.py download
    python dev/prepare_thumos14_event_pilot.py convert --v2e-entry ../v2e/v2e.py
    python dev/prepare_thumos14_event_pilot.py assemble
    python dev/prepare_thumos14_event_pilot.py validate

Use ``--video-ids video_validation_0000488 video_test_0000062`` and
``convert --stop-time 10`` for the initial end-to-end smoke test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = ROOT / "data" / "thumos14_events" / "longjump_pilot"
ANNOTATION_URLS = {
    "validation": (
        "https://www.crcv.ucf.edu/THUMOS14/Validation_set/"
        "TH14_Temporal_annotations_validation.zip"
    ),
    "test": (
        "https://www.crcv.ucf.edu/THUMOS14/test_set/"
        "TH14_Temporal_annotations_test.zip"
    ),
}
VIDEO_URLS = {
    "validation": (
        "http://www.crcv.ucf.edu/THUMOS14/Validation_set/videos/{video_id}.mp4"
    ),
    "test": (
        "http://www.crcv.ucf.edu/THUMOS14/test_set/"
        "TH14_test_set_mp4/{video_id}.mp4"
    ),
}
MANIFEST_FIELDS = (
    "video_id",
    "class_name",
    "official_subset",
    "split",
    "source_url",
    "source_path",
    "source_sha256",
    "source_bytes",
    "duration_s",
    "fps",
    "width",
    "height",
    "segments_json",
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--class-name", default="LongJump")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=None,
        help="Optional subset for a smoke test; defaults to every positive video.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = common_parser()

    prepare = subparsers.add_parser("prepare", parents=[common])
    prepare.add_argument("--validation-annotations", type=Path, default=None)
    prepare.add_argument("--test-annotations", type=Path, default=None)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)

    download = subparsers.add_parser("download", parents=[common])
    download.add_argument("--retries", type=int, default=3)

    probe = subparsers.add_parser("probe", parents=[common])
    probe.add_argument("--ffprobe", default="ffprobe")

    convert = subparsers.add_parser("convert", parents=[common])
    convert.add_argument("--v2e-entry", type=Path, required=True)
    convert.add_argument("--v2e-python", default=sys.executable)
    convert.add_argument(
        "--v2e-pythonpath",
        type=Path,
        default=ROOT / "dev" / "v2e_headless_stubs",
        help="Compatibility stubs for v2e's unused GUI/renderer imports on headless servers.",
    )
    convert.add_argument("--timestamp-resolution", type=float, default=0.003)
    convert.add_argument(
        "--disable-slomo",
        action="store_true",
        help=(
            "Generate events at source-frame timestamps without SuperSloMo. "
            "This is substantially faster and matches the original-rate synthetic-event protocol."
        ),
    )
    convert.add_argument("--pos-thres", type=float, default=0.2)
    convert.add_argument("--neg-thres", type=float, default=0.2)
    convert.add_argument("--sigma-thres", type=float, default=0.03)
    convert.add_argument("--cutoff-hz", type=float, default=15.0)
    convert.add_argument("--leak-rate-hz", type=float, default=0.01)
    convert.add_argument("--shot-noise-rate-hz", type=float, default=0.001)
    convert.add_argument("--output-width", type=int, default=346)
    convert.add_argument("--output-height", type=int, default=260)
    convert.add_argument("--batch-size", type=int, default=8)
    convert.add_argument("--stop-time", type=float, default=None)
    convert.add_argument("--force", action="store_true")

    assemble = subparsers.add_parser("assemble", parents=[common])
    assemble.add_argument("--output", type=Path, default=None)
    assemble.add_argument("--chunk-size", type=int, default=1_000_000)
    assemble.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", parents=[common])
    validate.add_argument("--data-path", type=Path, default=None)
    validate.add_argument("--ann-path", type=Path, default=None)
    validate.add_argument("--bin-width", type=float, default=0.033)
    validate.add_argument("--plot", action="store_true")
    return parser.parse_args()


def work_paths(work_dir: Path) -> dict[str, Path]:
    work_dir = resolve(work_dir)
    return {
        "root": work_dir,
        "annotations": work_dir / "official_annotations",
        "videos": work_dir / "videos",
        "v2e": work_dir / "v2e",
        "config": work_dir / "config" / "annotations",
        "manifest": work_dir / "manifest.csv",
        "data": work_dir / "preprocessed.h5",
        "report": work_dir / "validation_report.json",
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_download(url: str, destination: Path, retries: int = 3) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "event-penguins/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            partial.replace(destination)
            return
        except Exception as error:  # pragma: no cover - depends on network
            last_error = error
            print(f"[WARN] download attempt {attempt}/{retries} failed: {error}")
    raise RuntimeError(f"Could not download {url}") from last_error


def annotation_member(archive: zipfile.ZipFile, class_name: str, subset: str) -> str:
    suffix = "val.txt" if subset == "validation" else "test.txt"
    expected = f"{class_name}_{suffix}".lower()
    matches = [name for name in archive.namelist() if Path(name).name.lower() == expected]
    if len(matches) != 1:
        raise ValueError(f"Expected one {expected!r} member, found {matches}")
    return matches[0]


def parse_class_annotations(path: Path, class_name: str, subset: str) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    with zipfile.ZipFile(path) as archive:
        text = archive.read(annotation_member(archive, class_name, subset)).decode("utf-8")
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"Unexpected annotation row: {line!r}")
        video_id, start, end = fields
        segment = [float(start), float(end)]
        if not 0 <= segment[0] < segment[1]:
            raise ValueError(f"Invalid segment for {video_id}: {segment}")
        result.setdefault(video_id, []).append(segment)
    return {key: sorted(value) for key, value in sorted(result.items())}


def assign_validation_holdout(video_ids: list[str], fraction: float, seed: int) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if len(video_ids) < 2:
        return set()
    shuffled = list(video_ids)
    random.Random(seed).shuffle(shuffled)
    count = min(len(video_ids) - 1, max(1, int(math.ceil(len(video_ids) * fraction))))
    return set(shuffled[:count])


def selected_rows(rows: list[dict[str, str]], video_ids: list[str] | None) -> list[dict[str, str]]:
    if video_ids is None:
        return rows
    selected = set(video_ids)
    result = [row for row in rows if row["video_id"] in selected]
    missing = selected - {row["video_id"] for row in result}
    if missing:
        raise ValueError(f"Videos absent from manifest: {sorted(missing)}")
    return result


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}. Run prepare first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def prepare(args: argparse.Namespace) -> None:
    paths = work_paths(args.work_dir)
    paths["annotations"].mkdir(parents=True, exist_ok=True)
    annotation_paths = {
        "validation": resolve(args.validation_annotations)
        if args.validation_annotations
        else paths["annotations"] / "TH14_Temporal_annotations_validation.zip",
        "test": resolve(args.test_annotations)
        if args.test_annotations
        else paths["annotations"] / "TH14_Temporal_annotations_test.zip",
    }
    for subset, path in annotation_paths.items():
        if not path.exists():
            atomic_download(ANNOTATION_URLS[subset], path)

    by_subset = {
        subset: parse_class_annotations(path, args.class_name, subset)
        for subset, path in annotation_paths.items()
    }
    validation_holdout = assign_validation_holdout(
        list(by_subset["validation"]), args.validation_fraction, args.seed
    )
    rows = []
    for subset in ("validation", "test"):
        for video_id, segments in by_subset[subset].items():
            split = "test" if subset == "test" else ("val" if video_id in validation_holdout else "train")
            rows.append(
                {
                    "video_id": video_id,
                    "class_name": args.class_name,
                    "official_subset": subset,
                    "split": split,
                    "source_url": VIDEO_URLS[subset].format(video_id=video_id),
                    "source_path": str(paths["videos"] / subset / f"{video_id}.mp4"),
                    "source_sha256": "",
                    "source_bytes": "",
                    "duration_s": "",
                    "fps": "",
                    "width": "",
                    "height": "",
                    "segments_json": json.dumps(segments, separators=(",", ":")),
                }
            )
    write_manifest(paths["manifest"], rows)
    split_counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    print(f"[OK] {args.class_name}: {len(rows)} positive videos; splits={split_counts}")
    print(f"[OK] manifest={paths['manifest']}")


def download(args: argparse.Namespace) -> None:
    paths = work_paths(args.work_dir)
    rows = read_manifest(paths["manifest"])
    for row in selected_rows(rows, args.video_ids):
        destination = Path(row["source_path"])
        print(f"[DOWNLOAD] {row['video_id']} -> {destination}")
        atomic_download(row["source_url"], destination, args.retries)
        row["source_sha256"] = sha256_file(destination)
        row["source_bytes"] = str(destination.stat().st_size)
    write_manifest(paths["manifest"], rows)


def probe_video(path: Path, ffprobe: str) -> dict[str, float | int]:
    if shutil.which(ffprobe) is None:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not open {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frames <= 0:
                raise ValueError(f"OpenCV did not report valid FPS/frame count for {path}")
            return {
                "duration_s": frames / fps,
                "fps": fps,
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
        finally:
            capture.release()
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"ffprobe did not report a duration for {path}")
    return {
        "duration_s": float(duration),
        "fps": float(numerator) / float(denominator),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def probe(args: argparse.Namespace) -> None:
    paths = work_paths(args.work_dir)
    rows = read_manifest(paths["manifest"])
    for row in selected_rows(rows, args.video_ids):
        video_path = Path(row["source_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"Missing {video_path}; run download first")
        row.update({key: str(value) for key, value in probe_video(video_path, args.ffprobe).items()})
        if not row["source_sha256"]:
            row["source_sha256"] = sha256_file(video_path)
            row["source_bytes"] = str(video_path.stat().st_size)
        print(f"[PROBE] {row['video_id']} duration={row['duration_s']} fps={row['fps']}")
    write_manifest(paths["manifest"], rows)


def v2e_command(args: argparse.Namespace, row: dict[str, str], output_dir: Path) -> list[str]:
    command = [
        args.v2e_python,
        str(resolve(args.v2e_entry)),
        "-i",
        row["source_path"],
        "--output_folder",
        str(output_dir),
        "--overwrite",
        "--skip_video_output",
        "--no_preview",
        "--dvs_h5",
        "events.h5",
        "--output_width",
        str(args.output_width),
        "--output_height",
        str(args.output_height),
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
        "--dvs_emulator_seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
    ]
    if args.disable_slomo:
        command.append("--disable_slomo")
    else:
        command.extend(
            (
                f"--timestamp_resolution={args.timestamp_resolution}",
                "--auto_timestamp_resolution=False",
            )
        )
    if args.stop_time is not None:
        command.extend(("--stop_time", str(args.stop_time)))
    return command


def convert(args: argparse.Namespace) -> None:
    paths = work_paths(args.work_dir)
    rows = read_manifest(paths["manifest"])
    for row in selected_rows(rows, args.video_ids):
        source = Path(row["source_path"])
        if not source.exists():
            raise FileNotFoundError(f"Missing {source}; run download first")
        output_dir = paths["v2e"] / row["video_id"]
        output_h5 = output_dir / "events.h5"
        status_path = output_dir / "conversion.json"
        if output_h5.exists() and status_path.exists() and not args.force:
            print(f"[SKIP] {row['video_id']} already converted")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = v2e_command(args, row, output_dir)
        print(f"[V2E] {row['video_id']}")
        environment = os.environ.copy()
        pythonpath = str(resolve(args.v2e_pythonpath))
        if environment.get("PYTHONPATH"):
            pythonpath += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = pythonpath
        subprocess.run(command, check=True, env=environment)
        if not output_h5.exists():
            raise FileNotFoundError(f"v2e did not create {output_h5}")
        status = {
            "video_id": row["video_id"],
            "source_sha256": row["source_sha256"] or sha256_file(source),
            "v2e_entry": str(resolve(args.v2e_entry)),
            "v2e_entry_sha256": sha256_file(resolve(args.v2e_entry)),
            "v2e_pythonpath": pythonpath,
            "command": command,
            "stop_time_s": args.stop_time,
            "disable_slomo": bool(args.disable_slomo),
            "output_sha256": sha256_file(output_h5),
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def converted_duration(row: dict[str, str], status: dict[str, object], events: h5py.Dataset) -> float:
    source_duration = float(row["duration_s"]) if row["duration_s"] else 0.0
    stop_time = status.get("stop_time_s")
    requested = min(source_duration, float(stop_time)) if stop_time is not None and source_duration else source_duration
    last_event = float(events[-1, 0]) / 1e6 if len(events) else 0.0
    return max(requested, last_event)


def copy_v2e_events(source: h5py.Dataset, target_group: h5py.Group, chunk_size: int) -> h5py.Dataset:
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError(f"Expected v2e events with shape [N,4], got {source.shape}")
    target = target_group.create_dataset(
        "events",
        shape=source.shape,
        dtype=np.uint32,
        chunks=(min(max(1, len(source)), chunk_size), 4),
        compression="lzf",
    )
    for start in range(0, len(source), chunk_size):
        end = min(start + chunk_size, len(source))
        values = np.asarray(source[start:end], dtype=np.uint32)
        target[start:end, 0] = values[:, 1]
        target[start:end, 1] = values[:, 2]
        target[start:end, 2] = values[:, 0]
        target[start:end, 3] = values[:, 3]
    return target


def write_adapted_annotations(
    config_dir: Path,
    rows: list[dict[str, str]],
    durations: dict[str, float],
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    database = {}
    info_rows = []
    for index, row in enumerate(rows):
        duration = durations[row["video_id"]]
        segments = [
            segment
            for segment in json.loads(row["segments_json"])
            if 0 <= float(segment[0]) < float(segment[1]) <= duration + 1e-6
        ]
        database[row["video_id"]] = {
            "annotations": {
                "1": [
                    {
                        "label": "ed",
                        "source_label": row["class_name"],
                        "segment": [float(start), float(end)],
                    }
                    for start, end in segments
                ]
            }
        }
        info_rows.append(
            {
                "timestamp": row["video_id"],
                "recording_id": index,
                "roi_group_id": 1,
                "split": row["split"],
                "precipitation": "",
                "night": "",
                "ed_cnt": len(segments),
                "event_count": "",
                "duration_s": f"{duration:.9f}",
                "official_subset": row["official_subset"],
                "source_class": row["class_name"],
            }
        )
    (config_dir / "annotations.json").write_text(
        json.dumps({"version": "THUMOS14-v2e-pilot-v1", "database": database}, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = list(info_rows[0]) if info_rows else []
    with (config_dir / "recording_info.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(info_rows)


def assemble(args: argparse.Namespace) -> None:
    paths = work_paths(args.work_dir)
    rows = selected_rows(read_manifest(paths["manifest"]), args.video_ids)
    output = resolve(args.output) if args.output else paths["data"]
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; use --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    durations: dict[str, float] = {}
    try:
        with h5py.File(temporary, "w") as target:
            target.attrs["format"] = "event-penguins-[x,y,t_us,p]-v1"
            target.attrs["source"] = "THUMOS14 converted with v2e"
            for row in rows:
                source_path = paths["v2e"] / row["video_id"] / "events.h5"
                status_path = paths["v2e"] / row["video_id"] / "conversion.json"
                if not source_path.exists() or not status_path.exists():
                    raise FileNotFoundError(f"Missing v2e output for {row['video_id']}")
                status = json.loads(status_path.read_text(encoding="utf-8"))
                with h5py.File(source_path, "r") as source:
                    events = source["events"]
                    duration = converted_duration(row, status, events)
                    recording = target.create_group(row["video_id"])
                    recording.attrs["split"] = row["split"]
                    recording.attrs["official_subset"] = row["official_subset"]
                    recording.attrs["source_class"] = row["class_name"]
                    recording.attrs["duration_s"] = duration
                    recording.attrs["source_sha256"] = row["source_sha256"]
                    roi = recording.create_group("N01")
                    roi.attrs["height"] = 260
                    roi.attrs["width"] = 346
                    roi.attrs["duration_s"] = duration
                    copy_v2e_events(events, roi, args.chunk_size)
                    durations[row["video_id"]] = duration
                    print(f"[ASSEMBLE] {row['video_id']} events={len(events):,} duration={duration:.3f}s")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_adapted_annotations(paths["config"], rows, durations)
    print(f"[OK] canonical events={output}")
    print(f"[OK] annotations={paths['config'] / 'annotations.json'}")


def annotation_segments(ann_path: Path) -> dict[str, np.ndarray]:
    database = json.loads(ann_path.read_text(encoding="utf-8"))["database"]
    return {
        recording: np.asarray(
            [item["segment"] for item in value["annotations"].get("1", []) if item["label"] == "ed"],
            dtype=np.float64,
        ).reshape(-1, 2)
        for recording, value in database.items()
    }


def event_rate_diagnostics(events: np.ndarray, segments: np.ndarray, duration: float, bin_width: float) -> dict[str, float]:
    bins = np.arange(0.0, duration + bin_width, bin_width, dtype=np.float64)
    if len(bins) < 2:
        bins = np.asarray([0.0, max(duration, bin_width)])
    counts, edges = np.histogram(events[:, 2].astype(np.float64) / 1e6, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    inside = np.zeros(len(centers), dtype=bool)
    for start, end in segments:
        inside |= (centers >= start) & (centers < end)
    action_rate = float(counts[inside].mean() / bin_width) if inside.any() else float("nan")
    background_rate = float(counts[~inside].mean() / bin_width) if (~inside).any() else float("nan")
    ratio = action_rate / background_rate if background_rate > 0 else float("nan")
    return {
        "action_event_rate_hz": action_rate,
        "background_event_rate_hz": background_rate,
        "action_background_rate_ratio": ratio,
        "nonempty_bin_fraction": float((counts > 0).mean()),
    }


def validate_recording(
    recording: str,
    group: h5py.Group,
    segments: np.ndarray,
    bin_width: float,
) -> dict[str, object]:
    roi = group["N01"]
    events = roi["events"]
    duration = float(group.attrs["duration_s"])
    if events.ndim != 2 or events.shape[1] != 4 or len(events) == 0:
        raise ValueError(f"{recording}: invalid or empty events dataset {events.shape}")
    monotonic = True
    previous = -1
    x_min = y_min = 2**32 - 1
    x_max = y_max = 0
    polarities: set[int] = set()
    for start in range(0, len(events), 1_000_000):
        values = np.asarray(events[start : start + 1_000_000])
        timestamps = values[:, 2].astype(np.int64)
        monotonic &= bool(timestamps[0] >= previous and np.all(np.diff(timestamps) >= 0))
        previous = int(timestamps[-1])
        x_min = min(x_min, int(values[:, 0].min()))
        x_max = max(x_max, int(values[:, 0].max()))
        y_min = min(y_min, int(values[:, 1].min()))
        y_max = max(y_max, int(values[:, 1].max()))
        polarities.update(map(int, np.unique(values[:, 3])))
    if not monotonic:
        raise ValueError(f"{recording}: timestamps are not monotonic")
    if not (0 <= x_min <= x_max < int(roi.attrs["width"])):
        raise ValueError(f"{recording}: x coordinates out of bounds")
    if not (0 <= y_min <= y_max < int(roi.attrs["height"])):
        raise ValueError(f"{recording}: y coordinates out of bounds")
    if not polarities.issubset({0, 1}):
        raise ValueError(f"{recording}: invalid polarities {sorted(polarities)}")
    if len(segments) and (segments.min() < 0 or segments[:, 1].max() > duration + 1e-6):
        raise ValueError(f"{recording}: annotation outside converted duration")
    sampled = np.asarray(events)
    diagnostics = event_rate_diagnostics(sampled, segments, duration, bin_width)
    return {
        "recording": recording,
        "split": str(group.attrs["split"]),
        "duration_s": duration,
        "events": int(len(events)),
        "event_rate_hz": float(len(events) / max(duration, 1e-9)),
        "first_timestamp_s": float(events[0, 2]) / 1e6,
        "last_timestamp_s": float(events[-1, 2]) / 1e6,
        "x_range": [x_min, x_max],
        "y_range": [y_min, y_max],
        "polarities": sorted(polarities),
        "instances": int(len(segments)),
        **diagnostics,
    }


def plot_validation(report: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["recording"]).replace("video_", "") for row in report]
    ratios = [float(row["action_background_rate_ratio"]) for row in report]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.5))
    ax.bar(np.arange(len(labels)), ratios, color="#2A6F97")
    ax.axhline(1.0, color="#B23A48", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Taxa eventos accion / fondo")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=70, ha="right")
    ax.set_title("Diagnostico THUMOS14-E antes de executar os detectores")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def validate(args: argparse.Namespace) -> None:
    paths = work_paths(args.work_dir)
    data_path = resolve(args.data_path) if args.data_path else paths["data"]
    ann_path = resolve(args.ann_path) if args.ann_path else paths["config"] / "annotations.json"
    segments = annotation_segments(ann_path)
    report = []
    with h5py.File(data_path, "r") as handle:
        expected = set(segments)
        actual = set(handle.keys())
        if expected != actual:
            raise ValueError(f"HDF5/annotation recording mismatch: missing={expected-actual}, extra={actual-expected}")
        for recording in sorted(handle.keys()):
            row = validate_recording(recording, handle[recording], segments[recording], args.bin_width)
            report.append(row)
            print(
                f"[VALID] {recording} events={row['events']:,} instances={row['instances']} "
                f"action/background={row['action_background_rate_ratio']:.3f}"
            )
    payload = {
        "status": "ok",
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "annotation_path": str(ann_path),
        "annotation_sha256": sha256_file(ann_path),
        "bin_width_s": args.bin_width,
        "recordings": report,
    }
    paths["report"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot_validation(report, paths["root"] / "event_rate_diagnostic.png")
    print(f"[OK] validation report={paths['report']}")


def main() -> None:
    args = parse_args()
    args.work_dir = resolve(args.work_dir)
    command = globals()[args.command]
    command(args)


if __name__ == "__main__":
    main()
