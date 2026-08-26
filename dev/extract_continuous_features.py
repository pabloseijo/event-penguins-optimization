"""Extract frozen ATSN features on a fixed grid over complete ROI timelines.

The output is a memory-mapped ``[sum(T), 512]`` matrix plus one manifest row
per ROI. Extraction can be split safely across GPUs because every shard writes
disjoint matrix ranges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.augmented_tsn import AugmentedTsn
from src.classification import ProposalDataset


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adaptive_sample_durations(
    log_event_counts: np.ndarray,
    target_count: float,
    min_duration: float,
    max_duration: float,
) -> np.ndarray:
    """Choose a bounded integration duration from the local event density."""
    values = np.asarray(log_event_counts, dtype=np.float64)
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if not 0 < min_duration <= max_duration:
        raise ValueError("Invalid adaptive duration interval")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("log_event_counts must be finite and non-negative")
    event_counts = np.expm1(values)
    return np.clip(
        target_count / np.maximum(event_counts, 1.0),
        min_duration,
        max_duration,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Create the sequence index and feature matrix")
    index.add_argument("--data-path", default="data/preprocessed.h5")
    index.add_argument("--out-dir", default="tmp/temporalmaxer_continuous/source_features")
    index.add_argument("--splits", nargs="+", default=["train", "val"])
    index.add_argument("--recordings", nargs="+", default=None)
    index.add_argument("--grid-stride", type=float, default=0.5)
    index.add_argument("--window-duration", type=float, default=1.0)
    index.add_argument(
        "--sequence-duration",
        type=float,
        default=None,
        help=(
            "Fixed duration in seconds. By default each ROI uses its duration_s "
            "attribute, falling back to the last event timestamp."
        ),
    )
    index.add_argument("--feature-dim", type=int, default=512)
    index.add_argument("--adaptive-target-count", type=float, default=None)
    index.add_argument("--adaptive-min-duration", type=float, default=0.5)
    index.add_argument("--adaptive-max-duration", type=float, default=2.0)
    index.add_argument("--force", action="store_true")

    extract = subparsers.add_parser("extract", help="Fill one disjoint extraction shard")
    extract.add_argument("--data-path", default="data/preprocessed.h5")
    extract.add_argument("--model-path", default="models/model.pk")
    extract.add_argument("--out-dir", default="tmp/temporalmaxer_continuous/source_features")
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--batch-size", type=int, default=256)
    extract.add_argument("--num-workers", type=int, default=8)
    extract.add_argument("--decay", type=float, default=5e-6)
    extract.add_argument("--adaptive-event-stats-dir", default=None)
    extract.add_argument("--adaptive-target-count", type=float, default=None)
    extract.add_argument("--adaptive-min-duration", type=float, default=0.5)
    extract.add_argument("--adaptive-max-duration", type=float, default=2.0)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--data-parallel", action="store_true")
    extract.add_argument("--overwrite-shard", action="store_true")

    verify = subparsers.add_parser("verify", help="Validate the completed feature cache")
    verify.add_argument("--out-dir", default="tmp/temporalmaxer_continuous/source_features")
    return parser.parse_args()


def paths(out_dir: Path) -> dict[str, Path]:
    return {
        "sequences": out_dir / "sequences.csv",
        "features": out_dir / "frame_features.npy",
        "metadata": out_dir / "metadata.json",
        "timestamps": out_dir / "timestamp_cache",
    }


def build_index(args: argparse.Namespace) -> None:
    data_path = resolve(args.data_path)
    out_dir = resolve(args.out_dir)
    output = paths(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [output["sequences"], output["features"], output["metadata"]]
    if any(path.exists() for path in existing) and not args.force:
        raise FileExistsError(f"Cache already exists in {out_dir}; use --force to rebuild it")

    selected_recordings = set(args.recordings) if args.recordings else None
    selected_splits = set(args.splits)
    if args.adaptive_target_count is not None:
        if args.adaptive_target_count <= 0:
            raise ValueError("adaptive-target-count must be positive")
        if not 0 < args.adaptive_min_duration <= args.adaptive_max_duration:
            raise ValueError("Invalid adaptive duration interval")
    if args.sequence_duration is not None and args.sequence_duration <= 0:
        raise ValueError("sequence-duration must be positive")

    rows = []
    offset = 0
    with h5py.File(data_path, "r") as handle:
        for recording in sorted(handle.keys()):
            split = str(handle[recording].attrs.get("split", ""))
            if split not in selected_splits:
                continue
            if selected_recordings is not None and recording not in selected_recordings:
                continue
            for roi_key in sorted(handle[recording].keys()):
                roi_group = handle[recording][roi_key]
                if args.sequence_duration is not None:
                    duration_s = float(args.sequence_duration)
                elif "duration_s" in roi_group.attrs:
                    duration_s = float(roi_group.attrs["duration_s"])
                elif "duration_s" in handle[recording].attrs:
                    duration_s = float(handle[recording].attrs["duration_s"])
                else:
                    events = roi_group["events"]
                    if len(events) == 0:
                        raise ValueError(
                            f"{recording}/{roi_key} is empty and has no duration_s attribute"
                        )
                    duration_s = float(events[-1, 2]) / 1e6
                if not np.isfinite(duration_s) or duration_s <= 0:
                    raise ValueError(f"Invalid duration for {recording}/{roi_key}: {duration_s}")
                length = int(math.ceil(duration_s / args.grid_stride))
                roi_id = int(str(roi_key).lstrip("N"))
                rows.append(
                    {
                        "sequence_index": len(rows),
                        "rec_name": recording,
                        "roi_key": str(roi_key),
                        "roi_id": roi_id,
                        "split": split,
                        "offset": offset,
                        "length": length,
                        "duration_s": duration_s,
                    }
                )
                offset += length
    if not rows:
        raise ValueError("No ROI sequences matched the requested split/recording filters")

    sequence_frame = pd.DataFrame(rows)
    sequence_frame.to_csv(output["sequences"], index=False)
    matrix = np.lib.format.open_memmap(
        output["features"], mode="w+", dtype=np.float16, shape=(offset, args.feature_dim)
    )
    matrix[:] = np.nan
    matrix.flush()
    metadata = {
        "format_version": 1,
        "data_path": str(data_path),
        "feature_dim": int(args.feature_dim),
        "grid_stride_s": float(args.grid_stride),
        "window_duration_s": float(args.window_duration),
        "sequence_duration_s": (
            float(args.sequence_duration) if args.sequence_duration is not None else None
        ),
        "variable_sequence_duration": args.sequence_duration is None,
        "num_sequences": len(rows),
        "num_points": offset,
        "splits": sorted(selected_splits),
        "recordings": sorted(sequence_frame["rec_name"].unique().tolist()),
    }
    if args.adaptive_target_count is not None:
        metadata.update(
            {
                "window_strategy": "global-adaptive-event-count",
                "adaptive_target_count": float(args.adaptive_target_count),
                "adaptive_min_duration_s": float(args.adaptive_min_duration),
                "adaptive_max_duration_s": float(args.adaptive_max_duration),
            }
        )
    output["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[INDEX] sequences={len(rows)} points={offset} shape={matrix.shape} "
        f"path={out_dir}"
    )


class FrameEncoder(nn.Module):
    def __init__(self, model: AugmentedTsn) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_frames(images).squeeze(1)


def extract_shard(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    out_dir = resolve(args.out_dir)
    output = paths(out_dir)
    metadata = json.loads(output["metadata"].read_text(encoding="utf-8"))
    data_path = resolve(args.data_path)
    model_path = resolve(args.model_path)
    expected_data = Path(str(metadata["data_path"])).resolve()
    if data_path.resolve() != expected_data:
        raise ValueError(f"Extraction data path {data_path} != indexed data path {expected_data}")
    sequences = pd.read_csv(output["sequences"])
    sequences = sequences[sequences["sequence_index"] % args.num_shards == args.shard_index]
    status_path = out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if status_path.exists() and not args.overwrite_shard:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("complete"):
            print(f"[SKIP] shard {args.shard_index}/{args.num_shards} already complete")
            return

    proposal_parts = []
    global_indices = []
    stride_s = float(metadata["grid_stride_s"])
    for row in sequences.itertuples(index=False):
        centers_s = (np.arange(int(row.length), dtype=np.float64) + 0.5) * stride_s
        proposal_parts.append(
            pd.DataFrame(
                {
                    "rec_name": row.rec_name,
                    "roi_id": row.roi_key,
                    "t_start": centers_s * 1e6,
                    "t_end": centers_s * 1e6,
                }
            )
        )
        global_indices.append(np.arange(int(row.offset), int(row.offset + row.length)))
    proposals = pd.concat(proposal_parts, ignore_index=True)
    write_indices = np.concatenate(global_indices)
    adaptive_stats = None
    if args.adaptive_event_stats_dir is not None:
        if args.adaptive_target_count is None or args.adaptive_target_count <= 0:
            raise ValueError(
                "--adaptive-event-stats-dir requires a positive --adaptive-target-count"
            )
        if not 0 < args.adaptive_min_duration <= args.adaptive_max_duration:
            raise ValueError("Invalid adaptive duration interval")
        event_dir = resolve(args.adaptive_event_stats_dir)
        event_metadata = json.loads(
            (event_dir / "metadata.json").read_text(encoding="utf-8")
        )
        feature_names = list(event_metadata["feature_names"])
        if "log_event_count" not in feature_names:
            raise ValueError("Adaptive event statistics have no log_event_count feature")
        if int(event_metadata["num_points"]) != int(metadata["num_points"]):
            raise ValueError("Adaptive event statistics and feature index are not aligned")
        event_matrix = np.load(event_dir / "event_stats.npy", mmap_mode="r")
        log_counts = np.asarray(
            event_matrix[write_indices, feature_names.index("log_event_count")],
            dtype=np.float64,
        )
        durations_s = adaptive_sample_durations(
            log_event_counts=log_counts,
            target_count=args.adaptive_target_count,
            min_duration=args.adaptive_min_duration,
            max_duration=args.adaptive_max_duration,
        )
        proposals["sample_duration"] = durations_s * 1e6
        adaptive_stats = {
            "target_count": float(args.adaptive_target_count),
            "min_duration_s": float(durations_s.min()),
            "median_duration_s": float(np.median(durations_s)),
            "max_duration_s": float(durations_s.max()),
        }
    elif args.adaptive_target_count is not None:
        raise ValueError(
            "--adaptive-target-count requires --adaptive-event-stats-dir during extraction"
        )
    dataset = ProposalDataset(
        proposals=proposals,
        augment_fraction=0.0,
        data_path=str(data_path),
        num_tsn_samples=1,
        sample_duration=float(metadata["window_duration_s"]) * 1e6,
        decay=float(args.decay),
        cache_full_events=False,
        timestamp_cache_dir=str(output["timestamps"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    atsn = AugmentedTsn(2, num_tsn_samples=7, augment_factor=3)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    atsn.load_state_dict(state)
    encoder: nn.Module = FrameEncoder(atsn).to(device).eval()
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)

    feature_matrix = np.lib.format.open_memmap(output["features"], mode="r+")
    cursor = 0
    with torch.inference_mode():
        progress = tqdm(loader, desc=f"extract-{args.shard_index:02d}")
        for batch in progress:
            images = batch[0].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                encoded = encoder(images)
            encoded_numpy = encoded.float().cpu().numpy().astype(np.float16)
            end = cursor + len(encoded_numpy)
            feature_matrix[write_indices[cursor:end]] = encoded_numpy
            cursor = end
    feature_matrix.flush()
    if cursor != len(write_indices):
        raise RuntimeError(f"Extracted {cursor} rows but expected {len(write_indices)}")
    status_path.write_text(
        json.dumps(
            {
                "complete": True,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "num_sequences": len(sequences),
                "num_points": cursor,
                "data_path": str(data_path),
                "data_sha256": sha256_file(data_path),
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "index_metadata_sha256": sha256_file(output["metadata"]),
                "adaptive_window": adaptive_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] shard={args.shard_index}/{args.num_shards} points={cursor}")


def verify_cache(args: argparse.Namespace) -> None:
    out_dir = resolve(args.out_dir)
    output = paths(out_dir)
    metadata = json.loads(output["metadata"].read_text(encoding="utf-8"))
    sequences = pd.read_csv(output["sequences"])
    features = np.load(output["features"], mmap_mode="r")
    expected_shape = (int(metadata["num_points"]), int(metadata["feature_dim"]))
    if features.shape != expected_shape:
        raise ValueError(f"Feature shape {features.shape} != expected {expected_shape}")
    if int(sequences.iloc[-1]["offset"] + sequences.iloc[-1]["length"]) != len(features):
        raise ValueError("Sequence offsets do not cover the complete feature matrix")
    bad_rows = 0
    chunk = 8192
    norms = []
    for start in tqdm(range(0, len(features), chunk), desc="verify"):
        values = np.asarray(features[start : start + chunk], dtype=np.float32)
        bad_rows += int((~np.isfinite(values).all(axis=1)).sum())
        norms.append(np.linalg.norm(values, axis=1))
    norms_array = np.concatenate(norms)
    if bad_rows:
        raise ValueError(f"Cache contains {bad_rows} incomplete/non-finite rows")
    print(
        f"[OK] shape={features.shape} norm_mean={norms_array.mean():.4f} "
        f"norm_min={norms_array.min():.4f} norm_max={norms_array.max():.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.command == "index":
        build_index(args)
    elif args.command == "extract":
        extract_shard(args)
    else:
        verify_cache(args)


if __name__ == "__main__":
    main()
