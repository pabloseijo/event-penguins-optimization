"""Extract aligned recurrent TESPEC features for a proposal master.

Each ATSN temporal sample is represented as the official 10-bin, two-polarity
stacked histogram. Extraction follows proposal order semantically, sorts rows
only for efficient HDF5 access, and can resume after interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dev.train_temporalmaxer_dense import proposal_fingerprint, resolve
from src.tespec_encoder import TespecEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract TESPEC proposal features")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--histogram-bins", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-rows", type=int, default=256)
    parser.add_argument(
        "--corruption",
        choices=["none", "event_noise"],
        default="none",
    )
    parser.add_argument("--corruption-seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def expanded_tsn_samples(num_tsn_samples: int, augment_factor: int) -> int:
    return num_tsn_samples + 2 * int(np.ceil(num_tsn_samples / augment_factor))


def stacked_histogram(
    events: np.ndarray,
    height: int,
    width: int,
    bins: int,
    image_size: int,
) -> np.ndarray:
    """Construct the TESPEC/RVT two-polarity stacked histogram."""
    output = np.zeros((2 * bins, image_size, image_size), dtype=np.uint8)
    if len(events) == 0:
        return output
    x = np.clip(
        np.floor(events[:, 0].astype(np.float64) * image_size / max(width, 1)),
        0,
        image_size - 1,
    ).astype(np.int64)
    y = np.clip(
        np.floor(events[:, 1].astype(np.float64) * image_size / max(height, 1)),
        0,
        image_size - 1,
    ).astype(np.int64)
    time = events[:, 2].astype(np.float64)
    span = max(float(time[-1] - time[0]), 1.0)
    time_bin = np.minimum(
        np.floor((time - time[0]) * bins / span), bins - 1
    ).astype(np.int64)
    polarity = (events[:, 3] > 0).astype(np.int64)
    area = image_size * image_size
    flat_index = x + image_size * y + area * time_bin + bins * area * polarity
    counts = np.bincount(flat_index, minlength=2 * bins * area)
    return np.minimum(counts, 255).astype(np.uint8).reshape(output.shape)


def corrupt_stacked_sequence(
    sequence: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply source-only event thinning and sparse background activity."""
    corrupted = np.zeros_like(sequence)
    nonzero = np.nonzero(sequence)
    if len(nonzero[0]):
        polarity = nonzero[1] >= sequence.shape[1] // 2
        keep_by_polarity = rng.uniform(0.65, 1.0, size=2)
        probabilities = keep_by_polarity[polarity.astype(np.int64)]
        values = rng.binomial(sequence[nonzero], probabilities).astype(np.uint8)
        corrupted[nonzero] = values

    noise_fraction = float(rng.uniform(0.0, 0.15))
    noise_events = int(round(float(sequence.sum()) * noise_fraction))
    if noise_events > 0:
        flat = corrupted.reshape(-1)
        indices = rng.integers(0, len(flat), size=noise_events)
        unique, counts = np.unique(indices, return_counts=True)
        flat[unique] = np.minimum(
            flat[unique].astype(np.int64) + counts,
            255,
        ).astype(np.uint8)
    return corrupted


class TespecProposalDataset(Dataset):
    def __init__(self, proposals: pd.DataFrame, args: argparse.Namespace) -> None:
        self.proposals = proposals.reset_index(drop=True)
        self.data_path = str(resolve(args.data_path))
        self.num_segments = expanded_tsn_samples(args.num_tsn_samples, args.augment_factor)
        self.augment_fraction = 1.0 / args.augment_factor
        self.sample_duration_us = args.sample_duration * 1e6
        self.histogram_bins = args.histogram_bins
        self.image_size = args.image_size
        self.corruption = args.corruption
        self.corruption_seed = args.corruption_seed
        self._hf = None
        self._cached_key = None
        self._cached_events = None
        self._cached_timestamps = None
        self._cached_height = None
        self._cached_width = None

    def __len__(self) -> int:
        return len(self.proposals)

    def _get_roi(self, rec_name: str, roi_id: str):
        if self._hf is None:
            self._hf = h5py.File(self.data_path, "r")
        key = (rec_name, roi_id)
        if key != self._cached_key:
            group = self._hf[rec_name][roi_id]
            self._cached_events = np.asarray(group["events"])
            self._cached_timestamps = self._cached_events[:, 2]
            self._cached_height = int(group.attrs["height"])
            self._cached_width = int(group.attrs["width"])
            self._cached_key = key
        return (
            self._cached_events,
            self._cached_timestamps,
            self._cached_height,
            self._cached_width,
        )

    def __getitem__(self, index: int):
        row = self.proposals.iloc[index]
        events, timestamps, height, width = self._get_roi(
            str(row["rec_name"]), str(row["roi_id"])
        )
        start = float(row["t_start"])
        end = float(row["t_end"])
        duration = end - start
        sample_times = np.linspace(
            start - duration * self.augment_fraction,
            end + duration * self.augment_fraction,
            self.num_segments,
        )
        window_start = sample_times - 0.5 * self.sample_duration_us
        window_end = sample_times + 0.5 * self.sample_duration_us
        starts = np.searchsorted(timestamps, window_start)
        ends = np.searchsorted(timestamps, window_end)
        sequence = np.stack(
            [
                stacked_histogram(
                    events[left:right],
                    height,
                    width,
                    self.histogram_bins,
                    self.image_size,
                )
                for left, right in zip(starts, ends)
            ]
        )
        if self.corruption == "event_noise":
            rng = np.random.default_rng(
                int(self.corruption_seed) + int(row["_master_index"])
            )
            sequence = corrupt_stacked_sequence(sequence, rng)
        return torch.from_numpy(sequence), int(row["_master_index"])


def main() -> None:
    args = parse_args()
    if args.histogram_bins != 10:
        raise ValueError("The released TESPEC checkpoint expects exactly 10 bins per polarity")
    if args.image_size < 224:
        raise ValueError(
            "TESPEC Swin-T needs image-size >=224 to preserve its pretrained window-7 bias tables"
        )
    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    fingerprint = proposal_fingerprint(proposals)
    base_metadata = json.loads(
        (resolve(args.base_cache_dir) / "metadata.json").read_text(encoding="utf-8")
    )
    num_segments = expanded_tsn_samples(args.num_tsn_samples, args.augment_factor)
    for key, expected in (
        ("fingerprint", fingerprint),
        ("rows", len(proposals)),
        ("num_segments", num_segments),
    ):
        if base_metadata.get(key) != expected:
            raise ValueError(
                f"Base cache {key}={base_metadata.get(key)!r} does not match {expected!r}"
            )

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "event_features.npy"
    state_path = out_dir / "extraction_state.json"
    metadata_path = out_dir / "metadata.json"
    config = {
        "fingerprint": fingerprint,
        "rows": len(proposals),
        "num_segments": num_segments,
        "feature_dim": TespecEncoder.stage_dims[-1],
        "histogram_bins": args.histogram_bins,
        "image_size": args.image_size,
        "sample_duration": args.sample_duration,
        "checkpoint": str(resolve(args.checkpoint)),
        "corruption": args.corruption,
        "corruption_seed": args.corruption_seed,
    }
    completed = 0
    if args.restart:
        feature_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    elif state_path.exists() and feature_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in config.items()):
            raise ValueError("Existing TESPEC extraction state uses a different configuration")
        completed = int(state.get("completed", 0))
    elif metadata_path.exists() and feature_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in config.items()):
            print(f"[RESULTADO] Caché TESPEC xa completa: {feature_path}")
            return

    mode = "r+" if feature_path.exists() else "w+"
    features = np.lib.format.open_memmap(
        feature_path,
        mode=mode,
        dtype=np.float16,
        shape=(len(proposals), num_segments, TespecEncoder.stage_dims[-1]),
    )
    ordered = proposals.copy()
    ordered["_master_index"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(
        ["rec_name", "roi_id", "t_start", "t_end"], kind="stable"
    ).reset_index(drop=True)
    pending = ordered.iloc[completed:].reset_index(drop=True)
    dataset = TespecProposalDataset(pending, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = TespecEncoder(args.image_size)
    model.load_pretrained(resolve(args.checkpoint))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    offset = completed
    next_checkpoint = completed + max(args.checkpoint_rows, 1)
    progress = tqdm(loader, desc=f"tespec-features@{completed}", disable=args.quiet_progress)
    with torch.inference_mode():
        for sequence, master_indices in progress:
            sequence = sequence.to(device, dtype=torch.float16, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                batch_features = model(sequence)
            indices = master_indices.numpy().astype(np.int64)
            features[indices] = batch_features.float().cpu().numpy().astype(np.float16)
            offset += len(sequence)
            if offset >= next_checkpoint or offset == len(ordered):
                features.flush()
                state_path.write_text(
                    json.dumps({**config, "completed": offset}, indent=2),
                    encoding="utf-8",
                )
                next_checkpoint = offset + max(args.checkpoint_rows, 1)

    features.flush()
    metadata_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    state_path.unlink(missing_ok=True)
    print(f"[RESULTADO] Caché TESPEC completada: {features.shape} en {out_dir}")


if __name__ == "__main__":
    main()
