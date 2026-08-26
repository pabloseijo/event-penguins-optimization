"""Extract proposal-level TISM features for external-CV ranking experiments.

TISM projects events onto Time-Height and Time-Width maps. Each view stores
global event count and polarity through equivalent count/ON/OFF channels, so
the representation is invariant to translation on its orthogonal spatial axis.
The frozen dual-view feature is broadcast over dense temporal samples only when
loaded, avoiding eleven identical copies on disk.
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
from src.tism_encoder import FrozenTismEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen TISM proposal features")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-rows", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def _view_map(
    events: np.ndarray,
    start: float,
    end: float,
    spatial_column: int,
    spatial_extent: int,
    image_size: int,
) -> np.ndarray:
    if len(events) == 0:
        return np.zeros((3, image_size, image_size), dtype=np.uint8)
    duration = max(end - start, 1.0)
    time_index = np.clip(
        np.floor((events[:, 2].astype(np.float64) - start) * image_size / duration),
        0,
        image_size - 1,
    ).astype(np.int64)
    spatial_index = np.clip(
        np.floor(
            events[:, spatial_column].astype(np.float64)
            * image_size
            / max(spatial_extent, 1)
        ),
        0,
        image_size - 1,
    ).astype(np.int64)
    flat = time_index * image_size + spatial_index
    area = image_size * image_size
    on = np.bincount(flat[events[:, 3] > 0], minlength=area).reshape(
        image_size, image_size
    )
    off = np.bincount(flat[events[:, 3] <= 0], minlength=area).reshape(
        image_size, image_size
    )
    channels = np.stack((on + off, on, off)).astype(np.float32)
    channels = np.log1p(np.minimum(channels, 255.0)) / np.log(256.0)
    return np.rint(255.0 * channels).astype(np.uint8)


def tism_maps(
    events: np.ndarray,
    start: float,
    end: float,
    height: int,
    width: int,
    image_size: int = 224,
) -> np.ndarray:
    """Build T-H and T-W count/ON/OFF maps from one proposal."""
    time_height = _view_map(events, start, end, 1, height, image_size)
    time_width = _view_map(events, start, end, 0, width, image_size)
    return np.stack((time_height, time_width))


class TismProposalDataset(Dataset):
    def __init__(self, proposals: pd.DataFrame, args: argparse.Namespace) -> None:
        self.proposals = proposals.reset_index(drop=True)
        self.data_path = str(resolve(args.data_path))
        self.image_size = int(args.image_size)
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
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="right"))
        maps = tism_maps(events[left:right], start, end, height, width, self.image_size)
        return torch.from_numpy(maps), int(row["_master_index"])


def main() -> None:
    args = parse_args()
    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    fingerprint = proposal_fingerprint(proposals)
    base_metadata = json.loads(
        (resolve(args.base_cache_dir) / "metadata.json").read_text(encoding="utf-8")
    )
    for key, expected in (("fingerprint", fingerprint), ("rows", len(proposals))):
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
        "num_segments": int(base_metadata["num_segments"]),
        "stored_segments": 1,
        "broadcast_temporal": True,
        "feature_dim": FrozenTismEncoder.feature_dim,
        "image_size": int(args.image_size),
        "representation": "tism_count_on_off_imagenet_resnet18",
    }
    completed = 0
    if args.restart:
        feature_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    elif state_path.exists() and feature_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in config.items()):
            raise ValueError("Existing TISM extraction uses a different configuration")
        completed = int(state.get("completed", 0))
    elif metadata_path.exists() and feature_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in config.items()):
            print(f"[RESULTADO] Caché TISM xa completa: {feature_path}")
            return

    features = np.lib.format.open_memmap(
        feature_path,
        mode="r+" if feature_path.exists() else "w+",
        dtype=np.float16,
        shape=(len(proposals), 1, FrozenTismEncoder.feature_dim),
    )
    ordered = proposals.copy()
    ordered["_master_index"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(
        ["rec_name", "roi_id", "t_start", "t_end"], kind="stable"
    ).reset_index(drop=True)
    dataset = TismProposalDataset(ordered.iloc[completed:].reset_index(drop=True), args)
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
    model = FrozenTismEncoder().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    offset = completed
    next_checkpoint = completed + max(args.checkpoint_rows, 1)
    progress = tqdm(loader, desc=f"tism-features@{completed}", disable=args.quiet_progress)
    with torch.inference_mode():
        for views, master_indices in progress:
            views = views.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                batch_features = model(views)
            features[master_indices.numpy().astype(np.int64), 0] = (
                batch_features.float().cpu().numpy().astype(np.float16)
            )
            offset += len(views)
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
    print(f"[RESULTADO] Caché TISM completada: {features.shape} en {out_dir}")


if __name__ == "__main__":
    main()
