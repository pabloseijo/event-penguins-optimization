"""Boundary-Sensitive Pretraining on fixed windows from continuous source sequences."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.train_temporalmaxer_continuous import load_annotations  # noqa: E402
from src.bsp import (  # noqa: E402
    BoundaryTypeHead,
    DIFFERENT_CLASS,
    SAME_CLASS,
    NUM_BOUNDARY_TYPES,
    boundary_type_loss,
    synthesize_bsp_sequences,
)
from src.temporalmaxer_continuous import TemporalMaxerContinuous  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True)
class Window:
    offset: int
    rec_name: str


def clamp_window_start(start: int, sequence_length: int, window_length: int) -> int:
    if sequence_length < window_length:
        raise ValueError("BSP window is longer than the source sequence")
    return min(max(int(start), 0), sequence_length - window_length)


def overlaps_any(
    start_s: float,
    end_s: float,
    segments: np.ndarray,
) -> bool:
    if len(segments) == 0:
        return False
    return bool(np.any((segments[:, 0] < end_s) & (segments[:, 1] > start_s)))


def build_window_pools(
    sequences: pd.DataFrame,
    annotations: dict[tuple[str, int], np.ndarray],
    window_length: int,
    grid_stride_s: float,
    seed: int,
) -> tuple[list[Window], list[Window]]:
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for row in sequences.itertuples(index=False):
        length = int(row.length)
        if length < window_length:
            continue
        segments = annotations.get(
            (str(row.rec_name), int(row.roi_id)),
            np.empty((0, 2), dtype=np.float32),
        )
        for segment in segments:
            center = int(round(float(segment.mean()) / grid_stride_s))
            for jitter in (-window_length // 6, 0, window_length // 6):
                local_start = clamp_window_start(
                    center - window_length // 2 + jitter,
                    length,
                    window_length,
                )
                positives.append(
                    Window(int(row.offset) + local_start, str(row.rec_name))
                )

        target_negatives = max(3, 3 * len(segments))
        attempts = 0
        found = 0
        while found < target_negatives and attempts < 50 * target_negatives:
            local_start = int(rng.integers(0, length - window_length + 1))
            start_s = local_start * grid_stride_s
            end_s = (local_start + window_length) * grid_stride_s
            if not overlaps_any(start_s, end_s, segments):
                negatives.append(
                    Window(int(row.offset) + local_start, str(row.rec_name))
                )
                found += 1
            attempts += 1
    if len(positives) < 2 or not negatives:
        raise ValueError("BSP requires positive and background windows")
    return positives, negatives


def sample_cross_recording(
    windows: list[Window],
    excluded_recording: str,
    rng: np.random.Generator,
) -> int:
    candidates = [
        index for index, window in enumerate(windows)
        if window.rec_name != excluded_recording
    ]
    if not candidates:
        raise ValueError("BSP pairing requires at least two recordings")
    return int(candidates[int(rng.integers(len(candidates)))])


class ContinuousBSPDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        positives: list[Window],
        negatives: list[Window],
        window_length: int,
        count: int,
        seed: int,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.positives = positives
        self.negatives = negatives
        self.window_length = int(window_length)
        rng = np.random.default_rng(seed)
        self.boundary_types = np.resize(
            np.arange(NUM_BOUNDARY_TYPES, dtype=np.int64), count
        )
        rng.shuffle(self.boundary_types)
        self.primary = rng.integers(len(positives), size=count)
        self.secondary = self.primary.copy()
        for index, boundary_type in enumerate(self.boundary_types):
            primary_recording = positives[int(self.primary[index])].rec_name
            if boundary_type == DIFFERENT_CLASS:
                self.secondary[index] = sample_cross_recording(
                    negatives,
                    primary_recording,
                    rng,
                )
            elif boundary_type == SAME_CLASS:
                self.secondary[index] = sample_cross_recording(
                    positives,
                    primary_recording,
                    rng,
                )
        low = max(1, window_length // 3)
        high = min(window_length - 1, (2 * window_length) // 3)
        self.split_positions = rng.integers(low, high + 1, size=count)
        self.speed_rates = rng.choice(
            np.asarray([0.60, 0.75, 1.25, 1.50], dtype=np.float32),
            size=count,
        )

    def __len__(self) -> int:
        return len(self.boundary_types)

    def load_window(self, window: Window) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(
                self.features[
                    window.offset : window.offset + self.window_length
                ],
                dtype=np.float32,
            ).copy()
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        boundary_type = int(self.boundary_types[index])
        primary = self.load_window(self.positives[int(self.primary[index])])
        if boundary_type == DIFFERENT_CLASS:
            secondary_window = self.negatives[int(self.secondary[index])]
        else:
            secondary_window = self.positives[int(self.secondary[index])]
        return (
            primary,
            self.load_window(secondary_window),
            torch.tensor(boundary_type, dtype=torch.long),
            torch.tensor(int(self.split_positions[index]), dtype=torch.long),
            torch.tensor(float(self.speed_rates[index]), dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1"
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pyramid-levels", type=int, default=6)
    parser.add_argument("--head-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.window_length < 3 or args.samples < NUM_BOUNDARY_TYPES:
        raise ValueError("BSP requires a window >=3 and at least four samples")
    set_seed(args.seed + args.fold)
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads(
        (feature_dir / "metadata.json").read_text(encoding="utf-8")
    )
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    manifest = pd.read_csv(resolve(args.fold_manifest)).set_index("fold")
    validation_recordings = str(
        manifest.loc[args.fold, "val_record_names"]
    ).split()
    train_sequences = sequences[
        ~sequences["rec_name"].isin(validation_recordings)
    ].copy()
    positives, negatives = build_window_pools(
        train_sequences,
        load_annotations(resolve(args.ann_path)),
        args.window_length,
        float(metadata["grid_stride_s"]),
        args.seed + args.fold,
    )
    dataset = ContinuousBSPDataset(
        feature_dir / "frame_features.npy",
        positives,
        negatives,
        args.window_length,
        args.samples,
        args.seed + 1000 + args.fold,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = TemporalMaxerContinuous(
        input_dim=int(metadata["feature_dim"]),
        hidden_dim=args.hidden_dim,
        pyramid_levels=args.pyramid_levels,
        head_layers=args.head_layers,
        dropout=args.dropout,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.input_projection.parameters():
        parameter.requires_grad_(True)
    boundary_head = BoundaryTypeHead(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.input_projection.parameters(), "lr": args.lr},
            {"params": boundary_head.parameters(), "lr": args.classifier_lr},
        ],
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.input_projection.train()
        boundary_head.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for primary, secondary, boundary_types, splits, speeds in loader:
            primary = primary.to(device, non_blocking=True)
            secondary = secondary.to(device, non_blocking=True)
            boundary_types = boundary_types.to(device, non_blocking=True)
            splits = splits.to(device, non_blocking=True)
            speeds = speeds.to(device, non_blocking=True)
            synthetic = synthesize_bsp_sequences(
                primary,
                secondary,
                boundary_types,
                splits,
                speeds,
            )
            shared = model.input_projection(synthetic.transpose(1, 2))
            logits = boundary_head(shared)
            loss = boundary_type_loss(logits, boundary_types)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [*model.input_projection.parameters(), *boundary_head.parameters()],
                max_norm=5.0,
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(boundary_types)
            correct += int((logits.argmax(dim=1) == boundary_types).sum())
            seen += len(boundary_types)
        row = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "accuracy": correct / seen,
        }
        history.append(row)
        print(
            f"[BSP {epoch:02d}] loss={row['loss']:.6f} "
            f"accuracy={row['accuracy']:.4f}",
            flush=True,
        )

    out_path = resolve(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "metadata": metadata,
            "history": history,
            "positive_windows": len(positives),
            "negative_windows": len(negatives),
        },
        out_path,
    )


if __name__ == "__main__":
    main()
