"""Train the source-approved continuous detector on every source recording."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.train_temporalmaxer_continuous import (
    ContinuousSequenceDataset,
    hard_negative_sample_weights,
    collate_sequences,
    load_annotations,
    manifest_validation_recordings,
)
from src.temporalmaxer_continuous import TemporalMaxerContinuous


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--fold-manifest",
        default=None,
        help="Restrict final fitting to the validation-pool recordings declared by this manifest.",
    )
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pyramid-levels", type=int, default=6)
    parser.add_argument("--head-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument(
        "--hard-negative-recordings",
        default="",
        help="Comma-separated rec_name list to oversample during training (score-inversion domains).",
    )
    parser.add_argument(
        "--hard-negative-oversample",
        type=float,
        default=1.0,
        help="Sampling weight multiplier applied to hard-negative recordings (1.0 = disabled).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    if args.fold_manifest:
        allowed = manifest_validation_recordings(
            pd.read_csv(resolve(args.fold_manifest))
        )
        missing = allowed - set(sequences["rec_name"].astype(str))
        if missing:
            raise ValueError(f"Feature cache lacks validation recordings: {sorted(missing)}")
        sequences = sequences[sequences["rec_name"].astype(str).isin(allowed)].copy()
    annotations = load_annotations(
        resolve(args.ann_path), min_duration_s=args.min_action_duration
    )
    dataset = ContinuousSequenceDataset(
        feature_dir / "frame_features.npy", sequences, annotations
    )
    generator = torch.Generator().manual_seed(args.seed)
    hard_negative_recordings = {
        name.strip() for name in args.hard_negative_recordings.split(",") if name.strip()
    }
    sampler = None
    shuffle = True
    if hard_negative_recordings and args.hard_negative_oversample != 1.0:
        present = hard_negative_recordings & set(sequences["rec_name"].astype(str))
        if present:
            weights = hard_negative_sample_weights(
                sequences, hard_negative_recordings, args.hard_negative_oversample
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False
            print(
                f"[INFO] hard-negative oversampling active: {sorted(present)} "
                f"x{args.hard_negative_oversample}",
                flush=True,
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_sequences,
    )
    model = TemporalMaxerContinuous(
        input_dim=int(metadata["feature_dim"]),
        hidden_dim=args.hidden_dim,
        pyramid_levels=args.pyramid_levels,
        head_layers=args.head_layers,
        dropout=args.dropout,
        use_quality=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start_epoch = 1
    last_path = out_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1
    print(
        f"[INFO] source_roi={len(sequences)} records={sequences.rec_name.nunique()} "
        f"seed={args.seed} device={device}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("loss", "classification_loss", "regression_loss", "quality_loss")}
        batches = 0
        progress = tqdm(loader, desc=f"train-{epoch:02d}", disable=args.quiet_progress)
        for batch in progress:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            segments = [value.to(device, non_blocking=True) for value in batch["segments"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(features, mask)
                losses = model.losses(
                    output,
                    segments,
                    grid_stride_seconds=float(metadata["grid_stride_s"]),
                    regression_weight=args.regression_weight,
                    quality_weight=args.quality_weight,
                )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for key in totals:
                totals[key] += float(losses[key].detach())
        scheduler.step()
        row = {
            "epoch": epoch,
            **{key: value / max(batches, 1) for key, value in totals.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "history": history,
            "args": vars(args),
            "metadata": metadata,
        }
        atomic_save(state, last_path)
        print(f"[EPOCH {epoch:03d}] loss={row['loss']:.6f}", flush=True)
    final_state = {
        "model": model.state_dict(),
        "epoch": args.epochs,
        "args": vars(args),
        "metadata": metadata,
    }
    atomic_save(final_state, out_dir / "final.pt")
    print(f"[DONE] {out_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
