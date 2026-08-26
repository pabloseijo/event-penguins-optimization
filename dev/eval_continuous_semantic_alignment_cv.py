"""Align pseudo-background target statistics with labeled source background."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_feature_alignment_cv import evaluation_args
from dev.eval_temporalmaxer_continuous_test import load_models
from dev.train_temporalmaxer_continuous import (
    ContinuousSequenceDataset,
    collate_sequences,
    evaluate,
    load_annotations,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1"
    )
    parser.add_argument(
        "--checkpoint-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1"
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/cv_semantic_alignment_v1"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--background-quantile", type=float, default=0.8)
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def source_background_statistics(
    feature_path: Path,
    sequences: pd.DataFrame,
    excluded_recordings: set[str],
    annotations: dict[tuple[str, int], np.ndarray],
    grid_stride_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(feature_path, mmap_mode="r")
    total = None
    total_square = None
    count = 0
    selected = sequences[~sequences["rec_name"].isin(excluded_recordings)]
    for row in selected.itertuples(index=False):
        values = np.asarray(
            features[int(row.offset) : int(row.offset + row.length)], dtype=np.float64
        )
        centers = (np.arange(len(values), dtype=np.float64) + 0.5) * grid_stride_s
        background = np.ones(len(values), dtype=bool)
        for start, end in annotations.get(
            (str(row.rec_name), int(row.roi_id)), np.empty((0, 2))
        ):
            background &= (centers < start) | (centers > end)
        values = values[background]
        local_sum = values.sum(axis=0)
        local_square = np.square(values).sum(axis=0)
        total = local_sum if total is None else total + local_sum
        total_square = local_square if total_square is None else total_square + local_square
        count += len(values)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


@torch.no_grad()
def pseudo_background_statistics(
    model,
    loader: DataLoader,
    feature_path: Path,
    sequences: pd.DataFrame,
    quantile: float,
    device: torch.device,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    score_by_key = {}
    model.eval()
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            scores = model(features, mask)["classification_logits"][0].sigmoid()
        for recording, roi_id, values, length in zip(
            batch["rec_name"], batch["roi_id"], scores, batch["mask"].sum(dim=1)
        ):
            score_by_key[(str(recording), int(roi_id))] = (
                values[: int(length)].float().cpu().numpy()
            )

    matrix = np.load(feature_path, mmap_mode="r")
    result = {}
    for row in sequences.itertuples(index=False):
        key = (str(row.rec_name), int(row.roi_id))
        values = np.asarray(
            matrix[int(row.offset) : int(row.offset + row.length)], dtype=np.float32
        )
        scores = score_by_key[key]
        selected = scores <= np.quantile(scores, quantile)
        background = values[selected]
        result[key] = (
            background.mean(axis=0, dtype=np.float64).astype(np.float32),
            np.maximum(
                background.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4
            ),
        )
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.background_quantile < 1.0:
        raise ValueError("background-quantile must be in (0,1)")
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    feature_path = feature_dir / "frame_features.npy"
    metadata = json.loads((feature_dir / "metadata.json").read_text())
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    annotations = load_annotations(resolve(args.ann_path))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold in args.folds:
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        val_sequences = sequences[sequences["rec_name"].isin(recordings)].copy()
        checkpoint_path = resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = load_models(
            resolve(args.checkpoint_root),
            int(metadata["feature_dim"]),
            device,
            [checkpoint_path],
        )[0]
        raw_dataset = ContinuousSequenceDataset(feature_path, val_sequences, annotations={})
        raw_loader = DataLoader(
            raw_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_sequences,
        )
        target_stats = pseudo_background_statistics(
            model,
            raw_loader,
            feature_path,
            val_sequences,
            args.background_quantile,
            device,
        )
        source_mean, source_std = source_background_statistics(
            feature_path,
            sequences,
            set(recordings),
            annotations,
            float(metadata["grid_stride_s"]),
        )
        eval_args = evaluation_args(checkpoint, args.ann_path)
        for blend in args.blends:
            dataset = ContinuousSequenceDataset(
                feature_path,
                val_sequences,
                annotations={},
                feature_alignment_mean=source_mean,
                feature_alignment_std=source_std,
                feature_alignment_blend=blend,
                feature_alignment_target_stats=target_stats,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_sequences,
            )
            label = f"fold{fold:02d}_q{args.background_quantile:g}_blend{blend:g}"
            metrics = evaluate(
                model,
                loader,
                val_sequences,
                metadata,
                eval_args,
                device,
                out_dir / "predictions" / f"{label}.json",
            )
            rows.append(
                {
                    "fold": fold,
                    "background_quantile": args.background_quantile,
                    "blend": blend,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **metrics,
                }
            )
            print(f"[{label}] mAP={metrics['mAP']:.6f} AP07={metrics['AP@0.7']:.6f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "metrics.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
