"""Evaluate ViTTA-style target-to-source feature-statistic alignment."""

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

from dev.eval_temporalmaxer_continuous_test import load_models
from dev.train_temporalmaxer_continuous import (
    ContinuousSequenceDataset,
    collate_sequences,
    evaluate,
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
        "--out-dir", default="tmp/temporalmaxer_continuous/cv_feature_alignment_v1"
    )
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def source_statistics(
    feature_path: Path, sequences: pd.DataFrame, excluded_recordings: set[str]
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
        local_sum = values.sum(axis=0)
        local_square = np.square(values).sum(axis=0)
        total = local_sum if total is None else total + local_sum
        total_square = local_square if total_square is None else total_square + local_square
        count += len(values)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def evaluation_args(checkpoint: dict, ann_path: str) -> argparse.Namespace:
    saved = checkpoint["args"]
    return argparse.Namespace(
        ann_path=ann_path,
        quiet_progress=True,
        score_threshold=float(saved.get("score_threshold", 0.005)),
        pre_nms_topk=int(saved.get("pre_nms_topk", 500)),
        quality_power=float(saved.get("quality_power", 0.5)),
        soft_nms_sigma=float(saved.get("soft_nms_sigma", 0.5)),
        soft_nms_score_threshold=float(saved.get("soft_nms_score_threshold", 0.001)),
        max_predictions_per_roi=int(saved.get("max_predictions_per_roi", 200)),
    )


def main() -> None:
    args = parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.blends):
        raise ValueError("All blends must be in [0,1]")
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    feature_path = feature_dir / "frame_features.npy"
    metadata = json.loads((feature_dir / "metadata.json").read_text())
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold in args.folds:
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        val_sequences = sequences[sequences["rec_name"].isin(recordings)].copy()
        mean, std = source_statistics(feature_path, sequences, set(recordings))
        checkpoint_path = resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = load_models(
            resolve(args.checkpoint_root),
            int(metadata["feature_dim"]),
            device,
            [checkpoint_path],
        )[0]
        eval_args = evaluation_args(checkpoint, args.ann_path)
        for blend in args.blends:
            dataset = ContinuousSequenceDataset(
                feature_path,
                val_sequences,
                annotations={},
                feature_alignment_mean=mean,
                feature_alignment_std=std,
                feature_alignment_blend=blend,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_sequences,
            )
            label = f"fold{fold:02d}_blend{blend:g}"
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
                    "blend": blend,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **metrics,
                }
            )
            print(f"[{label}] mAP={metrics['mAP']:.6f} AP07={metrics['AP@0.7']:.6f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = []
    for blend, group in metrics.groupby("blend"):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary.append(
            {
                "blend": blend,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    result = pd.DataFrame(summary).sort_values("mean_mAP", ascending=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
