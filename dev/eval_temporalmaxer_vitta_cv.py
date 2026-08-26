"""Recording-disjoint CV for one-step ViTTA on the continuous detector."""

from __future__ import annotations

import argparse
import copy
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

from dev.eval_continuous_multi_rep_fusion_cv import evaluate as evaluate_prediction
from dev.eval_temporalmaxer_continuous_test import load_models
from dev.train_temporalmaxer_continuous import (
    ContinuousSequenceDataset,
    collate_sequences,
)
from src.temporalmaxer_continuous import TemporalMaxerContinuous
from src.utils import temporal_soft_nms


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
    parser.add_argument("--auxiliary-feature-dir", default=None)
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/cv_vitta_gradient_v1"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--source-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument(
        "--localization-consistency-weight",
        type=float,
        default=0.0,
        help="Equivariant offset consistency for event-aware localization adaptation.",
    )
    parser.add_argument(
        "--regression-statistics",
        action="store_true",
        help="Align localization-pyramid moments instead of classification moments.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_loader(
    feature_path: Path,
    sequences: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    auxiliary_feature_path: Path | None = None,
    auxiliary_mean: np.ndarray | None = None,
    auxiliary_std: np.ndarray | None = None,
) -> DataLoader:
    dataset = ContinuousSequenceDataset(
        feature_path,
        sequences,
        annotations={},
        auxiliary_feature_path=auxiliary_feature_path,
        auxiliary_mean=auxiliary_mean,
        auxiliary_std=auxiliary_std,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )


def masked_channel_moments(
    features: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-channel sum, squared sum, and valid temporal count."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    valid = mask.to(features.dtype)
    return (
        (features * valid).sum(dim=(0, 2)),
        (features.square() * valid).sum(dim=(0, 2)),
        valid.sum(),
    )


def moments_to_mean_variance(
    total: torch.Tensor, total_square: torch.Tensor, count: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = total / count.clamp_min(1.0)
    variance = total_square / count.clamp_min(1.0) - mean.square()
    return mean, variance.clamp_min(0.0)


def shift_temporal_view(features: torch.Tensor) -> torch.Tensor:
    shifted = torch.empty_like(features)
    shifted[:, 0] = features[:, 0]
    shifted[:, 1:] = features[:, :-1]
    return shifted


@torch.no_grad()
def source_statistics(
    model: TemporalMaxerContinuous,
    loader: DataLoader,
    device: torch.device,
    regression_statistics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    total = None
    total_square = None
    count = torch.zeros((), device=device)
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        output = model(features, mask)
        feature_key = (
            "regression_pyramid_features"
            if regression_statistics
            else "pyramid_features"
        )
        local_total, local_square, local_count = masked_channel_moments(
            output[feature_key][0], mask
        )
        total = local_total if total is None else total + local_total
        total_square = local_square if total_square is None else total_square + local_square
        count += local_count
    if total is None or total_square is None:
        raise ValueError("Cannot compute source statistics from an empty loader")
    return moments_to_mean_variance(total, total_square, count)


def prediction_consistency(first: dict, second: dict, mask: torch.Tensor) -> torch.Tensor:
    valid = mask[:, :-1] & mask[:, 1:]
    losses = []
    for key in ("classification_logits", "quality_logits"):
        first_probability = first[key][0][:, :-1].sigmoid()
        second_probability = second[key][0][:, 1:].sigmoid()
        losses.append(
            (first_probability - second_probability).abs()[valid].mean()
        )
    return torch.stack(losses).mean()


def localization_consistency(
    first: dict,
    second: dict,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Match offsets at temporally corresponding points after a one-bin shift."""
    valid = mask[:, :-1] & mask[:, 1:]
    first_offsets = first["offsets"][0][:, :-1]
    second_offsets = second["offsets"][0][:, 1:]
    scale = 1.0 + first_offsets.detach().abs()
    return ((first_offsets - second_offsets).abs() / scale)[valid].mean()


def adapt_recording(
    source_model: TemporalMaxerContinuous,
    loader: DataLoader,
    source_mean: torch.Tensor,
    source_variance: torch.Tensor,
    learning_rate: float,
    consistency_weight: float,
    localization_consistency_weight: float,
    regression_statistics: bool,
    device: torch.device,
) -> tuple[TemporalMaxerContinuous, dict[str, float]]:
    model = copy.deepcopy(source_model).to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0
    )
    batches = list(loader)
    if len(batches) != 1:
        raise ValueError("Each recording must fit in one adaptation batch")
    batch = batches[0]
    features = batch["features"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    shifted = shift_temporal_view(features)
    first = model(features, mask)
    second = model(shifted, mask)
    feature_key = (
        "regression_pyramid_features"
        if regression_statistics
        else "pyramid_features"
    )
    total_a, square_a, count_a = masked_channel_moments(first[feature_key][0], mask)
    total_b, square_b, count_b = masked_channel_moments(second[feature_key][0], mask)
    target_mean, target_variance = moments_to_mean_variance(
        total_a + total_b, square_a + square_b, count_a + count_b
    )
    alignment = (target_mean - source_mean).abs().mean()
    alignment += (target_variance - source_variance).abs().mean()
    consistency = prediction_consistency(first, second, mask)
    localization = localization_consistency(first, second, mask)
    loss = (
        alignment
        + consistency_weight * consistency
        + localization_consistency_weight * localization
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.eval()
    return model, {
        "alignment_loss": float(alignment.detach()),
        "consistency_loss": float(consistency.detach()),
        "localization_consistency_loss": float(localization.detach()),
        "adaptation_loss": float(loss.detach()),
    }


@torch.no_grad()
def predict_recording(
    model: TemporalMaxerContinuous,
    loader: DataLoader,
    grid_stride_s: float,
    device: torch.device,
) -> dict[str, dict[str, list[dict]]]:
    results: dict[str, dict[str, list[dict]]] = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        output = model(features, mask)
        candidates = model.decode(
            output,
            grid_stride_seconds=grid_stride_s,
            durations_seconds=batch["duration_s"],
            score_threshold=0.005,
            pre_nms_topk=200,
            quality_power=1.0,
        )
        for recording, roi_id, values in zip(
            batch["rec_name"], batch["roi_id"], candidates
        ):
            detections = temporal_soft_nms(
                values.float().cpu().numpy(), sigma=0.25, score_threshold=0.001
            )[:200]
            results.setdefault(recording, {})[str(int(roi_id))] = [
                {
                    "label": "ed",
                    "segment": [float(start), float(end)],
                    "score": float(score),
                }
                for start, end, score in detections
                if end - start >= 2.0
            ]
    return results


def main() -> None:
    args = parse_args()
    if (
        args.lr <= 0
        or args.consistency_weight < 0
        or args.localization_consistency_weight < 0
    ):
        raise ValueError("Learning rate must be positive and consistency weights non-negative")
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    feature_path = feature_dir / "frame_features.npy"
    metadata = json.loads((feature_dir / "metadata.json").read_text())
    auxiliary_path = None
    auxiliary_mean = None
    auxiliary_std = None
    auxiliary_dim = 0
    if args.auxiliary_feature_dir:
        auxiliary_dir = resolve(args.auxiliary_feature_dir)
        auxiliary_metadata = json.loads(
            (auxiliary_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if int(auxiliary_metadata["num_points"]) != int(metadata["num_points"]):
            raise ValueError("Base and auxiliary feature caches are not aligned")
        auxiliary_path = auxiliary_dir / "event_stats.npy"
        auxiliary_mean = np.asarray(auxiliary_metadata["mean"], dtype=np.float32)
        auxiliary_std = np.asarray(auxiliary_metadata["std"], dtype=np.float32)
        auxiliary_dim = int(auxiliary_metadata["feature_dim"])
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    adaptation_rows = []

    for fold in args.folds:
        val_recordings = str(manifest.loc[fold, "val_record_names"]).split()
        train_sequences = sequences[~sequences["rec_name"].isin(val_recordings)].copy()
        val_sequences = sequences[sequences["rec_name"].isin(val_recordings)].copy()
        checkpoint_path = (
            resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt"
        )
        source_model = load_models(
            resolve(args.checkpoint_root),
            int(metadata["feature_dim"]) + auxiliary_dim,
            device,
            [checkpoint_path],
        )[0]
        source_loader = make_loader(
            feature_path,
            train_sequences,
            args.source_batch_size,
            args.num_workers,
            device,
            auxiliary_path,
            auxiliary_mean,
            auxiliary_std,
        )
        source_mean, source_variance = source_statistics(
            source_model,
            source_loader,
            device,
            args.regression_statistics,
        )
        fold_results: dict[str, dict[str, list[dict]]] = {}
        for recording in val_recordings:
            recording_sequences = val_sequences[
                val_sequences["rec_name"] == recording
            ].copy()
            loader = make_loader(
                feature_path,
                recording_sequences,
                args.batch_size,
                args.num_workers,
                device,
                auxiliary_path,
                auxiliary_mean,
                auxiliary_std,
            )
            adapted, adaptation_metrics = adapt_recording(
                source_model,
                loader,
                source_mean,
                source_variance,
                args.lr,
                args.consistency_weight,
                args.localization_consistency_weight,
                args.regression_statistics,
                device,
            )
            fold_results.update(
                predict_recording(
                    adapted, loader, float(metadata["grid_stride_s"]), device
                )
            )
            adaptation_rows.append(
                {"fold": fold, "rec_name": recording, **adaptation_metrics}
            )
            print(
                f"[ADAPT] fold={fold} rec={recording} "
                f"loss={adaptation_metrics['adaptation_loss']:.6f}",
                flush=True,
            )
        prediction = {"version": "temporalmaxer-continuous-vitta-v1", "results": fold_results}
        metrics = evaluate_prediction(
            prediction,
            val_recordings,
            resolve(args.ann_path),
            out_dir / "predictions" / f"fold_{fold:02d}.json",
        )
        fold_rows.append(
            {
                "fold": fold,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **metrics,
            }
        )
        print(
            f"[FOLD] {fold} mAP={metrics['mAP']:.6f} AP07={metrics['AP@0.7']:.6f}",
            flush=True,
        )

    metrics = pd.DataFrame(fold_rows).sort_values("fold")
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(adaptation_rows).to_csv(out_dir / "adaptation.csv", index=False)
    weights = metrics["val_ed_instances"].to_numpy(np.float64)
    summary = {
        "mean_mAP": float(metrics["mAP"].mean()),
        "weighted_mAP": float(np.average(metrics["mAP"], weights=weights)),
        "worst_mAP": float(metrics["mAP"].min()),
        "mean_AP@0.7": float(metrics["AP@0.7"].mean()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
