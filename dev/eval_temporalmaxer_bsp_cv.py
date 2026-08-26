"""Source-only BSP adaptation of a frozen-ATSN TemporalMaxer detector.

The experiment implements the four-way boundary-sensitive pretext task from
Xu et al. (ICCV 2021) on cached ordered ATSN features. Only TemporalMaxer's
shared neck and the disposable BSP classifier are updated. Model selection is
performed on recording-disjoint source folds; this script never reads test.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dev.eval_temporalmaxer_groupdro_cv import add_ranking_fusions
from dev.train_temporalmaxer_dense import (
    DenseFeatureDataset,
    atomic_torch_save,
    build_targets,
    cache_paths,
    choose_training_indices,
    dense_loss,
    evaluate_variant,
    load_cache,
    make_model,
    map_to_master,
    score_model,
    softmax_ed,
    subset_targets,
)
from src.bsp import (
    DIFFERENT_CLASS,
    DIFFERENT_SPEED,
    NUM_BOUNDARY_TYPES,
    SAME_CLASS,
    SAME_SPEED,
    BoundaryTypeHead,
    boundary_type_loss,
    synthesize_bsp_sequences,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINABLE_PREFIXES = (
    "input_norm",
    "base_input_norm",
    "auxiliary_input_norm",
    "input_projection",
    "pyramid_fusion",
)


@dataclass(frozen=True)
class BSPSpecification:
    primary: np.ndarray
    secondary: np.ndarray
    boundary_type: np.ndarray
    split: np.ndarray
    speed_rate: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-master",
        default="tmp/temporalmaxer_dense/screened_folds_r5/master_proposals.csv",
    )
    parser.add_argument(
        "--train-fold-dir", default="tmp/temporalmaxer_dense/screened_folds_r5"
    )
    parser.add_argument(
        "--train-cache", default="tmp/temporalmaxer_dense/screened_cache"
    )
    parser.add_argument(
        "--eval-master",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument(
        "--eval-fold-dir", default="tmp/temporalmaxer_dense/hybrid_source"
    )
    parser.add_argument(
        "--eval-cache", default="tmp/temporalmaxer_dense/hybrid_source_cache"
    )
    parser.add_argument(
        "--checkpoint-root",
        default="tmp/temporalmaxer_dense/screened_cv_blend075_erm",
    )
    parser.add_argument(
        "--groupdro-root", default="tmp/quality_head/family_groupdro_cv"
    )
    parser.add_argument(
        "--out-root", default="tmp/temporalmaxer_dense/bsp_cv"
    )
    parser.add_argument(
        "--manifest", default="tmp/cv/recording_folds_r5/manifest.csv"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=140000)
    parser.add_argument("--bsp-samples", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--classifier-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bsp-weight", type=float, default=0.25)
    parser.add_argument("--l2sp-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260718)

    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--high-pos-tiou", type=float, default=0.7)
    parser.add_argument("--boundary-min-tiou", type=float, default=0.3)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--trident-weight", type=float, default=0.0)
    parser.add_argument("--qfl-beta", type=float, default=2.0)

    parser.add_argument("--boundary-blend", type=float, default=0.75)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pools_by_recording(
    proposals: pd.DataFrame, positions: np.ndarray
) -> dict[str, np.ndarray]:
    recordings = proposals["rec_name"].astype(str).to_numpy()
    return {
        recording: positions[recordings[positions] == recording]
        for recording in sorted(set(recordings[positions]))
    }


def _sample_other_recording(
    pools: dict[str, np.ndarray], excluded: str, rng: np.random.Generator
) -> int:
    recordings = [name for name, values in pools.items() if name != excluded and len(values)]
    if not recordings:
        recordings = [name for name, values in pools.items() if len(values)]
    if not recordings:
        raise ValueError("Cannot sample an empty BSP pool")
    recording = recordings[int(rng.integers(len(recordings)))]
    values = pools[recording]
    return int(values[int(rng.integers(len(values)))])


def build_bsp_specification(
    proposals: pd.DataFrame,
    quality: np.ndarray,
    count: int,
    num_segments: int,
    seed: int,
) -> BSPSpecification:
    """Build balanced source-only BSP pairs, crossing recordings for splices."""
    if len(proposals) != len(quality):
        raise ValueError("BSP proposals and quality targets are misaligned")
    if count < NUM_BOUNDARY_TYPES:
        raise ValueError("bsp_samples must be at least four")
    positive = np.flatnonzero(quality >= 0.5)
    negative = np.flatnonzero(quality < 0.1)
    if len(positive) < 2 or len(negative) < 1:
        raise ValueError("BSP requires at least two positives and one negative")
    positive_by_recording = _pools_by_recording(proposals, positive)
    negative_by_recording = _pools_by_recording(proposals, negative)
    recordings = proposals["rec_name"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)

    boundary_types = np.resize(np.arange(NUM_BOUNDARY_TYPES, dtype=np.int64), count)
    rng.shuffle(boundary_types)
    primary = rng.choice(positive, size=count, replace=True).astype(np.int64)
    secondary = primary.copy()
    for index, boundary_type in enumerate(boundary_types):
        recording = recordings[primary[index]]
        if boundary_type == DIFFERENT_CLASS:
            secondary[index] = _sample_other_recording(
                negative_by_recording, recording, rng
            )
        elif boundary_type == SAME_CLASS:
            secondary[index] = _sample_other_recording(
                positive_by_recording, recording, rng
            )

    low_split = max(1, num_segments // 3)
    high_split = min(num_segments - 1, (2 * num_segments) // 3 + 1)
    splits = rng.integers(low_split, high_split + 1, size=count, dtype=np.int64)
    speeds = rng.choice(
        np.asarray([0.60, 0.75, 1.25, 1.50], dtype=np.float32),
        size=count,
    ).astype(np.float32)
    return BSPSpecification(primary, secondary, boundary_types, splits, speeds)


class BSPPairDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        sampled_master_indices: np.ndarray,
        specification: BSPSpecification,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.master_indices = np.asarray(sampled_master_indices, dtype=np.int64)
        self.specification = specification

    def __len__(self) -> int:
        return len(self.specification.boundary_type)

    def __getitem__(self, index: int):
        primary = torch.from_numpy(
            np.asarray(
                self.features[
                    self.master_indices[self.specification.primary[index]]
                ],
                dtype=np.float32,
            ).copy()
        )
        secondary = torch.from_numpy(
            np.asarray(
                self.features[
                    self.master_indices[self.specification.secondary[index]]
                ],
                dtype=np.float32,
            ).copy()
        )
        return (
            primary,
            secondary,
            int(self.specification.boundary_type[index]),
            int(self.specification.split[index]),
            float(self.specification.speed_rate[index]),
        )


def configure_shared_neck(model: nn.Module) -> list[nn.Parameter]:
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(TRAINABLE_PREFIXES)
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise ValueError("No shared-neck parameters were selected for BSP")
    return trainable


def l2sp_loss(model: nn.Module, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = [
        (parameter - reference[name]).square().mean()
        for name, parameter in model.named_parameters()
        if name in reference
    ]
    if not terms:
        raise ValueError("L2-SP reference has no trainable parameters")
    return torch.stack(terms).mean()


def frozen_task_modules_eval(model: nn.Module) -> None:
    for name in (
        "class_tower",
        "boundary_tower",
        "action_head",
        "point_quality_head",
        "start_head",
        "end_head",
        "distance_head",
        "quality_head",
        "start_delta",
        "end_delta",
    ):
        getattr(model, name).eval()


@torch.no_grad()
def evaluate_source_fold(
    model: nn.Module,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    epoch: int,
) -> dict[str, float | str | int]:
    master = pd.read_csv(resolve(args.eval_master)).reset_index(drop=True)
    proposals = pd.read_csv(
        resolve(args.eval_fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
    ).reset_index(drop=True)
    _, logits, _ = load_cache(resolve(args.eval_cache))
    indices = map_to_master(master, proposals)
    scored = score_model(
        model,
        proposals,
        indices,
        cache_paths(resolve(args.eval_cache))["features"],
        logits,
        args,
        device,
    )
    groupdro = pd.read_csv(
        resolve(args.groupdro_root)
        / f"fold_{fold:02d}"
        / "cache"
        / "val_scores_qhead_qfl_only.csv"
    )
    positions = map_to_master(groupdro, proposals)
    selected = groupdro.iloc[positions].reset_index(drop=True)
    scored["quality_score"] = selected["quality_score"].to_numpy(dtype=np.float64)
    scored = add_ranking_fusions(scored)
    scored["blend050_t_start"] = 0.5 * (
        scored["t_start"] + scored["delta_t_start"]
    )
    scored["blend050_t_end"] = 0.5 * (
        scored["t_end"] + scored["delta_t_end"]
    )
    scored.to_csv(out_dir / f"scored_epoch_{epoch:02d}.csv", index=False)
    row = evaluate_variant(
        scored,
        "qhead_brem_score",
        "blend050",
        f"bsp_fold_{fold:02d}_epoch_{epoch:02d}",
        args,
        out_dir / "predictions",
    )
    pd.DataFrame([row]).to_csv(out_dir / f"metrics_epoch_{epoch:02d}.csv", index=False)
    return row


def load_base_model(
    fold: int,
    metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(
        resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    saved_args = checkpoint.get("args", {})
    for name in (
        "hidden_dim",
        "pyramid_levels",
        "dropout",
        "trident_bins",
        "event_features_only",
    ):
        if name in saved_args:
            setattr(args, name, saved_args[name])
    args.event_feature_cache_dir = None
    args.event_features_only = False
    model = make_model(metadata, args).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def train_fold(fold: int, args: argparse.Namespace, device: torch.device) -> dict:
    fold_seed = args.seed + fold
    set_seed(fold_seed)
    out_dir = resolve(args.out_root) / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    if best_path.exists() and not args.restart:
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))

    train_master = pd.read_csv(resolve(args.train_master)).reset_index(drop=True)
    train_proposals = pd.read_csv(
        resolve(args.train_fold_dir) / f"fold_{fold:02d}" / "train_proposals.csv"
    ).reset_index(drop=True)
    features, logits, metadata = load_cache(resolve(args.train_cache))
    train_master_indices = map_to_master(train_master, train_proposals)
    targets_all = build_targets(train_proposals, features.shape[1], args)
    cnn_scores = softmax_ed(np.asarray(logits[train_master_indices]))
    sampled = choose_training_indices(
        train_proposals,
        targets_all,
        cnn_scores,
        args.max_train_samples,
        fold_seed,
    )
    proposals = train_proposals.iloc[sampled].reset_index(drop=True)
    targets = subset_targets(targets_all, sampled)
    master_indices = train_master_indices[sampled]
    real_dataset = DenseFeatureDataset(
        cache_paths(resolve(args.train_cache))["features"],
        master_indices,
        targets,
    )
    bsp_specification = build_bsp_specification(
        proposals,
        targets.quality,
        min(args.bsp_samples, max(len(proposals), NUM_BOUNDARY_TYPES)),
        features.shape[1],
        fold_seed + 1000,
    )
    bsp_dataset = BSPPairDataset(
        cache_paths(resolve(args.train_cache))["features"],
        master_indices,
        bsp_specification,
    )
    real_generator = torch.Generator().manual_seed(fold_seed)
    bsp_generator = torch.Generator().manual_seed(fold_seed + 1)
    real_loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=real_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    bsp_loader = DataLoader(
        bsp_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=bsp_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model, base_checkpoint = load_base_model(fold, metadata, args, device)
    trainable = configure_shared_neck(model)
    boundary_head = BoundaryTypeHead(args.hidden_dim, args.dropout).to(device)
    reference = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable, "lr": args.lr},
            {"params": boundary_head.parameters(), "lr": args.classifier_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict] = []
    base_metrics_path = out_dir / "metrics_epoch_00.csv"
    if base_metrics_path.exists():
        base_row = pd.read_csv(base_metrics_path).iloc[0].to_dict()
    else:
        base_row = evaluate_source_fold(model, fold, args, device, out_dir, 0)
    best_selection = float(base_row["mAP"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    start_epoch = 1

    last_path = out_dir / "last.pt"
    if last_path.exists() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        boundary_head.load_state_dict(state["boundary_head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        real_generator.set_state(state["real_generator_state"].cpu())
        bsp_generator.set_state(state["bsp_generator_state"].cpu())
        history = state["history"]
        best_selection = float(state["best_selection"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_state"]
        stale_epochs = int(state["stale_epochs"])
        start_epoch = int(state["epoch"]) + 1
        print(f"[FOLD {fold:02d}] reanudando na época {start_epoch}")

    print(
        f"[FOLD {fold:02d}] train={len(real_dataset)} bsp={len(bsp_dataset)} "
        f"baseline={float(base_row['mAP']):.6f} device={device}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        frozen_task_modules_eval(model)
        boundary_head.train()
        bsp_iterator = iter(bsp_loader)
        totals = {"loss": 0.0, "dense": 0.0, "bsp": 0.0, "l2sp": 0.0, "correct": 0}
        seen_bsp = 0
        batches = 0
        for real_batch in real_loader:
            try:
                bsp_batch = next(bsp_iterator)
            except StopIteration:
                bsp_iterator = iter(bsp_loader)
                bsp_batch = next(bsp_iterator)
            (
                frame_features,
                quality_target,
                action_target,
                point_distance_target,
                start_target,
                end_target,
                delta_target,
                boundary_weight,
                _,
            ) = real_batch
            primary, secondary, boundary_types, split_positions, speed_rates = bsp_batch
            frame_features = frame_features.to(device, non_blocking=True)
            quality_target = quality_target.to(device, non_blocking=True)
            action_target = action_target.to(device, non_blocking=True)
            point_distance_target = point_distance_target.to(device, non_blocking=True)
            start_target = start_target.to(device, non_blocking=True)
            end_target = end_target.to(device, non_blocking=True)
            delta_target = delta_target.to(device, non_blocking=True)
            boundary_weight = boundary_weight.to(device, non_blocking=True)
            primary = primary.to(device, non_blocking=True)
            secondary = secondary.to(device, non_blocking=True)
            boundary_types = boundary_types.to(device, non_blocking=True)
            split_positions = split_positions.to(device, non_blocking=True)
            speed_rates = speed_rates.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(frame_features)
                sample_loss, _ = dense_loss(
                    output,
                    quality_target,
                    action_target,
                    point_distance_target,
                    start_target,
                    end_target,
                    delta_target,
                    boundary_weight,
                    args,
                )
                synthetic = synthesize_bsp_sequences(
                    primary,
                    secondary,
                    boundary_types,
                    split_positions,
                    speed_rates,
                )
                bsp_logits = boundary_head(model.encode_shared(synthetic))
                bsp_loss = boundary_type_loss(bsp_logits, boundary_types)
                regularization = l2sp_loss(model, reference)
                dense_mean = sample_loss.mean()
                loss = (
                    dense_mean
                    + args.bsp_weight * bsp_loss
                    + args.l2sp_weight * regularization
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [*trainable, *boundary_head.parameters()], max_norm=5.0
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach().cpu())
            totals["dense"] += float(dense_mean.detach().cpu())
            totals["bsp"] += float(bsp_loss.detach().cpu())
            totals["l2sp"] += float(regularization.detach().cpu())
            totals["correct"] += int(
                (bsp_logits.argmax(dim=1) == boundary_types).sum().detach().cpu()
            )
            seen_bsp += len(boundary_types)
            batches += 1
        scheduler.step()

        row = evaluate_source_fold(model, fold, args, device, out_dir, epoch)
        selection = float(row["mAP"])
        history_row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(batches, 1),
            "dense_loss": totals["dense"] / max(batches, 1),
            "bsp_loss": totals["bsp"] / max(batches, 1),
            "l2sp_loss": totals["l2sp"] / max(batches, 1),
            "bsp_accuracy": totals["correct"] / max(seen_bsp, 1),
            "selection_mAP": selection,
            "AP@0.5": float(row["AP@0.5"]),
            "AP@0.7": float(row["AP@0.7"]),
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        if selection > best_selection + 1e-6:
            best_selection = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        atomic_torch_save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "boundary_head": boundary_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "real_generator_state": real_generator.get_state(),
                "bsp_generator_state": bsp_generator.get_state(),
                "history": history,
                "best_selection": best_selection,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "stale_epochs": stale_epochs,
            },
            last_path,
        )
        print(
            f"[FOLD {fold:02d} EPOCH {epoch:02d}] "
            f"mAP={selection:.6f} AP07={float(row['AP@0.7']):.6f} "
            f"bsp_acc={history_row['bsp_accuracy']:.4f}"
        )
        if stale_epochs >= args.patience:
            break

    checkpoint = {
        "state_dict": best_state,
        "args": vars(args),
        "feature_dim": int(metadata["feature_dim"]),
        "best_epoch": best_epoch,
        "best_selection_mAP": best_selection,
        "base_checkpoint": str(
            resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt"
        ),
        "base_best_epoch": base_checkpoint.get("best_epoch"),
    }
    atomic_torch_save(checkpoint, best_path)
    summary = {
        "fold": fold,
        "baseline_mAP": float(base_row["mAP"]),
        "best_mAP": best_selection,
        "delta_mAP": best_selection - float(base_row["mAP"]),
        "best_epoch": best_epoch,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def aggregate(summaries: list[dict], args: argparse.Namespace) -> None:
    out_root = resolve(args.out_root)
    frame = pd.DataFrame(summaries).sort_values("fold")
    frame.to_csv(out_root / "fold_summary.csv", index=False)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    if len(frame) == 5 and set(frame["fold"]) == set(range(5)):
        weights = np.asarray(
            [manifest.loc[int(fold), "val_ed_instances"] for fold in frame["fold"]],
            dtype=np.float64,
        )
        aggregate_row = {
            "mean_baseline_mAP": float(frame["baseline_mAP"].mean()),
            "mean_best_mAP": float(frame["best_mAP"].mean()),
            "weighted_baseline_mAP": float(
                np.average(frame["baseline_mAP"], weights=weights)
            ),
            "weighted_best_mAP": float(
                np.average(frame["best_mAP"], weights=weights)
            ),
            "worst_baseline_mAP": float(frame["baseline_mAP"].min()),
            "worst_best_mAP": float(frame["best_mAP"].min()),
        }
        pd.DataFrame([aggregate_row]).to_csv(out_root / "cv_summary.csv", index=False)
        print(pd.DataFrame([aggregate_row]).to_string(index=False))
    else:
        print(frame.to_string(index=False))


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    resolve(args.out_root).mkdir(parents=True, exist_ok=True)
    summaries = [train_fold(fold, args, device) for fold in args.folds]
    aggregate(summaries, args)


if __name__ == "__main__":
    main()
