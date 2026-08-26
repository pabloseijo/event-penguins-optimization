"""Pilot class-balanced semantic domain adaptation on recording-disjoint CV.

The target fold is consumed without annotations during adaptation.  Source labels
define the action prior; that prior selects the same upper fraction of teacher
scores on target, avoiding an absolute pseudo-label threshold that is invalid
for the focal-loss score scale of this detector.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
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
        "--out-root", default="tmp/temporalmaxer_continuous/cv_sada_lite_v1"
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-lr", type=float, default=1e-5)
    parser.add_argument("--discriminator-lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--downstream-weight", type=float, default=2.0)
    parser.add_argument("--target-weight", type=float, default=0.2)
    parser.add_argument("--background-weight", type=float, default=0.3)
    parser.add_argument("--background-quantile", type=float, default=0.8)
    parser.add_argument("--reverse-beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--epochs-per-process", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.beta = beta
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.beta * gradient, None


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, features: torch.Tensor, beta: float) -> torch.Tensor:
        return self.layers(GradientReverse.apply(features, beta)).squeeze(1)


def make_loader(
    dataset: ContinuousSequenceDataset,
    args: argparse.Namespace,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_sequences,
    )


def repeat_loader(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def level_zero_source_targets(
    model,
    output: dict[str, list[torch.Tensor]],
    segments: list[torch.Tensor],
    grid_stride_s: float,
) -> torch.Tensor:
    lengths = [value.shape[1] for value in output["classification_logits"]]
    values = []
    device = output["classification_logits"][0].device
    for sequence_segments in segments:
        targets, _ = model.targets_for_sequence(
            lengths, sequence_segments.to(device) / grid_stride_s, device
        )
        values.append(targets[0].bool())
    return torch.stack(values)


@torch.no_grad()
def estimate_source_action_prior(
    model,
    loader: DataLoader,
    grid_stride_s: float,
    device: torch.device,
) -> float:
    positives = 0
    valid_points = 0
    model.eval()
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        output = model(features, mask)
        targets = level_zero_source_targets(
            model, output, batch["segments"], grid_stride_s
        )
        positives += int((targets & mask).sum())
        valid_points += int(mask.sum())
    if positives == 0 or valid_points == 0:
        raise ValueError("The source split has no valid action points")
    return positives / valid_points


def masked_level_features(
    output: dict[str, list[torch.Tensor]], selection: torch.Tensor
) -> torch.Tensor:
    return output["pyramid_features"][0].transpose(1, 2)[selection]


def target_pseudo_masks(
    teacher_output: dict[str, list[torch.Tensor]],
    valid_mask: torch.Tensor,
    action_prior: float,
    background_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = teacher_output["classification_logits"][0].sigmoid()
    valid_scores = scores[valid_mask]
    action_count = max(1, int(round(valid_scores.numel() * action_prior)))
    background_count = max(1, int(round(valid_scores.numel() * background_quantile)))
    order = valid_scores.argsort()
    flat_action = torch.zeros_like(valid_scores, dtype=torch.bool)
    flat_background = torch.zeros_like(valid_scores, dtype=torch.bool)
    flat_action[order[-action_count:]] = True
    flat_background[order[:background_count]] = True
    action = torch.zeros_like(valid_mask)
    background = torch.zeros_like(valid_mask)
    action[valid_mask] = flat_action
    background[valid_mask] = flat_background
    return action, background


def domain_loss(
    discriminator: DomainDiscriminator,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    beta: float,
    target_weight: float,
) -> torch.Tensor:
    loss = source_features.sum() * 0.0 + target_features.sum() * 0.0
    if source_features.shape[0] > 0:
        source_logits = discriminator(source_features, beta)
        loss = loss + F.binary_cross_entropy_with_logits(
            source_logits, torch.zeros_like(source_logits)
        )
    if target_features.shape[0] > 0:
        target_logits = discriminator(target_features, beta)
        loss = loss + target_weight * F.binary_cross_entropy_with_logits(
            target_logits, torch.ones_like(target_logits)
        )
    return loss


def main() -> None:
    args = parse_args()
    if not 0.0 < args.background_quantile < 1.0:
        raise ValueError("background-quantile must be in (0,1)")
    set_seed(args.seed + args.fold)
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    target_recordings = str(manifest.loc[args.fold, "val_record_names"]).split()
    source_sequences = sequences[~sequences["rec_name"].isin(target_recordings)].copy()
    target_sequences = sequences[sequences["rec_name"].isin(target_recordings)].copy()
    annotations = load_annotations(resolve(args.ann_path))
    feature_path = feature_dir / "frame_features.npy"
    source_dataset = ContinuousSequenceDataset(feature_path, source_sequences, annotations)
    target_dataset = ContinuousSequenceDataset(feature_path, target_sequences, annotations={})
    source_loader = make_loader(source_dataset, args, True, args.seed + args.fold)
    target_loader = make_loader(target_dataset, args, True, args.seed + 100 + args.fold)
    evaluation_loader = make_loader(target_dataset, args, False, args.seed)

    checkpoint_path = resolve(args.checkpoint_root) / f"fold_{args.fold:02d}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_models(
        resolve(args.checkpoint_root),
        int(metadata["feature_dim"]),
        device,
        [checkpoint_path],
    )[0]
    teacher = load_models(
        resolve(args.checkpoint_root),
        int(metadata["feature_dim"]),
        device,
        [checkpoint_path],
    )[0]
    teacher.requires_grad_(False).eval()
    action_discriminator = DomainDiscriminator(model.hidden_dim).to(device)
    background_discriminator = DomainDiscriminator(model.hidden_dim).to(device)
    action_prior = estimate_source_action_prior(
        model, make_loader(source_dataset, args, False, args.seed),
        float(metadata["grid_stride_s"]), device
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.model_lr},
            {
                "params": itertools.chain(
                    action_discriminator.parameters(), background_discriminator.parameters()
                ),
                "lr": args.discriminator_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    eval_args = evaluation_args(checkpoint, args.ann_path)
    mode = "control" if args.adv_weight == 0 else "sada"
    out_dir = resolve(args.out_root) / mode / f"fold_{args.fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    start_epoch = 1
    last_path = out_dir / "last.pt"
    if last_path.exists() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        action_discriminator.load_state_dict(state["action_discriminator"])
        background_discriminator.load_state_dict(state["background_discriminator"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1

    print(
        f"[INFO] fold={args.fold} mode={mode} source={len(source_dataset)} "
        f"target={len(target_dataset)} action_prior={action_prior:.8f}",
        flush=True,
    )
    grid_stride_s = float(metadata["grid_stride_s"])
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        action_discriminator.train()
        background_discriminator.train()
        source_iterator = repeat_loader(source_loader)
        target_iterator = repeat_loader(target_loader)
        totals = {"loss": 0.0, "downstream": 0.0, "action_adv": 0.0, "background_adv": 0.0}
        batches = max(len(source_loader), len(target_loader))
        for _ in range(batches):
            source_batch = next(source_iterator)
            target_batch = next(target_iterator)
            source_features = source_batch["features"].to(device, non_blocking=True)
            source_mask = source_batch["mask"].to(device, non_blocking=True)
            target_features = target_batch["features"].to(device, non_blocking=True)
            target_mask = target_batch["mask"].to(device, non_blocking=True)
            source_segments = [value.to(device) for value in source_batch["segments"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                source_output = model(source_features, source_mask)
                target_output = model(target_features, target_mask)
                with torch.no_grad():
                    teacher_output = teacher(target_features, target_mask)
                source_targets = level_zero_source_targets(
                    model, source_output, source_segments, grid_stride_s
                )
                source_action = source_targets & source_mask
                source_background = (~source_targets) & source_mask
                target_action, target_background = target_pseudo_masks(
                    teacher_output,
                    target_mask,
                    action_prior,
                    args.background_quantile,
                )
                downstream = model.losses(
                    source_output, source_segments, grid_stride_s
                )["loss"]
                if args.adv_weight > 0:
                    action_adv = domain_loss(
                        action_discriminator,
                        masked_level_features(source_output, source_action),
                        masked_level_features(target_output, target_action),
                        args.reverse_beta,
                        args.target_weight,
                    )
                    background_adv = domain_loss(
                        background_discriminator,
                        masked_level_features(source_output, source_background),
                        masked_level_features(target_output, target_background),
                        args.reverse_beta,
                        args.target_weight,
                    )
                else:
                    action_adv = downstream.detach() * 0.0
                    background_adv = downstream.detach() * 0.0
                loss = (
                    args.downstream_weight * downstream
                    + args.adv_weight
                    * (action_adv + args.background_weight * background_adv)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            clip_grad_norm_(action_discriminator.parameters(), 1.0)
            clip_grad_norm_(background_discriminator.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["downstream"] += float(downstream.detach())
            totals["action_adv"] += float(action_adv.detach())
            totals["background_adv"] += float(background_adv.detach())
        scheduler.step()

        metrics = evaluate(
            model,
            evaluation_loader,
            target_sequences,
            metadata,
            eval_args,
            device,
            out_dir / "predictions" / f"epoch_{epoch:03d}.json",
        )
        row = {
            "epoch": epoch,
            "action_prior": action_prior,
            **{key: value / batches for key, value in totals.items()},
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        portable_state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": checkpoint["args"],
            "adaptation_args": vars(args),
            "metadata": metadata,
        }
        previous_best = max(
            (float(value["mAP"]) for value in history[:-1]), default=-1.0
        )
        if float(metrics["mAP"]) > previous_best:
            torch.save(portable_state, out_dir / "best.pt")
            (out_dir / "metrics_best.json").write_text(
                json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8"
            )
        torch.save(
            {
                "model": model.state_dict(),
                "action_discriminator": action_discriminator.state_dict(),
                "background_discriminator": background_discriminator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "args": checkpoint["args"],
                "adaptation_args": vars(args),
                "metadata": metadata,
            },
            last_path,
        )
        print(
            f"[EPOCH {epoch:03d}] loss={row['loss']:.5f} "
            f"mAP={metrics['mAP']:.6f} AP07={metrics['AP@0.7']:.6f}",
            flush=True,
        )
        if args.epochs_per_process and epoch < args.epochs:
            if epoch - start_epoch + 1 >= args.epochs_per_process:
                raise SystemExit(75)


if __name__ == "__main__":
    main()
