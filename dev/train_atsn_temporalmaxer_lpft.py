"""Fine-tune ATSN layer4 with the dense TemporalMaxer localization objective.

The experiment is deliberately source-CV only. It starts from the original
ATSN weights and a recording-disjoint TemporalMaxer checkpoint, keeps all
BatchNorm statistics frozen, and updates only ResNet layer4 plus the dense
head. Training and raw-event validation are resumable because CiTIUS jobs can
be interrupted between batches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from dev.eval_temporalmaxer_groupdro_cv import add_ranking_fusions
from dev.train_atsn_lpft import FineTuneProposalDataset
from dev.train_temporalmaxer_dense import (
    DenseTargets,
    atomic_torch_save,
    autocast_context,
    build_targets,
    choose_training_indices,
    dense_loss,
    evaluate_variant,
    load_cache,
    map_to_master,
    softmax_ed,
    subset_targets,
)
from src.augmented_tsn import AugmentedTsn
from src.temporalmaxer_lite import TemporalMaxerLiteHead


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source"
    )
    parser.add_argument(
        "--master-proposals",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument(
        "--cache-dir", default="tmp/temporalmaxer_dense/hybrid_source_cache"
    )
    parser.add_argument(
        "--base-checkpoint-root",
        default="tmp/temporalmaxer_dense/screened_cv_blend075_erm",
    )
    parser.add_argument(
        "--boundary-reference-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_eval",
    )
    parser.add_argument(
        "--groupdro-root", default="tmp/quality_head/family_groupdro_cv"
    )
    parser.add_argument(
        "--out-root", default="tmp/temporalmaxer_dense/atsn_dense_lpft_cv"
    )
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Optional fixed subset of completed epochs to score.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-val-proposals", type=int, default=0)
    parser.add_argument("--lr-layer4", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument(
        "--train-block", choices=["layer4", "first"], default="layer4"
    )
    parser.add_argument("--freeze-detector", action="store_true")
    parser.add_argument("--event-drop-prob", type=float, default=0.0)
    parser.add_argument("--sample-duration-jitter", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--l2sp-weight", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--checkpoint-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")

    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)

    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--high-pos-tiou", type=float, default=0.7)
    parser.add_argument("--boundary-min-tiou", type=float, default=0.3)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--trident-weight", type=float, default=0.0)
    parser.add_argument("--qfl-beta", type=float, default=2.0)

    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DenseImageDataset(Dataset):
    """Attach dense targets to raw event time-surface sequences."""

    def __init__(
        self,
        proposals: pd.DataFrame,
        targets: DenseTargets | None,
        args: argparse.Namespace,
    ) -> None:
        self.targets = targets
        training = targets is not None
        image_proposals = proposals.copy()
        if training:
            image_proposals["label"] = 0
        self.images = FineTuneProposalDataset(
            proposals=image_proposals,
            augment_fraction=1.0 / args.augment_factor,
            data_path=resolve(args.data_path),
            num_tsn_samples=args.num_segments,
            sample_duration_s=args.sample_duration,
            decay=args.decay,
            require_label=training,
            sample_duration_jitter=(
                args.sample_duration_jitter if training else 0.0
            ),
            event_drop_prob=args.event_drop_prob if training else 0.0,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        images, local_index = self.images[index]
        if self.targets is None:
            return images, local_index
        return (
            images,
            torch.tensor(self.targets.quality[index], dtype=torch.float32),
            torch.from_numpy(self.targets.action[index]),
            torch.from_numpy(self.targets.point_distances[index]),
            torch.from_numpy(self.targets.start_distribution[index]),
            torch.from_numpy(self.targets.end_distribution[index]),
            torch.from_numpy(self.targets.deltas[index]),
            torch.tensor(self.targets.boundary_weight[index], dtype=torch.float32),
        )


class EndToEndDenseDetector(nn.Module):
    def __init__(self, atsn: AugmentedTsn, detector: TemporalMaxerLiteHead) -> None:
        super().__init__()
        self.atsn = atsn
        self.detector = detector

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.detector(self.atsn.encode_frames(images))

    def set_fine_tune_mode(self) -> None:
        self.eval()
        if any(parameter.requires_grad for parameter in self.detector.parameters()):
            self.detector.train()


def configure_backbone_trainable(
    atsn: AugmentedTsn, train_block: str
) -> list[str]:
    prefixes = (
        ("layer4.",)
        if train_block == "layer4"
        else ("conv1.", "bn1.", "layer1.")
    )
    names = []
    for name, parameter in atsn.backbone.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True
            names.append(name)
    if not names:
        raise RuntimeError(f"No ATSN parameters matched train_block={train_block}")
    return names


def load_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[EndToEndDenseDetector, dict]:
    atsn = AugmentedTsn(2, args.num_tsn_samples, args.augment_factor)
    try:
        atsn_state = torch.load(
            resolve(args.model_path), map_location="cpu", weights_only=True
        )
    except TypeError:
        atsn_state = torch.load(resolve(args.model_path), map_location="cpu")
    atsn.load_state_dict(atsn_state)
    for parameter in atsn.parameters():
        parameter.requires_grad = False
    trainable_backbone = configure_backbone_trainable(atsn, args.train_block)

    base_path = (
        resolve(args.base_checkpoint_root)
        / f"fold_{args.fold:02d}"
        / "best.pt"
    )
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    saved = base.get("args", {})
    detector = TemporalMaxerLiteHead(
        input_dim=int(base.get("feature_dim", 512)),
        hidden_dim=int(saved.get("hidden_dim", 128)),
        pyramid_levels=int(saved.get("pyramid_levels", 3)),
        dropout=float(saved.get("dropout", 0.15)),
        trident_bins=int(saved.get("trident_bins") or 0),
    )
    detector.load_state_dict(base["state_dict"])
    if args.freeze_detector:
        for parameter in detector.parameters():
            parameter.requires_grad = False
    model = EndToEndDenseDetector(atsn, detector).to(device)
    base["trainable_backbone"] = trainable_backbone
    return model, base


def layer4_snapshot(model: EndToEndDenseDetector) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.atsn.backbone.named_parameters()
        if parameter.requires_grad
    }


def l2sp_loss(
    model: EndToEndDenseDetector, reference: dict[str, torch.Tensor]
) -> torch.Tensor:
    losses = [
        (parameter - reference[name]).pow(2).mean()
        for name, parameter in model.atsn.backbone.named_parameters()
        if name in reference
    ]
    return torch.stack(losses).mean()


def optimizer_for(
    model: EndToEndDenseDetector, args: argparse.Namespace
) -> torch.optim.Optimizer:
    backbone = [
        parameter
        for parameter in model.atsn.backbone.parameters()
        if parameter.requires_grad
    ]
    groups = [{"params": backbone, "lr": args.lr_layer4}]
    detector = [
        parameter for parameter in model.detector.parameters() if parameter.requires_grad
    ]
    if detector:
        groups.append({"params": detector, "lr": args.lr_head})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def save_training_state(
    path: Path,
    model: EndToEndDenseDetector,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    completed: int,
    history: list[dict],
) -> None:
    atomic_torch_save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "completed": completed,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        path,
    )


def restore_training_state(
    path: Path,
    model: EndToEndDenseDetector,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, list[dict]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["cuda_rng_state_all"]]
        )
    return int(state["epoch"]), int(state["completed"]), list(state["history"])


def train_epochs(
    model: EndToEndDenseDetector,
    dataset: DenseImageDataset,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> list[dict]:
    reference = layer4_snapshot(model)
    optimizer = optimizer_for(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict] = []
    epoch = 1
    completed = 0
    last_path = out_dir / "last.pt"
    if last_path.exists() and not args.restart:
        epoch, completed, history = restore_training_state(
            last_path, model, optimizer, scheduler, scaler, device
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    while epoch <= args.epochs:
        epoch_path = out_dir / f"epoch_{epoch:02d}.pt"
        if epoch_path.exists() and not args.restart:
            epoch, _, history = restore_training_state(
                epoch_path, model, optimizer, scheduler, scaler, device
            )
            epoch += 1
            completed = 0
            save_training_state(
                last_path, model, optimizer, scheduler, scaler, epoch, completed, history
            )
            continue

        permutation = np.random.default_rng(args.seed + epoch).permutation(len(dataset))
        remaining = permutation[completed:]
        loader = DataLoader(
            Subset(dataset, remaining.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model.set_fine_tune_mode()
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            images, quality, action, point, start, end, deltas, boundary = batch
            images = images.to(device, non_blocking=True)
            targets = [
                value.to(device, non_blocking=True)
                for value in (quality, action, point, start, end, deltas, boundary)
            ]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(images)
                sample_loss, _ = dense_loss(output, *targets, args)
                loss = sample_loss.mean() + args.l2sp_weight * l2sp_loss(
                    model, reference
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(images)
            completed += batch_size
            epoch_loss += float(loss.detach().cpu()) * batch_size
            batches += 1
            if batches % max(args.checkpoint_batches, 1) == 0:
                save_training_state(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    completed,
                    history,
                )

        mean_loss = epoch_loss / max(len(remaining), 1)
        history.append({"epoch": epoch, "train_loss": mean_loss})
        scheduler.step()
        save_training_state(
            epoch_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            len(dataset),
            history,
        )
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        print(f"[EPOCH {epoch:02d}] train_loss={mean_loss:.6f}", flush=True)
        epoch += 1
        completed = 0
        save_training_state(
            last_path, model, optimizer, scheduler, scaler, epoch, completed, history
        )
    return history


OUTPUT_SHAPES = {
    "quality": (),
    "action": (),
    "point_score": (),
    "point_position": (),
    "point_distances": (2,),
    "deltas": (2,),
    "start_position": (),
    "end_position": (),
}


def open_output_arrays(
    cache_dir: Path, count: int, create: bool
) -> dict[str, np.memmap]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, tail in OUTPUT_SHAPES.items():
        path = cache_dir / f"{name}.npy"
        mode = "w+" if create or not path.exists() else "r+"
        arrays[name] = np.lib.format.open_memmap(
            path, mode=mode, dtype=np.float32, shape=(count, *tail)
        )
    return arrays


@torch.no_grad()
def score_raw_model(
    model: EndToEndDenseDetector,
    proposals: pd.DataFrame,
    args: argparse.Namespace,
    cache_dir: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    state_path = cache_dir / "state.json"
    completed = 0
    if state_path.exists():
        completed = int(json.loads(state_path.read_text(encoding="utf-8"))["completed"])
    arrays = open_output_arrays(cache_dir, len(proposals), create=completed == 0)
    if completed < len(proposals):
        dataset = DenseImageDataset(proposals, None, args)
        loader = DataLoader(
            Subset(dataset, range(completed, len(dataset))),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model.eval()
        relative_times = torch.linspace(
            -1.0 / args.augment_factor,
            1.0 + 1.0 / args.augment_factor,
            args.num_segments,
            device=device,
        )
        for images, local_indices in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                output = model(images)
            action = torch.sigmoid(output["action_logits"])
            point_quality = torch.sigmoid(output["point_quality_logits"])
            point_joint = action * point_quality
            topk = max(1, action.shape[1] // 4)
            point_index = point_joint.argmax(dim=1)
            batch_index = torch.arange(len(images), device=device)
            indices = local_indices.numpy().astype(np.int64)
            arrays["quality"][indices] = (
                torch.sigmoid(output["quality_logit"]).float().cpu().numpy()
            )
            arrays["action"][indices] = (
                action.topk(topk, dim=1).values.mean(dim=1).float().cpu().numpy()
            )
            arrays["point_score"][indices] = (
                point_joint.topk(topk, dim=1).values.mean(dim=1).float().cpu().numpy()
            )
            arrays["point_position"][indices] = (
                relative_times[point_index].float().cpu().numpy()
            )
            arrays["point_distances"][indices] = (
                output["boundary_distances"][batch_index, point_index]
                .float()
                .cpu()
                .numpy()
            )
            arrays["deltas"][indices] = output["boundary_deltas"].float().cpu().numpy()
            arrays["start_position"][indices] = (
                (torch.softmax(output["start_logits"], dim=1) * relative_times)
                .sum(dim=1)
                .float()
                .cpu()
                .numpy()
            )
            arrays["end_position"][indices] = (
                (torch.softmax(output["end_logits"], dim=1) * relative_times)
                .sum(dim=1)
                .float()
                .cpu()
                .numpy()
            )
            completed = int(indices[-1]) + 1
            for array in arrays.values():
                array.flush()
            atomic_json({"completed": completed}, state_path)
    return {name: np.asarray(array).copy() for name, array in arrays.items()}


def outputs_to_scored(
    proposals: pd.DataFrame,
    outputs: dict[str, np.ndarray],
    cnn_score: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    scored = proposals.reset_index(drop=True).copy()
    quality = outputs["quality"]
    point_score = outputs["point_score"]
    scored["cnn_score"] = cnn_score
    scored["dense_quality"] = quality
    scored["dense_action"] = outputs["action"]
    scored["dense_point"] = point_score
    scored["dense_score"] = np.sqrt(np.clip(quality * point_score, 0.0, 1.0))
    scored["brem_score"] = np.cbrt(
        np.clip(cnn_score * quality * point_score, 0.0, 1.0)
    )
    starts = scored["t_start"].to_numpy(dtype=np.float64)
    ends = scored["t_end"].to_numpy(dtype=np.float64)
    duration = np.maximum(ends - starts, 1.0)
    deltas = np.clip(outputs["deltas"], -args.max_boundary_delta, args.max_boundary_delta)
    scored["delta_t_start"] = np.maximum(0.0, starts + deltas[:, 0] * duration)
    scored["delta_t_end"] = ends + deltas[:, 1] * duration
    point_position = outputs["point_position"]
    point_distances = outputs["point_distances"]
    scored["point_t_start"] = np.maximum(
        0.0, starts + (point_position - point_distances[:, 0]) * duration
    )
    scored["point_t_end"] = starts + (
        point_position + point_distances[:, 1]
    ) * duration
    scored["distribution_t_start"] = np.maximum(
        0.0, starts + outputs["start_position"] * duration
    )
    scored["distribution_t_end"] = starts + outputs["end_position"] * duration
    for prefix in ("delta", "point", "distribution"):
        refined_start = scored[f"{prefix}_t_start"].to_numpy(
            dtype=np.float64, copy=True
        )
        refined_end = scored[f"{prefix}_t_end"].to_numpy(
            dtype=np.float64, copy=True
        )
        center = 0.5 * (refined_start + refined_end)
        short = refined_end - refined_start < 2.0e6
        refined_start[short] = np.maximum(0.0, center[short] - 1.0e6)
        refined_end[short] = refined_start[short] + 2.0e6
        scored[f"{prefix}_t_start"] = refined_start
        scored[f"{prefix}_t_end"] = refined_end
    return scored


def attach_external_controls(
    scored: pd.DataFrame,
    proposals: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    groupdro_path = (
        resolve(args.groupdro_root)
        / f"fold_{args.fold:02d}"
        / "cache"
        / "val_scores_qhead_qfl_only.csv"
    )
    groupdro = pd.read_csv(groupdro_path)
    groupdro = groupdro.iloc[map_to_master(groupdro, proposals)].reset_index(drop=True)
    scored["quality_score"] = groupdro["quality_score"].to_numpy(dtype=np.float64)
    scored = add_ranking_fusions(scored)
    reference_path = (
        resolve(args.boundary_reference_root)
        / f"scored_fold_{args.fold:02d}.csv"
    )
    reference = pd.read_csv(reference_path)
    reference = reference.iloc[map_to_master(reference, proposals)].reset_index(drop=True)
    scored["reference_t_start"] = 0.5 * (
        scored["t_start"].to_numpy(dtype=np.float64)
        + reference["delta_t_start"].to_numpy(dtype=np.float64)
    )
    scored["reference_t_end"] = 0.5 * (
        scored["t_end"].to_numpy(dtype=np.float64)
        + reference["delta_t_end"].to_numpy(dtype=np.float64)
    )
    scored["blend050_t_start"] = 0.5 * (
        scored["t_start"].to_numpy(dtype=np.float64)
        + scored["delta_t_start"].to_numpy(dtype=np.float64)
    )
    scored["blend050_t_end"] = 0.5 * (
        scored["t_end"].to_numpy(dtype=np.float64)
        + scored["delta_t_end"].to_numpy(dtype=np.float64)
    )
    return scored


def evaluate_epochs(
    model: EndToEndDenseDetector,
    val_proposals: pd.DataFrame,
    val_master_indices: np.ndarray,
    master_logits: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    eval_epochs = args.eval_epochs or list(range(1, args.epochs + 1))
    invalid = sorted(set(eval_epochs) - set(range(1, args.epochs + 1)))
    if invalid:
        raise ValueError(f"Evaluation epochs outside training range: {invalid}")
    for epoch in eval_epochs:
        metrics_path = out_dir / f"metrics_epoch_{epoch:02d}.csv"
        if metrics_path.exists():
            rows.extend(pd.read_csv(metrics_path).to_dict("records"))
            continue
        checkpoint = torch.load(
            out_dir / f"epoch_{epoch:02d}.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        outputs = score_raw_model(
            model,
            val_proposals,
            args,
            out_dir / f"eval_epoch_{epoch:02d}",
            device,
        )
        scored = outputs_to_scored(
            val_proposals,
            outputs,
            softmax_ed(np.asarray(master_logits[val_master_indices])),
            args,
        )
        scored = attach_external_controls(scored, val_proposals, args)
        scored.to_csv(out_dir / f"scored_epoch_{epoch:02d}.csv", index=False)
        epoch_rows = []
        for score, boundary in (
            ("quality_score", "reference"),
            ("quality_score", "blend050"),
            ("qhead_brem_score", "blend050"),
        ):
            row = evaluate_variant(
                scored,
                score,
                boundary,
                f"atsn_dense_lpft_fold_{args.fold:02d}_epoch_{epoch:02d}",
                args,
                out_dir / "predictions",
            )
            row["fold"] = args.fold
            row["epoch"] = epoch
            epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(metrics_path, index=False)
        rows.extend(epoch_rows)
        print(pd.DataFrame(epoch_rows).to_string(index=False), flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "all_metrics.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed + args.fold)
    device = choose_device(args.device)
    out_dir = resolve(args.out_root) / f"fold_{args.fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = resolve(args.fold_dir) / f"fold_{args.fold:02d}"
    val_proposals = pd.read_csv(fold_dir / "val_proposals.csv").reset_index(drop=True)
    val_recordings = set(val_proposals["rec_name"].astype(str))
    if args.max_val_proposals > 0:
        val_proposals = val_proposals.iloc[: args.max_val_proposals].reset_index(drop=True)
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    train_path = fold_dir / "train_proposals.csv"
    if train_path.exists():
        train_proposals = pd.read_csv(train_path).reset_index(drop=True)
    else:
        train_proposals = master[
            ~master["rec_name"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
    _, master_logits, metadata = load_cache(resolve(args.cache_dir))
    args.num_segments = int(metadata["num_segments"])
    train_master_indices = map_to_master(master, train_proposals)
    val_master_indices = map_to_master(master, val_proposals)
    targets = build_targets(train_proposals, args.num_segments, args)
    sampled = choose_training_indices(
        train_proposals,
        targets,
        softmax_ed(np.asarray(master_logits[train_master_indices])),
        args.max_train_samples,
        args.seed + args.fold,
    )
    sampled_proposals = train_proposals.iloc[sampled].reset_index(drop=True)
    sampled_targets = subset_targets(targets, sampled)
    pd.DataFrame(
        {
            "sample_kind": sampled_targets.sample_kind,
            "quality": sampled_targets.quality,
        }
    ).to_csv(out_dir / "sample_manifest.csv", index=False)
    dataset = DenseImageDataset(sampled_proposals, sampled_targets, args)
    model, base = load_model(args, device)
    print(
        f"[INFO] fold={args.fold} train={len(dataset)}/{len(train_proposals)} "
        f"val={len(val_proposals)} base_mAP={base.get('best_selection_mAP')} "
        f"block={args.train_block} detector_frozen={args.freeze_detector} "
        f"backbone_params={len(base['trainable_backbone'])}",
        flush=True,
    )
    train_epochs(model, dataset, args, out_dir, device)
    metrics = evaluate_epochs(
        model,
        val_proposals,
        val_master_indices,
        master_logits,
        args,
        out_dir,
        device,
    )
    selection = metrics[
        (metrics["score_column"] == "quality_score")
        & (metrics["boundary_mode"] == "blend050")
    ].sort_values("mAP", ascending=False)
    best_epoch = int(selection.iloc[0]["epoch"])
    best_checkpoint = torch.load(
        out_dir / f"epoch_{best_epoch:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    atomic_torch_save(
        {
            "state_dict": best_checkpoint["state_dict"],
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_selection_mAP": float(selection.iloc[0]["mAP"]),
        },
        out_dir / "best.pt",
    )
    summary = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "selection_mAP": float(selection.iloc[0]["mAP"]),
        "control_mAP": float(
            metrics[
                (metrics["epoch"] == best_epoch)
                & (metrics["score_column"] == "quality_score")
                & (metrics["boundary_mode"] == "reference")
            ].iloc[0]["mAP"]
        ),
    }
    atomic_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
