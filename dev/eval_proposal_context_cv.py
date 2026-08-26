"""Recording-disjoint proposal-context scoring inspired by P-GCN/ContextLoc.

Independent quality heads cannot model clusters of overlapping proposals. This
experiment orders proposals by temporal centre inside each ROI and applies a
small dilated TCN over frozen ATSN proposal features. Long ROI sequences are
processed with overlapping chunks and predictions are averaged on overlap.
The proposal lattice and the fixed TemporalMaxer boundary remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dev.eval_boundary_quality_router_cv import candidate_tiou
from dev.eval_temporal_boundary_router_cv import load_master_scores
from dev.train_temporalmaxer_dense import (
    atomic_torch_save,
    cache_paths,
    evaluate_variant,
    load_annotation_index,
    load_cache,
    map_to_master,
    stable_proposal_index,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProposalChunk:
    positions: np.ndarray
    recording: str
    roi_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-proposals",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument(
        "--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source"
    )
    parser.add_argument(
        "--cache-dir", default="tmp/temporalmaxer_dense/hybrid_source_cache"
    )
    parser.add_argument(
        "--reference-root",
        default="tmp/temporalmaxer_dense/boundary_quality_router_cv",
    )
    parser.add_argument(
        "--groupdro-root", default="tmp/quality_head/family_groupdro_cv"
    )
    parser.add_argument(
        "--manifest", default="tmp/cv/recording_folds_r5/manifest.csv"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/proposal_context_cv"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--chunk-stride", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--numeric-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--rank-pairs", type=int, default=32)
    parser.add_argument("--qfl-beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260719)

    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
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


def temporal_iou_aligned(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray:
    intersection = np.maximum(
        0.0, np.minimum(first_end, second_end) - np.maximum(first_start, second_start)
    )
    union = first_end - first_start + second_end - second_start - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def proposal_numeric_features(
    proposals: pd.DataFrame,
    reference: pd.DataFrame,
) -> np.ndarray:
    """Build domain-stable node features, including ROI-relative ranks and gaps."""
    if len(proposals) != len(reference) or not stable_proposal_index(proposals).equals(
        stable_proposal_index(reference)
    ):
        raise ValueError("Reference scores are not aligned with proposals")
    frame = proposals.reset_index(drop=True)
    starts = frame["t_start"].to_numpy(dtype=np.float64)
    ends = frame["t_end"].to_numpy(dtype=np.float64)
    duration = np.maximum(ends - starts, 1.0)
    center = 0.5 * (starts + ends)
    columns = [
        frame["score"].to_numpy(dtype=np.float64),
        reference["cnn_score"].to_numpy(dtype=np.float64),
        reference["dense_quality"].to_numpy(dtype=np.float64),
        reference["dense_action"].to_numpy(dtype=np.float64),
        reference["dense_point"].to_numpy(dtype=np.float64),
        reference["dense_score"].to_numpy(dtype=np.float64),
        reference["brem_score"].to_numpy(dtype=np.float64),
        np.log1p(duration / 1e6),
    ]
    numeric = np.stack(columns, axis=1)
    extra = np.zeros((len(frame), 7), dtype=np.float64)
    groups = frame.groupby(["rec_name", "roi_id"], sort=False).groups
    for local in groups.values():
        positions = np.asarray(list(local), dtype=np.int64)
        order = positions[np.argsort(center[positions], kind="stable")]
        size = len(order)
        extra[order, 0] = np.linspace(0.0, 1.0, size) if size > 1 else 0.5
        for output_column, values in enumerate(
            (
                frame["score"].to_numpy(dtype=np.float64),
                reference["cnn_score"].to_numpy(dtype=np.float64),
                reference["brem_score"].to_numpy(dtype=np.float64),
            ),
            start=1,
        ):
            ranks = pd.Series(values[positions]).rank(method="average", pct=True)
            extra[positions, output_column] = ranks.to_numpy(dtype=np.float64)
        if size > 1:
            previous = np.concatenate(([order[0]], order[:-1]))
            following = np.concatenate((order[1:], [order[-1]]))
            extra[order, 4] = (center[order] - center[previous]) / duration[order]
            extra[order, 5] = (center[following] - center[order]) / duration[order]
            previous_iou = temporal_iou_aligned(
                starts[order], ends[order], starts[previous], ends[previous]
            )
            following_iou = temporal_iou_aligned(
                starts[order], ends[order], starts[following], ends[following]
            )
            extra[order, 6] = 0.5 * (previous_iou + following_iou)
    output = np.concatenate((numeric, extra), axis=1)
    return np.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0).astype(
        np.float32
    )


def build_proposal_chunks(
    proposals: pd.DataFrame,
    chunk_length: int,
    stride: int,
) -> list[ProposalChunk]:
    if chunk_length < 2 or not 0 < stride <= chunk_length:
        raise ValueError("Require chunk_length >= 2 and stride in [1, chunk_length]")
    frame = proposals.reset_index(drop=True)
    center = 0.5 * (
        frame["t_start"].to_numpy(dtype=np.float64)
        + frame["t_end"].to_numpy(dtype=np.float64)
    )
    chunks: list[ProposalChunk] = []
    for (recording, roi_id), local in frame.groupby(
        ["rec_name", "roi_id"], sort=False
    ).groups.items():
        positions = np.asarray(list(local), dtype=np.int64)
        positions = positions[np.argsort(center[positions], kind="stable")]
        if len(positions) <= chunk_length:
            starts = [0]
        else:
            starts = list(range(0, len(positions) - chunk_length + 1, stride))
            final_start = len(positions) - chunk_length
            if starts[-1] != final_start:
                starts.append(final_start)
        for start in starts:
            selected = positions[start : start + chunk_length]
            chunks.append(
                ProposalChunk(selected, str(recording), str(roi_id))
            )
    return chunks


def chunks_cover_all(chunks: list[ProposalChunk], count: int) -> bool:
    covered = np.zeros(count, dtype=bool)
    for chunk in chunks:
        covered[chunk.positions] = True
    return bool(covered.all())


def quantile_match_scores(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Transfer the reference marginal distribution while retaining source order."""
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if source.shape != reference.shape:
        raise ValueError("Source and reference scores must have identical shapes")
    order = np.argsort(source, kind="stable")
    output = np.empty_like(source)
    output[order] = np.sort(reference, kind="stable")
    return output


def add_context_score_variants(scored: pd.DataFrame) -> pd.DataFrame:
    """Add conservative, marginally calibrated context fusions."""
    output = scored.copy()
    qhead = np.clip(output["quality_score"].to_numpy(dtype=np.float64), 1e-8, 1.0)
    context = np.clip(output["context_score"].to_numpy(dtype=np.float64), 1e-8, 1.0)
    calibrated = np.clip(quantile_match_scores(context, qhead), 1e-8, 1.0)
    output["context_calibrated"] = calibrated
    for weight in (0.05, 0.10, 0.25):
        suffix = f"w{int(round(weight * 100)):03d}"
        output[f"context_arithmetic_{suffix}"] = (
            (1.0 - weight) * qhead + weight * context
        )
        output[f"context_calibrated_geometric_{suffix}"] = np.power(
            qhead, 1.0 - weight
        ) * np.power(calibrated, weight)
    return output


class ProposalChunkDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        master_indices: np.ndarray,
        numeric: np.ndarray,
        proposals: pd.DataFrame,
        chunk_length: int,
        stride: int,
        targets: np.ndarray | None = None,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.master_indices = np.asarray(master_indices, dtype=np.int64)
        self.numeric = np.asarray(numeric, dtype=np.float32)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
        self.chunks = build_proposal_chunks(proposals, chunk_length, stride)
        if not chunks_cover_all(self.chunks, len(proposals)):
            raise ValueError("Proposal chunks do not cover every proposal")
        if len(self.numeric) != len(proposals) or len(self.master_indices) != len(proposals):
            raise ValueError("Proposal context inputs are misaligned")
        counts = pd.Series([chunk.recording for chunk in self.chunks]).value_counts()
        self.sampling_weights = np.asarray(
            [1.0 / float(counts[chunk.recording]) for chunk in self.chunks],
            dtype=np.float64,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int):
        positions = self.chunks[index].positions
        frame_features = torch.from_numpy(
            np.asarray(
                self.features[self.master_indices[positions]], dtype=np.float32
            ).copy()
        )
        numeric = torch.from_numpy(self.numeric[positions])
        target = (
            torch.zeros(len(positions), dtype=torch.float32)
            if self.targets is None
            else torch.from_numpy(self.targets[positions])
        )
        return (
            frame_features,
            numeric,
            target,
            torch.from_numpy(positions),
            float(self.sampling_weights[index]),
        )


def collate_chunks(batch):
    batch_size = len(batch)
    max_length = max(len(item[3]) for item in batch)
    segments = batch[0][0].shape[1]
    feature_dim = batch[0][0].shape[2]
    numeric_dim = batch[0][1].shape[1]
    frame_features = torch.zeros(
        batch_size, max_length, segments, feature_dim, dtype=torch.float32
    )
    numeric = torch.zeros(batch_size, max_length, numeric_dim, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_length, dtype=torch.float32)
    positions = torch.full((batch_size, max_length), -1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    weights = torch.zeros(batch_size, dtype=torch.float32)
    for index, (features, values, target, local_positions, weight) in enumerate(batch):
        length = len(local_positions)
        frame_features[index, :length] = features
        numeric[index, :length] = values
        targets[index, :length] = target
        positions[index, :length] = local_positions
        mask[index, :length] = True
        weights[index] = weight
    return frame_features, numeric, targets, positions, mask, weights


class ResidualContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.network(x.transpose(1, 2)).transpose(1, 2)
        output = self.norm(x + update)
        return output * mask.unsqueeze(2)


class ProposalContextTCN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        numeric_dim: int,
        hidden_dim: int = 128,
        numeric_hidden_dim: int = 32,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        frame_hidden = hidden_dim // 2
        self.frame_norm = nn.LayerNorm(feature_dim)
        self.frame_projection = nn.Sequential(
            nn.Linear(feature_dim, frame_hidden),
            nn.GELU(),
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, numeric_hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * frame_hidden + numeric_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context = nn.ModuleList(
            [ResidualContextBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4)]
        )
        self.quality_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)

    def forward(
        self,
        frame_features: torch.Tensor,
        numeric: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.frame_projection(self.frame_norm(frame_features))
        pooled = torch.cat((encoded.mean(dim=2), encoded.amax(dim=2)), dim=2)
        x = self.fusion(torch.cat((pooled, self.numeric_projection(numeric)), dim=2))
        x = x * mask.unsqueeze(2)
        for block in self.context:
            x = block(x, mask)
        return self.quality_head(x).squeeze(2)


def masked_quality_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    scale = torch.abs(targets - probability).pow(beta)
    weights = torch.where(targets > 0.0, 2.0 + 2.0 * targets, 1.0)
    loss = weights * scale * bce
    return (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def local_pairwise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    max_pairs: int,
) -> torch.Tensor:
    losses = []
    for sequence_logits, sequence_targets, sequence_mask in zip(logits, targets, mask):
        valid_logits = sequence_logits[sequence_mask]
        valid_targets = sequence_targets[sequence_mask]
        positive = torch.nonzero(valid_targets >= 0.5, as_tuple=False).flatten()
        negative = torch.nonzero(valid_targets < 0.1, as_tuple=False).flatten()
        pairs = min(len(positive), len(negative), max_pairs)
        if pairs == 0:
            continue
        positive = positive[torch.randperm(len(positive), device=logits.device)[:pairs]]
        negative = negative[torch.randperm(len(negative), device=logits.device)[:pairs]]
        losses.append(
            F.softplus(-(valid_logits[positive] - valid_logits[negative])).mean()
        )
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def make_loader(
    dataset: ProposalChunkDataset,
    args: argparse.Namespace,
    training: bool,
) -> DataLoader:
    sampler = None
    if training:
        sampler = WeightedRandomSampler(
            dataset.sampling_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_chunks,
    )


def train_model(
    model: ProposalContextTCN,
    dataset: ProposalChunkDataset,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> ProposalContextTCN:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    start_epoch = 1
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
    loader = make_loader(dataset, args, training=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for features, numeric, targets, _, mask, _ in loader:
            features = features.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(features, numeric, mask)
                per_chunk = masked_quality_focal_loss(
                    logits, targets, mask, args.qfl_beta
                )
                quality_loss = per_chunk.mean()
                rank_loss = local_pairwise_loss(
                    logits, targets, mask, args.rank_pairs
                )
                loss = quality_loss + args.rank_weight * rank_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        atomic_torch_save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": total / max(batches, 1),
            },
            checkpoint_path,
        )
        print(
            f"[EPOCH {epoch:02d}] loss={total / max(batches, 1):.6f}",
            flush=True,
        )
    return model


@torch.no_grad()
def predict_context_scores(
    model: ProposalContextTCN,
    dataset: ProposalChunkDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(dataset, args, training=False)
    score_sum = np.zeros(len(dataset.master_indices), dtype=np.float64)
    score_count = np.zeros(len(dataset.master_indices), dtype=np.int32)
    model.eval()
    for features, numeric, _, positions, mask, _ in loader:
        features = features.to(device, non_blocking=True)
        numeric = numeric.to(device, non_blocking=True)
        mask_device = mask.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            probability = torch.sigmoid(model(features, numeric, mask_device))
        probability = probability.float().cpu().numpy()
        for batch_index in range(len(positions)):
            valid = mask[batch_index].numpy()
            local = positions[batch_index, valid].numpy()
            score_sum[local] += probability[batch_index, valid]
            score_count[local] += 1
    if np.any(score_count == 0):
        raise ValueError("Context inference left proposals without predictions")
    return (score_sum / score_count).astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, _, metadata = load_cache(resolve(args.cache_dir))
    feature_path = cache_paths(resolve(args.cache_dir))["features"]
    annotations = load_annotation_index(resolve(args.ann_path))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed
    for fold in args.folds:
        fold_seed = base_seed + fold
        args.seed = fold_seed
        set_seed(fold_seed)
        fold_out = out_dir / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        if (fold_out / "metrics.csv").exists():
            continue
        val = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        held_out = set(val["rec_name"].astype(str))
        train_indices = np.flatnonzero(
            ~master["rec_name"].astype(str).isin(held_out).to_numpy()
        )
        val_indices = map_to_master(master, val)
        train = master.iloc[train_indices].reset_index(drop=True)
        reference_all = load_master_scores(
            resolve(args.reference_root), fold, "reference", master
        )
        train_reference = reference_all.iloc[train_indices].reset_index(drop=True)
        val_reference = reference_all.iloc[val_indices].reset_index(drop=True)
        train_numeric_raw = proposal_numeric_features(train, train_reference)
        mean = train_numeric_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(
            train_numeric_raw.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4
        )
        train_numeric = (train_numeric_raw - mean) / std
        val_numeric = (proposal_numeric_features(val, val_reference) - mean) / std
        train_targets = candidate_tiou(
            train,
            train["t_start"].to_numpy(dtype=np.float64)[:, None],
            train["t_end"].to_numpy(dtype=np.float64)[:, None],
            annotations,
        )[:, 0]
        train_dataset = ProposalChunkDataset(
            feature_path,
            train_indices,
            train_numeric,
            train,
            args.chunk_length,
            args.chunk_stride,
            train_targets,
        )
        val_dataset = ProposalChunkDataset(
            feature_path,
            val_indices,
            val_numeric,
            val,
            args.chunk_length,
            args.chunk_stride,
        )
        model = ProposalContextTCN(
            int(metadata["feature_dim"]),
            train_numeric.shape[1],
            args.hidden_dim,
            args.numeric_hidden_dim,
            args.dropout,
        ).to(device)
        model = train_model(
            model,
            train_dataset,
            args,
            device,
            fold_out / "last.pt",
        )
        context_score = predict_context_scores(
            model, val_dataset, args, device
        )
        groupdro_all = pd.read_csv(
            resolve(args.groupdro_root)
            / f"fold_{fold:02d}"
            / "cache"
            / "val_scores_qhead_qfl_only.csv"
        )
        groupdro = groupdro_all.iloc[
            map_to_master(groupdro_all, val)
        ].reset_index(drop=True)
        scored = val.copy()
        qhead = np.clip(
            groupdro["quality_score"].to_numpy(dtype=np.float64), 1e-8, 1.0
        )
        context = np.clip(context_score.astype(np.float64), 1e-8, 1.0)
        scored["quality_score"] = qhead
        scored["context_score"] = context
        scored["context_qhead_w025"] = np.power(qhead, 0.75) * np.power(context, 0.25)
        scored["context_qhead_w050"] = np.sqrt(qhead * context)
        scored = add_context_score_variants(scored)
        raw_start = scored["t_start"].to_numpy(dtype=np.float64)
        raw_end = scored["t_end"].to_numpy(dtype=np.float64)
        scored["reference_t_start"] = 0.5 * (
            raw_start + val_reference["delta_t_start"].to_numpy(dtype=np.float64)
        )
        scored["reference_t_end"] = 0.5 * (
            raw_end + val_reference["delta_t_end"].to_numpy(dtype=np.float64)
        )
        scored.to_csv(fold_out / "scored.csv", index=False)
        rows = []
        for score_column in (
            "quality_score",
            "context_score",
            "context_qhead_w025",
            "context_qhead_w050",
            "context_calibrated",
            "context_arithmetic_w005",
            "context_arithmetic_w010",
            "context_arithmetic_w025",
            "context_calibrated_geometric_w005",
            "context_calibrated_geometric_w010",
            "context_calibrated_geometric_w025",
        ):
            row = evaluate_variant(
                scored,
                score_column,
                "reference",
                f"proposal_context_fold_{fold:02d}",
                args,
                fold_out / "predictions",
            )
            row["fold"] = fold
            row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
            rows.append(row)
            pd.DataFrame(rows).to_csv(fold_out / "metrics_partial.csv", index=False)
        pd.DataFrame(rows).to_csv(fold_out / "metrics.csv", index=False)
        (fold_out / "configuration.json").write_text(
            json.dumps(
                {
                    "numeric_mean": mean.tolist(),
                    "numeric_std": std.tolist(),
                    "numeric_dim": int(train_numeric.shape[1]),
                    "train_chunks": len(train_dataset),
                    "val_chunks": len(val_dataset),
                    "train_recordings": sorted(train["rec_name"].astype(str).unique()),
                    "args": vars(args),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[FOLD {fold:02d}] completo", flush=True)

    metric_paths = [out_dir / f"fold_{fold:02d}" / "metrics.csv" for fold in range(5)]
    if not all(path.exists() for path in metric_paths):
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for score_column, group in metrics.groupby("score_column"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row: dict[str, float | str] = {"score_column": score_column}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
