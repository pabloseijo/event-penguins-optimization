"""Temporal contrastive adaptation of the event-based ATSN representation.

The experiment keeps the original classifier frozen and adapts only ResNet
``layer4`` so two semantics-preserving views of the same proposal have nearby
embeddings.  Overlapping proposals from the same recording/ROI are excluded
from the contrastive denominator to avoid treating alternate boundaries of the
same action as negatives.

Run from ``event_penguins/``.  A small validation-only example is::

    python dev/pretrain_atsn_contrastive.py \
        --train-proposals tmp/quality_head/boundary_smoke_inputs/proposals_train.csv \
        --val-proposals tmp/quality_head/boundary_smoke_inputs/proposals_val.csv \
        --max-train-proposals 256 --max-train-samples 64 \
        --max-val-proposals 64 --epochs 1 --num-workers 0 --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.augmented_tsn import AugmentedTsn
from src.classification import ProposalDataset, create_img_representation

from dev.train_atsn_lpft import (
    class_balanced_weights,
    distillation_loss,
    evaluate_scored,
    expanded_tsn_samples,
    focal_loss,
    label_proposals,
    limit_frame,
    limit_training_frame,
    load_or_generate_proposals,
    make_model,
    pairwise_rank_loss,
    print_label_summary,
    resolve_path,
    sampler_for_training,
    score_proposals,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptacion contrastiva temporal de ATSN con fc_cls conxelada."
    )
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--proto-path", default="tmp/prototype/ed_prototype.npy")
    parser.add_argument("--out-dir", default="tmp/atsn_contrastive/pilot")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-proposals", default=None)
    parser.add_argument("--val-proposals", default=None)
    parser.add_argument("--force-generate-proposals", action="store_true")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-backbone", type=float, default=2e-6)
    parser.add_argument("--lr-projector", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-hidden", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--false-negative-tiou", type=float, default=0.50)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=0.10)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--distill-weight", type=float, default=0.50)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--l2sp-weight", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--cb-beta", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")

    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--flap-neg-tiou", type=float, default=0.3)
    parser.add_argument("--min-gt-duration", type=float, default=2.0)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-positive-frac", type=float, default=0.40)
    parser.add_argument("--sampler-hard-neg-frac", type=float, default=0.40)
    parser.add_argument("--sampler-easy-neg-frac", type=float, default=0.20)

    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--temporal-scale-jitter", type=float, default=0.15)
    parser.add_argument("--temporal-shift-jitter", type=float, default=0.05)
    parser.add_argument("--sample-duration-jitter", type=float, default=0.25)
    parser.add_argument("--event-drop-prob", type=float, default=0.05)
    parser.add_argument("--fourier-amplitude-mix", action="store_true")
    parser.add_argument("--fourier-mix-strength", type=float, default=0.2)
    parser.add_argument("--fourier-mix-prob", type=float, default=0.5)
    parser.add_argument("--fourier-consistency-weight", type=float, default=0.0)
    parser.add_argument("--decay", type=float, default=5e-6)

    parser.add_argument("--min-ed-score", type=float, nargs="+", default=[0.02])
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])

    parser.add_argument("--max-train-proposals", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-proposals", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class ContrastiveProposalDataset(Dataset):
    """Return two independently augmented event views of one proposal."""

    _transform = ProposalDataset._transform

    def __init__(self, proposals: pd.DataFrame, args: argparse.Namespace, data_path: Path) -> None:
        self.proposals = proposals.reset_index(drop=True)
        self.data_path = str(data_path)
        self.num_tsn_samples = expanded_tsn_samples(args.num_tsn_samples, args.augment_factor)
        self.augment_fraction = 1.0 / args.augment_factor
        self.sample_duration_us = float(args.sample_duration) * 1e6
        self.decay = float(args.decay)
        self.temporal_scale_jitter = float(args.temporal_scale_jitter)
        self.temporal_shift_jitter = float(args.temporal_shift_jitter)
        self.sample_duration_jitter = float(args.sample_duration_jitter)
        self.event_drop_prob = float(args.event_drop_prob)
        self._hf = None
        self._cached_key = None
        self._cached_events = None
        self._cached_height = None
        self._cached_width = None

    def __len__(self) -> int:
        return len(self.proposals)

    def _get_h5(self):
        if self._hf is None:
            self._hf = h5py.File(self.data_path, "r")
        return self._hf

    def _get_roi_data(self, rec_name: str, roi_id: str):
        key = (rec_name, roi_id)
        if key != self._cached_key:
            roi = self._get_h5()[rec_name][roi_id]
            self._cached_events = np.asarray(roi["events"])
            self._cached_height = int(roi.attrs["height"])
            self._cached_width = int(roi.attrs["width"])
            self._cached_key = key
        return self._cached_events, self._cached_height, self._cached_width

    def _make_view(
        self,
        events: np.ndarray,
        height: int,
        width: int,
        t_start: float,
        t_end: float,
    ) -> torch.Tensor:
        duration = max(t_end - t_start, 1.0)
        center = 0.5 * (t_start + t_end)
        scale = math.exp(np.random.uniform(-self.temporal_scale_jitter, self.temporal_scale_jitter))
        shift = np.random.uniform(-self.temporal_shift_jitter, self.temporal_shift_jitter) * duration
        duration *= scale
        center += shift
        view_start = max(0.0, center - 0.5 * duration)
        view_end = view_start + duration

        augmented = duration * self.augment_fraction
        image_times = torch.linspace(
            view_start - augmented,
            view_end + augmented,
            self.num_tsn_samples,
        )
        sample_duration = self.sample_duration_us * math.exp(
            np.random.uniform(-self.sample_duration_jitter, self.sample_duration_jitter)
        )
        image_starts = image_times - 0.5 * sample_duration
        image_ends = image_times + 0.5 * sample_duration
        indices_start = np.searchsorted(events[:, 2], image_starts)
        indices_end = np.searchsorted(events[:, 2], image_ends)

        images = []
        for start, end in zip(indices_start, indices_end):
            image_events = events[start:end]
            if self.event_drop_prob > 0 and len(image_events) > 1:
                keep = np.random.random(len(image_events)) >= self.event_drop_prob
                if keep.any():
                    image_events = image_events[keep]
            images.append(
                create_img_representation(
                    image_events,
                    self.decay,
                    height,
                    width,
                    self._transform,
                )
            )
        return torch.stack(images)

    def __getitem__(self, idx: int):
        row = self.proposals.iloc[idx]
        rec_name = str(row["rec_name"])
        roi_id = str(row["roi_id"])
        t_start = float(row["t_start"])
        t_end = float(row["t_end"])
        events, height, width = self._get_roi_data(rec_name, roi_id)
        first = self._make_view(events, height, width, t_start, t_end)
        second = self._make_view(events, height, width, t_start, t_end)
        return (
            first,
            second,
            torch.tensor(int(row["label"]), dtype=torch.long),
            rec_name,
            roi_id,
            torch.tensor(t_start, dtype=torch.float32),
            torch.tensor(t_end, dtype=torch.float32),
        )


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=1)


def forward_embedding(model: AugmentedTsn, images: torch.Tensor) -> torch.Tensor:
    """Differentiable ATSN embedding immediately before dropout/fc_cls."""
    num_segments = images.shape[1]
    features = images.reshape((-1,) + images.shape[2:])
    features = model.backbone(features)["features"]
    features = model.avg_pool(features)
    features = features.reshape((-1, num_segments) + features.shape[1:])
    augment = model.num_augment
    start = model.consensus(features[:, :augment]).squeeze(1)
    main = model.consensus(features[:, augment:num_segments - augment]).squeeze(1)
    end = model.consensus(features[:, num_segments - augment:]).squeeze(1)
    return torch.cat((start, main, end), dim=1).flatten(1)


def logits_from_embedding(model: AugmentedTsn, embedding: torch.Tensor) -> torch.Tensor:
    return model.fc_cls(embedding)


def cross_recording_permutation(
    labels: torch.Tensor,
    rec_names: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """Choose same-class amplitude donors from another recording when possible."""
    batch_size = len(rec_names)
    donors = []
    for index in range(batch_size):
        same_label = [
            candidate
            for candidate in range(batch_size)
            if candidate != index
            and rec_names[candidate] != rec_names[index]
            and int(labels[candidate].item()) == int(labels[index].item())
        ]
        different_recording = [
            candidate
            for candidate in range(batch_size)
            if candidate != index and rec_names[candidate] != rec_names[index]
        ]
        candidates = same_label or different_recording
        if not candidates:
            candidates = [candidate for candidate in range(batch_size) if candidate != index]
        donor = index if not candidates else candidates[int(torch.randint(len(candidates), (1,)).item())]
        donors.append(donor)
    return torch.tensor(donors, dtype=torch.long, device=labels.device)


def fourier_amplitude_mix(
    images: torch.Tensor,
    donors: torch.Tensor,
    strength: float,
    probability: float,
) -> torch.Tensor:
    """Mix spatial Fourier amplitudes while preserving each event view's phase."""
    if strength <= 0 or probability <= 0 or len(images) < 2:
        return images
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    raw = (images * std + mean).clamp(0.0, 1.0)
    spectrum = torch.fft.rfft2(raw.float(), dim=(-2, -1))
    amplitude = spectrum.abs()
    unit_phase = spectrum / amplitude.clamp_min(1e-8)

    batch_size = images.shape[0]
    mix = torch.rand((batch_size, 1, 1, 1, 1), device=images.device) * strength
    enabled = torch.rand((batch_size, 1, 1, 1, 1), device=images.device) < probability
    mix = torch.where(enabled, mix, torch.zeros_like(mix))
    donor_amplitude = amplitude.index_select(0, donors)
    mixed_amplitude = amplitude + mix * (donor_amplitude - amplitude)
    mixed_raw = torch.fft.irfft2(
        mixed_amplitude * unit_phase,
        s=raw.shape[-2:],
        dim=(-2, -1),
    ).clamp(0.0, 1.0)
    return ((mixed_raw.to(images.dtype) - mean) / std).contiguous()


def configure_contrastive_model(model: AugmentedTsn) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    for name, parameter in model.backbone.named_parameters():
        if name.startswith("layer4."):
            parameter.requires_grad = True
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("Non se atoparon parametros layer4 no backbone ATSN.")
    return trainable


def temporal_iou_matrix(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    intersection = (
        torch.minimum(ends[:, None], ends[None, :])
        - torch.maximum(starts[:, None], starts[None, :])
    ).clamp_min(0.0)
    duration = (ends - starts).clamp_min(1e-12)
    union = duration[:, None] + duration[None, :] - intersection
    return intersection / union.clamp_min(1e-12)


def false_negative_mask(
    rec_names: list[str] | tuple[str, ...],
    roi_ids: list[str] | tuple[str, ...],
    starts: torch.Tensor,
    ends: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    batch_size = len(rec_names)
    same_stream = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=starts.device)
    for i in range(batch_size):
        for j in range(batch_size):
            same_stream[i, j] = rec_names[i] == rec_names[j] and roi_ids[i] == roi_ids[j]
    overlap = temporal_iou_matrix(starts, ends) >= threshold
    proposal_mask = same_stream & overlap
    proposal_mask.fill_diagonal_(False)
    return proposal_mask.repeat(2, 2)


def nt_xent_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    excluded_negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    batch_size = first.shape[0]
    if batch_size < 2:
        return (first.sum() + second.sum()) * 0.0
    embeddings = torch.cat((first, second), dim=0)
    similarities = embeddings @ embeddings.T / temperature
    diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=embeddings.device)
    allowed = ~(diagonal | excluded_negatives)
    positive_index = (torch.arange(2 * batch_size, device=embeddings.device) + batch_size) % (
        2 * batch_size
    )
    allowed[torch.arange(2 * batch_size, device=embeddings.device), positive_index] = True
    denominator = similarities.masked_fill(~allowed, -torch.inf).logsumexp(dim=1)
    positive = similarities[torch.arange(2 * batch_size, device=embeddings.device), positive_index]
    return (denominator - positive).mean()


def l2sp_for_parameters(
    named_parameters: list[tuple[str, nn.Parameter]],
    initial_parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = [
        (parameter - initial_parameters[name]).pow(2).sum()
        for name, parameter in named_parameters
    ]
    return torch.stack(terms).sum()


def build_train_loader(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    data_path: Path,
) -> DataLoader:
    dataset = ContrastiveProposalDataset(train_df, args, data_path)
    sampler = sampler_for_training(train_df, args)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )


def train_one_epoch(
    model: AugmentedTsn,
    teacher: AugmentedTsn,
    projector: ProjectionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    initial_parameters: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    teacher.eval()
    projector.train()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    totals = {
        "loss": 0.0,
        "contrastive": 0.0,
        "classification": 0.0,
        "rank": 0.0,
        "distill": 0.0,
        "fourier_consistency": 0.0,
        "l2sp": 0.0,
        "accuracy": 0.0,
    }
    count = 0
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    progress = tqdm(loader, desc="contrastive", leave=False, disable=args.quiet_progress)
    for first, second, labels, rec_names, roi_ids, starts, ends in progress:
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        starts = starts.to(device, non_blocking=True)
        ends = ends.to(device, non_blocking=True)
        excluded = false_negative_mask(
            list(rec_names),
            list(roi_ids),
            starts,
            ends,
            args.false_negative_tiou,
        )
        clean_second = second
        if args.fourier_amplitude_mix:
            donors = cross_recording_permutation(labels, list(rec_names))
            second = fourier_amplitude_mix(
                second,
                donors,
                args.fourier_mix_strength,
                args.fourier_mix_prob,
            )
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            first_embedding = forward_embedding(model, first)
            second_embedding = forward_embedding(model, second)
            first_logits = logits_from_embedding(model, first_embedding)
            second_logits = logits_from_embedding(model, second_embedding)
            logits = torch.cat((first_logits, second_logits), dim=0)
            repeated_labels = torch.cat((labels, labels), dim=0)

            contrastive = nt_xent_loss(
                projector(first_embedding),
                projector(second_embedding),
                excluded,
                args.temperature,
            )
            classification = focal_loss(
                logits,
                repeated_labels,
                class_weights,
                args.focal_gamma,
            )
            ranking = pairwise_rank_loss(logits, repeated_labels)
            with torch.no_grad():
                teacher_first_logits = teacher(first)
                teacher_second_logits = teacher(clean_second)
                teacher_logits = torch.cat((teacher_first_logits, teacher_second_logits), dim=0)
            distill = distillation_loss(logits, teacher_logits, args.distill_temperature)
            fourier_consistency = (
                distillation_loss(second_logits, teacher_second_logits, args.distill_temperature)
                if args.fourier_amplitude_mix and args.fourier_consistency_weight > 0
                else second_logits.sum() * 0.0
            )
            l2sp = l2sp_for_parameters(named_trainable, initial_parameters)
            loss = (
                args.contrastive_weight * contrastive
                + args.classification_weight * classification
                + args.rank_weight * ranking
                + args.distill_weight * distill
                + args.fourier_consistency_weight * fourier_consistency
                + args.l2sp_weight * l2sp
            )

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                args.grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()

        batch_items = int(repeated_labels.numel())
        count += batch_items
        values = {
            "loss": loss,
            "contrastive": contrastive,
            "classification": classification,
            "rank": ranking,
            "distill": distill,
            "fourier_consistency": fourier_consistency,
            "l2sp": l2sp,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().item()) * batch_items
        totals["accuracy"] += float((logits.argmax(dim=1) == repeated_labels).sum().item())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if count == 0:
        raise RuntimeError("O loader contrastivo non produciu batches; aumenta as mostras ou baixa batch-size.")
    return {f"train_{name}": value / count for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.num_workers = 0
        args.max_train_proposals = args.max_train_proposals or 256
        args.max_train_samples = args.max_train_samples or 32
        args.max_val_proposals = args.max_val_proposals or 32
        args.batch_size = min(args.batch_size, 4)

    set_seed(args.seed)
    data_path = resolve_path(args.data_path)
    ann_path = resolve_path(args.ann_path)
    proto_path = resolve_path(args.proto_path)
    out_dir = resolve_path(args.out_dir)
    cache_dir = out_dir / "cache"
    pred_dir = out_dir / "predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Device: {device}")

    train_props = load_or_generate_proposals(
        args.train_split,
        args.train_proposals,
        cache_dir / f"proposals_{args.train_split}.csv",
        data_path,
        ann_path,
        proto_path,
        out_dir,
        args.force_generate_proposals,
    )
    val_props = load_or_generate_proposals(
        args.val_split,
        args.val_proposals,
        cache_dir / f"proposals_{args.val_split}.csv",
        data_path,
        ann_path,
        proto_path,
        out_dir,
        args.force_generate_proposals,
    )
    train_props = limit_frame(train_props, args.max_train_proposals, args.seed)
    val_props = limit_frame(val_props, args.max_val_proposals, args.seed + 1)
    train_labeled = label_proposals(
        train_props,
        ann_path,
        args.train_split,
        args.pos_tiou,
        args.neg_tiou,
        args.flap_neg_tiou,
        args.min_gt_duration,
    )
    train_labeled.to_csv(cache_dir / f"labels_{args.train_split}.csv", index=False)
    print_label_summary("train", train_labeled)
    train_df = train_labeled[train_labeled["label"] >= 0].reset_index(drop=True)
    train_df = limit_training_frame(train_df, args.max_train_samples, args.seed + 2)
    if train_df.empty or train_df["label"].nunique() < 2:
        raise RuntimeError("O conxunto de adestramento non contén ambas as clases.")

    model = make_model(args, device)
    teacher = make_model(args, device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    layer4_parameters = configure_contrastive_model(model)
    model.eval()
    with torch.no_grad():
        embedding_dim = int(model.fc_cls.in_features)
    projector = ProjectionHead(
        embedding_dim,
        args.projection_hidden,
        args.projection_dim,
    ).to(device)
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in named_trainable
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": layer4_parameters, "lr": args.lr_backbone},
            {"params": projector.parameters(), "lr": args.lr_projector},
        ],
        weight_decay=args.weight_decay,
    )
    class_weights = class_balanced_weights(train_df["label"].astype(int), args.cb_beta, device)
    train_loader = build_train_loader(train_df, args, data_path)

    rows = []
    print("[INFO] Avaliando modelo base en val...")
    scored_base = score_proposals(model, val_props, args, data_path, device)
    scored_base.to_csv(cache_dir / "val_scores_base.csv", index=False)
    base_metrics = evaluate_scored(scored_base, ann_path, args, pred_dir, "epoch000_base")
    rows.append({"epoch": 0, "phase": "base", **base_metrics})
    print(
        f"[BASE] mAP={base_metrics['mAP']:.6f} AP@0.5={base_metrics['AP@0.5']:.6f} "
        f"AP@0.7={base_metrics['AP@0.7']:.6f}"
    )

    best_map = float(base_metrics["mAP"])
    best_epoch = 0
    candidate_path = out_dir / "best_contrastive.pk"
    torch.save(model.state_dict(), out_dir / "initial_model.pk")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            teacher,
            projector,
            train_loader,
            optimizer,
            class_weights,
            initial_parameters,
            args,
            device,
        )
        scored = score_proposals(model, val_props, args, data_path, device)
        scored.to_csv(cache_dir / f"val_scores_epoch{epoch:03d}.csv", index=False)
        metrics = evaluate_scored(scored, ann_path, args, pred_dir, f"epoch{epoch:03d}")
        rows.append({"epoch": epoch, "phase": "contrastive_layer4", **train_metrics, **metrics})
        print(
            f"[EPOCH {epoch:03d}] loss={train_metrics['train_loss']:.5f} "
            f"ctr={train_metrics['train_contrastive']:.5f} "
            f"mAP={metrics['mAP']:.6f} AP@0.5={metrics['AP@0.5']:.6f} "
            f"AP@0.7={metrics['AP@0.7']:.6f}"
        )
        if metrics["mAP"] > best_map:
            best_map = float(metrics["mAP"])
            best_epoch = epoch
            torch.save(model.state_dict(), candidate_path)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "projector": projector.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "args": vars(args),
                },
                out_dir / "best_checkpoint.pt",
            )
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    summary = {
        "base_val_mAP": float(base_metrics["mAP"]),
        "best_val_mAP": best_map,
        "delta_mAP": best_map - float(base_metrics["mAP"]),
        "best_epoch": best_epoch,
        "candidate_saved": candidate_path.exists(),
        "train_samples": int(len(train_df)),
        "val_proposals": int(len(val_props)),
        "trainable_backbone_parameters": int(sum(p.numel() for p in layer4_parameters)),
        "projector_parameters": int(sum(p.numel() for p in projector.parameters())),
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[RESULTADO] base_mAP={base_metrics['mAP']:.6f} best_mAP={best_map:.6f} "
        f"best_epoch={best_epoch} candidate_saved={candidate_path.exists()}"
    )


if __name__ == "__main__":
    main()
