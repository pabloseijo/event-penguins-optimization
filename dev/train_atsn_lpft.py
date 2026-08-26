"""Head-only LP-FT experiment for the Augmented TSN classifier.

This script is intentionally separate from the production pipeline. It builds a
proposal-level training set from train/val proposals, freezes the ResNet/TSN
backbone, and updates only fc_cls with class-balanced focal loss and hard
negatives from flap annotations.

Run from event_penguins/:
    python dev/train_atsn_lpft.py --epochs 10 --out-dir tmp/atsn_lpft/head_only

Smoke test:
    python dev/train_atsn_lpft.py \
        --train-split val --val-split val \
        --train-proposals tmp/fft_phase_analysis/val/generated_proposals_val.csv \
        --val-proposals tmp/fft_phase_analysis/val/generated_proposals_val.csv \
        --max-train-proposals 64 --max-val-proposals 64 \
        --epochs 1 --num-workers 0 --out-dir tmp/atsn_lpft/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.augmented_tsn import AugmentedTsn
from src.classification import (
    ProposalDataset,
    create_img_representation,
    create_multiscale_decay_img_representation,
    create_polarity_img_representation,
)
from src.evaluation import DetectionsEvaluator, segment_iou
from src.proposals import ProposalGenerator
from src.prototype import build_ed_prototype
from src.utils import temporal_soft_nms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning head-only de ATSN con poucos datos."
    )
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--proto-path", default="tmp/prototype/ed_prototype.npy")
    parser.add_argument("--out-dir", default="tmp/atsn_lpft/head_only")

    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-proposals", default=None)
    parser.add_argument("--val-proposals", default=None)
    parser.add_argument("--force-generate-proposals", action="store_true")

    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--flap-neg-tiou", type=float, default=0.3)
    parser.add_argument("--min-gt-duration", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-adapter", type=float, default=1e-3)
    parser.add_argument("--lr-temporal-head", type=float, default=1e-3)
    parser.add_argument("--unfreeze-layer4", action="store_true")
    parser.add_argument("--train-input-adapter", action="store_true")
    parser.add_argument("--train-temporal-difference-head", action="store_true")
    parser.add_argument("--temporal-hidden-channels", type=int, default=64)
    parser.add_argument("--freeze-classifier", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--cb-beta", type=float, default=0.999)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--l2sp-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-positive-frac", type=float, default=0.40)
    parser.add_argument("--sampler-hard-neg-frac", type=float, default=0.40)
    parser.add_argument("--sampler-easy-neg-frac", type=float, default=0.20)
    parser.add_argument("--quiet-progress", action="store_true")

    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--temporal-scale-jitter", type=float, default=0.0)
    parser.add_argument("--temporal-shift-jitter", type=float, default=0.0)
    parser.add_argument("--sample-duration-jitter", type=float, default=0.0)
    parser.add_argument("--event-drop-prob", type=float, default=0.0)
    parser.add_argument(
        "--event-representation",
        choices=["signed_repeat", "polarity_adapter", "multiscale_decay_adapter"],
        default="signed_repeat",
    )
    parser.add_argument(
        "--multiscale-decay-factors",
        type=float,
        nargs=3,
        default=[0.4, 1.0, 2.0],
    )
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


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_recordings(ann_path: Path, split: str) -> set[str]:
    info_path = ann_path.parent / "recording_info.csv"
    recordings: set[str] = set()
    with open(info_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                recordings.add(row["timestamp"])
    return recordings


def ensure_prototype(data_path: Path, ann_path: Path, proto_path: Path) -> np.ndarray:
    if proto_path.exists():
        return np.load(proto_path)
    print("[INFO] Construindo prototipo ED desde train...")
    proto_path.parent.mkdir(parents=True, exist_ok=True)
    prototype = build_ed_prototype(str(data_path), str(ann_path), split="train")
    np.save(proto_path, prototype)
    return prototype


def generate_proposals(
    split: str,
    data_path: Path,
    ann_path: Path,
    proto_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    prototype = ensure_prototype(data_path, ann_path, proto_path)
    gen = ProposalGenerator(
        bin_width=0.033,
        percentile=1.0,
        nms_threshold=0.95,
        data_path=str(data_path),
        output_dir=str(out_dir / f"proposal_logs_{split}"),
        use_adaptive_lambda=True,
        use_spatial_compactness=True,
        use_noise_penalization=True,
        use_dispersed_noise=True,
        use_periodicity=True,
        prototype=prototype,
    )
    return gen.run(split=split).reset_index(drop=True)


def load_or_generate_proposals(
    split: str,
    explicit_path: str | None,
    cache_path: Path,
    data_path: Path,
    ann_path: Path,
    proto_path: Path,
    out_dir: Path,
    force_generate: bool,
) -> pd.DataFrame:
    if explicit_path is not None:
        proposals = pd.read_csv(resolve_path(explicit_path))
        print(f"[INFO] Propostas {split} cargadas de {explicit_path}: {len(proposals)}")
        return proposals.reset_index(drop=True)
    if cache_path.exists() and not force_generate:
        proposals = pd.read_csv(cache_path)
        print(f"[INFO] Propostas {split} cargadas da cache: {len(proposals)}")
        return proposals.reset_index(drop=True)

    print(f"[INFO] Xerando propostas para split={split}...")
    proposals = generate_proposals(split, data_path, ann_path, proto_path, out_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(cache_path, index=False)
    print(f"[INFO] Propostas {split} gardadas en {cache_path}: {len(proposals)}")
    return proposals


def max_tiou_seconds(t_start_us: float, t_end_us: float, segments_s: np.ndarray) -> float:
    if segments_s.size == 0:
        return 0.0
    t_start = float(t_start_us) / 1e6
    t_end = float(t_end_us) / 1e6
    inter = np.maximum(0.0, np.minimum(t_end, segments_s[:, 1]) - np.maximum(t_start, segments_s[:, 0]))
    union = (t_end - t_start) + (segments_s[:, 1] - segments_s[:, 0]) - inter
    valid = union > 0
    if not np.any(valid):
        return 0.0
    return float(np.max(inter[valid] / union[valid]))


def build_annotation_index(
    ann_path: Path,
    split: str,
    min_duration: float,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], set[str]]:
    split_recs = load_split_recordings(ann_path, split)
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    index: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    for rec_name, rec_data in ann["database"].items():
        if rec_name not in split_recs:
            continue
        rec_index: dict[str, dict[str, list[list[float]]]] = {}
        for roi_key, roi_anns in rec_data.get("annotations", {}).items():
            if roi_key == "null":
                continue
            ed_segments = []
            flap_segments = []
            for item in roi_anns:
                start, end = map(float, item["segment"])
                if end - start < min_duration:
                    continue
                if item["label"] == "ed":
                    ed_segments.append([start, end])
                elif item["label"] in {"adult_flap", "chick_flap"}:
                    flap_segments.append([start, end])
            rec_index[roi_key] = {
                "ed": ed_segments,
                "flap": flap_segments,
            }
        index[rec_name] = rec_index

    np_index: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for rec_name, rec_data in index.items():
        np_index[rec_name] = {}
        for roi_key, values in rec_data.items():
            np_index[rec_name][roi_key] = {
                label: np.asarray(segments, dtype=np.float64).reshape(-1, 2)
                for label, segments in values.items()
            }
    return np_index, split_recs


def roi_to_ann_key(roi_id: str) -> str:
    return str(int(str(roi_id)[1:]))


def label_proposals(
    proposals: pd.DataFrame,
    ann_path: Path,
    split: str,
    pos_tiou: float,
    neg_tiou: float,
    flap_neg_tiou: float,
    min_gt_duration: float,
) -> pd.DataFrame:
    ann_index, split_recs = build_annotation_index(ann_path, split, min_gt_duration)
    rows = []
    for idx, row in proposals.reset_index(drop=True).iterrows():
        rec_name = row["rec_name"]
        if rec_name not in split_recs:
            continue
        roi_key = roi_to_ann_key(row["roi_id"])
        segments = ann_index.get(rec_name, {}).get(roi_key, {})
        best_ed = max_tiou_seconds(row["t_start"], row["t_end"], segments.get("ed", np.empty((0, 2))))
        best_flap = max_tiou_seconds(row["t_start"], row["t_end"], segments.get("flap", np.empty((0, 2))))

        label = -1
        sample_kind = "ignored"
        if best_ed >= pos_tiou:
            label = 1
            sample_kind = "positive"
        elif best_ed < neg_tiou:
            label = 0
            sample_kind = "hard_negative" if best_flap >= flap_neg_tiou else "easy_negative"

        labeled = row.to_dict()
        labeled.update(
            {
                "proposal_index": idx,
                "label": label,
                "sample_kind": sample_kind,
                "best_ed_tiou": best_ed,
                "best_flap_tiou": best_flap,
            }
        )
        rows.append(labeled)

    if not rows:
        return pd.DataFrame(columns=list(proposals.columns) + [
            "proposal_index",
            "label",
            "sample_kind",
            "best_ed_tiou",
            "best_flap_tiou",
        ])
    return pd.DataFrame(rows).reset_index(drop=True)


class FineTuneProposalDataset(Dataset):
    _transform = ProposalDataset._transform

    def __init__(
        self,
        proposals: pd.DataFrame,
        augment_fraction: float,
        data_path: Path,
        num_tsn_samples: int,
        sample_duration_s: float,
        decay: float,
        require_label: bool,
        temporal_scale_jitter: float = 0.0,
        temporal_shift_jitter: float = 0.0,
        sample_duration_jitter: float = 0.0,
        event_drop_prob: float = 0.0,
        event_representation: str = "signed_repeat",
        multiscale_decay_factors: tuple[float, float, float] = (0.4, 1.0, 2.0),
    ) -> None:
        self.proposals = proposals.reset_index(drop=True)
        self.augment_fraction = augment_fraction
        self.data_path = str(data_path)
        self.num_tsn_samples = num_tsn_samples
        self.sample_duration_us = sample_duration_s * 1e6
        self.decay = decay
        self.require_label = require_label
        self.temporal_scale_jitter = temporal_scale_jitter
        self.temporal_shift_jitter = temporal_shift_jitter
        self.sample_duration_jitter = sample_duration_jitter
        self.event_drop_prob = event_drop_prob
        self.event_representation = event_representation
        self.multiscale_decays = tuple(
            float(decay * factor) for factor in multiscale_decay_factors
        )
        self._hf = None
        self._cached_roi_key = None
        self._cached_roi_events = None
        self._cached_roi_height = None
        self._cached_roi_width = None

    def _get_h5(self):
        if self._hf is None:
            self._hf = h5py.File(self.data_path, "r")
        return self._hf

    def _get_roi_data(self, rec_name: str, roi_id: str):
        key = (rec_name, roi_id)
        if key != self._cached_roi_key:
            roi_group = self._get_h5()[rec_name][roi_id]
            self._cached_roi_events = np.asarray(roi_group["events"])
            self._cached_roi_height = int(roi_group.attrs["height"])
            self._cached_roi_width = int(roi_group.attrs["width"])
            self._cached_roi_key = key
        return (
            self._cached_roi_events,
            self._cached_roi_height,
            self._cached_roi_width,
        )

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, idx: int):
        row = self.proposals.iloc[idx]
        t_start = float(row["t_start"])
        t_end = float(row["t_end"])
        rec_name = row["rec_name"]
        roi_id = row["roi_id"]

        if self.require_label and (self.temporal_scale_jitter > 0 or self.temporal_shift_jitter > 0):
            duration = max(t_end - t_start, 1.0)
            center = 0.5 * (t_start + t_end)
            scale = math.exp(np.random.uniform(-self.temporal_scale_jitter, self.temporal_scale_jitter))
            shift = np.random.uniform(-self.temporal_shift_jitter, self.temporal_shift_jitter) * duration
            duration *= scale
            center += shift
            t_start = max(0.0, center - 0.5 * duration)
            t_end = t_start + duration

        roi_events, height, width = self._get_roi_data(rec_name, roi_id)

        t_delta = t_end - t_start
        t_aug_start = t_start - t_delta * self.augment_fraction
        t_aug_end = t_end + t_delta * self.augment_fraction

        sample_duration_us = self.sample_duration_us
        if self.require_label and self.sample_duration_jitter > 0:
            sample_duration_us *= math.exp(
                np.random.uniform(-self.sample_duration_jitter, self.sample_duration_jitter)
            )
        img_times = torch.linspace(t_aug_start, t_aug_end, self.num_tsn_samples)
        t_imgs_start = img_times - 0.5 * sample_duration_us
        t_imgs_end = img_times + 0.5 * sample_duration_us

        i_start = np.searchsorted(roi_events[:, 2], t_imgs_start)
        i_end = np.searchsorted(roi_events[:, 2], t_imgs_end)

        images = []
        for start_index, end_index in zip(i_start, i_end):
            events = roi_events[start_index:end_index]
            if self.require_label and self.event_drop_prob > 0 and len(events) > 1:
                keep = np.random.random(len(events)) >= self.event_drop_prob
                if keep.any():
                    events = events[keep]
            if self.event_representation == "polarity_adapter":
                image = create_polarity_img_representation(
                    events, self.decay, height, width, self._transform
                )
            elif self.event_representation == "multiscale_decay_adapter":
                image = create_multiscale_decay_img_representation(
                    events, self.multiscale_decays, height, width, self._transform
                )
            else:
                image = create_img_representation(
                    events, self.decay, height, width, self._transform
                )
            images.append(image)
        imgs = torch.stack(images)

        if self.require_label:
            return imgs, torch.tensor(int(row["label"]), dtype=torch.long)
        return imgs, idx


def expanded_tsn_samples(num_tsn_samples: int, augment_factor: int) -> int:
    augment_fraction = 1.0 / augment_factor
    num_aug_samples = int(math.ceil(augment_fraction * num_tsn_samples))
    return num_tsn_samples + 2 * num_aug_samples


def make_model(args: argparse.Namespace, device: torch.device) -> AugmentedTsn:
    use_adapter = args.event_representation in {
        "polarity_adapter",
        "multiscale_decay_adapter",
    }
    source_channel = 1 if args.event_representation == "multiscale_decay_adapter" else 0
    model = AugmentedTsn(
        2,
        args.num_tsn_samples,
        args.augment_factor,
        use_input_adapter=use_adapter,
        input_adapter_source_channel=source_channel,
        use_temporal_difference_head=args.train_temporal_difference_head,
        temporal_hidden_channels=args.temporal_hidden_channels,
    )
    try:
        state = torch.load(resolve_path(args.model_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(resolve_path(args.model_path), map_location=device)
    if use_adapter or args.train_temporal_difference_head:
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        allowed_missing_prefixes = []
        if use_adapter:
            allowed_missing_prefixes.append("input_adapter.")
        if args.train_temporal_difference_head:
            allowed_missing_prefixes.append("temporal_difference_head.")
        missing = [
            key
            for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unexpected or missing:
            raise RuntimeError(f"Incompatible ATSN checkpoint: missing={missing} unexpected={unexpected}")
    else:
        model.load_state_dict(state)
    model.to(device)
    return model


def configure_trainable_parameters(model: AugmentedTsn, args: argparse.Namespace) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if not args.freeze_classifier:
        for param in model.fc_cls.parameters():
            param.requires_grad = True
    if args.unfreeze_layer4:
        for name, param in model.backbone.named_parameters():
            if name.startswith("layer4."):
                param.requires_grad = True
    if args.train_input_adapter:
        if not isinstance(model.input_adapter, nn.Conv2d):
            raise ValueError("--train-input-adapter requires an adapter event representation")
        for param in model.input_adapter.parameters():
            param.requires_grad = True
    if args.train_temporal_difference_head:
        if isinstance(model.temporal_difference_head, nn.Identity):
            raise ValueError("Temporal difference head was not created")
        for param in model.temporal_difference_head.parameters():
            param.requires_grad = True


def set_head_train_mode(model: AugmentedTsn) -> None:
    model.eval()
    if any(parameter.requires_grad for parameter in model.fc_cls.parameters()):
        model.dropout.train()
        model.fc_cls.train()
    if any(parameter.requires_grad for parameter in model.input_adapter.parameters()):
        model.input_adapter.train()
    if any(parameter.requires_grad for parameter in model.temporal_difference_head.parameters()):
        model.temporal_difference_head.train()


def class_balanced_weights(labels: Iterable[int], beta: float, device: torch.device) -> torch.Tensor:
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights = weights / weights.sum() * 2.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    weights = class_weights[targets]
    return (weights * torch.pow(1.0 - pt, gamma) * ce).mean()


def pairwise_rank_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    scores = logits[:, 1] - logits[:, 0]
    positive = scores[targets == 1]
    negative = scores[targets == 0]
    if len(positive) == 0 or len(negative) == 0:
        return scores.sum() * 0.0
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def distillation_loss(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = F.log_softmax(logits / temperature, dim=1)
    teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean") * (temperature ** 2)


def l2sp_loss(model: AugmentedTsn, initial_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name in initial_parameters:
            terms.append((parameter - initial_parameters[name]).pow(2).sum())
    if not terms:
        return next(model.parameters()).sum() * 0.0
    return torch.stack(terms).sum()


def sampler_for_training(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> WeightedRandomSampler | None:
    if args.no_weighted_sampler:
        return None
    group_targets = {
        "positive": args.sampler_positive_frac,
        "hard_negative": args.sampler_hard_neg_frac,
        "easy_negative": args.sampler_easy_neg_frac,
    }
    counts = train_df["sample_kind"].value_counts().to_dict()
    active_total = sum(group_targets[k] for k in group_targets if counts.get(k, 0) > 0)
    weights = []
    for kind in train_df["sample_kind"]:
        if counts.get(kind, 0) == 0:
            weights.append(0.0)
            continue
        target = group_targets.get(kind, 0.0) / max(active_total, 1e-12)
        weights.append(target / counts[kind])
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_loader(
    df: pd.DataFrame,
    args: argparse.Namespace,
    data_path: Path,
    require_label: bool,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    dataset = FineTuneProposalDataset(
        proposals=df,
        augment_fraction=1.0 / args.augment_factor,
        data_path=data_path,
        num_tsn_samples=expanded_tsn_samples(args.num_tsn_samples, args.augment_factor),
        sample_duration_s=args.sample_duration,
        decay=args.decay,
        require_label=require_label,
        temporal_scale_jitter=args.temporal_scale_jitter if require_label else 0.0,
        temporal_shift_jitter=args.temporal_shift_jitter if require_label else 0.0,
        sample_duration_jitter=args.sample_duration_jitter if require_label else 0.0,
        event_drop_prob=args.event_drop_prob if require_label else 0.0,
        event_representation=args.event_representation,
        multiscale_decay_factors=tuple(args.multiscale_decay_factors),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: AugmentedTsn,
    teacher: AugmentedTsn | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    initial_parameters: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    set_head_train_mode(model)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    component_totals = {"classification": 0.0, "rank": 0.0, "distill": 0.0, "l2sp": 0.0}

    progress = tqdm(loader, desc="train", leave=False, disable=args.quiet_progress)
    for imgs, labels in progress:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        classification = focal_loss(logits, labels, class_weights, args.focal_gamma)
        ranking = pairwise_rank_loss(logits, labels)
        if teacher is not None and args.distill_weight > 0:
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            distill = distillation_loss(logits, teacher_logits, args.distill_temperature)
        else:
            distill = logits.sum() * 0.0
        l2sp = l2sp_loss(model, initial_parameters) if args.l2sp_weight > 0 else logits.sum() * 0.0
        loss = (
            args.classification_weight * classification
            + args.rank_weight * ranking
            + args.distill_weight * distill
            + args.l2sp_weight * l2sp
        )
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
        optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += float(loss.item()) * int(labels.numel())
        running_correct += int((preds == labels).sum().item())
        running_total += int(labels.numel())
        for name, value in (
            ("classification", classification),
            ("rank", ranking),
            ("distill", distill),
            ("l2sp", l2sp),
        ):
            component_totals[name] += float(value.item()) * int(labels.numel())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = {
        "train_loss": running_loss / max(running_total, 1),
        "train_acc": running_correct / max(running_total, 1),
    }
    metrics.update(
        {f"train_{name}": total / max(running_total, 1) for name, total in component_totals.items()}
    )
    return metrics


@torch.no_grad()
def score_proposals(
    model: AugmentedTsn,
    proposals: pd.DataFrame,
    args: argparse.Namespace,
    data_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    loader = build_loader(
        proposals,
        args,
        data_path,
        require_label=False,
        shuffle=False,
        sampler=None,
    )
    scores = np.zeros(len(proposals), dtype=np.float64)
    progress = tqdm(loader, desc="score", leave=False, disable=args.quiet_progress)
    for imgs, indices in progress:
        logits = model(imgs.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        scores[np.asarray(indices, dtype=np.int64)] = probs
    out = proposals.reset_index(drop=True).copy()
    out["cnn_score"] = scores
    return out


def load_gt(valid_sequences: list[str], ann_path: Path) -> pd.DataFrame:
    with open(ann_path, encoding="utf-8") as f:
        db = json.load(f)["database"]

    rows = []
    for rec, value in db.items():
        if rec not in valid_sequences:
            continue
        for roi, annotations in value["annotations"].items():
            if roi == "null":
                continue
            for ann in annotations:
                if ann["label"] != "ed":
                    continue
                start, end = map(float, ann["segment"])
                if end - start < 2.0:
                    continue
                rows.append(
                    {
                        "video-id": f"{rec}_{int(roi)}",
                        "rec": rec,
                        "roi": int(roi),
                        "t-start": start,
                        "t-end": end,
                    }
                )
    return pd.DataFrame(rows)


def predictions_to_df(prediction: dict) -> pd.DataFrame:
    rows = []
    for rec, rois in prediction["results"].items():
        for roi, detections in rois.items():
            for det in detections:
                start, end = det["segment"]
                if end - start < 2:
                    continue
                rows.append(
                    {
                        "video-id": f"{rec}_{int(roi)}",
                        "rec": rec,
                        "roi": int(roi),
                        "t-start": float(start),
                        "t-end": float(end),
                        "score": float(det["score"]),
                    }
                )
    return pd.DataFrame(rows)


def best_iou_by_gt(gt: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
    if pred.empty:
        return np.zeros(len(gt))
    grouped = {key: grp.reset_index(drop=True) for key, grp in pred.groupby("video-id")}
    best = []
    for _, row in gt.iterrows():
        candidates = grouped.get(row["video-id"])
        if candidates is None or candidates.empty:
            best.append(0.0)
            continue
        iou = segment_iou(
            np.asarray([row["t-start"], row["t-end"]]),
            candidates[["t-start", "t-end"]].to_numpy(dtype=np.float64),
        )
        best.append(float(iou.max()))
    return np.asarray(best, dtype=np.float64)


def build_prediction(
    scored: pd.DataFrame,
    min_ed_score: float,
    args: argparse.Namespace,
) -> dict:
    result: dict[str, dict[int, list[dict]]] = {
        rec: {int(roi[1:]): [] for roi in grp["roi_id"].unique()}
        for rec, grp in scored.groupby("rec_name")
    }
    selected = scored[scored["cnn_score"] >= min_ed_score].copy()
    if selected.empty:
        return {"version": "VERSION 0.0", "results": result}

    durations_s = (selected["t_end"].to_numpy() - selected["t_start"].to_numpy()) / 1e6
    excess = np.maximum(0.0, durations_s - args.duration_dmax)
    selected["final_score"] = selected["cnn_score"].to_numpy() * np.exp(-excess / args.duration_sigma)

    for (rec, roi_id), grp in selected.groupby(["rec_name", "roi_id"]):
        arr = grp[["t_start", "t_end", "final_score"]].to_numpy(dtype=np.float64)
        processed = temporal_soft_nms(
            arr,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
        )
        result[rec][int(roi_id[1:])] = [
            {
                "label": "ed",
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(score),
            }
            for start, end, score in processed
            if (float(end) - float(start)) / 1e6 >= 2.0
        ]
    return {"version": "VERSION 0.0", "results": result}


def evaluate_scored(
    scored: pd.DataFrame,
    ann_path: Path,
    args: argparse.Namespace,
    pred_dir: Path,
    prefix: str,
) -> dict:
    valid_sequences = sorted(scored["rec_name"].unique())
    gt = load_gt(valid_sequences, ann_path)
    best_metrics = None

    for min_score in args.min_ed_score:
        prediction = build_prediction(scored, min_score, args)
        pred_path = pred_dir / f"{prefix}_min{min_score:.3f}.json"
        pred_path.write_text(json.dumps(prediction), encoding="utf-8")
        evaluator = DetectionsEvaluator(
            ground_truth_filename=str(ann_path),
            prediction_filename=str(pred_path),
            tiou_thresholds=np.asarray(args.tiou, dtype=np.float64),
            valid_labels="ed",
            valid_sequences=valid_sequences,
            min_duration=2.0,
        )
        mean_ap = evaluator.run()
        pred_df = predictions_to_df(prediction)
        best_iou = best_iou_by_gt(gt, pred_df)
        metrics = {
            "prefix": prefix,
            "min_ed_score": float(min_score),
            "n_pred": int(len(pred_df)),
            "mAP": float(mean_ap),
            "AP@0.1": float(evaluator.mAP[0]) if len(evaluator.mAP) > 0 else float("nan"),
            "AP@0.3": float(evaluator.mAP[1]) if len(evaluator.mAP) > 1 else float("nan"),
            "AP@0.5": float(evaluator.mAP[2]) if len(evaluator.mAP) > 2 else float("nan"),
            "AP@0.7": float(evaluator.mAP[3]) if len(evaluator.mAP) > 3 else float("nan"),
            "recall@0.1": float((best_iou >= 0.1).mean()) if len(best_iou) else float("nan"),
            "recall@0.3": float((best_iou >= 0.3).mean()) if len(best_iou) else float("nan"),
            "recall@0.5": float((best_iou >= 0.5).mean()) if len(best_iou) else float("nan"),
            "missed@0.1": int((best_iou < 0.1).sum()) if len(best_iou) else 0,
            "missed@0.5": int((best_iou < 0.5).sum()) if len(best_iou) else 0,
        }
        if best_metrics is None or metrics["mAP"] > best_metrics["mAP"]:
            best_metrics = metrics

    assert best_metrics is not None
    return best_metrics


def limit_frame(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def limit_training_frame(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    targets = {
        "positive": 0.40,
        "hard_negative": 0.40,
        "easy_negative": 0.20,
    }
    parts = []
    used = 0
    for kind, frac in targets.items():
        group = df[df["sample_kind"] == kind]
        if group.empty:
            continue
        take = min(len(group), max(1, int(round(max_rows * frac))))
        parts.append(group.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
        used += take

    selected = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame(columns=df.columns)
    if used < max_rows:
        remaining = df.drop(index=selected.index, errors="ignore")
        if not remaining.empty:
            take = min(len(remaining), max_rows - used)
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(n=take, random_state=int(rng.integers(0, 1_000_000))),
                ],
                ignore_index=False,
            )

    selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return selected


def print_label_summary(name: str, labels_df: pd.DataFrame) -> None:
    counts = labels_df["sample_kind"].value_counts().to_dict()
    usable = labels_df[labels_df["label"] >= 0]
    print(
        f"[INFO] {name}: propostas={len(labels_df)} usable={len(usable)} "
        f"pos={counts.get('positive', 0)} hard_neg={counts.get('hard_negative', 0)} "
        f"easy_neg={counts.get('easy_negative', 0)} ignored={counts.get('ignored', 0)}"
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.num_workers = 0
        args.max_train_proposals = args.max_train_proposals or 64
        args.max_train_samples = args.max_train_samples or 32
        args.max_val_proposals = args.max_val_proposals or 64

    set_seed(args.seed)
    data_path = resolve_path(args.data_path)
    ann_path = resolve_path(args.ann_path)
    proto_path = resolve_path(args.proto_path)
    model_path = resolve_path(args.model_path)
    out_dir = resolve_path(args.out_dir)
    cache_dir = out_dir / "cache"
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

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
    val_labeled = label_proposals(
        val_props,
        ann_path,
        args.val_split,
        args.pos_tiou,
        args.neg_tiou,
        args.flap_neg_tiou,
        args.min_gt_duration,
    )
    train_labeled.to_csv(cache_dir / f"labels_{args.train_split}.csv", index=False)
    val_labeled.to_csv(cache_dir / f"labels_{args.val_split}.csv", index=False)

    print_label_summary("train", train_labeled)
    print_label_summary("val", val_labeled)

    train_df = train_labeled[train_labeled["label"] >= 0].reset_index(drop=True)
    if args.max_train_samples is not None:
        train_df = limit_training_frame(train_df, args.max_train_samples, args.seed + 2)
    if train_df.empty or train_df["label"].nunique() < 2:
        raise RuntimeError("O conxunto de adestramento non ten positivas e negativas suficientes.")

    model = make_model(args, device)
    teacher = make_model(args, device) if args.distill_weight > 0 else None
    if teacher is not None:
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()
    configure_trainable_parameters(model, args)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    parameter_groups = []
    classifier_parameters = [parameter for parameter in model.fc_cls.parameters() if parameter.requires_grad]
    if classifier_parameters:
        parameter_groups.append({"params": classifier_parameters, "lr": args.lr_head})
    if args.unfreeze_layer4:
        layer4_parameters = [
            parameter
            for name, parameter in model.backbone.named_parameters()
            if name.startswith("layer4.") and parameter.requires_grad
        ]
        parameter_groups.append({"params": layer4_parameters, "lr": args.lr_backbone})
    adapter_parameters = [
        parameter for parameter in model.input_adapter.parameters() if parameter.requires_grad
    ]
    if adapter_parameters:
        parameter_groups.append({"params": adapter_parameters, "lr": args.lr_adapter})
    temporal_parameters = [
        parameter
        for parameter in model.temporal_difference_head.parameters()
        if parameter.requires_grad
    ]
    if temporal_parameters:
        parameter_groups.append({"params": temporal_parameters, "lr": args.lr_temporal_head})
    if not parameter_groups:
        raise ValueError("No trainable parameters were selected")
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    class_weights = class_balanced_weights(train_df["label"].astype(int), args.cb_beta, device)
    sampler = sampler_for_training(train_df, args)
    train_loader = build_loader(
        train_df,
        args,
        data_path,
        require_label=True,
        shuffle=sampler is None,
        sampler=sampler,
    )

    rows = []
    print("[INFO] Avaliando modelo base en val...")
    scored_base = score_proposals(model, val_props, args, data_path, device)
    scored_base.to_csv(cache_dir / "val_scores_base.csv", index=False)
    base_metrics = evaluate_scored(scored_base, ann_path, args, pred_dir, "epoch000_base")
    rows.append({"epoch": 0, "phase": "base", "train_loss": np.nan, "train_acc": np.nan, **base_metrics})
    print(
        f"[BASE] mAP={base_metrics['mAP']:.4f} AP@0.5={base_metrics['AP@0.5']:.4f} "
        f"recall@0.5={base_metrics['recall@0.5']:.4f} n_pred={base_metrics['n_pred']}"
    )

    best_map = base_metrics["mAP"]
    best_epoch = 0
    best_path = out_dir / "best_head_only.pk"
    torch.save(model.state_dict(), out_dir / "initial_model.pk")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            teacher,
            train_loader,
            optimizer,
            class_weights,
            initial_parameters,
            args,
            device,
        )
        scored = score_proposals(model, val_props, args, data_path, device)
        scored.to_csv(cache_dir / f"val_scores_epoch{epoch:03d}.csv", index=False)
        eval_metrics = evaluate_scored(scored, ann_path, args, pred_dir, f"epoch{epoch:03d}")
        phase = "lpft_layer4_rank" if args.unfreeze_layer4 else "lpft_head"
        row = {"epoch": epoch, "phase": phase, **train_metrics, **eval_metrics}
        rows.append(row)

        print(
            f"[EPOCH {epoch:03d}] loss={train_metrics['train_loss']:.4f} "
            f"acc={train_metrics['train_acc']:.4f} mAP={eval_metrics['mAP']:.4f} "
            f"AP@0.5={eval_metrics['AP@0.5']:.4f} recall@0.5={eval_metrics['recall@0.5']:.4f}"
        )

        if eval_metrics["mAP"] > best_map:
            best_map = eval_metrics["mAP"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": eval_metrics,
                    "args": vars(args),
                },
                out_dir / "best_checkpoint.pt",
            )

        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    if not best_path.exists():
        torch.save(model.state_dict(), best_path)

    summary = {
        "best_epoch": best_epoch,
        "best_val_mAP": best_map,
        "base_val_mAP": base_metrics["mAP"],
        "delta_mAP": best_map - base_metrics["mAP"],
        "train_samples": int(len(train_df)),
        "train_positive": int((train_df["label"] == 1).sum()),
        "train_negative": int((train_df["label"] == 0).sum()),
        "train_hard_negative": int((train_df["sample_kind"] == "hard_negative").sum()),
        "val_proposals": int(len(val_props)),
        "best_model": str(best_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[RESULTADO] best_epoch={best_epoch} base_mAP={base_metrics['mAP']:.4f} best_mAP={best_map:.4f}")
    print(f"[INFO] Saidas gardadas en {out_dir}")


if __name__ == "__main__":
    main()
