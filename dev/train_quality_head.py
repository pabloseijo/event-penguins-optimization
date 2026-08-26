"""Train a proposal quality/boundary head on top of frozen ATSN embeddings.

This experiment targets the mAP gap revealed by the proposal oracle: proposals
already contain high-quality segments, but CNN class scores do not rank them by
temporal quality. Unlike head-only LP-FT, this script uses a continuous tIoU
target and leaves the ATSN backbone/classifier untouched.

The optional boundary branch predicts normalized start/end offsets for each
proposal, following the same idea as temporal boundary regression: learn to turn
noisy proposals near a GT action into better localized detections instead of
tuning hand-picked trimming thresholds.

Run from event_penguins/:
    python dev/train_quality_head.py \
        --train-proposals tmp/atsn_lpft/head_only_pilot/cache/proposals_train.csv \
        --val-proposals tmp/atsn_lpft/head_only_pilot/cache/proposals_val.csv \
        --out-dir tmp/quality_head/atsn_frozen
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.augmented_tsn import AugmentedTsn
from src.classification import ProposalDataset
from src.evaluation import DetectionsEvaluator, segment_iou
from src.rank_sort_loss import rank_sort_loss
from src.temporalmaxer_lite import temporal_aware_normalization_perturbation
from src.utils import temporal_soft_nms


ROOT = Path(__file__).resolve().parents[1]


NUMERIC_COLUMNS = [
    "cnn_score",
    "cnn_margin",
    "proposal_score_robust",
    "source_score_robust",
    "proposal_roi_rank",
    "proposal_roi_z",
    "cnn_roi_rank",
    "cnn_roi_z",
    "duration_log",
    "duration_penalty",
    "cnn_x_prop",
    "cnn_x_duration_penalty",
    "is_lattice",
    "is_base_variant",
    "is_expand_left",
    "is_expand_right",
    "is_expand_both",
    "is_trim_left",
    "is_trim_right",
    "is_trim_both",
    "is_shift",
    "is_center_duration",
    "variant_param",
    "center_duration_s",
    "family_size_log",
    "family_cnn_rank",
    "family_cnn_delta_max",
    "family_cnn_delta_mean",
    "family_prop_rank",
    "family_prop_delta_max",
    "family_duration_rank",
    "family_duration_log_ratio",
    "family_base_cnn_max",
    "family_expand_cnn_max",
    "family_trim_cnn_max",
    "family_shift_cnn_max",
    "family_center_cnn_max",
    "family_expand_minus_base",
    "family_trim_minus_base",
    "family_shift_minus_base",
    "family_center_minus_base",
    "family_expand_left_cnn_max",
    "family_expand_right_cnn_max",
    "family_expand_both_cnn_max",
    "family_expand_lr_delta",
    "family_expand_both_minus_sides",
]


@dataclass(frozen=True)
class QualityConfig:
    name: str
    hidden: int
    dropout: float
    qfl_weight: float
    rank_weight: float
    distill_weight: float
    high_pos_frac: float
    pos_frac: float
    semi_frac: float
    hard_neg_frac: float
    easy_neg_frac: float


CONFIGS = [
    QualityConfig("qhead_qfl_rank", 256, 0.20, 1.0, 1.0, 0.0, 0.10, 0.25, 0.25, 0.25, 0.15),
    QualityConfig("qhead_qfl_rank_distill", 256, 0.20, 1.0, 1.0, 0.05, 0.10, 0.25, 0.25, 0.25, 0.15),
    QualityConfig("qhead_qfl_only", 256, 0.20, 1.0, 0.0, 0.0, 0.10, 0.30, 0.25, 0.20, 0.15),
    QualityConfig("qhead_iou70_rank", 256, 0.20, 1.0, 1.25, 0.0, 0.30, 0.20, 0.15, 0.25, 0.10),
    QualityConfig("qhead_iou70_focus", 384, 0.25, 1.2, 1.50, 0.0, 0.35, 0.15, 0.15, 0.25, 0.10),
    QualityConfig("qhead_hardneg_rank", 256, 0.25, 1.0, 1.25, 0.0, 0.10, 0.25, 0.20, 0.35, 0.10),
    QualityConfig("qhead_hardneg_conservative", 256, 0.20, 1.0, 0.75, 0.02, 0.10, 0.30, 0.20, 0.30, 0.10),
    QualityConfig("qhead_multitiou", 256, 0.20, 1.0, 0.50, 0.02, 0.10, 0.30, 0.25, 0.20, 0.15),
    QualityConfig("qhead_multitiou_rank", 384, 0.25, 1.0, 1.00, 0.02, 0.10, 0.25, 0.25, 0.25, 0.15),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-ATSN proposal quality head.")
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--train-proposals", default=None)
    parser.add_argument("--val-proposals", required=True)
    parser.add_argument("--train-repr", default=None)
    parser.add_argument("--val-repr", default=None)
    parser.add_argument(
        "--drop-temporal-descriptor-groups",
        nargs="*",
        choices=["norm", "adjacent_cos", "delta", "center_cos", "spectral"],
        default=[],
        help="Ablate named groups from the 47 descriptors appended by extract_temporal_descriptors.py.",
    )
    parser.add_argument("--out-dir", default="tmp/quality_head/atsn_frozen")
    parser.add_argument(
        "--reuse-labeled-cache",
        action="store_true",
        help="Reuse train/val quality-label CSVs already written in the output cache.",
    )
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume each config from its epoch-level last checkpoint.",
    )
    parser.add_argument("--eval-checkpoint", default=None)
    parser.add_argument("--eval-label", default="eval")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Write checkpoint scores without running Soft-NMS/AP evaluation.",
    )
    parser.add_argument(
        "--skip-baseline-evaluation",
        action="store_true",
        help="Skip the unchanged frozen-CNN baseline during training runs.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional subset of config names to train, e.g. qhead_qfl_only.",
    )

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--high-pos-tiou", type=float, default=0.7)
    parser.add_argument("--rank-pos-tiou", type=float, default=None)
    parser.add_argument("--rank-output-threshold", type=float, default=None)
    parser.add_argument("--local-rank-weight", type=float, default=0.0)
    parser.add_argument("--local-rank-min-tiou", type=float, default=0.1)
    parser.add_argument("--local-rank-min-gap", type=float, default=0.15)
    parser.add_argument("--local-rank-pairs-per-gt", type=int, default=128)
    parser.add_argument("--high-iou-loss-weight", type=float, default=0.0)
    parser.add_argument("--multi-quality-head", action="store_true")
    parser.add_argument("--quality-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--semi-tiou", type=float, default=0.1)
    parser.add_argument("--flap-neg-tiou", type=float, default=0.3)
    parser.add_argument("--hard-score", type=float, default=0.3)
    parser.add_argument("--hard-cnn-rank", type=float, default=1.1)
    parser.add_argument("--hard-neg-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-neg-top-frac", type=float, default=1.0)
    parser.add_argument(
        "--oof-hardness-csv",
        default=None,
        help="Cross-fitted proposal scores used only to mine training negatives.",
    )
    parser.add_argument(
        "--oof-hardness-threshold",
        type=float,
        default=0.1,
        help="Promote negative proposals at or above this OOF score to hard negatives.",
    )
    parser.add_argument("--hard-rank-neg-prob", type=float, default=0.75)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument(
        "--head-architecture",
        choices=["mlp", "temporal_completeness", "surrounding_contrastive"],
        default="mlp",
        help=(
            "Quality-head architecture. temporal_completeness preserves the ATSN "
            "start/main/end roles and models their relations with shared weights."
        ),
    )
    parser.add_argument(
        "--decoupled-boundary-head",
        action="store_true",
        help="Use independent quality and boundary towers to avoid task interference.",
    )
    parser.add_argument("--min-gt-duration", type=float, default=2.0)
    parser.add_argument(
        "--context-roles",
        nargs="*",
        choices=["previous", "next", "expanded", "center", "fixed_center"],
        default=[],
        help="Extra label-free temporal windows scored by the frozen ATSN.",
    )
    parser.add_argument(
        "--context-window-scale",
        type=float,
        default=1.0,
        help="Relative context duration, or fixed seconds for the fixed_center role.",
    )
    parser.add_argument("--train-context-dir", default=None)
    parser.add_argument("--val-context-dir", default=None)
    parser.add_argument(
        "--context-feature-mode",
        choices=["logits", "embeddings"],
        default="logits",
        help="Store scalar ATSN logits or full context embeddings for the quality head.",
    )

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--repr-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--recording-balanced-sampling", action="store_true")
    parser.add_argument(
        "--group-dro",
        action="store_true",
        help="Optimize recording-level worst-group quality risk with GroupDRO.",
    )
    parser.add_argument("--group-dro-eta", type=float, default=0.01)
    parser.add_argument(
        "--tanp-sigma",
        type=float,
        default=0.0,
        help=(
            "TANP strength for the ordered ATSN start/main/end embedding roles; "
            "zero preserves the original training path."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--weight-average-start-epoch",
        type=int,
        default=0,
        help="Uniformly average epoch-end weights from this 1-based epoch; zero disables it.",
    )
    parser.add_argument(
        "--weight-average-interval",
        type=int,
        default=1,
        help="Number of epochs between weight-average snapshots.",
    )
    parser.add_argument("--qfl-beta", type=float, default=2.0)
    parser.add_argument(
        "--rank-sort-weight",
        type=float,
        default=0.0,
        help="Auxiliary Rank & Sort loss weight; zero disables it.",
    )
    parser.add_argument("--rank-sort-delta", type=float, default=0.5)
    parser.add_argument("--rank-sort-max-positives", type=int, default=128)
    parser.add_argument("--rank-sort-max-negatives", type=int, default=384)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-min-tiou", type=float, default=0.3)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--use-boundary-refinement", dest="use_boundary_refinement", action="store_true", default=True)
    parser.add_argument("--no-boundary-refinement", dest="use_boundary_refinement", action="store_false")
    parser.add_argument("--pair-batch-size", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet-progress", action="store_true")

    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)

    parser.add_argument("--min-score", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument(
        "--score-cols",
        nargs="+",
        default=None,
        help="Optional subset of scored columns to evaluate.",
    )
    parser.add_argument(
        "--pre-nms-topk-per-roi",
        type=int,
        default=0,
        help="Keep only the top-K scored proposals per ROI before Soft-NMS. 0 disables this filter.",
    )
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])

    parser.add_argument("--max-train-proposals", type=int, default=None)
    parser.add_argument("--max-val-proposals", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_labeled_frame(csv_path: Path) -> pd.DataFrame:
    """Load a prepared label frame through a faster binary cache when available."""
    pickle_path = csv_path.with_suffix(".pkl")
    if pickle_path.exists() and pickle_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return pd.read_pickle(pickle_path)
    frame = pd.read_csv(csv_path)
    temporary = pickle_path.with_suffix(".pkl.tmp")
    frame.to_pickle(temporary)
    temporary.replace(pickle_path)
    return frame


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expanded_tsn_samples(num_tsn_samples: int, augment_factor: int) -> int:
    augment_fraction = 1.0 / augment_factor
    num_aug_samples = int(math.ceil(augment_fraction * num_tsn_samples))
    return num_tsn_samples + 2 * num_aug_samples


def load_split_recordings(ann_path: Path, split: str) -> set[str]:
    info_path = ann_path.parent / "recording_info.csv"
    recordings: set[str] = set()
    with open(info_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split == "trainval":
                if row["split"] in {"train", "val"}:
                    recordings.add(row["timestamp"])
            elif row["split"] == split:
                recordings.add(row["timestamp"])
    return recordings


def split_from_proposals(proposals: pd.DataFrame, ann_path: Path) -> str:
    recs = set(proposals["rec_name"].unique())
    for split in ["train", "val", "test"]:
        if recs <= load_split_recordings(ann_path, split):
            return split
    # Folds de validación cruzada que usan gravacións de train+val
    if recs <= load_split_recordings(ann_path, "trainval"):
        return "trainval"
    raise ValueError("Could not infer split from proposal recording names.")


def make_model(args: argparse.Namespace, device: torch.device) -> AugmentedTsn:
    model = AugmentedTsn(2, args.num_tsn_samples, args.augment_factor)
    try:
        state = torch.load(resolve_path(args.model_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(resolve_path(args.model_path), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def forward_embedding(model: AugmentedTsn, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_segs = imgs.shape[1]
    x = imgs.reshape((-1,) + imgs.shape[2:])
    x = model.backbone(x)["features"]
    x = model.avg_pool(x)
    x = x.reshape((-1, num_segs) + x.shape[1:])

    a = model.num_augment
    start = model.consensus(x[:, :a]).squeeze(1)
    main = model.consensus(x[:, a:num_segs - a]).squeeze(1)
    end = model.consensus(x[:, num_segs - a:]).squeeze(1)
    emb = torch.cat((start, main, end), dim=1).flatten(1)
    logits = model.fc_cls(model.dropout(emb))
    return emb, logits


def collect_or_load_representations(
    proposals: pd.DataFrame,
    repr_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if repr_path.exists():
        data = np.load(repr_path, allow_pickle=True)
        embeddings = data["embeddings"]
        logits = data["logits"]
        if len(embeddings) != len(proposals) or len(logits) != len(proposals):
            raise ValueError(f"{repr_path} does not match proposals length {len(proposals)}")
        print(f"[INFO] Representacións cargadas: {repr_path} emb={embeddings.shape} logits={logits.shape}")
        return embeddings, logits

    print(f"[INFO] Extraendo embeddings ATSN: {repr_path}")
    model = make_model(args, device)
    dataset = ProposalDataset(
        proposals.reset_index(drop=True),
        augment_fraction=1.0 / args.augment_factor,
        data_path=str(resolve_path(args.data_path)),
        num_tsn_samples=expanded_tsn_samples(args.num_tsn_samples, args.augment_factor),
        sample_duration=args.sample_duration * 1e6,
        decay=args.decay,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.repr_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_embeddings = []
    all_logits = []
    progress = tqdm(loader, desc="repr", disable=args.quiet_progress)
    for imgs, *_ in progress:
        emb, logits = forward_embedding(model, imgs.to(device, non_blocking=True))
        all_embeddings.append(emb.detach().cpu().numpy().astype(np.float16))
        all_logits.append(logits.detach().cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(all_embeddings, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    repr_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(repr_path, embeddings=embeddings, logits=logits)
    print(f"[INFO] Representacións gardadas: {repr_path} emb={embeddings.shape} logits={logits.shape}")
    return embeddings, logits


def proposal_fingerprint(proposals: pd.DataFrame) -> str:
    columns = ["rec_name", "roi_id", "t_start", "t_end"]
    hashed = pd.util.hash_pandas_object(proposals[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def transform_context_proposals(
    proposals: pd.DataFrame,
    role: str,
    scale: float,
) -> pd.DataFrame:
    if scale <= 0:
        raise ValueError("--context-window-scale must be positive")
    out = proposals.copy()
    start = proposals["t_start"].to_numpy(dtype=np.float64)
    end = proposals["t_end"].to_numpy(dtype=np.float64)
    duration = np.maximum(end - start, 1.0)

    if role == "previous":
        context_start = np.maximum(0.0, start - scale * duration)
        context_end = np.maximum(start, context_start + 1.0)
    elif role == "next":
        context_start = end
        context_end = end + scale * duration
    elif role == "expanded":
        padding = 0.5 * scale * duration
        context_start = np.maximum(0.0, start - padding)
        context_end = end + padding
    elif role == "center":
        center = 0.5 * (start + end)
        context_duration = np.maximum(0.5 * scale * duration, 1.0)
        context_start = np.maximum(0.0, center - 0.5 * context_duration)
        context_end = context_start + context_duration
    elif role == "fixed_center":
        center = 0.5 * (start + end)
        context_duration = scale * 1e6
        context_start = np.maximum(0.0, center - 0.5 * context_duration)
        context_end = context_start + context_duration
    else:
        raise ValueError(f"Unknown context role: {role}")

    out["t_start"] = context_start
    out["t_end"] = context_end
    return out.reset_index(drop=True)


@torch.no_grad()
def collect_or_load_context_logits(
    proposals: pd.DataFrame,
    context_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    roles = list(dict.fromkeys(args.context_roles))
    if not roles:
        return {}

    context_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, np.ndarray] = {}
    model = None
    for role in roles:
        transformed = transform_context_proposals(proposals, role, args.context_window_scale)
        fingerprint = proposal_fingerprint(transformed)
        cache_path = context_dir / f"{role}_scale{args.context_window_scale:g}_logits.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            logits = data["logits"]
            cached_fingerprint = str(data["fingerprint"].item()) if "fingerprint" in data else ""
            if len(logits) != len(proposals) or cached_fingerprint != fingerprint:
                raise ValueError(f"{cache_path} does not match the requested context proposals")
            print(f"[INFO] Contexto cargado: role={role} path={cache_path} logits={logits.shape}")
            outputs[role] = logits
            continue

        if model is None:
            model = make_model(args, device)
        extraction_order = transformed.sort_values(
            ["rec_name", "roi_id", "t_start", "t_end"],
            kind="stable",
        ).index.to_numpy(dtype=np.int64)
        extraction_frame = transformed.loc[extraction_order].reset_index(drop=True)
        dataset = ProposalDataset(
            extraction_frame,
            augment_fraction=1.0 / args.augment_factor,
            data_path=str(resolve_path(args.data_path)),
            num_tsn_samples=expanded_tsn_samples(args.num_tsn_samples, args.augment_factor),
            sample_duration=args.sample_duration * 1e6,
            decay=args.decay,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.repr_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        all_logits = []
        progress = tqdm(loader, desc=f"context:{role}", disable=args.quiet_progress)
        for imgs, *_ in progress:
            _, logits = forward_embedding(model, imgs.to(device, non_blocking=True))
            all_logits.append(logits.detach().cpu().numpy().astype(np.float32))
        sorted_logits = np.concatenate(all_logits, axis=0)
        role_logits = np.empty_like(sorted_logits)
        role_logits[extraction_order] = sorted_logits
        np.savez(cache_path, logits=role_logits, fingerprint=np.asarray(fingerprint))
        print(f"[INFO] Contexto gardado: role={role} path={cache_path} logits={role_logits.shape}")
        outputs[role] = role_logits
    return outputs


@torch.no_grad()
def collect_or_load_context_representations(
    proposals: pd.DataFrame,
    context_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    roles = list(dict.fromkeys(args.context_roles))
    if not roles:
        return {}, {}

    context_dir.mkdir(parents=True, exist_ok=True)
    embeddings_out: dict[str, np.ndarray] = {}
    logits_out: dict[str, np.ndarray] = {}
    model = None
    for role in roles:
        transformed = transform_context_proposals(proposals, role, args.context_window_scale)
        fingerprint = proposal_fingerprint(transformed)
        cache_path = context_dir / f"{role}_scale{args.context_window_scale:g}_repr.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            embeddings = data["embeddings"]
            logits = data["logits"]
            cached_fingerprint = str(data["fingerprint"].item()) if "fingerprint" in data else ""
            if len(embeddings) != len(proposals) or len(logits) != len(proposals) or cached_fingerprint != fingerprint:
                raise ValueError(f"{cache_path} does not match the requested context proposals")
            print(
                f"[INFO] Contexto cargado: role={role} path={cache_path} "
                f"emb={embeddings.shape} logits={logits.shape}"
            )
            embeddings_out[role] = embeddings
            logits_out[role] = logits
            continue

        if model is None:
            model = make_model(args, device)
        extraction_order = transformed.sort_values(
            ["rec_name", "roi_id", "t_start", "t_end"],
            kind="stable",
        ).index.to_numpy(dtype=np.int64)
        extraction_frame = transformed.loc[extraction_order].reset_index(drop=True)
        dataset = ProposalDataset(
            extraction_frame,
            augment_fraction=1.0 / args.augment_factor,
            data_path=str(resolve_path(args.data_path)),
            num_tsn_samples=expanded_tsn_samples(args.num_tsn_samples, args.augment_factor),
            sample_duration=args.sample_duration * 1e6,
            decay=args.decay,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.repr_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        all_embeddings = []
        all_logits = []
        progress = tqdm(loader, desc=f"context-repr:{role}", disable=args.quiet_progress)
        for imgs, *_ in progress:
            embeddings, logits = forward_embedding(model, imgs.to(device, non_blocking=True))
            all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float16))
            all_logits.append(logits.detach().cpu().numpy().astype(np.float32))
        sorted_embeddings = np.concatenate(all_embeddings, axis=0)
        sorted_logits = np.concatenate(all_logits, axis=0)
        role_embeddings = np.empty_like(sorted_embeddings)
        role_logits = np.empty_like(sorted_logits)
        role_embeddings[extraction_order] = sorted_embeddings
        role_logits[extraction_order] = sorted_logits
        np.savez(
            cache_path,
            embeddings=role_embeddings,
            logits=role_logits,
            fingerprint=np.asarray(fingerprint),
        )
        print(
            f"[INFO] Contexto gardado: role={role} path={cache_path} "
            f"emb={role_embeddings.shape} logits={role_logits.shape}"
        )
        embeddings_out[role] = role_embeddings
        logits_out[role] = role_logits
    return embeddings_out, logits_out


def collect_or_load_context_features(
    proposals: pd.DataFrame,
    context_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if args.context_feature_mode == "embeddings":
        return collect_or_load_context_representations(proposals, context_dir, args, device)
    return {}, collect_or_load_context_logits(proposals, context_dir, args, device)


def combine_context_embeddings(
    proposal_embeddings: np.ndarray,
    context_embeddings: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    if args.context_feature_mode != "embeddings":
        return proposal_embeddings
    if set(context_embeddings) != {"previous", "next"}:
        raise ValueError(
            "Embedding context currently requires exactly --context-roles previous next"
        )
    previous = context_embeddings["previous"]
    following = context_embeddings["next"]
    if len(previous) != len(proposal_embeddings) or len(following) != len(proposal_embeddings):
        raise ValueError("Context embeddings do not match proposal embeddings")
    return np.concatenate((previous, proposal_embeddings, following), axis=1)


def softmax_ed(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp[:, 1] / exp.sum(axis=1)


def robust01(values: pd.Series | np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros(len(arr), dtype=np.float64)
    if finite.sum() == 0:
        return out
    qlo, qhi = np.nanpercentile(arr[finite], [lo, hi])
    if qhi <= qlo + 1e-12:
        return out
    out[finite] = np.clip((arr[finite] - qlo) / (qhi - qlo), 0.0, 1.0)
    return out


def max_tiou_seconds(t_start_us: float, t_end_us: float, segments_s: np.ndarray) -> float:
    best, _, _ = best_match_seconds(t_start_us, t_end_us, segments_s)
    return best


def best_match_seconds(t_start_us: float, t_end_us: float, segments_s: np.ndarray) -> tuple[float, float, float]:
    if segments_s.size == 0:
        return 0.0, float("nan"), float("nan")
    t_start = float(t_start_us) / 1e6
    t_end = float(t_end_us) / 1e6
    inter = np.maximum(0.0, np.minimum(t_end, segments_s[:, 1]) - np.maximum(t_start, segments_s[:, 0]))
    union = (t_end - t_start) + (segments_s[:, 1] - segments_s[:, 0]) - inter
    valid = union > 0
    if not np.any(valid):
        return 0.0, float("nan"), float("nan")
    tiou = np.zeros(len(segments_s), dtype=np.float64)
    tiou[valid] = inter[valid] / union[valid]
    best_idx = int(np.argmax(tiou))
    return float(tiou[best_idx]), float(segments_s[best_idx, 0]), float(segments_s[best_idx, 1])


def build_annotation_index(
    ann_path: Path,
    split: str,
    min_duration: float,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], set[str]]:
    split_recs = load_split_recordings(ann_path, split)
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for rec_name, rec_data in ann["database"].items():
        if rec_name not in split_recs:
            continue
        out[rec_name] = {}
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
            out[rec_name][roi_key] = {
                "ed": np.asarray(ed_segments, dtype=np.float64).reshape(-1, 2),
                "flap": np.asarray(flap_segments, dtype=np.float64).reshape(-1, 2),
            }
    return out, split_recs


def roi_to_ann_key(roi_id: str) -> str:
    return str(int(str(roi_id)[1:]))


def add_lattice_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recover label-free sibling groups from invertible lattice transforms."""
    df = df.copy()
    start = df["t_start"].to_numpy(dtype=np.float64)
    end = df["t_end"].to_numpy(dtype=np.float64)
    duration = np.maximum(end - start, 1.0)
    parent_start = start.copy()
    parent_end = end.copy()
    variant = df["variant"].fillna("").astype(str)
    lattice = df["is_lattice"].to_numpy(dtype=bool)
    param = df["variant_param"].to_numpy(dtype=np.float64)

    def set_one_sided(mask: np.ndarray, denominator: np.ndarray, anchor_end: bool) -> None:
        base_duration = duration[mask] / denominator[mask]
        if anchor_end:
            parent_end[mask] = end[mask]
            parent_start[mask] = parent_end[mask] - base_duration
        else:
            parent_start[mask] = start[mask]
            parent_end[mask] = parent_start[mask] + base_duration

    expand_left = lattice & variant.str.startswith("expand_left_").to_numpy()
    expand_right = lattice & variant.str.startswith("expand_right_").to_numpy()
    expand_both = lattice & variant.str.startswith("expand_both_").to_numpy()
    trim_left = lattice & variant.str.startswith("trim_left_").to_numpy()
    trim_right = lattice & variant.str.startswith("trim_right_").to_numpy()
    trim_both = lattice & variant.str.startswith("trim_both_").to_numpy()
    shift = lattice & variant.str.startswith("shift_").to_numpy()
    center_duration = lattice & variant.str.startswith("center_dur_").to_numpy()

    set_one_sided(expand_left, 1.0 + param, anchor_end=True)
    set_one_sided(expand_right, 1.0 + param, anchor_end=False)
    set_one_sided(trim_left, np.maximum(1.0 - param, 1e-6), anchor_end=True)
    set_one_sided(trim_right, np.maximum(1.0 - param, 1e-6), anchor_end=False)
    for mask, denominator in (
        (expand_both, 1.0 + 2.0 * param),
        (trim_both, np.maximum(1.0 - 2.0 * param, 1e-6)),
    ):
        base_duration = duration[mask] / denominator[mask]
        center = 0.5 * (start[mask] + end[mask])
        parent_start[mask] = center - 0.5 * base_duration
        parent_end[mask] = center + 0.5 * base_duration
    parent_start[shift] = start[shift] - param[shift] * duration[shift]
    parent_end[shift] = end[shift] - param[shift] * duration[shift]
    center = 0.5 * (start[center_duration] + end[center_duration])
    parent_start[center_duration] = center
    parent_end[center_duration] = center

    source_score = pd.to_numeric(
        df["source_score"] if "source_score" in df else df["score"], errors="coerce"
    ).fillna(df["score"]).to_numpy(dtype=np.float64)
    family_keys = pd.DataFrame(
        {
            "rec_name": df["rec_name"].to_numpy(),
            "roi_id": df["roi_id"].to_numpy(),
            "parent_start_bin": np.rint(parent_start / 10_000.0).astype(np.int64),
            "parent_end_bin": np.rint(parent_end / 10_000.0).astype(np.int64),
            "parent_source_score": np.round(source_score, 6),
        },
        index=df.index,
    )
    family_id = family_keys.groupby(list(family_keys.columns), sort=False, dropna=False).ngroup()
    df["_family_id"] = family_id
    grouped = df.groupby("_family_id", sort=False)
    family_size = grouped["cnn_score"].transform("size").to_numpy(dtype=np.float64)
    cnn_max = grouped["cnn_score"].transform("max").to_numpy(dtype=np.float64)
    cnn_mean = grouped["cnn_score"].transform("mean").to_numpy(dtype=np.float64)
    prop_max = grouped["score"].transform("max").to_numpy(dtype=np.float64)
    median_duration = grouped["duration_s"].transform("median").to_numpy(dtype=np.float64)
    df["family_size_log"] = np.log1p(family_size)
    df["family_cnn_rank"] = grouped["cnn_score"].rank(method="average", pct=True).fillna(0.0)
    df["family_cnn_delta_max"] = df["cnn_score"].to_numpy(dtype=np.float64) - cnn_max
    df["family_cnn_delta_mean"] = df["cnn_score"].to_numpy(dtype=np.float64) - cnn_mean
    df["family_prop_rank"] = grouped["score"].rank(method="average", pct=True).fillna(0.0)
    df["family_prop_delta_max"] = df["score"].to_numpy(dtype=np.float64) - prop_max
    df["family_duration_rank"] = grouped["duration_s"].rank(method="average", pct=True).fillna(0.0)
    df["family_duration_log_ratio"] = np.log(
        np.maximum(df["duration_s"].to_numpy(dtype=np.float64), 1e-6)
        / np.maximum(median_duration, 1e-6)
    )
    role_masks = {
        "base": variant.eq("base").to_numpy(),
        "expand": variant.str.startswith("expand_").to_numpy(),
        "expand_left": variant.str.startswith("expand_left_").to_numpy(),
        "expand_right": variant.str.startswith("expand_right_").to_numpy(),
        "expand_both": variant.str.startswith("expand_both_").to_numpy(),
        "trim": variant.str.startswith("trim_").to_numpy(),
        "shift": variant.str.startswith("shift_").to_numpy(),
        "center": variant.str.startswith("center_dur_").to_numpy(),
    }
    family_mean = grouped["cnn_score"].transform("mean")
    role_scores = {}
    for role, mask in role_masks.items():
        masked = df["cnn_score"].where(mask)
        role_scores[role] = masked.groupby(df["_family_id"], sort=False).transform("max")
    base_score = role_scores["base"].fillna(family_mean)
    df["family_base_cnn_max"] = base_score
    for role in ["expand", "trim", "shift", "center"]:
        role_score = role_scores[role].fillna(base_score)
        df[f"family_{role}_cnn_max"] = role_score
        df[f"family_{role}_minus_base"] = role_score - base_score
    for role in ["expand_left", "expand_right", "expand_both"]:
        df[f"family_{role}_cnn_max"] = role_scores[role].fillna(base_score)
    left_score = df["family_expand_left_cnn_max"]
    right_score = df["family_expand_right_cnn_max"]
    both_score = df["family_expand_both_cnn_max"]
    df["family_expand_lr_delta"] = left_score - right_score
    df["family_expand_both_minus_sides"] = both_score - np.maximum(left_score, right_score)
    return df.drop(columns=["_family_id"])


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proposal_score_robust"] = robust01(df["score"])
    df["source_score_robust"] = robust01(df["source_score"] if "source_score" in df.columns else df["score"])
    df["duration_s"] = (df["t_end"] - df["t_start"]) / 1e6
    df["duration_log"] = np.log1p(np.maximum(df["duration_s"], 0.0))
    excess = np.maximum(0.0, df["duration_s"].to_numpy(dtype=np.float64) - 60.0)
    df["duration_penalty"] = np.exp(-excess / 20.0)
    df["cnn_x_prop"] = df["cnn_score"] * df["proposal_score_robust"]
    df["cnn_x_duration_penalty"] = df["cnn_score"] * df["duration_penalty"]
    source = df["source"].fillna("").astype(str) if "source" in df.columns else pd.Series("", index=df.index)
    variant = df["variant"].fillna("").astype(str) if "variant" in df.columns else pd.Series("", index=df.index)
    df["is_lattice"] = source.eq("lattice").astype(float)
    df["is_base_variant"] = variant.eq("base").astype(float)
    for prefix in [
        "expand_left",
        "expand_right",
        "expand_both",
        "trim_left",
        "trim_right",
        "trim_both",
        "shift",
    ]:
        df[f"is_{prefix}"] = variant.str.startswith(prefix).astype(float)
    df["is_center_duration"] = variant.str.startswith("center_dur").astype(float)
    param = pd.to_numeric(
        variant.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    ).fillna(0.0)
    df["variant_param"] = param.astype(float)
    center_duration = pd.to_numeric(
        variant.str.extract(r"center_dur_([-+]?\d+(?:\.\d+)?)s", expand=False),
        errors="coerce",
    ).fillna(0.0)
    df["center_duration_s"] = center_duration.astype(float)
    df = add_lattice_family_features(df)
    df["cnn_roi_rank"] = df.groupby(["rec_name", "roi_id"])["cnn_score"].rank(method="average", pct=True).fillna(0.0)
    df["proposal_roi_rank"] = df.groupby(["rec_name", "roi_id"])["score"].rank(method="average", pct=True).fillna(0.0)
    df["cnn_roi_z"] = 0.0
    df["proposal_roi_z"] = 0.0
    for _, idx in df.groupby(["rec_name", "roi_id"]).groups.items():
        for src, dst in [("cnn_score", "cnn_roi_z"), ("score", "proposal_roi_z")]:
            vals = df.loc[idx, src].to_numpy(dtype=np.float64)
            std = vals.std()
            if std > 1e-12:
                z = (vals - vals.mean()) / std
                df.loc[idx, dst] = 1.0 / (1.0 + np.exp(-z))
    return df


def prepare_frame(
    proposals: pd.DataFrame,
    logits: np.ndarray,
    split: str,
    ann_path: Path,
    args: argparse.Namespace,
    context_logits: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    df = proposals.reset_index(drop=True).copy()
    df["logit_bg"] = logits[:, 0].astype(np.float64)
    df["logit_ed"] = logits[:, 1].astype(np.float64)
    df["cnn_margin"] = df["logit_ed"] - df["logit_bg"]
    df["cnn_score"] = softmax_ed(logits, args.temperature)
    for role, role_logits in (context_logits or {}).items():
        if len(role_logits) != len(df):
            raise ValueError(f"Context role {role} has {len(role_logits)} rows, expected {len(df)}")
        margin_col = f"context_{role}_cnn_margin"
        score_col = f"context_{role}_cnn_score"
        df[margin_col] = role_logits[:, 1].astype(np.float64) - role_logits[:, 0].astype(np.float64)
        df[score_col] = softmax_ed(role_logits, args.temperature)
    previous_score = "context_previous_cnn_score"
    next_score = "context_next_cnn_score"
    if previous_score in df and next_score in df:
        adjacent_scores = df[[previous_score, next_score]].to_numpy(dtype=np.float64)
        adjacent_margins = df[["context_previous_cnn_margin", "context_next_cnn_margin"]].to_numpy(dtype=np.float64)
        df["context_max_cnn_score"] = adjacent_scores.max(axis=1)
        df["context_mean_cnn_score"] = adjacent_scores.mean(axis=1)
        df["context_cnn_contrast"] = df["cnn_score"] - df["context_max_cnn_score"]
        df["context_cnn_symmetry"] = np.abs(adjacent_scores[:, 0] - adjacent_scores[:, 1])
        df["context_margin_contrast"] = df["cnn_margin"] - adjacent_margins.max(axis=1)
    df = add_rank_features(df)

    ann_index, split_recs = build_annotation_index(ann_path, split, args.min_gt_duration)
    rows = []
    for idx, row in df.iterrows():
        rec_name = row["rec_name"]
        if rec_name not in split_recs:
            continue
        roi_key = roi_to_ann_key(row["roi_id"])
        segments = ann_index.get(rec_name, {}).get(roi_key, {})
        best_ed, gt_start_s, gt_end_s = best_match_seconds(
            row["t_start"], row["t_end"], segments.get("ed", np.empty((0, 2)))
        )
        best_flap, _, _ = best_match_seconds(
            row["t_start"], row["t_end"], segments.get("flap", np.empty((0, 2)))
        )

        cnn_rank = float(row.get("cnn_roi_rank", 0.0))
        hardness_score = max(float(row["cnn_score"]), cnn_rank)
        cnn_hard_negative = float(row["cnn_score"]) >= args.hard_score or cnn_rank >= args.hard_cnn_rank

        if best_ed >= args.high_pos_tiou:
            sample_kind = "high_positive"
        elif best_ed >= args.pos_tiou:
            sample_kind = "positive"
        elif best_ed >= args.semi_tiou:
            sample_kind = "semi_positive"
        elif best_flap >= args.flap_neg_tiou or cnn_hard_negative:
            sample_kind = "hard_negative"
        else:
            sample_kind = "easy_negative"

        quality_target = best_ed if best_ed >= args.neg_tiou else 0.0
        sample_weight = 1.0
        if sample_kind == "hard_negative" and best_ed < args.neg_tiou:
            sample_weight += args.hard_neg_loss_weight * hardness_score
        duration_us = max(float(row["t_end"]) - float(row["t_start"]), 1.0)
        has_boundary_target = (
            best_ed >= args.boundary_min_tiou
            and np.isfinite(gt_start_s)
            and np.isfinite(gt_end_s)
        )
        if has_boundary_target:
            start_delta = (gt_start_s * 1e6 - float(row["t_start"])) / duration_us
            end_delta = (gt_end_s * 1e6 - float(row["t_end"])) / duration_us
            start_delta = float(np.clip(start_delta, -args.max_boundary_delta, args.max_boundary_delta))
            end_delta = float(np.clip(end_delta, -args.max_boundary_delta, args.max_boundary_delta))
            boundary_weight = float(np.clip(best_ed, 0.0, 1.0))
        else:
            start_delta = 0.0
            end_delta = 0.0
            boundary_weight = 0.0
        labeled = row.to_dict()
        labeled.update(
            {
                "proposal_index": idx,
                "best_ed_tiou": best_ed,
                "best_flap_tiou": best_flap,
                "quality_target": float(np.clip(quality_target, 0.0, 1.0)),
                "gt_start_s": gt_start_s,
                "gt_end_s": gt_end_s,
                "start_delta_target": start_delta,
                "end_delta_target": end_delta,
                "boundary_weight": boundary_weight,
                "sample_kind": sample_kind,
                "hardness_score": hardness_score,
                "sample_weight": sample_weight,
            }
        )
        rows.append(labeled)
    return pd.DataFrame(rows).reset_index(drop=True)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    context = sorted(
        col
        for col in df.columns
        if col.startswith("context_") and pd.api.types.is_numeric_dtype(df[col])
    )
    return [*NUMERIC_COLUMNS, *context]


OOF_HARDNESS_KEYS = ["rec_name", "roi_id", "t_start", "t_end"]


def apply_oof_hardness(
    frame: pd.DataFrame,
    hardness: pd.DataFrame,
    threshold: float,
    neg_tiou: float,
) -> tuple[pd.DataFrame, int]:
    """Promote cross-fitted false positives without exposing validation labels."""
    required = {*OOF_HARDNESS_KEYS, "oof_quality_score"}
    missing = sorted(required - set(hardness.columns))
    if missing:
        raise ValueError(f"OOF hardness file misses columns: {missing}")
    if hardness.duplicated(OOF_HARDNESS_KEYS).any():
        raise ValueError("OOF hardness keys must be unique")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("OOF hardness threshold must be in [0,1]")

    merged = frame.merge(
        hardness[OOF_HARDNESS_KEYS + ["oof_quality_score"]],
        on=OOF_HARDNESS_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["oof_quality_score"].isna().any():
        raise ValueError(
            f"OOF hardness is missing {int(merged['oof_quality_score'].isna().sum())} proposals"
        )
    negative = merged["quality_target"].to_numpy(dtype=np.float64) < neg_tiou
    promoted = negative & (
        merged["oof_quality_score"].to_numpy(dtype=np.float64) >= threshold
    )
    merged.loc[promoted, "sample_kind"] = "hard_negative"
    merged["hardness_score"] = np.maximum(
        merged["hardness_score"].to_numpy(dtype=np.float64),
        merged["oof_quality_score"].to_numpy(dtype=np.float64),
    )
    return merged, int(promoted.sum())


def sample_indices(
    train_df: pd.DataFrame,
    cfg: QualityConfig,
    max_samples: int,
    seed: int,
    hard_neg_top_frac: float,
) -> np.ndarray:
    if len(train_df) <= max_samples:
        return train_df.index.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    targets = {
        "high_positive": cfg.high_pos_frac,
        "positive": cfg.pos_frac,
        "semi_positive": cfg.semi_frac,
        "hard_negative": cfg.hard_neg_frac,
        "easy_negative": cfg.easy_neg_frac,
    }
    chosen = []
    used = 0
    for kind, frac in targets.items():
        group = train_df[train_df["sample_kind"] == kind]
        if group.empty or frac <= 0:
            continue
        take = min(len(group), max(1, int(round(max_samples * frac))))
        if kind == "hard_negative" and 0.0 < hard_neg_top_frac < 1.0 and "hardness_score" in group.columns:
            pool_size = min(len(group), max(take, int(math.ceil(len(group) * hard_neg_top_frac))))
            group = group.sort_values("hardness_score", ascending=False).head(pool_size)
        chosen.append(rng.choice(group.index.to_numpy(dtype=np.int64), size=take, replace=False))
        used += take
    if used < max_samples:
        selected = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        remaining = train_df.drop(index=selected, errors="ignore")
        if not remaining.empty:
            take = min(len(remaining), max_samples - used)
            chosen.append(rng.choice(remaining.index.to_numpy(dtype=np.int64), size=take, replace=False))
    out = np.concatenate(chosen) if chosen else train_df.index.to_numpy(dtype=np.int64)
    rng.shuffle(out)
    return out


def build_local_rank_pairs(
    train_df: pd.DataFrame,
    min_tiou: float,
    min_gap: float,
    pairs_per_gt: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample better/worse proposals matched to the same training GT instance."""
    if pairs_per_gt <= 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)
    valid = (
        (train_df["quality_target"].to_numpy(dtype=np.float64) >= min_tiou)
        & np.isfinite(train_df["gt_start_s"].to_numpy(dtype=np.float64))
        & np.isfinite(train_df["gt_end_s"].to_numpy(dtype=np.float64))
    )
    candidates = train_df.loc[valid]
    if candidates.empty:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)

    rng = np.random.default_rng(seed)
    better_rows: list[int] = []
    worse_rows: list[int] = []
    gaps: list[float] = []
    group_columns = ["rec_name", "roi_id", "gt_start_s", "gt_end_s"]
    for _, group in candidates.groupby(group_columns, sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        quality = group["quality_target"].to_numpy(dtype=np.float64)
        eligible_better = np.flatnonzero(quality - quality.min() >= min_gap)
        if len(eligible_better) == 0:
            continue
        for _ in range(pairs_per_gt):
            better_pos = int(rng.choice(eligible_better))
            eligible_worse = np.flatnonzero(quality <= quality[better_pos] - min_gap)
            if len(eligible_worse) == 0:
                continue
            worse_pos = int(rng.choice(eligible_worse))
            better_rows.append(int(indices[better_pos]))
            worse_rows.append(int(indices[worse_pos]))
            gaps.append(float(quality[better_pos] - quality[worse_pos]))
    return (
        np.asarray(better_rows, dtype=np.int64),
        np.asarray(worse_rows, dtype=np.int64),
        np.asarray(gaps, dtype=np.float32),
    )


def recording_balanced_indices(
    train_df: pd.DataFrame,
    cfg: QualityConfig,
    max_samples: int,
    seed: int,
    hard_neg_top_frac: float,
) -> np.ndarray:
    """Balance proposal categories inside an equal quota for each recording."""
    recordings = sorted(train_df["rec_name"].unique())
    if not recordings or len(train_df) <= max_samples:
        return train_df.index.to_numpy(dtype=np.int64)
    quota = int(math.ceil(max_samples / len(recordings)))
    selected = []
    for offset, recording in enumerate(recordings):
        group = train_df[train_df["rec_name"] == recording]
        selected.append(
            sample_indices(group, cfg, quota, seed + offset + 1, hard_neg_top_frac)
        )
    indices = np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    elif len(indices) < max_samples:
        remaining = train_df.index.difference(indices).to_numpy(dtype=np.int64)
        take = min(len(remaining), max_samples - len(indices))
        if take > 0:
            indices = np.concatenate((indices, rng.choice(remaining, size=take, replace=False)))
    rng.shuffle(indices)
    return indices


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (train_x - mean) / std, (other_x - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def standardize_for_head(
    train_x: np.ndarray,
    other_x: np.ndarray,
    embedding_dim: int,
    numeric_dim: int,
    architecture: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    shared_role_architectures = {"temporal_completeness", "surrounding_contrastive"}
    if architecture not in shared_role_architectures or embedding_dim <= 0:
        return standardize(train_x, other_x)
    if embedding_dim % 3 != 0:
        raise ValueError(
            f"{architecture} requires an embedding divisible into three temporal roles, "
            f"got {embedding_dim} dimensions"
        )

    role_dim = embedding_dim // 3
    train_roles = train_x[:, :embedding_dim].reshape(-1, 3, role_dim)
    # Frozen ATSN embeddings are high-dimensional; a deterministic training-only
    # subset estimates their moments accurately without scanning billions of values.
    stats_stride = max(1, math.ceil(len(train_roles) / 20_000))
    stats_roles = train_roles[::stats_stride]
    role_mean = stats_roles.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    role_std = stats_roles.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    role_std = np.where(role_std < 1e-6, 1.0, role_std).astype(np.float32)
    mean_parts = [np.tile(role_mean, 3)]
    std_parts = [np.tile(role_std, 3)]

    if numeric_dim > 0:
        train_num = train_x[:, embedding_dim:embedding_dim + numeric_dim]
        num_mean = train_num.mean(axis=0, dtype=np.float64).astype(np.float32)
        num_std = train_num.std(axis=0, dtype=np.float64).astype(np.float32)
        num_std = np.where(num_std < 1e-6, 1.0, num_std).astype(np.float32)
        mean_parts.append(num_mean)
        std_parts.append(num_std)

    mean = np.concatenate(mean_parts).astype(np.float32)
    std = np.concatenate(std_parts).astype(np.float32)
    return (train_x - mean) / std, (other_x - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def tanp_standardized_roles(
    standardized: torch.Tensor,
    embedding_dim: int,
    embedding_mean: torch.Tensor,
    embedding_std: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Perturb three ordered ATSN roles in raw feature space, then re-standardize."""
    if sigma == 0:
        return standardized
    if embedding_dim <= 0 or embedding_dim % 3 != 0:
        raise ValueError("TANP requires an ATSN embedding with three temporal roles")
    if embedding_mean.numel() != embedding_dim or embedding_std.numel() != embedding_dim:
        raise ValueError("TANP scaler dimensions do not match the ATSN embedding")

    embedding = standardized[:, :embedding_dim]
    raw = embedding * embedding_std + embedding_mean
    temporal = raw.reshape(len(raw), 3, embedding_dim // 3).transpose(1, 2)
    perturbed = temporal_aware_normalization_perturbation(temporal, sigma)
    output = standardized.clone()
    output[:, :embedding_dim] = (
        perturbed.transpose(1, 2).reshape(len(raw), embedding_dim) - embedding_mean
    ) / embedding_std
    return output


TEMPORAL_DESCRIPTOR_GROUP_RANGES = {
    "norm": (0, 11),
    "adjacent_cos": (11, 21),
    "delta": (21, 31),
    "center_cos": (31, 42),
    "spectral": (42, 47),
}


def drop_temporal_descriptor_groups(
    embeddings: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    """Drop selected groups from the known 47-dimensional descriptor tail."""
    if not groups:
        return embeddings
    descriptor_dim = 47
    if embeddings.ndim != 2 or embeddings.shape[1] < descriptor_dim:
        raise ValueError(
            "Temporal descriptor ablation requires a 2D embedding with a 47-dimensional tail"
        )
    keep = np.ones(embeddings.shape[1], dtype=bool)
    descriptor_start = embeddings.shape[1] - descriptor_dim
    for group in groups:
        start, end = TEMPORAL_DESCRIPTOR_GROUP_RANGES[group]
        keep[descriptor_start + start : descriptor_start + end] = False
    return embeddings[:, keep]


class QualityHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, out_dim: int = 1) -> None:
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, max(hidden // 2, 32)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(hidden // 2, 32), out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class DecoupledQualityBoundaryHead(nn.Module):
    """Independent MLP towers for proposal quality and boundary offsets."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        dropout: float,
        num_quality_outputs: int,
    ) -> None:
        super().__init__()
        self.quality = QualityHead(in_dim, hidden, dropout, out_dim=num_quality_outputs)
        self.boundary = QualityHead(in_dim, hidden, dropout, out_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quality = self.quality(x)
        if quality.ndim == 1:
            quality = quality.unsqueeze(1)
        boundary = self.boundary(x)
        return torch.cat((quality, boundary), dim=1)


class TemporalCompletenessHead(nn.Module):
    """Quality head with an explicit start/main/end temporal inductive bias."""

    def __init__(
        self,
        embedding_dim: int,
        numeric_dim: int,
        hidden: int,
        dropout: float,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 3 != 0:
            raise ValueError("TemporalCompletenessHead requires three ATSN embedding roles")
        self.embedding_dim = embedding_dim
        self.numeric_dim = numeric_dim
        branch_dim = embedding_dim // 3
        role_dim = max(32, min(96, hidden // 4))
        numeric_hidden = max(16, min(64, numeric_dim * 2)) if numeric_dim > 0 else 0

        self.role_projection = nn.Sequential(
            nn.Linear(branch_dim, role_dim),
            nn.LayerNorm(role_dim),
            nn.ReLU(),
        )
        self.numeric_projection = (
            nn.Sequential(
                nn.Linear(numeric_dim, numeric_hidden),
                nn.LayerNorm(numeric_hidden),
                nn.ReLU(),
            )
            if numeric_dim > 0
            else None
        )

        fused_dim = 7 * role_dim + numeric_hidden
        inner_dim = max(hidden // 2, 32)
        self.output = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        roles = x[:, :self.embedding_dim].reshape(x.shape[0], 3, -1)
        projected = self.role_projection(roles)
        start, main, end = projected.unbind(dim=1)
        context_mean = 0.5 * (start + end)
        temporal = torch.cat(
            (
                start,
                main,
                end,
                main - start,
                main - end,
                torch.abs(start - end),
                torch.abs(main - context_mean),
            ),
            dim=1,
        )
        if self.numeric_projection is not None:
            numeric = x[:, self.embedding_dim:self.embedding_dim + self.numeric_dim]
            temporal = torch.cat((temporal, self.numeric_projection(numeric)), dim=1)
        out = self.output(temporal)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class SurroundingContrastiveHead(nn.Module):
    """P-MIL-style outer-inner contrast over previous/inside/next embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        numeric_dim: int,
        hidden: int,
        dropout: float,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 3 != 0:
            raise ValueError("SurroundingContrastiveHead requires previous/inside/next embeddings")
        self.embedding_dim = embedding_dim
        self.numeric_dim = numeric_dim
        window_dim = embedding_dim // 3
        projected_dim = max(64, min(192, hidden // 2))
        numeric_hidden = max(16, min(64, numeric_dim * 2)) if numeric_dim > 0 else 0

        self.window_projection = nn.Sequential(
            nn.Linear(window_dim, projected_dim),
            nn.LayerNorm(projected_dim),
        )
        self.numeric_projection = (
            nn.Sequential(
                nn.Linear(numeric_dim, numeric_hidden),
                nn.LayerNorm(numeric_hidden),
                nn.ReLU(),
            )
            if numeric_dim > 0
            else None
        )
        fused_dim = 3 * projected_dim + numeric_hidden
        inner_dim = max(hidden // 2, 32)
        self.output = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = x[:, :self.embedding_dim].reshape(x.shape[0], 3, -1)
        previous, inside, following = self.window_projection(windows).unbind(dim=1)
        contrast = torch.cat((inside - previous, inside, inside - following), dim=1)
        if self.numeric_projection is not None:
            numeric = x[:, self.embedding_dim:self.embedding_dim + self.numeric_dim]
            contrast = torch.cat((contrast, self.numeric_projection(numeric)), dim=1)
        out = self.output(contrast)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def make_quality_head(
    args: argparse.Namespace,
    in_dim: int,
    numeric_dim: int,
    hidden: int,
    dropout: float,
    out_dim: int,
) -> nn.Module:
    embedding_dim = 0 if args.numeric_only else in_dim - numeric_dim
    if args.decoupled_boundary_head:
        if out_dim < 3:
            raise ValueError("A decoupled boundary head requires quality outputs plus two offsets")
        return DecoupledQualityBoundaryHead(
            in_dim=in_dim,
            hidden=hidden,
            dropout=dropout,
            num_quality_outputs=out_dim - 2,
        )
    if args.head_architecture == "temporal_completeness" and embedding_dim > 0:
        return TemporalCompletenessHead(
            embedding_dim=embedding_dim,
            numeric_dim=numeric_dim,
            hidden=hidden,
            dropout=dropout,
            out_dim=out_dim,
        )
    if args.head_architecture == "surrounding_contrastive" and embedding_dim > 0:
        return SurroundingContrastiveHead(
            embedding_dim=embedding_dim,
            numeric_dim=numeric_dim,
            hidden=hidden,
            dropout=dropout,
            out_dim=out_dim,
        )
    return QualityHead(in_dim, hidden, dropout, out_dim=out_dim)


def split_head_output(raw: torch.Tensor, num_quality_outputs: int = 1) -> tuple[torch.Tensor, torch.Tensor | None]:
    if raw.ndim == 1:
        return raw, None
    quality_logits = raw[:, :num_quality_outputs]
    if num_quality_outputs == 1:
        quality_logits = quality_logits.squeeze(1)
    offsets = raw[:, num_quality_outputs:num_quality_outputs + 2]
    return quality_logits, offsets if offsets.shape[1] == 2 else None


def quality_thresholds(args: argparse.Namespace) -> list[float]:
    if not args.multi_quality_head:
        return []
    return sorted(float(t) for t in args.quality_thresholds)


def quality_targets(best_tiou: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    thresholds = quality_thresholds(args)
    if not thresholds:
        return best_tiou.astype(np.float32)
    targets = [
        np.where(best_tiou >= thr, best_tiou, 0.0).astype(np.float32)
        for thr in thresholds
    ]
    return np.stack(targets, axis=1)


def num_quality_outputs(args: argparse.Namespace) -> int:
    return max(1, len(quality_thresholds(args)))


def nearest_threshold_index(thresholds: list[float], target: float) -> int:
    if not thresholds:
        return 0
    arr = np.asarray(thresholds, dtype=np.float64)
    return int(np.argmin(np.abs(arr - target)))


def rank_output_index(args: argparse.Namespace) -> int:
    thresholds = quality_thresholds(args)
    if not thresholds:
        return 0
    target = (
        args.rank_output_threshold
        if args.rank_output_threshold is not None
        else (args.rank_pos_tiou if args.rank_pos_tiou is not None else args.pos_tiou)
    )
    return nearest_threshold_index(thresholds, float(target))


def select_quality_for_rank(logits: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    return logits[:, rank_output_index(args)]


def select_quality_for_distill(pred: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if pred.ndim == 1:
        return pred
    thresholds = quality_thresholds(args)
    return pred[:, nearest_threshold_index(thresholds, 0.5)]


def quality_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    high_tiou: float,
    high_iou_weight: float,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    scale = torch.pow(torch.abs(target - pred), beta)
    weights = torch.where(target > 0, 2.0 + 2.0 * target, torch.ones_like(target))
    if high_iou_weight > 0:
        weights = weights + high_iou_weight * (target >= high_tiou).float()
    if sample_weight is not None:
        if target.ndim > sample_weight.ndim:
            sample_weight = sample_weight.unsqueeze(1)
        weights = weights * sample_weight
    loss = weights * scale * bce
    if loss.ndim > 1:
        loss = loss.mean(dim=tuple(range(1, loss.ndim)))
    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError(f"Unsupported quality focal loss reduction: {reduction}")
    return loss.mean()


def group_dro_reduce(
    per_sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    group_weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Exponentially upweight recordings with high current-batch risk."""
    present = torch.unique(group_ids, sorted=True)
    losses = torch.stack([per_sample_loss[group_ids == group].mean() for group in present])
    with torch.no_grad():
        update = torch.exp(torch.clamp(eta * losses.detach(), max=20.0))
        group_weights[present] *= update
        group_weights /= group_weights.sum().clamp_min(1e-12)
    present_weights = group_weights[present]
    present_weights = present_weights / present_weights.sum().clamp_min(1e-12)
    return torch.sum(present_weights * losses)


def rank_sort_subset(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_positives: int,
    max_negatives: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a quality-spanning positive set and the hardest negatives."""
    positive = torch.nonzero(targets > 0, as_tuple=False).flatten()
    negative = torch.nonzero(targets == 0, as_tuple=False).flatten()
    if max_positives > 0 and len(positive) > max_positives:
        ordered = positive[torch.argsort(targets[positive])]
        positions = torch.linspace(
            0,
            len(ordered) - 1,
            max_positives,
            device=logits.device,
        ).round().long()
        positive = ordered[positions]
    if max_negatives > 0 and len(negative) > max_negatives:
        hard_positions = torch.topk(
            logits[negative].detach(),
            k=max_negatives,
        ).indices
        negative = negative[hard_positions]
    selected = torch.cat((positive, negative))
    return logits[selected], targets[selected]


@torch.no_grad()
def update_weight_average(
    averaged_model: nn.Module,
    current_model: nn.Module,
    num_averaged: int,
) -> int:
    """Add one model snapshot to a uniform parameter-and-buffer average."""
    averaged_state = averaged_model.state_dict()
    current_state = current_model.state_dict()
    if averaged_state.keys() != current_state.keys():
        raise ValueError("Models used for weight averaging do not share the same state")

    alpha = 1.0 / float(num_averaged + 1)
    for name, averaged_value in averaged_state.items():
        current_value = current_state[name].detach()
        if averaged_value.is_floating_point():
            if num_averaged == 0:
                averaged_value.copy_(current_value)
            else:
                averaged_value.lerp_(current_value, alpha)
        else:
            averaged_value.copy_(current_value)
    return num_averaged + 1


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_config(
    cfg: QualityConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    pred_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    if args.recording_balanced_sampling:
        train_idx = recording_balanced_indices(
            train_df, cfg, args.max_train_samples, args.seed, args.hard_neg_top_frac
        )
        recording_counts = train_df.loc[train_idx, "rec_name"].value_counts()
        print(
            f"[INFO] Recording-balanced samples={len(train_idx)} records={len(recording_counts)} "
            f"min={recording_counts.min()} max={recording_counts.max()}"
        )
    else:
        train_idx = sample_indices(train_df, cfg, args.max_train_samples, args.seed, args.hard_neg_top_frac)
    sub = train_df.loc[train_idx].reset_index(drop=True)
    train_emb_sub = train_embeddings[train_idx].astype(np.float32)
    val_emb = val_embeddings.astype(np.float32)

    feature_columns = numeric_feature_columns(train_df)
    missing_val_columns = sorted(set(feature_columns) - set(val_df.columns))
    if missing_val_columns:
        raise ValueError(f"Validation frame is missing numeric features: {missing_val_columns}")
    train_num = sub[feature_columns].to_numpy(dtype=np.float32)
    val_num = val_df[feature_columns].to_numpy(dtype=np.float32)
    if args.numeric_only:
        train_x_raw = train_num
        val_x_raw = val_num
    else:
        train_x_raw = np.concatenate([train_emb_sub, train_num], axis=1)
        val_x_raw = np.concatenate([val_emb, val_num], axis=1)
    embedding_dim = 0 if args.numeric_only else train_emb_sub.shape[1]
    numeric_dim = train_num.shape[1]
    train_x, val_x, scaler = standardize_for_head(
        train_x_raw,
        val_x_raw,
        embedding_dim=embedding_dim,
        numeric_dim=numeric_dim,
        architecture=args.head_architecture,
    )
    if args.tanp_sigma < 0:
        raise ValueError("--tanp-sigma must be non-negative")
    if args.tanp_sigma > 0 and embedding_dim % 3 != 0:
        raise ValueError("TANP requires embeddings divisible into start/main/end roles")
    tanp_mean = torch.tensor(
        scaler["mean"][:embedding_dim], dtype=torch.float32, device=device
    )
    tanp_std = torch.tensor(
        scaler["std"][:embedding_dim], dtype=torch.float32, device=device
    )

    y_scalar = sub["quality_target"].to_numpy(dtype=np.float32)
    y = quality_targets(y_scalar, args)
    base = sub["cnn_score"].to_numpy(dtype=np.float32)
    start_delta = sub["start_delta_target"].to_numpy(dtype=np.float32)
    end_delta = sub["end_delta_target"].to_numpy(dtype=np.float32)
    boundary_weight = sub["boundary_weight"].to_numpy(dtype=np.float32)
    sample_weight = sub.get("sample_weight", pd.Series(1.0, index=sub.index)).to_numpy(dtype=np.float32)
    recording_names = sorted(sub["rec_name"].astype(str).unique())
    recording_to_group = {name: index for index, name in enumerate(recording_names)}
    group_ids = sub["rec_name"].astype(str).map(recording_to_group).to_numpy(dtype=np.int64).copy()
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(y),
        torch.from_numpy(base),
        torch.from_numpy(start_delta),
        torch.from_numpy(end_delta),
        torch.from_numpy(boundary_weight),
        torch.from_numpy(sample_weight),
        torch.from_numpy(group_ids),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    n_quality = num_quality_outputs(args)
    head_outputs = n_quality + 2 if args.boundary_loss_weight > 0 else n_quality
    model = make_quality_head(
        args,
        in_dim=train_x.shape[1],
        numeric_dim=numeric_dim,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        out_dim=head_outputs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.weight_average_start_epoch < 0:
        raise ValueError("--weight-average-start-epoch must be non-negative")
    if args.weight_average_interval <= 0:
        raise ValueError("--weight-average-interval must be positive")
    averaged_model = None
    num_averaged = 0
    if args.weight_average_start_epoch > 0:
        expected_snapshots = 1 + (
            args.epochs - args.weight_average_start_epoch
        ) // args.weight_average_interval
        if expected_snapshots < 2:
            raise ValueError("Weight averaging requires at least two epoch snapshots")
        averaged_model = copy.deepcopy(model)
        averaged_model.requires_grad_(False)
    if args.group_dro and args.group_dro_eta <= 0:
        raise ValueError("--group-dro-eta must be positive")
    group_weights = torch.full(
        (len(recording_names),),
        1.0 / max(len(recording_names), 1),
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(args.seed)
    rank_pos_tiou = args.rank_pos_tiou if args.rank_pos_tiou is not None else args.pos_tiou
    high_pool = np.where(y_scalar >= rank_pos_tiou)[0]
    low_pool = np.where(y_scalar < args.neg_tiou)[0]
    hard_pool = np.where((y_scalar < args.neg_tiou) & (sub["sample_kind"].to_numpy() == "hard_negative"))[0]
    if len(hard_pool) > 0 and 0.0 < args.hard_neg_top_frac < 1.0 and "hardness_score" in sub.columns:
        keep = min(len(hard_pool), max(1, int(math.ceil(len(hard_pool) * args.hard_neg_top_frac))))
        order = np.argsort(-sub.iloc[hard_pool]["hardness_score"].to_numpy(dtype=np.float64))
        hard_pool = hard_pool[order[:keep]]
    if len(hard_pool) == 0:
        hard_pool = low_pool
    local_better, local_worse, local_gaps = build_local_rank_pairs(
        sub,
        min_tiou=args.local_rank_min_tiou,
        min_gap=args.local_rank_min_gap,
        pairs_per_gt=args.local_rank_pairs_per_gt,
        seed=args.seed,
    )
    if args.local_rank_weight > 0:
        print(
            f"[INFO] Local rank pairs={len(local_better)} "
            f"min_tiou={args.local_rank_min_tiou:g} min_gap={args.local_rank_min_gap:g}"
        )

    best_metrics = None
    best_state = None
    best_num_averaged = 0
    start_epoch = 0
    last_path = out_dir / f"last_{cfg.name}.pt"
    if args.resume_training and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        group_weights.copy_(state["group_weights"].to(device))
        rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
            )
        num_averaged = int(state["num_averaged"])
        if averaged_model is not None and state.get("averaged_state") is not None:
            averaged_model.load_state_dict(state["averaged_state"])
        best_metrics = state.get("best_metrics")
        best_state = state.get("best_state")
        best_num_averaged = int(state.get("best_num_averaged", 0))
        start_epoch = int(state["epoch"])
        print(f"[INFO] Resuming {cfg.name} after epoch {start_epoch}")

    def persist_training_state(epoch_number: int) -> None:
        if not args.resume_training:
            return
        atomic_torch_save(
            {
                "epoch": epoch_number,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "group_weights": group_weights.detach().cpu(),
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                "averaged_state": (
                    averaged_model.state_dict() if averaged_model is not None else None
                ),
                "num_averaged": num_averaged,
                "best_metrics": best_metrics,
                "best_state": best_state,
                "best_num_averaged": best_num_averaged,
            },
            last_path,
        )

    progress = tqdm(range(start_epoch, args.epochs), desc=cfg.name, disable=args.quiet_progress)
    for epoch in progress:
        model.train()
        losses = []
        for xb, yb, base_b, start_b, end_b, boundary_w_b, sample_w_b, group_b in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            base_b = base_b.to(device)
            start_b = start_b.to(device)
            end_b = end_b.to(device)
            boundary_w_b = boundary_w_b.to(device)
            sample_w_b = sample_w_b.to(device)
            group_b = group_b.to(device)
            xb = tanp_standardized_roles(
                xb,
                embedding_dim,
                tanp_mean,
                tanp_std,
                args.tanp_sigma,
            )
            raw = model(xb)
            logits, offsets = split_head_output(raw, n_quality)
            qfl_per_sample = quality_focal_loss(
                logits,
                yb,
                args.qfl_beta,
                args.high_pos_tiou,
                args.high_iou_loss_weight,
                sample_w_b,
                reduction="none",
            )
            qfl = (
                group_dro_reduce(
                    qfl_per_sample,
                    group_b,
                    group_weights,
                    args.group_dro_eta,
                )
                if args.group_dro
                else qfl_per_sample.mean()
            )
            pred = torch.sigmoid(logits)
            distill = F.mse_loss(select_quality_for_distill(pred, args), base_b)
            rank_sort = torch.tensor(0.0, device=device)
            if args.rank_sort_weight > 0:
                rank_logits = select_quality_for_rank(logits, args)
                rank_targets = yb if yb.ndim == 1 else yb[:, rank_output_index(args)]
                rank_logits, rank_targets = rank_sort_subset(
                    rank_logits,
                    rank_targets,
                    args.rank_sort_max_positives,
                    args.rank_sort_max_negatives,
                )
                ranking_error, sorting_error = rank_sort_loss(
                    rank_logits,
                    rank_targets,
                    delta=args.rank_sort_delta,
                )
                rank_sort = ranking_error + sorting_error
            boundary = torch.tensor(0.0, device=device)
            if offsets is not None and args.boundary_loss_weight > 0:
                target_offsets = torch.stack((start_b, end_b), dim=1)
                per_sample = F.smooth_l1_loss(offsets, target_offsets, reduction="none").mean(dim=1)
                denom = boundary_w_b.sum().clamp_min(1.0)
                boundary = (per_sample * boundary_w_b).sum() / denom

            if len(high_pool) > 0 and len(low_pool) > 0 and cfg.rank_weight > 0:
                n_pairs = min(args.pair_batch_size, len(high_pool), len(low_pool))
                pos_idx = rng.choice(high_pool, size=n_pairs, replace=len(high_pool) < n_pairs)
                neg_src = hard_pool if rng.random() < args.hard_rank_neg_prob and len(hard_pool) > 0 else low_pool
                neg_idx = rng.choice(neg_src, size=n_pairs, replace=len(neg_src) < n_pairs)
                pair_idx = np.concatenate([pos_idx, neg_idx])
                pair_x = torch.from_numpy(train_x[pair_idx]).to(device)
                pair_x = tanp_standardized_roles(
                    pair_x,
                    embedding_dim,
                    tanp_mean,
                    tanp_std,
                    args.tanp_sigma,
                )
                pair_raw = model(pair_x)
                pair_logits, _ = split_head_output(pair_raw, n_quality)
                pair_rank_logits = select_quality_for_rank(pair_logits, args)
                rank = F.softplus(-(pair_rank_logits[:n_pairs] - pair_rank_logits[n_pairs:])).mean()
            else:
                rank = torch.tensor(0.0, device=device)

            if args.local_rank_weight > 0 and len(local_better) > 0:
                n_local = min(args.pair_batch_size, len(local_better))
                local_choice = rng.choice(len(local_better), size=n_local, replace=len(local_better) < n_local)
                better_idx = local_better[local_choice]
                worse_idx = local_worse[local_choice]
                local_pair_idx = np.concatenate((better_idx, worse_idx))
                local_x = torch.from_numpy(train_x[local_pair_idx]).to(device)
                local_x = tanp_standardized_roles(
                    local_x,
                    embedding_dim,
                    tanp_mean,
                    tanp_std,
                    args.tanp_sigma,
                )
                local_raw = model(local_x)
                local_logits, _ = split_head_output(local_raw, n_quality)
                local_logits = select_quality_for_rank(local_logits, args)
                gap_weight = torch.from_numpy(local_gaps[local_choice]).to(device)
                local_losses = F.softplus(-(local_logits[:n_local] - local_logits[n_local:]))
                local_rank = (local_losses * gap_weight).sum() / gap_weight.sum().clamp_min(1e-6)
            else:
                local_rank = torch.tensor(0.0, device=device)

            loss = (
                cfg.qfl_weight * qfl
                + cfg.rank_weight * rank
                + args.local_rank_weight * local_rank
                + cfg.distill_weight * distill
                + args.rank_sort_weight * rank_sort
                + args.boundary_loss_weight * boundary
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        epoch_number = epoch + 1
        should_average = (
            averaged_model is not None
            and epoch_number >= args.weight_average_start_epoch
            and (epoch_number - args.weight_average_start_epoch) % args.weight_average_interval == 0
        )
        if should_average:
            num_averaged = update_weight_average(averaged_model, model, num_averaged)

        persist_training_state(epoch_number)
        should_eval = epoch_number % max(args.eval_every, 1) == 0 or epoch == args.epochs - 1
        if should_eval:
            has_weight_average = averaged_model is not None and num_averaged >= 2
            eval_model = averaged_model if has_weight_average else model
            val_scored = score_val(eval_model, val_df, val_x, device, args)
            epoch_rows = evaluate_all_scores(val_scored, cfg.name, args, pred_dir, epoch)
            epoch_best = max(epoch_rows, key=lambda row: row["mAP"])
            progress.set_postfix(loss=f"{np.mean(losses):.4f}", mAP=f"{epoch_best['mAP']:.4f}")
            eligible_checkpoint = averaged_model is None or has_weight_average
            if eligible_checkpoint and (best_metrics is None or epoch_best["mAP"] > best_metrics["mAP"]):
                best_metrics = epoch_best
                best_state = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
                best_num_averaged = num_averaged
        else:
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")

        persist_training_state(epoch_number)

    if best_metrics is None or best_state is None:
        has_weight_average = averaged_model is not None and num_averaged >= 2
        eval_model = averaged_model if has_weight_average else model
        val_scored = score_val(eval_model, val_df, val_x, device, args)
        epoch_rows = evaluate_all_scores(
            val_scored,
            cfg.name,
            args,
            pred_dir,
            max(args.epochs - 1, 0),
        )
        best_metrics = max(epoch_rows, key=lambda row: row["mAP"])
        best_state = {
            key: value.detach().cpu()
            for key, value in eval_model.state_dict().items()
        }
        best_num_averaged = num_averaged
        persist_training_state(args.epochs)
    assert best_metrics is not None and best_state is not None
    model.load_state_dict(best_state)
    best_scored = score_val(model, val_df, val_x, device, args)
    checkpoint = {
        "state_dict": best_state,
        "config": cfg.__dict__,
        "numeric_columns": feature_columns,
        "head_outputs": head_outputs,
        "quality_thresholds": quality_thresholds(args),
        "multi_quality_head": bool(args.multi_quality_head),
        "numeric_only": bool(args.numeric_only),
        "head_architecture": args.head_architecture,
        "embedding_dim": int(embedding_dim),
        "numeric_dim": int(numeric_dim),
        "scaler": scaler,
        "args": vars(args),
        "recording_groups": recording_names,
        "group_dro_weights": group_weights.detach().cpu().tolist() if args.group_dro else None,
        "weight_average_count": best_num_averaged,
        "best_metrics": best_metrics,
    }
    torch.save(checkpoint, out_dir / f"{cfg.name}.pt")
    last_path.unlink(missing_ok=True)
    return best_scored, best_metrics


def add_refined_boundaries(df: pd.DataFrame, offsets: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    duration = np.maximum(out["t_end"].to_numpy(dtype=np.float64) - out["t_start"].to_numpy(dtype=np.float64), 1.0)
    delta_start = np.clip(offsets[:, 0].astype(np.float64), -args.max_boundary_delta, args.max_boundary_delta)
    delta_end = np.clip(offsets[:, 1].astype(np.float64), -args.max_boundary_delta, args.max_boundary_delta)
    refined_start = out["t_start"].to_numpy(dtype=np.float64) + delta_start * duration
    refined_end = out["t_end"].to_numpy(dtype=np.float64) + delta_end * duration

    min_duration_us = args.min_gt_duration * 1e6
    center = 0.5 * (refined_start + refined_end)
    too_short = refined_end - refined_start < min_duration_us
    refined_start[too_short] = center[too_short] - 0.5 * min_duration_us
    refined_end[too_short] = center[too_short] + 0.5 * min_duration_us
    refined_start = np.maximum(0.0, refined_start)
    refined_end = np.maximum(refined_start + min_duration_us, refined_end)

    out["boundary_delta_start"] = delta_start
    out["boundary_delta_end"] = delta_end
    out["refined_t_start"] = refined_start
    out["refined_t_end"] = refined_end
    return out


@torch.no_grad()
def score_val(
    model: QualityHead,
    val_df: pd.DataFrame,
    val_x: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    model.eval()
    preds = []
    offsets_out = []
    n_quality = num_quality_outputs(args)
    for start in range(0, len(val_x), 8192):
        xb = torch.from_numpy(val_x[start:start + 8192]).to(device)
        raw = model(xb)
        logits, offsets = split_head_output(raw, n_quality)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        if offsets is not None:
            offsets_out.append(offsets.detach().cpu().numpy())
    quality = np.concatenate(preds, axis=0)
    if quality.ndim == 1:
        quality = quality[:, None]
    thresholds = quality_thresholds(args)
    out = val_df.copy()
    out["quality_score"] = np.clip(quality[:, nearest_threshold_index(thresholds, 0.5)] if thresholds else quality[:, 0], 0.0, 1.0)
    out["quality_x_cnn"] = out["quality_score"] * out["cnn_score"]
    out["sqrt_quality_x_cnn"] = np.sqrt(np.clip(out["quality_x_cnn"], 0.0, 1.0))
    out["quality_avg_cnn"] = 0.5 * out["quality_score"] + 0.5 * out["cnn_score"]
    if thresholds:
        eps = 1e-9
        for idx, thr in enumerate(thresholds):
            name = f"quality_t{int(round(thr * 100)):03d}"
            out[name] = np.clip(quality[:, idx], 0.0, 1.0)
            out[f"{name}_x_cnn"] = out[name] * out["cnn_score"]
            out[f"{name}_sqrt_cnn"] = np.sqrt(np.clip(out[f"{name}_x_cnn"], 0.0, 1.0))
            out[f"{name}_avg_cnn"] = 0.5 * out[name] + 0.5 * out["cnn_score"]
        q_cols = [f"quality_t{int(round(thr * 100)):03d}" for thr in thresholds]
        out["quality_multi_mean"] = out[q_cols].mean(axis=1)
        out["quality_multi_geom"] = np.exp(np.log(np.clip(out[q_cols].to_numpy(dtype=np.float64), eps, 1.0)).mean(axis=1))
        out["quality_multi_mean_avg_cnn"] = 0.5 * out["quality_multi_mean"] + 0.5 * out["cnn_score"]
        out["quality_multi_geom_avg_cnn"] = 0.5 * out["quality_multi_geom"] + 0.5 * out["cnn_score"]
        if 0.5 in thresholds and 0.7 in thresholds:
            q05 = f"quality_t{int(round(0.5 * 100)):03d}"
            q07 = f"quality_t{int(round(0.7 * 100)):03d}"
            out["quality_t050_t070_mean"] = 0.5 * out[q05] + 0.5 * out[q07]
            out["quality_t050_t070_geom"] = np.sqrt(np.clip(out[q05] * out[q07], 0.0, 1.0))
            out["quality_t050_t070_mean_avg_cnn"] = 0.5 * out["quality_t050_t070_mean"] + 0.5 * out["cnn_score"]
            out["quality_t050_t070_geom_avg_cnn"] = 0.5 * out["quality_t050_t070_geom"] + 0.5 * out["cnn_score"]
    is_lattice = (
        out["source"].fillna("").astype(str).eq("lattice").to_numpy(dtype=np.float64)
        if "source" in out.columns
        else np.zeros(len(out), dtype=np.float64)
    )
    base_priority_050 = 1.0 - 0.50 * is_lattice
    base_priority_080 = 1.0 - 0.80 * is_lattice
    out["quality_score_base_priority_050"] = out["quality_score"] * base_priority_050
    out["quality_score_base_priority_080"] = out["quality_score"] * base_priority_080
    out["quality_avg_cnn_base_priority_050"] = out["quality_avg_cnn"] * base_priority_050
    out["quality_avg_cnn_base_priority_080"] = out["quality_avg_cnn"] * base_priority_080
    if offsets_out:
        out = add_refined_boundaries(out, np.concatenate(offsets_out, axis=0), args)
    return out


def load_gt(
    valid_sequences: list[str], ann_path: Path, min_duration: float = 2.0
) -> pd.DataFrame:
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
                if end - start < min_duration:
                    continue
                rows.append({"video-id": f"{rec}_{int(roi)}", "t-start": start, "t-end": end})
    return pd.DataFrame(rows)


def predictions_to_df(prediction: dict, min_duration: float = 2.0) -> pd.DataFrame:
    rows = []
    for rec, rois in prediction["results"].items():
        for roi, detections in rois.items():
            for det in detections:
                start, end = det["segment"]
                if end - start < min_duration:
                    continue
                rows.append(
                    {
                        "video-id": f"{rec}_{int(roi)}",
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


def build_prediction(df: pd.DataFrame, score_col: str, min_score: float, args: argparse.Namespace) -> dict:
    result: dict[str, dict[int, list[dict]]] = {
        rec: {int(roi[1:]): [] for roi in grp["roi_id"].unique()}
        for rec, grp in df.groupby("rec_name")
    }
    selected = df[df[score_col] >= min_score].copy()
    if selected.empty:
        return {"version": f"quality_head:{score_col}", "results": result}

    start_col = "t_start"
    end_col = "t_end"
    if args.use_boundary_refinement and {"refined_t_start", "refined_t_end"} <= set(selected.columns):
        start_col = "refined_t_start"
        end_col = "refined_t_end"

    scores = selected[score_col].to_numpy(dtype=np.float64).copy()
    durations_s = (selected[end_col].to_numpy() - selected[start_col].to_numpy()) / 1e6
    excess = np.maximum(0.0, durations_s - args.duration_dmax)
    scores *= np.exp(-excess / args.duration_sigma)
    selected["final_score"] = scores

    for (rec, roi_id), grp in selected.groupby(["rec_name", "roi_id"]):
        if args.pre_nms_topk_per_roi > 0 and len(grp) > args.pre_nms_topk_per_roi:
            grp = grp.sort_values("final_score", ascending=False).head(args.pre_nms_topk_per_roi)
        arr = grp[[start_col, end_col, "final_score"]].to_numpy(dtype=np.float64)
        processed = temporal_soft_nms(arr, sigma=args.soft_nms_sigma, score_threshold=args.soft_nms_score_threshold)
        result[rec][int(roi_id[1:])] = [
            {
                "label": "ed",
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(score),
            }
            for start, end, score in processed
            if (float(end) - float(start)) / 1e6 >= args.min_gt_duration
        ]
    return {"version": f"quality_head:{score_col}", "results": result}


def evaluate_score(
    df: pd.DataFrame,
    score_col: str,
    label: str,
    args: argparse.Namespace,
    pred_dir: Path,
    suffix: str,
) -> list[dict]:
    ann_path = resolve_path(args.ann_path)
    valid_sequences = sorted(df["rec_name"].unique())
    gt = load_gt(valid_sequences, ann_path, args.min_gt_duration)
    rows = []
    for min_score in args.min_score:
        prediction = build_prediction(df, score_col, min_score, args)
        pred_path = pred_dir / f"{label}_{suffix}_{score_col}_min{min_score:.3f}.json"
        pred_path.write_text(json.dumps(prediction), encoding="utf-8")
        evaluator = DetectionsEvaluator(
            ground_truth_filename=str(ann_path),
            prediction_filename=str(pred_path),
            tiou_thresholds=np.asarray(args.tiou, dtype=np.float64),
            valid_labels="ed",
            valid_sequences=valid_sequences,
            min_duration=args.min_gt_duration,
        )
        mean_ap = evaluator.run()
        pred_df = predictions_to_df(prediction, args.min_gt_duration)
        best_iou = best_iou_by_gt(gt, pred_df)
        rows.append(
            {
                "variant": label,
                "score_col": score_col,
                "min_score": float(min_score),
                "n_pred": int(len(pred_df)),
                "mAP": float(mean_ap),
                "AP@0.1": float(evaluator.mAP[0]),
                "AP@0.3": float(evaluator.mAP[1]),
                "AP@0.5": float(evaluator.mAP[2]),
                "AP@0.7": float(evaluator.mAP[3]),
                "recall@0.1": float((best_iou >= 0.1).mean()) if len(best_iou) else float("nan"),
                "recall@0.3": float((best_iou >= 0.3).mean()) if len(best_iou) else float("nan"),
                "recall@0.5": float((best_iou >= 0.5).mean()) if len(best_iou) else float("nan"),
                "recall@0.7": float((best_iou >= 0.7).mean()) if len(best_iou) else float("nan"),
                "missed@0.1": int((best_iou < 0.1).sum()) if len(best_iou) else 0,
                "missed@0.5": int((best_iou < 0.5).sum()) if len(best_iou) else 0,
                "missed@0.7": int((best_iou < 0.7).sum()) if len(best_iou) else 0,
            }
        )
    return rows


def evaluate_all_scores(
    df: pd.DataFrame,
    label: str,
    args: argparse.Namespace,
    pred_dir: Path,
    epoch: int | str,
) -> list[dict]:
    cols = [
        "quality_score",
        "quality_x_cnn",
        "sqrt_quality_x_cnn",
        "quality_avg_cnn",
        "quality_score_base_priority_050",
        "quality_score_base_priority_080",
        "quality_avg_cnn_base_priority_050",
        "quality_avg_cnn_base_priority_080",
    ]
    dynamic_cols = [
        col for col in df.columns
        if col.startswith("quality_t") or col.startswith("quality_multi_")
    ]
    for col in dynamic_cols:
        if col not in cols and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if args.score_cols:
        missing = sorted(set(args.score_cols) - set(cols))
        if missing:
            raise ValueError(f"Unknown score columns: {missing}. Available: {cols}")
        cols = args.score_cols
    rows = []
    for col in cols:
        rows.extend(evaluate_score(df, col, label, args, pred_dir, suffix=f"epoch{epoch}"))
    return rows


def limit_frame(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def print_summary(name: str, df: pd.DataFrame) -> None:
    counts = df["sample_kind"].value_counts().to_dict()
    q = df["quality_target"].to_numpy(dtype=np.float64)
    print(
        f"[INFO] {name}: n={len(df)} pos={counts.get('positive', 0)} "
        f"semi={counts.get('semi_positive', 0)} hard={counts.get('hard_negative', 0)} "
        f"easy={counts.get('easy_negative', 0)} q_mean={q.mean():.4f} q_max={q.max():.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.min_gt_duration < 0:
        raise ValueError("--min-gt-duration must be non-negative")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.num_workers = 0
        args.max_train_proposals = args.max_train_proposals or 256
        args.max_val_proposals = args.max_val_proposals or 256
        args.max_train_samples = min(args.max_train_samples, 256)
        args.quiet_progress = True

    set_seed(args.seed)
    out_dir = resolve_path(args.out_dir)
    cache_dir = out_dir / "cache"
    pred_dir = out_dir / "predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ann_path = resolve_path(args.ann_path)
    val_props = pd.read_csv(resolve_path(args.val_proposals)).reset_index(drop=True)
    val_props = limit_frame(val_props, args.max_val_proposals, args.seed + 1)
    val_split = split_from_proposals(val_props, ann_path)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")
    if args.eval_checkpoint:
        try:
            checkpoint = torch.load(resolve_path(args.eval_checkpoint), map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resolve_path(args.eval_checkpoint), map_location=device)
        if checkpoint.get("multi_quality_head") or checkpoint.get("quality_thresholds"):
            args.multi_quality_head = True
            args.quality_thresholds = [float(t) for t in checkpoint.get("quality_thresholds", args.quality_thresholds)]
        args.numeric_only = bool(checkpoint.get("numeric_only", args.numeric_only))
        args.head_architecture = str(checkpoint.get("head_architecture", "mlp"))
        checkpoint_args = checkpoint.get("args", {})
        args.decoupled_boundary_head = bool(
            checkpoint_args.get("decoupled_boundary_head", args.decoupled_boundary_head)
        )
        args.context_roles = list(checkpoint_args.get("context_roles", args.context_roles))
        args.context_window_scale = float(checkpoint_args.get("context_window_scale", args.context_window_scale))
        args.context_feature_mode = str(checkpoint_args.get("context_feature_mode", args.context_feature_mode))
        if not args.drop_temporal_descriptor_groups:
            args.drop_temporal_descriptor_groups = list(
                checkpoint_args.get("drop_temporal_descriptor_groups", [])
            )

        print(f"[INFO] Eval proposals={len(val_props)} split={val_split}")
        val_repr_path = resolve_path(args.val_repr) if args.val_repr else cache_dir / "eval_repr.npz"
        val_embeddings, val_logits = collect_or_load_representations(val_props, val_repr_path, args, device)
        val_embeddings = drop_temporal_descriptor_groups(
            val_embeddings,
            args.drop_temporal_descriptor_groups,
        )
        val_context_dir = resolve_path(args.val_context_dir) if args.val_context_dir else cache_dir / "eval_context"
        val_context_embeddings, val_context_logits = collect_or_load_context_features(
            val_props, val_context_dir, args, device
        )
        val_embeddings = combine_context_embeddings(val_embeddings, val_context_embeddings, args)
        val_df = prepare_frame(val_props, val_logits, val_split, ann_path, args, val_context_logits)
        val_df.to_csv(cache_dir / f"{args.eval_label}_quality_labels.csv", index=False)
        print_summary(args.eval_label, val_df)

        cfg_data = checkpoint["config"]
        scaler = checkpoint["scaler"]
        val_num = val_df[checkpoint["numeric_columns"]].to_numpy(dtype=np.float32)
        if args.numeric_only:
            val_x_raw = val_num
        else:
            val_x_raw = np.concatenate([val_embeddings.astype(np.float32), val_num], axis=1)
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
        val_x = (val_x_raw - mean) / std
        head_outputs = int(checkpoint.get("head_outputs", 1))
        numeric_dim = int(checkpoint.get("numeric_dim", len(checkpoint["numeric_columns"])))
        model = make_quality_head(
            args,
            in_dim=val_x.shape[1],
            numeric_dim=numeric_dim,
            hidden=int(cfg_data["hidden"]),
            dropout=float(cfg_data["dropout"]),
            out_dim=head_outputs,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        val_scored = score_val(model, val_df, val_x.astype(np.float32), device, args)
        val_scored.to_csv(cache_dir / f"{args.eval_label}_scores_{Path(args.eval_checkpoint).stem}.csv", index=False)

        if args.skip_evaluation:
            print(f"[INFO] Scores written for {args.eval_label}; evaluation skipped")
            return

        rows = evaluate_score(val_df, "cnn_score", "base_cnn", args, pred_dir, suffix=f"{args.eval_label}_baseline")
        rows.extend(evaluate_all_scores(val_scored, Path(args.eval_checkpoint).stem, args, pred_dir, args.eval_label))
        summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
        summary.to_csv(out_dir / f"summary_{args.eval_label}.csv", index=False)
        best = summary.iloc[0].to_dict()
        (out_dir / f"best_{args.eval_label}.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print("\n[RESULTADO EVAL]")
        print(summary.head(20).to_string(index=False))
        return

    if args.train_proposals is None:
        raise ValueError("--train-proposals is required unless --eval-checkpoint is used")

    train_props = pd.read_csv(resolve_path(args.train_proposals)).reset_index(drop=True)
    train_props = limit_frame(train_props, args.max_train_proposals, args.seed)
    train_split = split_from_proposals(train_props, ann_path)
    print(f"[INFO] Train proposals={len(train_props)} split={train_split}; val proposals={len(val_props)} split={val_split}")

    train_repr_path = resolve_path(args.train_repr) if args.train_repr else cache_dir / "train_repr.npz"
    val_repr_path = resolve_path(args.val_repr) if args.val_repr else cache_dir / "val_repr.npz"
    train_embeddings, train_logits = collect_or_load_representations(train_props, train_repr_path, args, device)
    val_embeddings, val_logits = collect_or_load_representations(val_props, val_repr_path, args, device)
    train_embeddings = drop_temporal_descriptor_groups(
        train_embeddings,
        args.drop_temporal_descriptor_groups,
    )
    val_embeddings = drop_temporal_descriptor_groups(
        val_embeddings,
        args.drop_temporal_descriptor_groups,
    )
    train_context_dir = resolve_path(args.train_context_dir) if args.train_context_dir else cache_dir / "train_context"
    val_context_dir = resolve_path(args.val_context_dir) if args.val_context_dir else cache_dir / "val_context"
    train_context_embeddings, train_context_logits = collect_or_load_context_features(
        train_props, train_context_dir, args, device
    )
    val_context_embeddings, val_context_logits = collect_or_load_context_features(
        val_props, val_context_dir, args, device
    )
    train_embeddings = combine_context_embeddings(train_embeddings, train_context_embeddings, args)
    val_embeddings = combine_context_embeddings(val_embeddings, val_context_embeddings, args)

    train_labels_path = cache_dir / "train_quality_labels.csv"
    val_labels_path = cache_dir / "val_quality_labels.csv"
    reuse_labels = (
        args.reuse_labeled_cache
        and train_labels_path.exists()
        and val_labels_path.exists()
    )
    if reuse_labels:
        train_df = load_labeled_frame(train_labels_path)
        val_df = load_labeled_frame(val_labels_path)
        if len(train_df) != len(train_props) or len(val_df) != len(val_props):
            raise ValueError(
                "Cached quality labels do not match proposal rows: "
                f"train {len(train_df)}/{len(train_props)}, val {len(val_df)}/{len(val_props)}"
            )
        print(f"[INFO] Reused labeled cache from {cache_dir}")
    else:
        train_df = prepare_frame(
            train_props,
            train_logits,
            train_split,
            ann_path,
            args,
            train_context_logits,
        )
        val_df = prepare_frame(
            val_props,
            val_logits,
            val_split,
            ann_path,
            args,
            val_context_logits,
        )
        train_df.to_csv(train_labels_path, index=False)
        val_df.to_csv(val_labels_path, index=False)
    if args.oof_hardness_csv:
        hardness = pd.read_csv(resolve_path(args.oof_hardness_csv))
        train_df, promoted = apply_oof_hardness(
            train_df,
            hardness,
            args.oof_hardness_threshold,
            args.neg_tiou,
        )
        print(
            f"[INFO] OOF hard-negative mining promoted={promoted} "
            f"threshold={args.oof_hardness_threshold:g}"
        )
    print_summary("train", train_df)
    print_summary("val", val_df)

    if args.skip_baseline_evaluation:
        baseline_rows = []
        best_base = {"mAP": -float("inf")}
    else:
        baseline_rows = evaluate_score(
            val_df,
            "cnn_score",
            "base_cnn",
            args,
            pred_dir,
            suffix="baseline",
        )
        best_base = max(baseline_rows, key=lambda row: row["mAP"])
        print(
            f"[BASE] mAP={best_base['mAP']:.4f} AP@0.5={best_base['AP@0.5']:.4f} "
            f"AP@0.7={best_base['AP@0.7']:.4f} n={best_base['n_pred']} min={best_base['min_score']}"
        )

    selected_configs = CONFIGS
    if args.configs:
        requested = set(args.configs)
        selected_configs = [cfg for cfg in CONFIGS if cfg.name in requested]
        missing = sorted(requested - {cfg.name for cfg in selected_configs})
        if missing:
            raise ValueError(f"Unknown configs: {missing}")

    rows = baseline_rows
    best = best_base
    for cfg in selected_configs:
        val_scored, metrics = train_config(
            cfg, train_df, val_df, train_embeddings, val_embeddings,
            args, device, out_dir, pred_dir,
        )
        val_scored.to_csv(cache_dir / f"val_scores_{cfg.name}.csv", index=False)
        rows.append(metrics)
        print(
            f"[{cfg.name}] best={metrics['score_col']} mAP={metrics['mAP']:.4f} "
            f"AP@0.5={metrics['AP@0.5']:.4f} AP@0.7={metrics['AP@0.7']:.4f} "
            f"n={metrics['n_pred']} min={metrics['min_score']}"
        )
        if metrics["mAP"] > best["mAP"]:
            best = metrics

    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("\n[RESULTADO]")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
