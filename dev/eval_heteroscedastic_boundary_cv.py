"""Cross-fit a candidate-conditioned uncertainty-aware boundary head.

The head pools frozen dense ATSN features immediately outside and inside both
candidate boundaries. It predicts a Gaussian mean/variance for each relative
boundary offset plus proposal tIoU. Training candidates are jittered only on
source recordings so arbitrary proposals that match the same action learn a
consistent correction. The official test split is never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.diagnose_atsn_linear_separability import load_annotations  # noqa: E402
from dev.eval_actionness_quality_head_cv import recording_weights  # noqa: E402
from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_final_boundary_gradient_cv import (  # noqa: E402
    frame_prediction,
    prediction_frame,
    source_prediction_path,
)
from src.evaluation import segment_iou  # noqa: E402


ROLE_COUNT = 6
SCALAR_COLUMNS = (
    "log_duration",
    "score",
    "score_global_rank",
    "score_recording_rank",
    "score_roi_rank",
    "context_ratio",
    "start_cosine_distance",
    "end_cosine_distance",
    "start_relative_l2",
    "end_relative_l2",
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument(
        "--feature-dir",
        default="tmp/temporalmaxer_continuous/source_features_v1",
    )
    parser.add_argument("--feature-array-name", default="frame_features.npy")
    parser.add_argument(
        "--sequences-path",
        default=None,
        help="Defaults to sequences.csv inside --feature-dir.",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/heteroscedastic_boundary_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--positive-tiou", type=float, default=0.1)
    parser.add_argument("--max-relative-offset", type=float, default=0.5)
    parser.add_argument("--jitter-copies", type=int, default=3)
    parser.add_argument("--jitter-fraction", type=float, default=0.15)
    parser.add_argument("--context-ratio", type=float, default=0.25)
    parser.add_argument("--min-context-s", type=float, default=1.0)
    parser.add_argument("--max-context-s", type=float, default=4.0)
    parser.add_argument(
        "--blends", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--score-weights", type=float, nargs="+", default=[0.1, 0.2, 0.3]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026])
    return parser.parse_args()


@dataclass
class DenseFeatureStore:
    feature_dir: Path
    feature_array_name: str = "frame_features.npy"
    sequences_path: Path | None = None

    def __post_init__(self) -> None:
        metadata = json.loads(
            (self.feature_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.stride_s = float(metadata["grid_stride_s"])
        self.feature_dim = int(metadata["feature_dim"])
        self.features = np.load(
            self.feature_dir / self.feature_array_name, mmap_mode="r"
        )
        sequence_path = (
            self.sequences_path
            if self.sequences_path is not None
            else self.feature_dir / "sequences.csv"
        )
        sequences = pd.read_csv(sequence_path)
        self.rows = {
            (str(row.rec_name), int(row.roi_id)): row
            for row in sequences.itertuples(index=False)
        }

    def sequence(self, recording: str, roi_id: int) -> np.ndarray:
        row = self.rows[(str(recording), int(roi_id))]
        offset = int(row.offset)
        length = int(row.length)
        return np.asarray(self.features[offset : offset + length], dtype=np.float32)

    def duration(self, recording: str, roi_id: int) -> float:
        return float(self.rows[(str(recording), int(roi_id))].duration_s)


def add_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["score_global_rank"] = output["score"].rank(
        method="average", pct=True
    )
    output["score_recording_rank"] = output.groupby("rec_name")["score"].rank(
        method="average", pct=True
    )
    output["score_roi_rank"] = output.groupby(["rec_name", "roi_id"])[
        "score"
    ].rank(method="average", pct=True)
    return output


def interval_mean(
    sequence: np.ndarray,
    start_s: float,
    end_s: float,
    stride_s: float,
) -> np.ndarray:
    """Mean a half-open temporal interval, falling back to its closest point."""
    if len(sequence) == 0:
        raise ValueError("Cannot pool an empty feature sequence")
    low = max(0, int(math.floor(start_s / stride_s)))
    high = min(len(sequence), int(math.ceil(end_s / stride_s)))
    if high <= low:
        center = 0.5 * (start_s + end_s)
        index = int(np.clip(round(center / stride_s), 0, len(sequence) - 1))
        return sequence[index].astype(np.float32, copy=False)
    return sequence[low:high].mean(axis=0, dtype=np.float64).astype(np.float32)


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(first) * np.linalg.norm(second)),
        1e-8,
    )
    return float(1.0 - np.dot(first, second) / denominator)


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(
        0.5 * float(np.linalg.norm(first) + np.linalg.norm(second)),
        1e-8,
    )
    return float(np.linalg.norm(first - second) / denominator)


def candidate_descriptor(
    sequence: np.ndarray,
    start_s: float,
    end_s: float,
    score: float,
    ranks: tuple[float, float, float],
    stride_s: float,
    context_ratio: float,
    min_context_s: float,
    max_context_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    duration = max(float(end_s - start_s), 1e-6)
    context = float(
        np.clip(context_ratio * duration, min_context_s, max_context_s)
    )
    roles = np.stack(
        (
            interval_mean(sequence, start_s - context, start_s, stride_s),
            interval_mean(sequence, start_s, start_s + context, stride_s),
            interval_mean(sequence, end_s - context, end_s, stride_s),
            interval_mean(sequence, end_s, end_s + context, stride_s),
            interval_mean(sequence, start_s, end_s, stride_s),
            interval_mean(
                sequence,
                start_s - context,
                end_s + context,
                stride_s,
            ),
        )
    ).astype(np.float32)
    scalars = np.asarray(
        (
            np.log1p(duration),
            float(score),
            *map(float, ranks),
            context / duration,
            cosine_distance(roles[0], roles[1]),
            cosine_distance(roles[2], roles[3]),
            relative_l2(roles[0], roles[1]),
            relative_l2(roles[2], roles[3]),
        ),
        dtype=np.float32,
    )
    return roles, scalars


def extract_descriptors(
    frame: pd.DataFrame,
    store: DenseFeatureStore,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    role_values = []
    scalar_values = []
    sequence_cache: dict[tuple[str, int], np.ndarray] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.rec_name), int(row.roi_id))
        if key not in sequence_cache:
            sequence_cache[key] = store.sequence(*key)
        roles, scalars = candidate_descriptor(
            sequence_cache[key],
            float(row.t_start),
            float(row.t_end),
            float(row.score),
            (
                float(row.score_global_rank),
                float(row.score_recording_rank),
                float(row.score_roi_rank),
            ),
            store.stride_s,
            args.context_ratio,
            args.min_context_s,
            args.max_context_s,
        )
        role_values.append(roles)
        scalar_values.append(scalars)
    return np.stack(role_values), np.stack(scalar_values)


def add_targets(
    frame: pd.DataFrame,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
    max_relative_offset: float,
) -> pd.DataFrame:
    qualities = []
    offsets = []
    for row in frame.itertuples(index=False):
        targets = annotations.get((str(row.rec_name), int(row.roi_id)), [])
        duration = max(float(row.t_end) - float(row.t_start), 1e-6)
        if not targets:
            qualities.append(0.0)
            offsets.append((0.0, 0.0))
            continue
        overlaps = segment_iou(
            np.asarray([row.t_start, row.t_end], dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
        )
        target_index = int(np.argmax(overlaps))
        target_start, target_end = targets[target_index]
        qualities.append(float(overlaps[target_index]))
        offsets.append(
            (
                np.clip(
                    (float(target_start) - float(row.t_start)) / duration,
                    -max_relative_offset,
                    max_relative_offset,
                ),
                np.clip(
                    (float(target_end) - float(row.t_end)) / duration,
                    -max_relative_offset,
                    max_relative_offset,
                ),
            )
        )
    output = frame.copy()
    output["target_tiou"] = qualities
    output["target_start_delta"] = [value[0] for value in offsets]
    output["target_end_delta"] = [value[1] for value in offsets]
    return output


def jitter_candidates(
    frame: pd.DataFrame,
    store: DenseFeatureStore,
    copies: int,
    fraction: float,
    seed: int,
    minimum_duration_s: float = 2.0,
) -> pd.DataFrame:
    if copies < 0 or fraction < 0:
        raise ValueError("Jitter copies and fraction must be non-negative")
    parts = [frame.assign(jitter_copy=0)]
    rng = np.random.default_rng(seed)
    for copy_index in range(1, copies + 1):
        jittered = frame.copy()
        starts = jittered["t_start"].to_numpy(np.float64).copy()
        ends = jittered["t_end"].to_numpy(np.float64).copy()
        durations = np.maximum(ends - starts, minimum_duration_s)
        starts += rng.uniform(-fraction, fraction, len(frame)) * durations
        ends += rng.uniform(-fraction, fraction, len(frame)) * durations
        for index, row in enumerate(jittered.itertuples(index=False)):
            limit = store.duration(str(row.rec_name), int(row.roi_id))
            starts[index] = np.clip(starts[index], 0.0, limit)
            ends[index] = np.clip(ends[index], 0.0, limit)
            if ends[index] - starts[index] < minimum_duration_s:
                center = np.clip(
                    0.5 * (starts[index] + ends[index]),
                    0.5 * minimum_duration_s,
                    max(0.5 * minimum_duration_s, limit - 0.5 * minimum_duration_s),
                )
                starts[index] = max(0.0, center - 0.5 * minimum_duration_s)
                ends[index] = min(limit, center + 0.5 * minimum_duration_s)
        jittered["t_start"] = starts
        jittered["t_end"] = ends
        jittered["jitter_copy"] = copy_index
        parts.append(jittered)
    return pd.concat(parts, ignore_index=True)


class HeteroscedasticBoundaryHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        scalar_dim: int,
        hidden_dim: int,
        dropout: float,
        max_relative_offset: float,
    ) -> None:
        super().__init__()
        self.max_relative_offset = float(max_relative_offset)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.role_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(ROLE_COUNT * hidden_dim + scalar_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 5),
        )
        with torch.no_grad():
            self.output[-1].bias.zero_()
            self.output[-1].bias[2:4].fill_(-2.0)

    def forward(
        self,
        roles: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if roles.ndim != 3 or roles.shape[1] != ROLE_COUNT:
            raise ValueError(f"Expected roles [B,{ROLE_COUNT},D]")
        projected = self.role_projection(self.feature_norm(roles)).flatten(1)
        output = self.output(torch.cat((projected, scalars), dim=1))
        mean = torch.tanh(output[:, :2]) * self.max_relative_offset
        log_variance = output[:, 2:4].clamp(-6.0, 2.0)
        return mean, log_variance, output[:, 4]


def boundary_quality_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    quality_logit: torch.Tensor,
    target_offsets: torch.Tensor,
    target_quality: torch.Tensor,
    sample_weight: torch.Tensor,
    positive_tiou: float,
    boundary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = quality_logit.sigmoid()
    qfl = F.binary_cross_entropy_with_logits(
        quality_logit, target_quality, reduction="none"
    ) * (target_quality - probabilities).abs().square()
    quality_loss = (qfl * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    positive = target_quality >= positive_tiou
    if positive.any():
        nll = 0.5 * (
            torch.exp(-log_variance) * (mean - target_offsets).square()
            + log_variance
        ).mean(dim=1)
        positive_weight = (
            sample_weight[positive] * target_quality[positive].square()
        )
        boundary_loss = (
            nll[positive] * positive_weight
        ).sum() / positive_weight.sum().clamp_min(1e-8)
    else:
        boundary_loss = mean.sum() * 0.0
    return (
        quality_loss + boundary_weight * boundary_loss,
        quality_loss,
        boundary_loss,
    )


def standardize_scalars(
    scalars: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = scalars.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        scalars.std(axis=0, dtype=np.float64).astype(np.float32),
        1e-4,
    )
    return (scalars - mean) / std, mean, std


def fit_head(
    frame: pd.DataFrame,
    roles: np.ndarray,
    scalars: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[HeteroscedasticBoundaryHead, np.ndarray, np.ndarray]:
    set_seed(seed)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = HeteroscedasticBoundaryHead(
            int(state["feature_dim"]),
            int(state["scalar_dim"]),
            int(state["hidden_dim"]),
            float(state["dropout"]),
            float(state["max_relative_offset"]),
        ).to(device)
        model.load_state_dict(state["state_dict"])
        return (
            model,
            np.asarray(state["scalar_mean"], dtype=np.float32),
            np.asarray(state["scalar_std"], dtype=np.float32),
        )
    standardized, scalar_mean, scalar_std = standardize_scalars(scalars)
    target_offsets = frame[
        ["target_start_delta", "target_end_delta"]
    ].to_numpy(np.float32)
    target_quality = frame["target_tiou"].to_numpy(np.float32)
    sample_weight = recording_weights(frame).astype(np.float32)
    sample_weight /= max(float(sample_weight.mean()), 1e-8)
    dataset = TensorDataset(
        torch.from_numpy(roles),
        torch.from_numpy(standardized.astype(np.float32)),
        torch.from_numpy(target_offsets),
        torch.from_numpy(target_quality),
        torch.from_numpy(sample_weight),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = HeteroscedasticBoundaryHead(
        roles.shape[2],
        scalars.shape[1],
        args.hidden_dim,
        args.dropout,
        args.max_relative_offset,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            batch = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                prediction = model(batch[0], batch[1])
                loss, _, _ = boundary_quality_loss(
                    *prediction,
                    batch[2],
                    batch[3],
                    batch[4],
                    args.positive_tiou,
                    args.boundary_weight,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scalar_mean": scalar_mean,
            "scalar_std": scalar_std,
            "feature_dim": roles.shape[2],
            "scalar_dim": scalars.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "max_relative_offset": args.max_relative_offset,
            "seed": seed,
        },
        checkpoint_path,
    )
    return model, scalar_mean, scalar_std


@torch.no_grad()
def predict_ensemble(
    models: list[tuple[HeteroscedasticBoundaryHead, np.ndarray, np.ndarray]],
    roles: np.ndarray,
    scalars: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_means = []
    all_variances = []
    all_quality = []
    for model, scalar_mean, scalar_std in models:
        standardized = (scalars - scalar_mean) / scalar_std
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(roles),
                torch.from_numpy(standardized.astype(np.float32)),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        model.eval()
        means = []
        variances = []
        qualities = []
        for role_batch, scalar_batch in loader:
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                mean, log_variance, quality = model(
                    role_batch.to(device),
                    scalar_batch.to(device),
                )
            means.append(mean.float().cpu().numpy())
            variances.append(log_variance.exp().float().cpu().numpy())
            qualities.append(quality.sigmoid().float().cpu().numpy())
        all_means.append(np.concatenate(means))
        all_variances.append(np.concatenate(variances))
        all_quality.append(np.concatenate(qualities))
    mean_stack = np.stack(all_means)
    variance_stack = np.stack(all_variances)
    ensemble_mean = mean_stack.mean(axis=0)
    ensemble_variance = np.maximum(
        (variance_stack + np.square(mean_stack)).mean(axis=0)
        - np.square(ensemble_mean),
        1e-8,
    )
    return (
        ensemble_mean,
        ensemble_variance,
        np.stack(all_quality).mean(axis=0),
    )


def apply_refinement(
    frame: pd.DataFrame,
    mean: np.ndarray,
    variance: np.ndarray,
    quality: np.ndarray,
    blend: float,
    reliability_weighted: bool,
    minimum_duration_s: float = 2.0,
) -> pd.DataFrame:
    starts = frame["t_start"].to_numpy(np.float64)
    ends = frame["t_end"].to_numpy(np.float64)
    durations = np.maximum(ends - starts, minimum_duration_s)
    uncertainty = np.sqrt(np.maximum(variance, 1e-8)).mean(axis=1)
    reliability = np.clip(quality, 0.0, 1.0) * np.exp(-2.0 * uncertainty)
    factor = blend * (reliability if reliability_weighted else 1.0)
    refined_starts = starts + factor * durations * mean[:, 0]
    refined_ends = ends + factor * durations * mean[:, 1]
    valid = refined_ends - refined_starts >= minimum_duration_s
    output = frame.copy()
    output["t_start"] = np.where(valid, refined_starts, starts)
    output["t_end"] = np.where(valid, refined_ends, ends)
    output["boundary_quality"] = quality
    output["boundary_uncertainty"] = uncertainty
    output["boundary_reliability"] = reliability
    output["predicted_start_delta"] = mean[:, 0]
    output["predicted_end_delta"] = mean[:, 1]
    return output


def rerank_with_local_quality(
    frame: pd.DataFrame,
    local_quality: np.ndarray,
    weight: float,
) -> pd.DataFrame:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Score weight must be in [0,1]")
    output = frame.copy()
    original_rank = output["score"].rank(
        method="average", pct=True
    ).to_numpy(np.float64)
    local_rank = pd.Series(local_quality).rank(
        method="average", pct=True
    ).to_numpy(np.float64)
    output["score"] = (1.0 - weight) * original_rank + weight * local_rank
    return output


def load_source_frames(source_root: Path) -> pd.DataFrame:
    parts = []
    for fold in range(5):
        prediction = json.loads(
            source_prediction_path(source_root, fold).read_text(encoding="utf-8")
        )
        parts.append(add_rank_features(prediction_frame(prediction, fold)))
    return pd.concat(parts, ignore_index=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        rows.append(
            {
                "variant": variant,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mAP", ascending=False)


def main() -> None:
    args = parse_args()
    source_root = resolve(args.source_root)
    feature_dir = resolve(args.feature_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    annotations = load_annotations(resolve(args.ann_path), min_duration_s=2.0)
    store = DenseFeatureStore(
        feature_dir,
        args.feature_array_name,
        resolve(args.sequences_path) if args.sequences_path else None,
    )
    source = load_source_frames(source_root)
    device = torch.device(args.device)
    rows = []

    for fold in args.folds:
        fold_dir = out_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_base = source[source["fold"] != fold].reset_index(drop=True)
        validation = source[source["fold"] == fold].reset_index(drop=True)
        train = jitter_candidates(
            train_base,
            store,
            args.jitter_copies,
            args.jitter_fraction,
            seed=1000 + fold,
        )
        train = add_targets(train, annotations, args.max_relative_offset)
        train_roles, train_scalars = extract_descriptors(train, store, args)
        validation_roles, validation_scalars = extract_descriptors(
            validation, store, args
        )
        models = []
        for seed in args.seeds:
            models.append(
                fit_head(
                    train,
                    train_roles,
                    train_scalars,
                    args,
                    seed + fold * 100,
                    device,
                    fold_dir / f"head_seed_{seed}.pt",
                )
            )
        mean, variance, quality = predict_ensemble(
            models,
            validation_roles,
            validation_scalars,
            args.batch_size,
            device,
        )
        np.savez_compressed(
            fold_dir / "boundary_outputs.npz",
            mean=mean,
            variance=variance,
            quality=quality,
        )
        variants: list[tuple[str, pd.DataFrame]] = [("control", validation)]
        uncertainty = np.sqrt(np.maximum(variance, 1e-8)).mean(axis=1)
        reliability = np.clip(quality, 0.0, 1.0) * np.exp(-2.0 * uncertainty)
        reliable_boundary = apply_refinement(
            validation,
            mean,
            variance,
            quality,
            blend=1.0,
            reliability_weighted=True,
        )
        for blend in args.blends:
            label = int(round(100 * blend))
            variants.append(
                (
                    f"mean_w{label:03d}",
                    apply_refinement(
                        validation,
                        mean,
                        variance,
                        quality,
                        blend,
                        reliability_weighted=False,
                    ),
                )
            )
            variants.append(
                (
                    f"reliable_w{label:03d}",
                    apply_refinement(
                        validation,
                        mean,
                        variance,
                        quality,
                        blend,
                        reliability_weighted=True,
                    ),
                )
            )
        for score_weight in args.score_weights:
            label = int(round(100 * score_weight))
            variants.append(
                (
                    f"quality_score_w{label:03d}",
                    rerank_with_local_quality(
                        validation, quality, score_weight
                    ),
                )
            )
            variants.append(
                (
                    f"reliability_score_w{label:03d}",
                    rerank_with_local_quality(
                        validation, reliability, score_weight
                    ),
                )
            )
            variants.append(
                (
                    f"reliable_boundary_quality_score_w{label:03d}",
                    rerank_with_local_quality(
                        reliable_boundary, quality, score_weight
                    ),
                )
            )
            variants.append(
                (
                    f"reliable_boundary_reliability_score_w{label:03d}",
                    rerank_with_local_quality(
                        reliable_boundary, reliability, score_weight
                    ),
                )
            )
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for variant, frame in variants:
            prediction = frame_prediction(
                frame,
                f"source-heteroscedastic-boundary-{variant}",
            )
            prediction_path = fold_dir / "predictions" / f"{variant}.json"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        prediction_path,
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / "metrics_partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
