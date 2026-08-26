"""Train and evaluate full-ROI TemporalMaxer with recording-disjoint folds."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.evaluation import DetectionsEvaluator
from src.temporalmaxer_continuous import TemporalMaxerContinuous
from src.utils import temporal_soft_nms


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features")
    parser.add_argument("--feature-array-name", default="frame_features.npy")
    parser.add_argument(
        "--sequences-path",
        default=None,
        help="Sequence index path; defaults to sequences.csv inside feature-dir.",
    )
    parser.add_argument(
        "--standardize-features",
        action="store_true",
        help="Apply channel mean/std stored in the feature metadata.",
    )
    parser.add_argument("--auxiliary-feature-dir", default=None)
    parser.add_argument(
        "--cross-layer-task-decoupling",
        action="store_true",
        help=(
            "Use base features only for classification and an independent "
            "base+auxiliary projection for quality and localization."
        ),
    )
    parser.add_argument(
        "--feature-normalization",
        choices=("none", "temporal-center", "temporal-zscore"),
        default="none",
        help="Remove recording/ROI feature statistics without using labels.",
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--min-action-duration",
        type=float,
        default=2.0,
        help=(
            "Minimum GT and decoded duration in seconds. The 2 s default preserves "
            "the source ED recipe; THUMOS14 target training must pass 0.0."
        ),
    )
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pyramid-levels", type=int, default=6)
    parser.add_argument("--head-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--tanp",
        action="store_true",
        help="Apply temporal-aware normalization perturbation after input projection.",
    )
    parser.add_argument("--tanp-std", type=float, default=0.75)
    parser.add_argument("--tanp-probability", type=float, default=1.0)
    parser.add_argument(
        "--background-mix",
        action="store_true",
        help="Mix every training sequence with a median background from another recording.",
    )
    parser.add_argument("--background-mix-probability", type=float, default=1.0)
    parser.add_argument(
        "--mixstyle",
        action="store_true",
        help=(
            "Mix per-channel temporal feature statistics with another source "
            "recording while preserving normalized temporal content."
        ),
    )
    parser.add_argument("--mixstyle-probability", type=float, default=0.5)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument(
        "--pal-action-transplant",
        action="store_true",
        help=(
            "Paste a source ED instance into background from another training "
            "recording, following UP-TAL's cross-background PAL principle."
        ),
    )
    parser.add_argument("--pal-action-transplant-probability", type=float, default=0.5)
    parser.add_argument("--pal-action-transplant-margin-bins", type=int, default=2)
    parser.add_argument("--pal-blend-min", type=float, default=1.0)
    parser.add_argument("--pal-blend-max", type=float, default=1.0)
    parser.add_argument("--pal-consistency-weight", type=float, default=0.0)
    parser.add_argument("--pal-consistency-temperature", type=float, default=0.07)
    parser.add_argument(
        "--temporal-reversal-probability",
        type=float,
        default=0.0,
        help="Reverse training sequences and their boundaries with this probability.",
    )
    parser.add_argument(
        "--hard-negative-recordings",
        default="",
        help="Comma-separated rec_name list to oversample during training (score-inversion domains).",
    )
    parser.add_argument(
        "--hard-negative-oversample",
        type=float,
        default=1.0,
        help="Sampling weight multiplier applied to hard-negative recordings (1.0 = disabled).",
    )
    parser.add_argument(
        "--unlabeled-feature-dir",
        default=None,
        help="Feature dir for a Mean-Teacher consistency loss on unlabeled target-domain recordings.",
    )
    parser.add_argument("--unlabeled-recordings", nargs="+", default=None)
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="Weight of the student/EMA-teacher classification consistency loss (0 = disabled).",
    )
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument(
        "--temporal-order",
        action="store_true",
        help="Add GLAD's self-supervised temporal clip order task.",
    )
    parser.add_argument("--temporal-order-chunks", type=int, default=3)
    parser.add_argument("--temporal-order-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--reg-max", type=int, default=0)
    parser.add_argument(
        "--trident-bins",
        type=int,
        default=0,
        help="Use an official-style Trident relative boundary head with this radius.",
    )
    parser.add_argument(
        "--neck-type",
        choices=("maxpool", "attention"),
        default="maxpool",
        help=(
            "Temporal neck. 'maxpool' is TemporalMaxer (default, used for every result "
            "so far); 'attention' adds ActionFormer-style windowed self-attention at "
            "each pyramid level, for the architecture comparison."
        ),
    )
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--attention-window",
        type=int,
        default=19,
        help="Odd number of bins each position attends over when --neck-type=attention.",
    )
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--center-sampling-radius", type=float, default=0.0)
    parser.add_argument("--empty-sequence-weight", type=float, default=1.0)
    parser.add_argument("--use-boundary-heads", action="store_true")
    parser.add_argument("--boundary-weight", type=float, default=0.25)
    parser.add_argument("--boundary-target-sigma", type=float, default=1.0)
    parser.add_argument("--boundary-refine-radius", type=float, default=2.0)
    parser.add_argument("--boundary-refine-blend", type=float, default=0.5)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument(
        "--rank-sort",
        action="store_true",
        help="Replace focal classification with IoU-targeted Rank & Sort loss.",
    )
    parser.add_argument("--rank-sort-delta", type=float, default=0.5)
    parser.add_argument(
        "--group-dro",
        action="store_true",
        help="Optimize worst-recording risk using per-sequence GroupDRO losses.",
    )
    parser.add_argument("--group-dro-eta", type=float, default=0.01)
    parser.add_argument("--disable-quality", action="store_true")
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.005)
    parser.add_argument("--pre-nms-topk", type=int, default=500)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--max-predictions-per-roi", type=int, default=200)
    parser.add_argument(
        "--tiou",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7],
        help="Evaluation thresholds; THUMOS14-E uses 0.3 0.4 0.5 0.6 0.7.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Warm-start model weights only (not optimizer/scheduler) from this checkpoint, "
        "e.g. for short self-training fine-tunes. Ignored if last.pt already exists in out-dir.",
    )
    parser.add_argument(
        "--epochs-per-process",
        type=int,
        default=0,
        help="Exit with code 75 after N epochs so a supervisor can resume from last.pt.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_annotations(
    path: Path, min_duration_s: float = 2.0
) -> dict[tuple[str, int], np.ndarray]:
    if min_duration_s < 0:
        raise ValueError("min_duration_s must be non-negative")
    database = json.loads(path.read_text(encoding="utf-8"))["database"]
    result: dict[tuple[str, int], np.ndarray] = {}
    for recording, value in database.items():
        for roi, annotations in value.get("annotations", {}).items():
            if roi == "null":
                continue
            segments = [
                [float(item["segment"][0]), float(item["segment"][1])]
                for item in annotations
                if item["label"] == "ed"
                and float(item["segment"][1]) - float(item["segment"][0])
                >= min_duration_s
            ]
            result[(recording, int(roi))] = np.asarray(segments, dtype=np.float32).reshape(-1, 2)
    return result


def split_fold_sequences(
    sequences: pd.DataFrame, fold_manifest: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select only the manifest-declared train/validation recordings."""
    required = {"fold", "train_record_names", "val_record_names"}
    missing = required - set(fold_manifest.columns)
    if missing:
        raise ValueError(f"Fold manifest lacks columns {sorted(missing)}")
    if fold_manifest["fold"].duplicated().any():
        raise ValueError("Fold manifest contains duplicate fold identifiers")
    indexed = fold_manifest.set_index("fold")
    if fold not in indexed.index:
        raise ValueError(f"Fold {fold} is not present in the fold manifest")
    row = indexed.loc[fold]
    train_recordings = set(str(row["train_record_names"]).split())
    val_recordings = set(str(row["val_record_names"]).split())
    if not train_recordings or not val_recordings:
        raise ValueError("Training or validation recording list is empty")
    overlap = train_recordings & val_recordings
    if overlap:
        raise ValueError(f"Train/validation recording overlap: {sorted(overlap)}")
    available = set(sequences["rec_name"].astype(str))
    missing_recordings = (train_recordings | val_recordings) - available
    if missing_recordings:
        raise ValueError(
            f"Fold manifest references missing recordings: {sorted(missing_recordings)}"
        )
    train_sequences = sequences[
        sequences["rec_name"].astype(str).isin(train_recordings)
    ].copy()
    val_sequences = sequences[
        sequences["rec_name"].astype(str).isin(val_recordings)
    ].copy()
    if train_sequences.empty or val_sequences.empty:
        raise ValueError("Training or validation sequence split is empty")
    return train_sequences, val_sequences


def manifest_validation_recordings(fold_manifest: pd.DataFrame) -> set[str]:
    """Return the declared validation-pool recordings across all CV rows."""
    required = {"train_record_names", "val_record_names"}
    missing = required - set(fold_manifest.columns)
    if missing:
        raise ValueError(f"Fold manifest lacks columns {sorted(missing)}")
    recordings: set[str] = set()
    for column in required:
        for value in fold_manifest[column]:
            recordings.update(str(value).split())
    if not recordings:
        raise ValueError("Fold manifest does not declare any recordings")
    return recordings


def normalize_temporal_features(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return features
    if mode not in {"temporal-center", "temporal-zscore"}:
        raise ValueError(f"Unknown feature normalization: {mode}")
    normalized = features - features.mean(axis=0, dtype=np.float64).astype(np.float32)
    if mode == "temporal-zscore":
        std = normalized.std(axis=0, dtype=np.float64).astype(np.float32)
        normalized /= np.maximum(std, 1e-4)
    return normalized


def align_temporal_feature_statistics(
    features: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    blend: float,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> np.ndarray:
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Feature-alignment blend must be in [0,1]")
    if target_mean is None:
        target_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    if target_std is None:
        target_std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    aligned = (features - target_mean) / np.maximum(target_std, 1e-4)
    aligned = aligned * source_std + source_mean
    return (1.0 - blend) * features + blend * aligned


def feature_background_mix(
    features: np.ndarray, donor_background: np.ndarray, mix_ratio: float
) -> np.ndarray:
    """GLAD-style background mixing in frozen feature space."""
    if not 0.0 <= mix_ratio <= 1.0:
        raise ValueError("Background mix ratio must be in [0,1]")
    if donor_background.shape != (features.shape[1],):
        raise ValueError("Donor background dimension does not match the sequence")
    return (1.0 - mix_ratio) * features + mix_ratio * donor_background[None, :]


def feature_mixstyle(
    features: np.ndarray,
    donor_mean: np.ndarray,
    donor_std: np.ndarray,
    recipient_weight: float,
) -> np.ndarray:
    """Mix temporal feature statistics while preserving recipient content."""
    if not 0.0 <= recipient_weight <= 1.0:
        raise ValueError("MixStyle recipient weight must lie in [0,1]")
    if donor_mean.shape != (features.shape[1],) or donor_std.shape != (
        features.shape[1],
    ):
        raise ValueError("MixStyle donor statistics have an invalid dimension")
    recipient_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    recipient_std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    normalized = (features - recipient_mean) / np.maximum(recipient_std, 1e-6)
    mixed_mean = (
        recipient_weight * recipient_mean
        + (1.0 - recipient_weight) * donor_mean
    )
    mixed_std = (
        recipient_weight * recipient_std
        + (1.0 - recipient_weight) * donor_std
    )
    return normalized * np.maximum(mixed_std, 1e-6) + mixed_mean


def reverse_temporal_sample(
    features: np.ndarray,
    segments: np.ndarray,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse a sequence and map [start,end] annotations equivariantly."""
    if duration_s <= 0:
        raise ValueError("Sequence duration must be positive")
    reversed_segments = segments.copy()
    if len(reversed_segments):
        reversed_segments = np.column_stack(
            (
                duration_s - segments[:, 1],
                duration_s - segments[:, 0],
            )
        ).astype(np.float32)
        reversed_segments = reversed_segments[np.argsort(reversed_segments[:, 0])]
    return features[::-1].copy(), reversed_segments


@dataclass(frozen=True)
class ActionRegion:
    sequence_index: int
    rec_name: str
    crop_start: int
    crop_end: int
    segment_start_offset_s: float
    segment_end_offset_s: float


def build_action_regions(
    sequences: pd.DataFrame,
    annotations: dict[tuple[str, int], np.ndarray],
    grid_stride_s: float,
) -> list[ActionRegion]:
    """Index source-only ED regions while retaining sub-bin GT offsets."""
    if grid_stride_s <= 0:
        raise ValueError("PAL grid stride must be positive")
    regions = []
    for sequence_index, row in sequences.reset_index(drop=True).iterrows():
        segments = annotations.get(
            (str(row["rec_name"]), int(row["roi_id"])),
            np.empty((0, 2), dtype=np.float32),
        )
        for start_s, end_s in segments:
            crop_start = max(0, int(np.floor(float(start_s) / grid_stride_s)))
            crop_end = min(
                int(row["length"]),
                int(np.ceil(float(end_s) / grid_stride_s)),
            )
            if crop_end <= crop_start:
                continue
            crop_start_s = crop_start * grid_stride_s
            regions.append(
                ActionRegion(
                    sequence_index=int(sequence_index),
                    rec_name=str(row["rec_name"]),
                    crop_start=crop_start,
                    crop_end=crop_end,
                    segment_start_offset_s=float(start_s) - crop_start_s,
                    segment_end_offset_s=float(end_s) - crop_start_s,
                )
            )
    return regions


def valid_transplant_starts(
    sequence_length: int,
    region_length: int,
    segments: np.ndarray,
    grid_stride_s: float,
    margin_bins: int,
) -> np.ndarray:
    """Return placements whose pasted crop does not overlap labelled actions."""
    if sequence_length < region_length:
        return np.empty(0, dtype=np.int64)
    if grid_stride_s <= 0 or margin_bins < 0:
        raise ValueError("Invalid PAL stride or margin")
    occupied = np.zeros(sequence_length, dtype=bool)
    for start_s, end_s in segments:
        start = max(0, int(np.floor(float(start_s) / grid_stride_s)) - margin_bins)
        end = min(
            sequence_length,
            int(np.ceil(float(end_s) / grid_stride_s)) + margin_bins,
        )
        occupied[start:end] = True
    valid = np.convolve(
        occupied.astype(np.int8),
        np.ones(region_length, dtype=np.int16),
        mode="valid",
    )
    return np.flatnonzero(valid == 0).astype(np.int64)


def transplant_action_region(
    recipient_features: np.ndarray,
    recipient_segments: np.ndarray,
    donor_features: np.ndarray,
    donor_region: ActionRegion,
    destination_start: int,
    grid_stride_s: float,
    blend_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend one real ED region into background and append its translated annotation."""
    if not 0.0 <= blend_ratio <= 1.0:
        raise ValueError("PAL blend ratio must lie in [0,1]")
    region_length = donor_region.crop_end - donor_region.crop_start
    destination_end = destination_start + region_length
    if destination_start < 0 or destination_end > len(recipient_features):
        raise ValueError("PAL destination falls outside the recipient sequence")
    donor_crop = donor_features[donor_region.crop_start : donor_region.crop_end]
    if donor_crop.shape != recipient_features[destination_start:destination_end].shape:
        raise ValueError("PAL donor and recipient feature dimensions do not match")
    output_features = recipient_features.copy()
    recipient_crop = recipient_features[destination_start:destination_end]
    output_features[destination_start:destination_end] = (
        blend_ratio * donor_crop + (1.0 - blend_ratio) * recipient_crop
    )
    destination_time_s = destination_start * grid_stride_s
    transplanted = np.asarray(
        [
            [
                destination_time_s + donor_region.segment_start_offset_s,
                destination_time_s + donor_region.segment_end_offset_s,
            ]
        ],
        dtype=np.float32,
    )
    output_segments = np.concatenate((recipient_segments, transplanted), axis=0)
    output_segments = output_segments[np.argsort(output_segments[:, 0])]
    return output_features, output_segments


class ContinuousSequenceDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        sequences: pd.DataFrame,
        annotations: dict[tuple[str, int], np.ndarray],
        auxiliary_feature_path: Path | None = None,
        auxiliary_mean: np.ndarray | None = None,
        auxiliary_std: np.ndarray | None = None,
        feature_normalization: str = "none",
        feature_alignment_mean: np.ndarray | None = None,
        feature_alignment_std: np.ndarray | None = None,
        feature_alignment_blend: float = 0.0,
        feature_alignment_target_stats: dict[
            tuple[str, int], tuple[np.ndarray, np.ndarray]
        ]
        | None = None,
        feature_channel_mean: np.ndarray | None = None,
        feature_channel_std: np.ndarray | None = None,
        background_mix_probability: float = 0.0,
        mixstyle_probability: float = 0.0,
        mixstyle_alpha: float = 0.1,
        temporal_reversal_probability: float = 0.0,
        action_transplant_probability: float = 0.0,
        action_transplant_stride_s: float = 0.5,
        action_transplant_margin_bins: int = 2,
        action_transplant_blend_min: float = 1.0,
        action_transplant_blend_max: float = 1.0,
    ) -> None:
        self.feature_path = feature_path
        self.sequences = sequences.reset_index(drop=True)
        self.annotations = annotations
        self._features = None
        self.auxiliary_feature_path = auxiliary_feature_path
        self.auxiliary_mean = auxiliary_mean
        self.auxiliary_std = auxiliary_std
        self.feature_normalization = feature_normalization
        if self.feature_normalization not in {"none", "temporal-center", "temporal-zscore"}:
            raise ValueError(f"Unknown feature normalization: {self.feature_normalization}")
        self.feature_alignment_mean = feature_alignment_mean
        self.feature_alignment_std = feature_alignment_std
        self.feature_alignment_blend = float(feature_alignment_blend)
        self.feature_alignment_target_stats = feature_alignment_target_stats
        self.feature_channel_mean = feature_channel_mean
        self.feature_channel_std = feature_channel_std
        self.background_mix_probability = float(background_mix_probability)
        self.mixstyle_probability = float(mixstyle_probability)
        self.mixstyle_alpha = float(mixstyle_alpha)
        self.temporal_reversal_probability = float(temporal_reversal_probability)
        self.action_transplant_probability = float(action_transplant_probability)
        self.action_transplant_stride_s = float(action_transplant_stride_s)
        self.action_transplant_margin_bins = int(action_transplant_margin_bins)
        self.action_transplant_blend_min = float(action_transplant_blend_min)
        self.action_transplant_blend_max = float(action_transplant_blend_max)
        if not 0.0 <= self.background_mix_probability <= 1.0:
            raise ValueError("Background mix probability must be in [0,1]")
        if not 0.0 <= self.mixstyle_probability <= 1.0:
            raise ValueError("MixStyle probability must be in [0,1]")
        if self.mixstyle_alpha <= 0:
            raise ValueError("MixStyle alpha must be positive")
        if not 0.0 <= self.temporal_reversal_probability <= 1.0:
            raise ValueError("Temporal reversal probability must be in [0,1]")
        if not 0.0 <= self.action_transplant_probability <= 1.0:
            raise ValueError("PAL action transplant probability must be in [0,1]")
        if self.action_transplant_stride_s <= 0:
            raise ValueError("PAL action transplant stride must be positive")
        if self.action_transplant_margin_bins < 0:
            raise ValueError("PAL action transplant margin must be non-negative")
        if not (
            0.0
            <= self.action_transplant_blend_min
            <= self.action_transplant_blend_max
            <= 1.0
        ):
            raise ValueError("PAL blend bounds must satisfy 0 <= min <= max <= 1")
        if (feature_channel_mean is None) != (feature_channel_std is None):
            raise ValueError("Feature standardization requires both mean and std")
        if (feature_alignment_mean is None) != (feature_alignment_std is None):
            raise ValueError("Feature alignment requires both source mean and source std")
        if feature_alignment_mean is not None and self.feature_normalization != "none":
            raise ValueError("Feature normalization and source alignment are mutually exclusive")
        self._auxiliary_features = None
        self.action_regions = (
            build_action_regions(
                self.sequences,
                self.annotations,
                self.action_transplant_stride_s,
            )
            if self.action_transplant_probability > 0
            else []
        )
        if self.action_transplant_probability > 0 and not self.action_regions:
            raise ValueError("PAL action transplant requires source ED instances")
        self.background_bank = None
        self.background_recordings = None
        if self.background_mix_probability > 0:
            feature_matrix = self._get_features()
            backgrounds = []
            recordings = []
            for row in self.sequences.itertuples(index=False):
                start = int(row.offset)
                end = start + int(row.length)
                values = np.asarray(feature_matrix[start:end], dtype=np.float32)
                backgrounds.append(np.median(values, axis=0).astype(np.float32))
                recordings.append(str(row.rec_name))
            self.background_bank = np.stack(backgrounds)
            self.background_recordings = np.asarray(recordings)
        self.mixstyle_mean_bank = None
        self.mixstyle_std_bank = None
        self.mixstyle_recordings = None
        if self.mixstyle_probability > 0:
            means = []
            standard_deviations = []
            recordings = []
            for index, row in self.sequences.iterrows():
                values = self._load_sequence_features(int(index))
                means.append(
                    values.mean(axis=0, dtype=np.float64).astype(np.float32)
                )
                standard_deviations.append(
                    values.std(axis=0, dtype=np.float64).astype(np.float32)
                )
                recordings.append(str(row["rec_name"]))
            self.mixstyle_mean_bank = np.stack(means)
            self.mixstyle_std_bank = np.stack(standard_deviations)
            self.mixstyle_recordings = np.asarray(recordings)

    def _get_features(self) -> np.ndarray:
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode="r")
        return self._features

    def _get_auxiliary_features(self) -> np.ndarray | None:
        if self.auxiliary_feature_path is None:
            return None
        if self._auxiliary_features is None:
            self._auxiliary_features = np.load(self.auxiliary_feature_path, mmap_mode="r")
        return self._auxiliary_features

    def __len__(self) -> int:
        return len(self.sequences)

    def _load_sequence_features(self, index: int) -> np.ndarray:
        row = self.sequences.iloc[int(index)]
        start = int(row["offset"])
        end = start + int(row["length"])
        features = np.asarray(self._get_features()[start:end], dtype=np.float32).copy()
        if self.feature_channel_mean is not None:
            features = (features - self.feature_channel_mean) / np.maximum(
                self.feature_channel_std, 1e-6
            )
        features = normalize_temporal_features(features, self.feature_normalization)
        if self.feature_alignment_mean is not None:
            target_stats = None
            if self.feature_alignment_target_stats is not None:
                target_stats = self.feature_alignment_target_stats[
                    (str(row["rec_name"]), int(row["roi_id"]))
                ]
            features = align_temporal_feature_statistics(
                features,
                self.feature_alignment_mean,
                self.feature_alignment_std,
                self.feature_alignment_blend,
                *(target_stats or (None, None)),
            )
        auxiliary = self._get_auxiliary_features()
        if auxiliary is not None:
            auxiliary_values = np.asarray(auxiliary[start:end], dtype=np.float32).copy()
            auxiliary_values = (auxiliary_values - self.auxiliary_mean) / self.auxiliary_std
            features = np.concatenate((features, auxiliary_values), axis=1)
        return features

    def __getitem__(self, index: int) -> dict:
        row = self.sequences.iloc[index]
        features = self._load_sequence_features(index)
        if (
            self.mixstyle_mean_bank is not None
            and np.random.random() < self.mixstyle_probability
        ):
            donor_candidates = np.flatnonzero(
                self.mixstyle_recordings != str(row["rec_name"])
            )
            if len(donor_candidates) == 0:
                raise ValueError("MixStyle requires at least two recordings")
            donor_index = int(np.random.choice(donor_candidates))
            recipient_weight = float(
                np.random.beta(self.mixstyle_alpha, self.mixstyle_alpha)
            )
            features = feature_mixstyle(
                features,
                self.mixstyle_mean_bank[donor_index],
                self.mixstyle_std_bank[donor_index],
                recipient_weight,
            )
        if (
            self.background_bank is not None
            and np.random.random() < self.background_mix_probability
        ):
            donor_candidates = np.flatnonzero(
                self.background_recordings != str(row["rec_name"])
            )
            if len(donor_candidates) == 0:
                raise ValueError("Background mixing requires at least two recordings")
            donor_index = int(np.random.choice(donor_candidates))
            features = feature_background_mix(
                features,
                self.background_bank[donor_index],
                float(np.random.uniform(0.0, 1.0)),
            )
        segments = self.annotations.get(
            (str(row["rec_name"]), int(row["roi_id"])), np.empty((0, 2), dtype=np.float32)
        ).copy()
        pal_donor_features = None
        pal_recipient_span = None
        pal_donor_span = None
        if (
            self.action_regions
            and np.random.random() < self.action_transplant_probability
        ):
            donor_candidates = [
                region
                for region in self.action_regions
                if region.rec_name != str(row["rec_name"])
                and region.crop_end - region.crop_start <= len(features)
            ]
            if donor_candidates:
                donor_region = donor_candidates[
                    int(np.random.randint(len(donor_candidates)))
                ]
                starts = valid_transplant_starts(
                    len(features),
                    donor_region.crop_end - donor_region.crop_start,
                    segments,
                    self.action_transplant_stride_s,
                    self.action_transplant_margin_bins,
                )
                if len(starts):
                    donor_features = self._load_sequence_features(
                        donor_region.sequence_index
                    )
                    destination_start = int(np.random.choice(starts))
                    blend_ratio = self.action_transplant_blend_min
                    if (
                        self.action_transplant_blend_max
                        > self.action_transplant_blend_min
                    ):
                        blend_ratio = float(
                            np.random.uniform(
                                self.action_transplant_blend_min,
                                self.action_transplant_blend_max,
                            )
                        )
                    features, segments = transplant_action_region(
                        features,
                        segments,
                        donor_features,
                        donor_region,
                        destination_start,
                        self.action_transplant_stride_s,
                        blend_ratio,
                    )
                    pal_donor_features = donor_features
                    pal_recipient_span = (
                        destination_start,
                        destination_start
                        + donor_region.crop_end
                        - donor_region.crop_start,
                    )
                    pal_donor_span = (
                        donor_region.crop_start,
                        donor_region.crop_end,
                    )
        if (
            self.temporal_reversal_probability > 0.0
            and np.random.random() < self.temporal_reversal_probability
        ):
            features, segments = reverse_temporal_sample(
                features,
                segments,
                float(row["duration_s"]),
            )
            if pal_recipient_span is not None:
                start, end = pal_recipient_span
                pal_recipient_span = (len(features) - end, len(features) - start)
        return {
            "features": torch.from_numpy(features),
            "segments": torch.from_numpy(segments),
            "rec_name": str(row["rec_name"]),
            "roi_id": int(row["roi_id"]),
            "duration_s": float(row["duration_s"]),
            "pal_donor_features": (
                None
                if pal_donor_features is None
                else torch.from_numpy(pal_donor_features)
            ),
            "pal_recipient_span": pal_recipient_span,
            "pal_donor_span": pal_donor_span,
        }


def collate_sequences(batch: list[dict]) -> dict:
    lengths = torch.as_tensor([len(item["features"]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    feature_dim = int(batch[0]["features"].shape[1])
    features = torch.zeros(len(batch), max_length, feature_dim, dtype=torch.float32)
    mask = torch.arange(max_length)[None, :] < lengths[:, None]
    for index, item in enumerate(batch):
        features[index, : lengths[index]] = item["features"]
    paired = [
        (index, item)
        for index, item in enumerate(batch)
        if item.get("pal_donor_features") is not None
    ]
    donor_features = torch.empty((0, 0, feature_dim), dtype=torch.float32)
    donor_mask = torch.empty((0, 0), dtype=torch.bool)
    recipient_indices = torch.empty(0, dtype=torch.long)
    recipient_spans = torch.empty((0, 2), dtype=torch.long)
    donor_spans = torch.empty((0, 2), dtype=torch.long)
    if paired:
        donor_lengths = torch.as_tensor(
            [len(item["pal_donor_features"]) for _, item in paired],
            dtype=torch.long,
        )
        donor_features = torch.zeros(
            len(paired),
            int(donor_lengths.max()),
            feature_dim,
            dtype=torch.float32,
        )
        donor_mask = (
            torch.arange(int(donor_lengths.max()))[None, :]
            < donor_lengths[:, None]
        )
        for pair_index, (_, item) in enumerate(paired):
            donor = item["pal_donor_features"]
            donor_features[pair_index, : len(donor)] = donor
        recipient_indices = torch.as_tensor(
            [index for index, _ in paired], dtype=torch.long
        )
        recipient_spans = torch.as_tensor(
            [item["pal_recipient_span"] for _, item in paired], dtype=torch.long
        )
        donor_spans = torch.as_tensor(
            [item["pal_donor_span"] for _, item in paired], dtype=torch.long
        )
    return {
        "features": features,
        "mask": mask,
        "segments": [item["segments"] for item in batch],
        "rec_name": [item["rec_name"] for item in batch],
        "roi_id": [item["roi_id"] for item in batch],
        "duration_s": torch.as_tensor([item["duration_s"] for item in batch]),
        "pal_donor_features": donor_features,
        "pal_donor_mask": donor_mask,
        "pal_recipient_indices": recipient_indices,
        "pal_recipient_spans": recipient_spans,
        "pal_donor_spans": donor_spans,
    }


def hard_negative_sample_weights(
    sequences: pd.DataFrame, hard_recordings: set[str], oversample: float
) -> np.ndarray:
    """Upweight sequences from recordings known to induce background/ranking failures."""
    weights = np.ones(len(sequences), dtype=np.float64)
    is_hard = sequences["rec_name"].astype(str).isin(hard_recordings).to_numpy()
    weights[is_hard] = oversample
    return weights


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    shuffle: bool,
    generator: torch.Generator,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    sampler = None
    if shuffle and sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=generator if shuffle and sampler is None else None,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_sequences,
    )


def apply_soft_nms(candidates: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    if candidates.numel() == 0:
        return np.empty((0, 3), dtype=np.float64)
    detections = candidates.float().cpu().numpy().astype(np.float64)
    detections = temporal_soft_nms(
        detections,
        sigma=args.soft_nms_sigma,
        score_threshold=args.soft_nms_score_threshold,
    )
    return detections[: args.max_predictions_per_roi]


def slice_model_output(output: dict, index: int) -> dict:
    sliced = {}
    for key, levels in output.items():
        sliced[key] = [
            value[index : index + 1] if value is not None else None for value in levels
        ]
    return sliced


def group_dro_reduce(
    sample_losses: torch.Tensor,
    sample_groups: torch.Tensor,
    group_weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    if eta <= 0:
        raise ValueError("GroupDRO eta must be positive")
    unique_groups = sample_groups.unique(sorted=True)
    group_risks = torch.stack(
        [sample_losses[sample_groups == group].mean() for group in unique_groups]
    )
    with torch.no_grad():
        group_weights[unique_groups] *= torch.exp(eta * group_risks.detach())
        group_weights /= group_weights.sum()
    observed_weights = group_weights[unique_groups]
    observed_weights = observed_weights / observed_weights.sum().clamp_min(1e-12)
    return (observed_weights * group_risks).sum()


def pal_region_consistency_loss(
    recipient_features: torch.Tensor,
    donor_features: torch.Tensor,
    recipient_indices: torch.Tensor,
    recipient_spans: torch.Tensor,
    donor_spans: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Align the same action crop across its original and transplanted contexts."""
    if temperature <= 0:
        raise ValueError("PAL consistency temperature must be positive")
    if not (
        len(recipient_indices)
        == len(recipient_spans)
        == len(donor_features)
        == len(donor_spans)
    ):
        raise ValueError("PAL consistency metadata has incompatible lengths")
    if len(recipient_indices) == 0:
        return recipient_features.sum() * 0.0

    recipient_regions = []
    donor_regions = []
    for pair_index, recipient_index in enumerate(recipient_indices.tolist()):
        recipient_start, recipient_end = recipient_spans[pair_index].tolist()
        donor_start, donor_end = donor_spans[pair_index].tolist()
        if (
            recipient_end <= recipient_start
            or donor_end <= donor_start
            or recipient_end > recipient_features.shape[-1]
            or donor_end > donor_features.shape[-1]
        ):
            raise ValueError("PAL consistency span falls outside encoded features")
        recipient_regions.append(
            recipient_features[
                recipient_index, :, recipient_start:recipient_end
            ].mean(dim=-1)
        )
        donor_regions.append(
            donor_features[pair_index, :, donor_start:donor_end].mean(dim=-1)
        )
    recipient_regions = torch.nn.functional.normalize(
        torch.stack(recipient_regions), dim=-1
    )
    donor_regions = torch.nn.functional.normalize(
        torch.stack(donor_regions), dim=-1
    )
    if len(recipient_regions) == 1:
        return 1.0 - (recipient_regions * donor_regions).sum(dim=-1).mean()
    logits = recipient_regions @ donor_regions.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    )


@torch.no_grad()
def evaluate(
    model: TemporalMaxerContinuous,
    loader: DataLoader,
    sequences: pd.DataFrame,
    metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
    prediction_path: Path,
) -> dict[str, float | int]:
    model.eval()
    results = {
        recording: {str(int(roi)): [] for roi in group["roi_id"].unique()}
        for recording, group in sequences.groupby("rec_name")
    }
    for batch in tqdm(loader, desc="val", disable=args.quiet_progress):
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(features, mask)
        candidates = model.decode(
            output,
            grid_stride_seconds=float(metadata["grid_stride_s"]),
            durations_seconds=batch["duration_s"],
            score_threshold=args.score_threshold,
            pre_nms_topk=args.pre_nms_topk,
            quality_power=args.quality_power,
            min_duration_seconds=args.min_action_duration,
        )
        for rec_name, roi_id, roi_candidates in zip(
            batch["rec_name"], batch["roi_id"], candidates
        ):
            detections = apply_soft_nms(roi_candidates, args)
            results[rec_name][str(int(roi_id))] = [
                {
                    "label": "ed",
                    "segment": [float(start), float(end)],
                    "score": float(score),
                }
                for start, end, score in detections
            ]
    prediction = {"version": "temporalmaxer-continuous-v1", "results": results}
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    valid_recordings = sorted(sequences["rec_name"].unique().tolist())
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(resolve(args.ann_path)),
        prediction_filename=str(prediction_path),
        tiou_thresholds=np.asarray(args.tiou, dtype=np.float64),
        valid_sequences=valid_recordings,
        valid_labels=["ed"],
        min_duration=args.min_action_duration,
    )
    mean_ap = float(evaluator.run())
    metrics: dict[str, float | int] = {
        "mAP": mean_ap,
        "n_predictions": sum(len(value) for rois in results.values() for value in rois.values()),
    }
    for threshold, value in zip(args.tiou, evaluator.mAP):
        metrics[f"AP@{threshold:.1f}"] = float(value)
    return metrics


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.pal_consistency_weight < 0:
        raise ValueError("PAL consistency weight must be non-negative")
    if args.pal_consistency_weight > 0 and not args.pal_action_transplant:
        raise ValueError("PAL consistency requires --pal-action-transplant")
    if args.pal_consistency_temperature <= 0:
        raise ValueError("PAL consistency temperature must be positive")
    if args.min_action_duration < 0:
        raise ValueError("Minimum action duration must be non-negative")
    if not args.tiou or any(not 0.0 <= value <= 1.0 for value in args.tiou):
        raise ValueError("tIoU thresholds must be in [0,1]")
    set_seed(args.seed + args.fold)
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    auxiliary_path = None
    auxiliary_mean = None
    auxiliary_std = None
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
        metadata = {
            **metadata,
            "auxiliary_feature_dim": int(auxiliary_metadata["feature_dim"]),
            "auxiliary_mean": auxiliary_metadata["mean"],
            "auxiliary_std": auxiliary_metadata["std"],
        }
    if args.cross_layer_task_decoupling and auxiliary_path is None:
        raise ValueError(
            "--cross-layer-task-decoupling requires --auxiliary-feature-dir"
        )
    sequences_path = (
        resolve(args.sequences_path) if args.sequences_path else feature_dir / "sequences.csv"
    )
    sequences = pd.read_csv(sequences_path)
    fold_manifest = pd.read_csv(resolve(args.fold_manifest))
    train_sequences, val_sequences = split_fold_sequences(
        sequences, fold_manifest, args.fold
    )
    val_recordings = sorted(val_sequences["rec_name"].astype(str).unique())

    annotations = load_annotations(
        resolve(args.ann_path), min_duration_s=args.min_action_duration
    )
    feature_path = feature_dir / args.feature_array_name
    feature_mean = None
    feature_std = None
    if args.standardize_features:
        if "mean" not in metadata or "std" not in metadata:
            raise ValueError("Feature metadata has no mean/std for standardization")
        feature_mean = np.asarray(metadata["mean"], dtype=np.float32)
        feature_std = np.asarray(metadata["std"], dtype=np.float32)
    train_dataset = ContinuousSequenceDataset(
        feature_path,
        train_sequences,
        annotations,
        auxiliary_path,
        auxiliary_mean,
        auxiliary_std,
        args.feature_normalization,
        feature_channel_mean=feature_mean,
        feature_channel_std=feature_std,
        background_mix_probability=(
            args.background_mix_probability if args.background_mix else 0.0
        ),
        mixstyle_probability=(
            args.mixstyle_probability if args.mixstyle else 0.0
        ),
        mixstyle_alpha=args.mixstyle_alpha,
        temporal_reversal_probability=args.temporal_reversal_probability,
        action_transplant_probability=(
            args.pal_action_transplant_probability
            if args.pal_action_transplant
            else 0.0
        ),
        action_transplant_stride_s=float(metadata["grid_stride_s"]),
        action_transplant_margin_bins=args.pal_action_transplant_margin_bins,
        action_transplant_blend_min=args.pal_blend_min,
        action_transplant_blend_max=args.pal_blend_max,
    )
    val_dataset = ContinuousSequenceDataset(
        feature_path,
        val_sequences,
        annotations,
        auxiliary_path,
        auxiliary_mean,
        auxiliary_std,
        args.feature_normalization,
        feature_channel_mean=feature_mean,
        feature_channel_std=feature_std,
    )
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    hard_negative_recordings = {
        name.strip() for name in args.hard_negative_recordings.split(",") if name.strip()
    }
    train_sample_weights = None
    if hard_negative_recordings and args.hard_negative_oversample != 1.0:
        present = hard_negative_recordings & set(train_sequences["rec_name"].astype(str))
        if present:
            train_sample_weights = hard_negative_sample_weights(
                train_sequences, hard_negative_recordings, args.hard_negative_oversample
            )
            print(
                f"[INFO] hard-negative oversampling active: {sorted(present)} "
                f"x{args.hard_negative_oversample}",
                flush=True,
            )
    train_loader = make_loader(train_dataset, args, True, generator, train_sample_weights)
    val_loader = make_loader(val_dataset, args, False, generator)

    model = TemporalMaxerContinuous(
        input_dim=int(metadata["feature_dim"]) + int(metadata.get("auxiliary_feature_dim", 0)),
        hidden_dim=args.hidden_dim,
        pyramid_levels=args.pyramid_levels,
        head_layers=args.head_layers,
        dropout=args.dropout,
        use_quality=not args.disable_quality,
        reg_max=args.reg_max,
        trident_bins=args.trident_bins,
        neck_type=args.neck_type,
        attention_heads=args.attention_heads,
        attention_window=args.attention_window,
        center_sampling_radius=args.center_sampling_radius,
        use_boundary_heads=args.use_boundary_heads,
        boundary_refine_radius_seconds=(
            args.boundary_refine_radius if args.use_boundary_heads else 0.0
        ),
        boundary_refine_blend=args.boundary_refine_blend,
        tanp_std=args.tanp_std if args.tanp else 0.0,
        tanp_probability=args.tanp_probability,
        use_temporal_order=args.temporal_order,
        temporal_order_chunks=args.temporal_order_chunks,
        classification_input_dim=(
            int(metadata["feature_dim"])
            if args.cross_layer_task_decoupling
            else None
        ),
    ).to(device)
    training_recordings = sorted(train_sequences["rec_name"].astype(str).unique())
    recording_to_group = {
        recording: index for index, recording in enumerate(training_recordings)
    }
    group_weights = torch.full(
        (len(training_recordings),),
        1.0 / len(training_recordings),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_map = -1.0
    best_epoch = 0
    stale = 0
    start_epoch = 1

    last_path = out_dir / "last.pt"
    if last_path.exists() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best_map = float(state["best_map"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
        if args.group_dro and state.get("group_weights") is not None:
            group_weights.copy_(state["group_weights"].to(device))
        start_epoch = int(state["epoch"]) + 1
    elif args.init_checkpoint:
        init_state = torch.load(resolve(args.init_checkpoint), map_location=device, weights_only=False)
        model.load_state_dict(init_state["model"])
        print(f"[INFO] warm-started model weights from {args.init_checkpoint}", flush=True)

    if args.pal_action_transplant:
        # One-epoch supervisors restart workers, so advance their reproducible RNG per resume.
        generator.manual_seed(args.seed + args.fold + 100_003 * (start_epoch - 1))

    teacher_model = None
    unlabeled_iterator = None
    if args.consistency_weight > 0 and args.unlabeled_feature_dir and args.unlabeled_recordings:
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        unlabeled_dir = resolve(args.unlabeled_feature_dir)
        unlabeled_sequences = pd.read_csv(unlabeled_dir / "sequences.csv")
        unlabeled_sequences = unlabeled_sequences[
            unlabeled_sequences["rec_name"].isin(args.unlabeled_recordings)
        ].copy()
        if unlabeled_sequences.empty:
            raise ValueError("No unlabeled sequences matched --unlabeled-recordings")
        unlabeled_dataset = ContinuousSequenceDataset(
            unlabeled_dir / args.feature_array_name,
            unlabeled_sequences,
            annotations={},
            feature_normalization=args.feature_normalization,
            feature_channel_mean=feature_mean,
            feature_channel_std=feature_std,
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=min(args.batch_size, len(unlabeled_dataset)),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_sequences,
        )
        unlabeled_iterator = itertools.cycle(unlabeled_loader)
        print(
            f"[INFO] Mean-Teacher consistency active: {len(unlabeled_sequences)} unlabeled ROIs "
            f"from {sorted(set(unlabeled_sequences.rec_name))}, weight={args.consistency_weight}, "
            f"ema_decay={args.ema_decay}",
            flush=True,
        )

    print(
        f"[INFO] fold={args.fold} train_roi={len(train_sequences)} val_roi={len(val_sequences)} "
        f"train_records={train_sequences.rec_name.nunique()} val_records={val_recordings} device={device}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {
            key: 0.0
            for key in (
                "loss",
                "classification_loss",
                "regression_loss",
                "distribution_loss",
                "quality_loss",
                "boundary_loss",
                "temporal_order_loss",
                "pal_consistency_loss",
            )
        }
        batches = 0
        progress = tqdm(train_loader, desc=f"train-{epoch:02d}", disable=args.quiet_progress)
        for batch in progress:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            segments = [value.to(device, non_blocking=True) for value in batch["segments"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(features, mask)
                standard_losses = model.losses(
                    output,
                    segments,
                    grid_stride_seconds=float(metadata["grid_stride_s"]),
                    regression_weight=args.regression_weight,
                    quality_weight=args.quality_weight,
                    distribution_weight=args.distribution_weight,
                    empty_sequence_weight=args.empty_sequence_weight,
                    boundary_weight=args.boundary_weight,
                    boundary_target_sigma=args.boundary_target_sigma,
                    rank_sort=args.rank_sort,
                    rank_sort_delta=args.rank_sort_delta,
                )
                temporal_order_loss = model.temporal_order_loss(
                    output["pyramid_features"][0], output["masks"][0]
                )
                standard_losses["temporal_order_loss"] = temporal_order_loss
                standard_losses["loss"] = (
                    standard_losses["loss"]
                    + args.temporal_order_weight * temporal_order_loss
                )
                if args.group_dro:
                    sample_loss_dicts = [
                        model.losses(
                            slice_model_output(output, index),
                            [segments[index]],
                            grid_stride_seconds=float(metadata["grid_stride_s"]),
                            regression_weight=args.regression_weight,
                            quality_weight=args.quality_weight,
                            distribution_weight=args.distribution_weight,
                            empty_sequence_weight=args.empty_sequence_weight,
                            boundary_weight=args.boundary_weight,
                            boundary_target_sigma=args.boundary_target_sigma,
                            rank_sort=args.rank_sort,
                            rank_sort_delta=args.rank_sort_delta,
                        )
                        for index in range(len(segments))
                    ]
                    sample_losses = torch.stack(
                        [value["loss"] for value in sample_loss_dicts]
                    )
                    sample_groups = torch.as_tensor(
                        [recording_to_group[value] for value in batch["rec_name"]],
                        dtype=torch.long,
                        device=device,
                    )
                    robust_loss = group_dro_reduce(
                        sample_losses,
                        sample_groups,
                        group_weights,
                        args.group_dro_eta,
                    )
                    losses = {**standard_losses, "loss": robust_loss}
                else:
                    losses = standard_losses
                pal_consistency_loss = output["pyramid_features"][0].sum() * 0.0
                if (
                    args.pal_consistency_weight > 0
                    and len(batch["pal_recipient_indices"]) > 0
                ):
                    donor_features = batch["pal_donor_features"].to(
                        device, non_blocking=True
                    )
                    donor_mask = batch["pal_donor_mask"].to(
                        device, non_blocking=True
                    )
                    donor_output = model(donor_features, donor_mask)
                    pal_consistency_loss = pal_region_consistency_loss(
                        output["pyramid_features"][0],
                        donor_output["pyramid_features"][0],
                        batch["pal_recipient_indices"].to(device),
                        batch["pal_recipient_spans"].to(device),
                        batch["pal_donor_spans"].to(device),
                        args.pal_consistency_temperature,
                    )
                    losses["loss"] = (
                        losses["loss"]
                        + args.pal_consistency_weight * pal_consistency_loss
                    )
                losses = {
                    **losses,
                    "pal_consistency_loss": pal_consistency_loss,
                }
                if teacher_model is not None:
                    unlabeled_batch = next(unlabeled_iterator)
                    unlabeled_features = unlabeled_batch["features"].to(device, non_blocking=True)
                    unlabeled_mask = unlabeled_batch["mask"].to(device, non_blocking=True)
                    student_out = model(unlabeled_features, unlabeled_mask)
                    with torch.no_grad():
                        teacher_out = teacher_model(unlabeled_features, unlabeled_mask)
                    consistency_terms = []
                    for level_mask, student_logits, teacher_logits in zip(
                        student_out["masks"],
                        student_out["classification_logits"],
                        teacher_out["classification_logits"],
                    ):
                        diff = torch.sigmoid(student_logits) - torch.sigmoid(teacher_logits)
                        consistency_terms.append((diff.pow(2) * level_mask).sum() / level_mask.sum().clamp_min(1))
                    consistency_loss = sum(consistency_terms) / len(consistency_terms)
                    losses = {**losses, "consistency_loss": consistency_loss}
                    losses["loss"] = losses["loss"] + args.consistency_weight * consistency_loss
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if teacher_model is not None:
                with torch.no_grad():
                    for teacher_param, student_param in zip(
                        teacher_model.parameters(), model.parameters()
                    ):
                        teacher_param.mul_(args.ema_decay).add_(student_param, alpha=1 - args.ema_decay)
            batches += 1
            for key in totals:
                totals[key] += float(losses[key].detach())
            progress.set_postfix(loss=f"{totals['loss'] / batches:.4f}")
        scheduler.step()

        prediction_path = out_dir / "predictions" / f"epoch_{epoch:03d}.json"
        metrics = evaluate(
            model, val_loader, val_sequences, metadata, args, device, prediction_path
        )
        row = {
            "epoch": epoch,
            **{key: value / max(batches, 1) for key, value in totals.items()},
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        improved = float(metrics["mAP"]) > best_map
        if improved:
            best_map = float(metrics["mAP"])
            best_epoch = epoch
            stale = 0
            atomic_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "args": vars(args),
                    "metadata": metadata,
                    "group_weights": group_weights.detach().cpu() if args.group_dro else None,
                    "group_recordings": training_recordings if args.group_dro else None,
                },
                out_dir / "best.pt",
            )
            (out_dir / "metrics_best.json").write_text(
                json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8"
            )
        else:
            stale += 1
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_map": best_map,
                "best_epoch": best_epoch,
                "stale": stale,
                "group_weights": group_weights.detach().cpu() if args.group_dro else None,
            },
            last_path,
        )
        print(
            f"[EPOCH {epoch:03d}] loss={row['loss']:.4f} mAP={metrics['mAP']:.6f} "
            f"AP07={metrics['AP@0.7']:.6f} best={best_map:.6f}@{best_epoch}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"[EARLY-STOP] no improvement for {stale} epochs")
            break
        epochs_this_process = epoch - start_epoch + 1
        if (
            args.epochs_per_process > 0
            and epochs_this_process >= args.epochs_per_process
            and epoch < args.epochs
        ):
            print(f"[CHECKPOINT-EXIT] epoch={epoch}; supervisor should resume", flush=True)
            raise SystemExit(75)

    print(json.dumps({"fold": args.fold, "best_epoch": best_epoch, "best_mAP": best_map}))


if __name__ == "__main__":
    main()
