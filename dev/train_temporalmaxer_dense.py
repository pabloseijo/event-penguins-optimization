"""Train a dense TemporalMaxer-lite head on ordered ATSN frame features.

The original AugmentedTSN reduces eleven temporal samples to three averages
before classification. This experiment freezes the ATSN encoder, caches all
eleven 512-D features, and learns dense actionness, boundary distributions,
proposal quality, and boundary offsets. Model selection is recording-disjoint;
the official test split is only supported through explicit checkpoint eval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.augmented_tsn import AugmentedTsn
from src.classification import ProposalDataset
from src.evaluation import DetectionsEvaluator
from src.temporalmaxer_lite import TemporalMaxerLiteHead
from src.utils import temporal_soft_nms


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]


@dataclass
class DenseTargets:
    quality: np.ndarray
    action: np.ndarray
    point_distances: np.ndarray
    start_distribution: np.ndarray
    end_distribution: np.ndarray
    deltas: np.ndarray
    boundary_weight: np.ndarray
    sample_kind: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense TemporalMaxer-lite proposal head")
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--master-proposals", required=True)
    parser.add_argument("--train-proposals", default=None)
    parser.add_argument("--val-proposals", default=None)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--event-feature-cache-dir",
        default=None,
        help="Optional aligned [N,T,D] ON/OFF and local-spectrum feature cache.",
    )
    parser.add_argument(
        "--event-features-only",
        action="store_true",
        help="Use only the aligned event representation, without ATSN features.",
    )
    parser.add_argument(
        "--corrupted-event-feature-cache-dir",
        default=None,
        help="Optional aligned source-only corrupted event view used during training.",
    )
    parser.add_argument(
        "--corrupted-event-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--seed-master-proposals", default=None)
    parser.add_argument("--seed-cache-dir", default=None)
    parser.add_argument("--out-dir", default="tmp/temporalmaxer_dense/run")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--eval-checkpoint", default=None)
    parser.add_argument("--eval-label", default="eval")
    parser.add_argument("--restart", action="store_true")

    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--repr-batch-size", type=int, default=32)
    parser.add_argument("--cache-checkpoint-rows", type=int, default=1600)
    parser.add_argument(
        "--timestamp-cache-dir",
        default="tmp/temporalmaxer_dense/roi_timestamps",
    )
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pyramid-levels", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--group-dro", action="store_true")
    parser.add_argument("--group-dro-eta", type=float, default=0.01)
    parser.add_argument(
        "--tanp-sigma",
        type=float,
        default=0.0,
        help="AAAI 2026 temporal-aware normalization perturbation strength.",
    )
    parser.add_argument("--trc-weight", type=float, default=0.0)
    parser.add_argument("--trc-topk", type=int, default=3)

    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--high-pos-tiou", type=float, default=0.7)
    parser.add_argument("--boundary-min-tiou", type=float, default=0.3)
    parser.add_argument(
        "--min-action-duration",
        type=float,
        default=2.0,
        help="Minimum GT and prediction duration in seconds; THUMOS14-E uses 0.",
    )
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--trident-bins", type=int, default=0)
    parser.add_argument("--trident-weight", type=float, default=0.5)
    parser.add_argument(
        "--selection-score",
        choices=["cnn_score", "dense_score", "brem_score"],
        default="cnn_score",
    )
    parser.add_argument(
        "--selection-boundary",
        choices=["raw", "blend", "trident"],
        default="blend",
    )
    parser.add_argument("--qfl-beta", type=float, default=2.0)

    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument(
        "--fast-selection-eval",
        action="store_true",
        help="Evaluate only the blend variant during training; final evaluation stays complete.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def expanded_tsn_samples(num_tsn_samples: int, augment_factor: int) -> int:
    return num_tsn_samples + 2 * int(math.ceil(num_tsn_samples / augment_factor))


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def minimum_action_duration(args: argparse.Namespace) -> float:
    value = float(getattr(args, "min_action_duration", 2.0))
    if value < 0:
        raise ValueError("min_action_duration must be non-negative")
    return value


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_atsn(args: argparse.Namespace, device: torch.device) -> AugmentedTsn:
    model = AugmentedTsn(2, args.num_tsn_samples, args.augment_factor)
    model_path = resolve(args.model_path)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def proposal_fingerprint(proposals: pd.DataFrame) -> str:
    missing = sorted(set(KEY_COLUMNS) - set(proposals.columns))
    if missing:
        raise ValueError(f"Proposal file misses key columns: {missing}")
    hashed = pd.util.hash_pandas_object(proposals[KEY_COLUMNS], index=False)
    return hashlib.sha256(hashed.to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "features": cache_dir / "frame_features.npy",
        "logits": cache_dir / "logits.npy",
        "state": cache_dir / "extraction_state.json",
        "metadata": cache_dir / "metadata.json",
    }


def valid_complete_cache(
    proposals: pd.DataFrame,
    cache_dir: Path,
    num_segments: int,
) -> bool:
    paths = cache_paths(cache_dir)
    if not all(paths[key].exists() for key in ("features", "logits", "metadata")):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    expected = {
        "rows": len(proposals),
        "num_segments": num_segments,
        "fingerprint": proposal_fingerprint(proposals),
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def extract_representations(
    proposals: pd.DataFrame,
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    num_segments = expanded_tsn_samples(args.num_tsn_samples, args.augment_factor)
    if valid_complete_cache(proposals, cache_dir, num_segments):
        print(f"[INFO] Dense cache reutilizada: {cache_dir}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    fingerprint = proposal_fingerprint(proposals)
    completed = 0
    feature_dim = 512
    can_resume = paths["state"].exists() and paths["features"].exists() and paths["logits"].exists()
    if can_resume:
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
        if (
            state.get("fingerprint") == fingerprint
            and state.get("rows") == len(proposals)
            and state.get("num_segments") == num_segments
        ):
            completed = int(state.get("completed", 0))

    mode = "r+" if completed > 0 else "w+"
    features = np.lib.format.open_memmap(
        paths["features"],
        mode=mode,
        dtype=np.float16,
        shape=(len(proposals), num_segments, feature_dim),
    )
    logits = np.lib.format.open_memmap(
        paths["logits"],
        mode=mode,
        dtype=np.float32,
        shape=(len(proposals), 2),
    )
    if completed == 0 and args.seed_master_proposals and args.seed_cache_dir:
        seed_master = pd.read_csv(resolve(args.seed_master_proposals)).reset_index(drop=True)
        seed_cache_dir = resolve(args.seed_cache_dir)
        if not valid_complete_cache(seed_master, seed_cache_dir, num_segments):
            raise ValueError("Seed cache does not match its proposal master")
        seed_features, seed_logits, seed_metadata = load_cache(seed_cache_dir)
        seed_index = stable_proposal_index(seed_master)
        if not seed_index.is_unique:
            raise ValueError("Seed master proposal identities are not unique")
        target_to_seed = seed_index.get_indexer(stable_proposal_index(proposals))
        first_missing = np.flatnonzero(target_to_seed < 0)
        seed_count = int(first_missing[0]) if len(first_missing) else len(proposals)
        if np.any(target_to_seed[seed_count:] >= 0):
            raise ValueError("Reusable seed proposals are not a contiguous target prefix")
        seed_positions = target_to_seed[:seed_count]
        if (
            seed_features.shape != (len(seed_master), num_segments, feature_dim)
            or seed_logits.shape != (len(seed_master), 2)
            or int(seed_metadata["num_segments"]) != num_segments
        ):
            raise ValueError("Seed cache shapes do not match the target cache")
        for start in range(0, seed_count, 10_000):
            end = min(seed_count, start + 10_000)
            source = seed_positions[start:end]
            features[start:end] = seed_features[source]
            logits[start:end] = seed_logits[source]
        features.flush()
        logits.flush()
        completed = seed_count
        paths["state"].write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "rows": len(proposals),
                    "num_segments": num_segments,
                    "completed": completed,
                }
            ),
            encoding="utf-8",
        )
        print(f"[INFO] Caché sementada con {completed} propostas")

    dataset = ProposalDataset(
        proposals.reset_index(drop=True),
        augment_fraction=1.0 / args.augment_factor,
        data_path=str(resolve(args.data_path)),
        num_tsn_samples=num_segments,
        sample_duration=args.sample_duration * 1e6,
        decay=args.decay,
        cache_full_events=False,
        timestamp_cache_dir=str(resolve(args.timestamp_cache_dir)),
    )
    pending = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(
        pending,
        batch_size=args.repr_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_atsn(args, device)
    offset = completed
    next_checkpoint = completed + max(args.cache_checkpoint_rows, 1)
    progress = tqdm(
        loader,
        desc=f"dense-features@{completed}",
        disable=args.quiet_progress,
    )
    with torch.inference_mode():
        for images, *_ in progress:
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                batch_logits, batch_features = model.forward_with_frame_features(images)
            size = len(images)
            features[offset:offset + size] = batch_features.float().cpu().numpy().astype(np.float16)
            logits[offset:offset + size] = batch_logits.float().cpu().numpy()
            offset += size
            if offset >= next_checkpoint or offset == len(dataset):
                features.flush()
                logits.flush()
                paths["state"].write_text(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "rows": len(proposals),
                            "num_segments": num_segments,
                            "completed": offset,
                        }
                    ),
                    encoding="utf-8",
                )
                next_checkpoint = offset + max(args.cache_checkpoint_rows, 1)

    features.flush()
    logits.flush()
    paths["metadata"].write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "rows": len(proposals),
                "num_segments": num_segments,
                "feature_dim": feature_dim,
                "model_path": str(resolve(args.model_path)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["state"].unlink(missing_ok=True)
    print(f"[RESULTADO] Dense cache completada: {features.shape} en {cache_dir}")


def load_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    paths = cache_paths(cache_dir)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    features = np.load(paths["features"], mmap_mode="r")
    logits = np.load(paths["logits"], mmap_mode="r")
    return features, logits, metadata


def event_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "features": cache_dir / "event_features.npy",
        "metadata": cache_dir / "metadata.json",
    }


def load_event_cache(
    cache_dir: Path,
    base_metadata: dict,
) -> tuple[np.ndarray, dict]:
    paths = event_cache_paths(cache_dir)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    features = np.load(paths["features"], mmap_mode="r")
    for key in ("fingerprint", "rows", "num_segments"):
        if metadata.get(key) != base_metadata.get(key):
            raise ValueError(
                f"Event cache {key}={metadata.get(key)!r} does not match "
                f"dense cache {base_metadata.get(key)!r}"
            )
    stored_segments = int(
        metadata.get("stored_segments", base_metadata["num_segments"])
    )
    if stored_segments != int(base_metadata["num_segments"]):
        if stored_segments != 1 or not metadata.get("broadcast_temporal", False):
            raise ValueError(
                "Static event caches must store one segment and set "
                "broadcast_temporal=true"
            )
    expected_shape = (
        int(base_metadata["rows"]),
        stored_segments,
        int(metadata["feature_dim"]),
    )
    if features.shape != expected_shape:
        raise ValueError(
            f"Event cache shape {features.shape} does not match {expected_shape}"
        )
    return features, metadata


def event_feature_configuration(
    args: argparse.Namespace,
    base_metadata: dict,
) -> tuple[Path | None, int]:
    cache_arg = getattr(args, "event_feature_cache_dir", None)
    if not cache_arg:
        return None, 0
    cache_dir = resolve(cache_arg)
    _, metadata = load_event_cache(cache_dir, base_metadata)
    return event_cache_paths(cache_dir)["features"], int(metadata["feature_dim"])


@torch.no_grad()
def event_blank_feature(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """Encode the current pipeline's time-surface for an empty event window."""
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        empty_time_surface = (torch.ones((1, 1, 3, 224, 224), device=device) - mean) / std
        model = load_atsn(args, device)
        feature = model.encode_frames(empty_time_surface).squeeze(0).squeeze(0).float()
        del model
        return feature
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def drop_one_event_frame(
    frame_features: torch.Tensor,
    blank_feature: torch.Tensor,
) -> torch.Tensor:
    """Replace one temporal sample per proposal by the no-event ATSN feature."""
    corrupted = frame_features.clone()
    positions = torch.randint(
        frame_features.shape[1],
        (frame_features.shape[0],),
        device=frame_features.device,
    )
    rows = torch.arange(frame_features.shape[0], device=frame_features.device)
    corrupted[rows, positions] = blank_feature.to(frame_features.dtype)
    return corrupted


def stable_proposal_index(proposals: pd.DataFrame) -> pd.MultiIndex:
    """Build a CSV-roundtrip-stable identity for proposal boundaries."""
    keys = proposals[KEY_COLUMNS].copy()
    keys["rec_name"] = keys["rec_name"].astype(str)
    keys["roi_id"] = keys["roi_id"].astype(str)
    for column in ("t_start", "t_end"):
        values = keys[column].to_numpy(dtype=np.float64)
        keys[column] = np.rint(values * 1_000.0).astype(np.int64)
    return pd.MultiIndex.from_frame(keys)


def map_to_master(master: pd.DataFrame, subset: pd.DataFrame) -> np.ndarray:
    master_index = stable_proposal_index(master)
    if not master_index.is_unique:
        raise ValueError("Master proposal keys are not unique after nanosecond quantization")
    subset_index = stable_proposal_index(subset)
    mapped = master_index.get_indexer(subset_index)
    if np.any(mapped < 0):
        raise ValueError(f"{int((mapped < 0).sum())} proposals are missing from the master cache")
    return mapped.astype(np.int64)


def load_annotation_index(
    path: Path, min_action_duration: float = 2.0
) -> dict[tuple[str, str], np.ndarray]:
    if min_action_duration < 0:
        raise ValueError("min_action_duration must be non-negative")
    database = json.loads(path.read_text(encoding="utf-8"))["database"]
    out: dict[tuple[str, str], np.ndarray] = {}
    for recording, value in database.items():
        for roi, annotations in value.get("annotations", {}).items():
            if roi == "null":
                continue
            segments = [
                list(map(float, item["segment"]))
                for item in annotations
                if item["label"] == "ed"
                and float(item["segment"][1]) - float(item["segment"][0])
                >= min_action_duration
            ]
            out[(recording, str(int(roi)))] = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    return out


def roi_key(roi_id: str) -> str:
    return str(int(str(roi_id)[1:]))


def best_match(
    start_us: float,
    end_us: float,
    segments_s: np.ndarray,
) -> tuple[float, float, float]:
    if segments_s.size == 0:
        return 0.0, float("nan"), float("nan")
    start_s = start_us / 1e6
    end_s = end_us / 1e6
    intersections = np.maximum(
        0.0,
        np.minimum(end_s, segments_s[:, 1]) - np.maximum(start_s, segments_s[:, 0]),
    )
    unions = (end_s - start_s) + (segments_s[:, 1] - segments_s[:, 0]) - intersections
    tiou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    index = int(np.argmax(tiou))
    return float(tiou[index]), float(segments_s[index, 0]), float(segments_s[index, 1])


def build_targets(
    proposals: pd.DataFrame,
    num_segments: int,
    args: argparse.Namespace,
) -> DenseTargets:
    annotations = load_annotation_index(resolve(args.ann_path), minimum_action_duration(args))
    count = len(proposals)
    quality = np.zeros(count, dtype=np.float32)
    action = np.zeros((count, num_segments), dtype=np.float32)
    point_distances = np.zeros((count, num_segments, 2), dtype=np.float32)
    start_distribution = np.zeros((count, num_segments), dtype=np.float32)
    end_distribution = np.zeros((count, num_segments), dtype=np.float32)
    deltas = np.zeros((count, 2), dtype=np.float32)
    boundary_weight = np.zeros(count, dtype=np.float32)
    sample_kind = np.full(count, 4, dtype=np.int8)
    augment_fraction = 1.0 / args.augment_factor
    relative_times = np.linspace(
        -augment_fraction,
        1.0 + augment_fraction,
        num_segments,
        dtype=np.float64,
    )
    sigma = max(float(relative_times[1] - relative_times[0]), 1e-3)

    iterator = proposals.reset_index(drop=True).iterrows()
    for index, row in tqdm(iterator, total=count, desc="dense-targets", disable=args.quiet_progress):
        duration_us = max(float(row["t_end"]) - float(row["t_start"]), 1.0)
        segments = annotations.get((str(row["rec_name"]), roi_key(row["roi_id"])), np.empty((0, 2)))
        tiou, gt_start_s, gt_end_s = best_match(float(row["t_start"]), float(row["t_end"]), segments)
        quality[index] = tiou if tiou >= args.neg_tiou else 0.0
        if tiou >= args.high_pos_tiou:
            sample_kind[index] = 0
        elif tiou >= args.pos_tiou:
            sample_kind[index] = 1
        elif tiou >= args.neg_tiou:
            sample_kind[index] = 2
        else:
            sample_kind[index] = 4

        if tiou < args.neg_tiou or not np.isfinite(gt_start_s) or not np.isfinite(gt_end_s):
            continue
        start_relative = (gt_start_s * 1e6 - float(row["t_start"])) / duration_us
        end_relative = (gt_end_s * 1e6 - float(row["t_start"])) / duration_us
        inside = (relative_times >= start_relative) & (relative_times <= end_relative)
        if not np.any(inside):
            center_relative = 0.5 * (start_relative + end_relative)
            inside[int(np.argmin(np.abs(relative_times - center_relative)))] = True
        action[index, inside] = 1.0
        point_distances[index, inside, 0] = (
            relative_times[inside] - start_relative
        ).astype(np.float32)
        point_distances[index, inside, 1] = (
            end_relative - relative_times[inside]
        ).astype(np.float32)

        if tiou >= args.boundary_min_tiou:
            start_target = np.exp(-0.5 * ((relative_times - start_relative) / sigma) ** 2)
            end_target = np.exp(-0.5 * ((relative_times - end_relative) / sigma) ** 2)
            start_distribution[index] = (start_target / max(start_target.sum(), 1e-12)).astype(np.float32)
            end_distribution[index] = (end_target / max(end_target.sum(), 1e-12)).astype(np.float32)
            deltas[index, 0] = np.clip(
                (gt_start_s * 1e6 - float(row["t_start"])) / duration_us,
                -args.max_boundary_delta,
                args.max_boundary_delta,
            )
            deltas[index, 1] = np.clip(
                (gt_end_s * 1e6 - float(row["t_end"])) / duration_us,
                -args.max_boundary_delta,
                args.max_boundary_delta,
            )
            boundary_weight[index] = tiou
    return DenseTargets(
        quality=quality,
        action=action,
        point_distances=point_distances,
        start_distribution=start_distribution,
        end_distribution=end_distribution,
        deltas=deltas,
        boundary_weight=boundary_weight,
        sample_kind=sample_kind,
    )


class DenseFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        master_indices: np.ndarray,
        targets: DenseTargets | None = None,
        group_ids: np.ndarray | None = None,
        event_feature_path: Path | None = None,
        event_features_only: bool = False,
        corrupted_event_feature_path: Path | None = None,
        corrupted_event_probability: float = 0.0,
        event_scale: np.ndarray | None = None,
        event_bias: np.ndarray | None = None,
    ) -> None:
        self.feature_path = feature_path
        self.features = np.load(feature_path, mmap_mode="r")
        self.master_indices = np.asarray(master_indices, dtype=np.int64)
        self.targets = targets
        self.group_ids = group_ids
        self.event_features_only = event_features_only
        self.corrupted_event_probability = float(corrupted_event_probability)
        self.event_scale = event_scale
        self.event_bias = event_bias
        if not 0.0 <= self.corrupted_event_probability <= 1.0:
            raise ValueError("corrupted_event_probability must be in [0, 1]")
        self.event_features = (
            np.load(event_feature_path, mmap_mode="r")
            if event_feature_path is not None
            else None
        )
        self.corrupted_event_features = (
            np.load(corrupted_event_feature_path, mmap_mode="r")
            if corrupted_event_feature_path is not None
            else None
        )
        if self.event_features is not None:
            aligned = self.event_features.shape[0] == self.features.shape[0]
            temporal = self.event_features.shape[1] in (1, self.features.shape[1])
            if not aligned or not temporal:
                raise ValueError(
                    "Event features are not aligned with dense features: "
                    f"{self.event_features.shape} vs {self.features.shape}"
                )
        if self.event_features_only and self.event_features is None:
            raise ValueError("event_features_only requires an aligned event feature cache")
        if self.corrupted_event_features is not None:
            if self.event_features is None:
                raise ValueError("A corrupted event cache requires a clean event cache")
            if self.corrupted_event_features.shape != self.event_features.shape:
                raise ValueError(
                    "Corrupted event features are not aligned with clean features: "
                    f"{self.corrupted_event_features.shape} vs {self.event_features.shape}"
                )
        for name, affine in (("event_scale", event_scale), ("event_bias", event_bias)):
            if affine is not None:
                if self.event_features is None:
                    raise ValueError(f"{name} requires an event feature cache")
                expected = (len(self.master_indices), self.event_features.shape[-1])
                if affine.shape != expected:
                    raise ValueError(f"{name} shape {affine.shape} does not match {expected}")

    def __len__(self) -> int:
        return len(self.master_indices)

    def __getitem__(self, index: int):
        feature = torch.from_numpy(
            np.asarray(self.features[self.master_indices[index]], dtype=np.float32).copy()
        )
        if self.event_features is not None:
            event_source = self.event_features
            if (
                self.targets is not None
                and self.corrupted_event_features is not None
                and torch.rand(()) < self.corrupted_event_probability
            ):
                event_source = self.corrupted_event_features
            event_feature = torch.from_numpy(
                np.asarray(
                    event_source[self.master_indices[index]],
                    dtype=np.float32,
                ).copy()
            )
            if event_feature.shape[0] == 1 and feature.shape[0] != 1:
                event_feature = event_feature.expand(feature.shape[0], -1)
            if self.event_scale is not None:
                scale = torch.from_numpy(
                    np.asarray(self.event_scale[index], dtype=np.float32).copy()
                )
                event_feature = event_feature * scale
            if self.event_bias is not None:
                bias = torch.from_numpy(
                    np.asarray(self.event_bias[index], dtype=np.float32).copy()
                )
                event_feature = event_feature + bias
            feature = (
                event_feature
                if self.event_features_only
                else torch.cat((feature, event_feature), dim=1)
            )
        if self.targets is None:
            return feature, int(index)
        group_id = -1 if self.group_ids is None else int(self.group_ids[index])
        return (
            feature,
            torch.tensor(self.targets.quality[index], dtype=torch.float32),
            torch.from_numpy(self.targets.action[index]),
            torch.from_numpy(self.targets.point_distances[index]),
            torch.from_numpy(self.targets.start_distribution[index]),
            torch.from_numpy(self.targets.end_distribution[index]),
            torch.from_numpy(self.targets.deltas[index]),
            torch.tensor(self.targets.boundary_weight[index], dtype=torch.float32),
            torch.tensor(group_id, dtype=torch.long),
        )


def softmax_ed(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.astype(np.float64).max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def choose_training_indices(
    proposals: pd.DataFrame,
    targets: DenseTargets,
    cnn_scores: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if max_samples <= 0 or len(proposals) <= max_samples:
        return np.arange(len(proposals), dtype=np.int64)
    rng = np.random.default_rng(seed)
    kinds = targets.sample_kind.copy()
    hard_negative = (kinds == 4) & (cnn_scores >= np.quantile(cnn_scores[kinds == 4], 0.80))
    kinds[hard_negative] = 3
    fractions = {0: 0.15, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.10}
    recordings = sorted(proposals["rec_name"].astype(str).unique())
    quota = int(math.ceil(max_samples / max(len(recordings), 1)))
    selected: list[np.ndarray] = []
    for recording in recordings:
        recording_idx = np.flatnonzero(proposals["rec_name"].astype(str).to_numpy() == recording)
        local: list[np.ndarray] = []
        for kind, fraction in fractions.items():
            pool = recording_idx[kinds[recording_idx] == kind]
            take = min(len(pool), max(1, int(round(quota * fraction)))) if len(pool) else 0
            if take:
                local.append(rng.choice(pool, size=take, replace=False))
        chosen = np.concatenate(local) if local else np.empty(0, dtype=np.int64)
        if len(chosen) < min(quota, len(recording_idx)):
            remaining = np.setdiff1d(recording_idx, chosen, assume_unique=False)
            take = min(len(remaining), quota - len(chosen))
            if take:
                chosen = np.concatenate((chosen, rng.choice(remaining, size=take, replace=False)))
        selected.append(chosen)
    output = np.unique(np.concatenate(selected))
    if len(output) < max_samples:
        remaining = np.setdiff1d(np.arange(len(proposals)), output, assume_unique=False)
        take = min(len(remaining), max_samples - len(output))
        output = np.concatenate((output, rng.choice(remaining, size=take, replace=False)))
    if len(output) > max_samples:
        output = rng.choice(output, size=max_samples, replace=False)
    rng.shuffle(output)
    return output.astype(np.int64)


def subset_targets(targets: DenseTargets, indices: np.ndarray) -> DenseTargets:
    return DenseTargets(
        quality=targets.quality[indices],
        action=targets.action[indices],
        point_distances=targets.point_distances[indices],
        start_distribution=targets.start_distribution[indices],
        end_distribution=targets.end_distribution[indices],
        deltas=targets.deltas[indices],
        boundary_weight=targets.boundary_weight[indices],
        sample_kind=targets.sample_kind[indices],
    )


def dense_loss(
    output: dict[str, torch.Tensor],
    quality_target: torch.Tensor,
    action_target: torch.Tensor,
    point_distance_target: torch.Tensor,
    start_target: torch.Tensor,
    end_target: torch.Tensor,
    delta_target: torch.Tensor,
    boundary_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    quality_probability = torch.sigmoid(output["quality_logit"])
    quality_bce = F.binary_cross_entropy_with_logits(
        output["quality_logit"], quality_target, reduction="none"
    )
    quality_loss = quality_bce * (quality_probability - quality_target).abs().pow(args.qfl_beta)

    point_quality_target = action_target * quality_target.unsqueeze(1)
    point_quality_probability = torch.sigmoid(output["point_quality_logits"])
    point_quality_bce = F.binary_cross_entropy_with_logits(
        output["point_quality_logits"], point_quality_target, reduction="none"
    )
    point_quality_loss = (
        point_quality_bce
        * (point_quality_probability - point_quality_target).abs().pow(args.qfl_beta)
    ).mean(dim=1)
    quality_loss = 0.5 * (quality_loss + point_quality_loss)

    action_bce = F.binary_cross_entropy_with_logits(
        output["action_logits"], action_target, reduction="none"
    )
    action_weights = 1.0 + action_target
    action_loss = (action_bce * action_weights).mean(dim=1)

    valid_boundary = boundary_weight > 0
    distribution_loss = torch.zeros_like(quality_loss)
    delta_loss = torch.zeros_like(quality_loss)
    trident_loss = torch.zeros_like(quality_loss)
    if valid_boundary.any():
        start_ce = -(start_target[valid_boundary] * F.log_softmax(
            output["start_logits"][valid_boundary], dim=1
        )).sum(dim=1)
        end_ce = -(end_target[valid_boundary] * F.log_softmax(
            output["end_logits"][valid_boundary], dim=1
        )).sum(dim=1)
        distribution_loss[valid_boundary] = 0.5 * (start_ce + end_ce) * boundary_weight[valid_boundary]
        regression = F.smooth_l1_loss(
            output["boundary_deltas"][valid_boundary],
            delta_target[valid_boundary],
            reduction="none",
        ).mean(dim=1)
        point_weights = action_target[valid_boundary] * boundary_weight[valid_boundary].unsqueeze(1)
        point_regression = F.smooth_l1_loss(
            output["boundary_distances"][valid_boundary],
            point_distance_target[valid_boundary],
            reduction="none",
        ).mean(dim=2)
        point_regression = (point_regression * point_weights).sum(dim=1) / point_weights.sum(
            dim=1
        ).clamp_min(1e-6)
        delta_loss[valid_boundary] = 0.5 * (
            regression * boundary_weight[valid_boundary] + point_regression
        )
        if output.get("trident_offsets_bins") is not None:
            step = (1.0 + 2.0 / args.augment_factor) / max(action_target.shape[1] - 1, 1)
            predicted_offsets = output["trident_offsets_bins"][valid_boundary] * step
            target_offsets = point_distance_target[valid_boundary]
            intersection = (
                torch.minimum(predicted_offsets[:, :, 0], target_offsets[:, :, 0])
                + torch.minimum(predicted_offsets[:, :, 1], target_offsets[:, :, 1])
            )
            union = (
                predicted_offsets.sum(dim=2)
                + target_offsets.sum(dim=2)
                - intersection
            ).clamp_min(1e-6)
            point_iou_loss = 1.0 - intersection / union
            trident_loss[valid_boundary] = (
                point_iou_loss * point_weights
            ).sum(dim=1) / point_weights.sum(dim=1).clamp_min(1e-6)

    sample_loss = (
        args.quality_weight * quality_loss
        + args.action_weight * action_loss
        + args.distribution_weight * distribution_loss
        + args.boundary_weight * delta_loss
        + getattr(args, "trident_weight", 0.0) * trident_loss
    )
    metrics = {
        "quality_loss": float(quality_loss.mean().detach().cpu()),
        "action_loss": float(action_loss.mean().detach().cpu()),
        "distribution_loss": float(distribution_loss.mean().detach().cpu()),
        "boundary_loss": float(delta_loss.mean().detach().cpu()),
        "trident_loss": float(trident_loss.mean().detach().cpu()),
    }
    return sample_loss, metrics


def temporal_robust_consistency_loss(
    clean_output: dict[str, torch.Tensor],
    corrupted_output: dict[str, torch.Tensor],
    delta_target: torch.Tensor,
    boundary_weight: torch.Tensor,
    augment_factor: int,
    topk: int,
) -> torch.Tensor:
    """Action-centric TRC over dense point-boundary tIoU error distributions."""
    distances_clean = clean_output["boundary_distances"]
    distances_corrupted = corrupted_output["boundary_distances"]
    batch_size, num_points, _ = distances_clean.shape
    losses = distances_clean.new_zeros(batch_size)
    valid = boundary_weight > 0
    if not valid.any():
        return losses

    relative_times = torch.linspace(
        -1.0 / augment_factor,
        1.0 + 1.0 / augment_factor,
        num_points,
        device=distances_clean.device,
        dtype=distances_clean.dtype,
    )
    gt_start = delta_target[:, 0]
    gt_end = 1.0 + delta_target[:, 1]
    gt_center = 0.5 * (gt_start + gt_end)
    nearest = (relative_times.unsqueeze(0) - gt_center.unsqueeze(1)).abs().topk(
        min(max(int(topk), 1), num_points),
        dim=1,
        largest=False,
    ).indices
    selected_times = relative_times.expand(batch_size, -1).gather(1, nearest)
    gather_index = nearest.unsqueeze(2).expand(-1, -1, 2)

    def error_distribution(distances: torch.Tensor) -> torch.Tensor:
        selected = distances.gather(1, gather_index)
        predicted_start = selected_times - selected[:, :, 0]
        predicted_end = selected_times + selected[:, :, 1]
        intersection = (
            torch.minimum(predicted_end, gt_end.unsqueeze(1))
            - torch.maximum(predicted_start, gt_start.unsqueeze(1))
        ).clamp_min(0.0)
        union = (
            predicted_end - predicted_start
            + gt_end.unsqueeze(1) - gt_start.unsqueeze(1)
            - intersection
        ).clamp_min(1e-6)
        tiou = intersection / union
        return (1.0 - tiou).clamp(1e-6, 1.0)

    clean_error = error_distribution(distances_clean)
    corrupted_error = error_distribution(distances_corrupted)
    target = 0.5 * (clean_error + corrupted_error)
    reverse_kl_clean = target * (target.log() - clean_error.log())
    reverse_kl_corrupted = target * (target.log() - corrupted_error.log())
    per_sample = 0.5 * (reverse_kl_clean + reverse_kl_corrupted).sum(dim=1)
    losses[valid] = per_sample[valid] * boundary_weight[valid]
    return losses


def group_risk(
    sample_loss: torch.Tensor,
    group_ids: torch.Tensor,
    weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    present = torch.unique(group_ids[group_ids >= 0])
    if len(present) == 0:
        return sample_loss.mean()
    losses = torch.stack([sample_loss[group_ids == group].mean() for group in present])
    with torch.no_grad():
        weights[present] *= torch.exp(eta * losses.detach())
        weights /= weights.sum().clamp_min(1e-12)
    local_weights = weights[present]
    local_weights = local_weights / local_weights.sum().clamp_min(1e-12)
    return (local_weights * losses).sum()


@torch.no_grad()
def score_model(
    model: TemporalMaxerLiteHead,
    proposals: pd.DataFrame,
    master_indices: np.ndarray,
    feature_path: Path,
    master_logits: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    event_scale: np.ndarray | None = None,
    event_bias: np.ndarray | None = None,
) -> pd.DataFrame:
    event_feature_path = None
    if getattr(args, "event_feature_cache_dir", None):
        base_metadata = json.loads(
            cache_paths(feature_path.parent)["metadata"].read_text(encoding="utf-8")
        )
        event_feature_path, _ = event_feature_configuration(args, base_metadata)
    dataset = DenseFeatureDataset(
        feature_path,
        master_indices,
        event_feature_path=event_feature_path,
        event_features_only=getattr(args, "event_features_only", False),
        event_scale=event_scale,
        event_bias=event_bias,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(args.batch_size, 512),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    quality_batches = []
    action_batches = []
    point_score_batches = []
    point_position_batches = []
    point_distance_batches = []
    trident_distance_batches = []
    delta_batches = []
    start_position_batches = []
    end_position_batches = []
    model.eval()
    relative_times = torch.linspace(
        -1.0 / args.augment_factor,
        1.0 + 1.0 / args.augment_factor,
        dataset.features.shape[1],
        device=device,
    )
    for features, _ in loader:
        features = features.to(device, non_blocking=True)
        with autocast_context(device):
            output = model(features)
        quality_batches.append(torch.sigmoid(output["quality_logit"]).float().cpu().numpy())
        action_probability = torch.sigmoid(output["action_logits"])
        point_quality_probability = torch.sigmoid(output["point_quality_logits"])
        point_joint = action_probability * point_quality_probability
        topk = max(1, action_probability.shape[1] // 4)
        action_batches.append(action_probability.topk(topk, dim=1).values.mean(dim=1).float().cpu().numpy())
        point_score_batches.append(
            point_joint.topk(topk, dim=1).values.mean(dim=1).float().cpu().numpy()
        )
        point_index = point_joint.argmax(dim=1)
        batch_index = torch.arange(len(features), device=device)
        point_position_batches.append(relative_times[point_index].float().cpu().numpy())
        point_distance_batches.append(
            output["boundary_distances"][batch_index, point_index].float().cpu().numpy()
        )
        if output.get("trident_offsets_bins") is not None:
            step = (1.0 + 2.0 / args.augment_factor) / max(action_probability.shape[1] - 1, 1)
            trident_distance_batches.append(
                (
                    output["trident_offsets_bins"][batch_index, point_index] * step
                ).float().cpu().numpy()
            )
        delta_batches.append(output["boundary_deltas"].float().cpu().numpy())
        start_probability = torch.softmax(output["start_logits"], dim=1)
        end_probability = torch.softmax(output["end_logits"], dim=1)
        start_position_batches.append((start_probability * relative_times).sum(dim=1).float().cpu().numpy())
        end_position_batches.append((end_probability * relative_times).sum(dim=1).float().cpu().numpy())

    quality = np.concatenate(quality_batches)
    action = np.concatenate(action_batches)
    point_score = np.concatenate(point_score_batches)
    point_position = np.concatenate(point_position_batches)
    point_distances = np.concatenate(point_distance_batches)
    trident_distances = (
        np.concatenate(trident_distance_batches) if trident_distance_batches else None
    )
    deltas = np.concatenate(delta_batches)
    start_position = np.concatenate(start_position_batches)
    end_position = np.concatenate(end_position_batches)
    cnn_score = softmax_ed(np.asarray(master_logits[master_indices]))
    scored = proposals.reset_index(drop=True).copy()
    scored["cnn_score"] = cnn_score
    scored["dense_quality"] = quality
    scored["dense_action"] = action
    scored["dense_point"] = point_score
    scored["dense_score"] = np.sqrt(np.clip(quality * point_score, 0.0, 1.0))
    scored["brem_score"] = np.cbrt(np.clip(cnn_score * quality * point_score, 0.0, 1.0))

    starts = scored["t_start"].to_numpy(dtype=np.float64)
    ends = scored["t_end"].to_numpy(dtype=np.float64)
    duration = np.maximum(ends - starts, 1.0)
    delta_start = np.clip(deltas[:, 0], -args.max_boundary_delta, args.max_boundary_delta)
    delta_end = np.clip(deltas[:, 1], -args.max_boundary_delta, args.max_boundary_delta)
    scored["delta_t_start"] = np.maximum(0.0, starts + delta_start * duration)
    scored["delta_t_end"] = ends + delta_end * duration
    blend = float(args.boundary_blend)
    scored["blend_t_start"] = (1.0 - blend) * starts + blend * scored["delta_t_start"]
    scored["blend_t_end"] = (1.0 - blend) * ends + blend * scored["delta_t_end"]
    scored["distribution_t_start"] = np.maximum(0.0, starts + start_position * duration)
    scored["distribution_t_end"] = starts + end_position * duration
    scored["point_t_start"] = np.maximum(
        0.0,
        starts + (point_position - point_distances[:, 0]) * duration,
    )
    scored["point_t_end"] = starts + (
        point_position + point_distances[:, 1]
    ) * duration
    if trident_distances is not None:
        scored["trident_t_start"] = np.maximum(
            0.0,
            starts + (point_position - trident_distances[:, 0]) * duration,
        )
        scored["trident_t_end"] = starts + (
            point_position + trident_distances[:, 1]
        ) * duration
    boundary_prefixes = ["delta", "blend", "distribution", "point"]
    if trident_distances is not None:
        boundary_prefixes.append("trident")
    for prefix in boundary_prefixes:
        refined_start = scored[f"{prefix}_t_start"].to_numpy(dtype=np.float64, copy=True)
        refined_end = scored[f"{prefix}_t_end"].to_numpy(dtype=np.float64, copy=True)
        center = 0.5 * (refined_start + refined_end)
        min_duration = minimum_action_duration(args) * 1e6
        short = refined_end - refined_start < min_duration
        refined_start[short] = np.maximum(0.0, center[short] - 0.5 * min_duration)
        refined_end[short] = refined_start[short] + min_duration
        scored[f"{prefix}_t_start"] = refined_start
        scored[f"{prefix}_t_end"] = refined_end
    return scored


def build_prediction(
    scored: pd.DataFrame,
    score_column: str,
    boundary_mode: str,
    args: argparse.Namespace,
) -> dict:
    start_column = "t_start" if boundary_mode == "raw" else f"{boundary_mode}_t_start"
    end_column = "t_end" if boundary_mode == "raw" else f"{boundary_mode}_t_end"
    result = {
        recording: {int(str(roi)[1:]): [] for roi in group["roi_id"].unique()}
        for recording, group in scored.groupby("rec_name")
    }
    selected = scored[scored[score_column] >= args.min_score].copy()
    if selected.empty:
        return {"version": f"temporalmaxer:{score_column}:{boundary_mode}", "results": result}
    durations = (
        selected[end_column].to_numpy(dtype=np.float64)
        - selected[start_column].to_numpy(dtype=np.float64)
    ) / 1e6
    penalties = np.exp(-np.maximum(0.0, durations - args.duration_dmax) / args.duration_sigma)
    selected["final_score"] = selected[score_column].to_numpy(dtype=np.float64) * penalties
    for (recording, roi), group in selected.groupby(["rec_name", "roi_id"]):
        if args.pre_nms_topk_per_roi > 0 and len(group) > args.pre_nms_topk_per_roi:
            group = group.nlargest(args.pre_nms_topk_per_roi, "final_score")
        candidates = group[[start_column, end_column, "final_score"]].to_numpy(dtype=np.float64)
        detections = temporal_soft_nms(
            candidates,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
        )
        result[recording][int(str(roi)[1:])] = [
            {
                "label": "ed",
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(score),
            }
            for start, end, score in detections
            if float(end) - float(start) >= minimum_action_duration(args) * 1e6
        ]
    return {"version": f"temporalmaxer:{score_column}:{boundary_mode}", "results": result}


def evaluate_variant(
    scored: pd.DataFrame,
    score_column: str,
    boundary_mode: str,
    label: str,
    args: argparse.Namespace,
    prediction_dir: Path,
) -> dict[str, float | str | int]:
    prediction = build_prediction(scored, score_column, boundary_mode, args)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_dir / f"{label}_{score_column}_{boundary_mode}.json"
    path.write_text(json.dumps(prediction), encoding="utf-8")
    sequences = sorted(scored["rec_name"].astype(str).unique())
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(resolve(args.ann_path)),
        prediction_filename=str(path),
        tiou_thresholds=np.asarray(args.tiou, dtype=np.float64),
        valid_labels="ed",
        valid_sequences=sequences,
        min_duration=minimum_action_duration(args),
    )
    mean_ap = float(evaluator.run())
    count = sum(
        len(detections)
        for rois in prediction["results"].values()
        for detections in rois.values()
    )
    row: dict[str, float | str | int] = {
        "label": label,
        "score_column": score_column,
        "boundary_mode": boundary_mode,
        "mAP": mean_ap,
        "n_predictions": count,
    }
    for threshold, value in zip(args.tiou, evaluator.mAP):
        row[f"AP@{threshold:.1f}"] = float(value)
    return row


def evaluate_model(
    model: TemporalMaxerLiteHead,
    proposals: pd.DataFrame,
    master_indices: np.ndarray,
    feature_path: Path,
    master_logits: np.ndarray,
    label: str,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    variants: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    scored = score_model(
        model,
        proposals,
        master_indices,
        feature_path,
        master_logits,
        args,
        device,
    )
    if variants is None:
        variants = [
            ("cnn_score", "raw"),
            ("cnn_score", "delta"),
            ("cnn_score", "blend"),
            ("cnn_score", "point"),
            ("dense_score", "raw"),
            ("brem_score", "raw"),
            ("brem_score", "delta"),
            ("brem_score", "distribution"),
        ]
    metrics = [
        evaluate_variant(scored, score, boundary, label, args, out_dir / "predictions")
        for score, boundary in variants
    ]
    return scored, metrics


def make_model(metadata: dict, args: argparse.Namespace) -> TemporalMaxerLiteHead:
    _, event_feature_dim = event_feature_configuration(args, metadata)
    event_only = bool(getattr(args, "event_features_only", False))
    if event_only and event_feature_dim <= 0:
        raise ValueError("event_features_only requires --event-feature-cache-dir")
    input_dim = (
        event_feature_dim
        if event_only
        else int(metadata["feature_dim"]) + event_feature_dim
    )
    return TemporalMaxerLiteHead(
        input_dim=input_dim,
        auxiliary_dim=0 if event_only else event_feature_dim,
        hidden_dim=args.hidden_dim,
        pyramid_levels=args.pyramid_levels,
        dropout=args.dropout,
        trident_bins=getattr(args, "trident_bins", 0),
        tanp_sigma=getattr(args, "tanp_sigma", 0.0),
    )


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(
    master: pd.DataFrame,
    train_proposals: pd.DataFrame,
    val_proposals: pd.DataFrame,
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    features, logits, metadata = load_cache(cache_dir)
    event_feature_path, event_feature_dim = event_feature_configuration(args, metadata)
    corrupted_event_feature_path = None
    if args.corrupted_event_feature_cache_dir:
        if event_feature_path is None:
            raise ValueError("Corrupted event training requires --event-feature-cache-dir")
        corrupted_dir = resolve(args.corrupted_event_feature_cache_dir)
        _, corrupted_metadata = load_event_cache(corrupted_dir, metadata)
        if int(corrupted_metadata["feature_dim"]) != event_feature_dim:
            raise ValueError("Clean and corrupted event caches use different feature dimensions")
        corrupted_event_feature_path = event_cache_paths(corrupted_dir)["features"]
    train_master_indices = map_to_master(master, train_proposals)
    val_master_indices = map_to_master(master, val_proposals)
    train_targets_all = build_targets(train_proposals, features.shape[1], args)
    train_cnn_scores = softmax_ed(np.asarray(logits[train_master_indices]))
    sampled = choose_training_indices(
        train_proposals,
        train_targets_all,
        train_cnn_scores,
        args.max_train_samples,
        args.seed,
    )
    sampled_proposals = train_proposals.iloc[sampled].reset_index(drop=True)
    sampled_targets = subset_targets(train_targets_all, sampled)
    sampled_master_indices = train_master_indices[sampled]
    recordings = sorted(sampled_proposals["rec_name"].astype(str).unique())
    recording_to_group = {recording: index for index, recording in enumerate(recordings)}
    group_ids = sampled_proposals["rec_name"].astype(str).map(recording_to_group).to_numpy(dtype=np.int64)
    dataset = DenseFeatureDataset(
        cache_paths(cache_dir)["features"],
        sampled_master_indices,
        sampled_targets,
        group_ids,
        event_feature_path=event_feature_path,
        event_features_only=args.event_features_only,
        corrupted_event_feature_path=corrupted_event_feature_path,
        corrupted_event_probability=args.corrupted_event_probability,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = make_model(metadata, args).to(device)
    blank_feature = event_blank_feature(args, device) if args.trc_weight > 0 else None
    if blank_feature is not None and event_feature_dim:
        event_blank = torch.zeros(event_feature_dim, device=device)
        blank_feature = (
            event_blank
            if args.event_features_only
            else torch.cat((blank_feature, event_blank), dim=0)
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    group_weights = torch.full(
        (len(recordings),),
        1.0 / max(len(recordings), 1),
        device=device,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_selection = -float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    last_path = out_dir / "last.pt"
    if last_path.exists() and not args.restart:
        try:
            state = torch.load(last_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        group_weights.copy_(state["group_weights"].to(device))
        generator.set_state(state["generator_state"].cpu())
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
            )
        history = state["history"]
        best_selection = float(state["best_selection"])
        best_state = state["best_state"]
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        start_epoch = int(state["epoch"]) + 1
        print(
            f"[INFO] Reanudando tras época {state['epoch']}; "
            f"mellor época {best_epoch}, stale={stale_epochs}"
        )
    print(
        f"[INFO] train={len(sampled)}/{len(train_proposals)} val={len(val_proposals)} "
        f"records={len(recordings)} device={device}"
    )
    if args.selection_boundary == "trident" and args.trident_bins <= 0:
        raise ValueError("Trident selection requires --trident-bins > 0")
    selection_variants = (
        [(args.selection_score, args.selection_boundary)]
        if args.fast_selection_eval
        else [
            ("cnn_score", "raw"),
            ("cnn_score", "delta"),
            ("cnn_score", "blend"),
            ("cnn_score", "point"),
            *([("cnn_score", "trident")] if args.trident_bins > 0 else []),
        ]
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        component_sum = {key: 0.0 for key in (
            "quality_loss", "action_loss", "distribution_loss", "boundary_loss",
            "trident_loss", "trc_loss"
        )}
        batches = 0
        progress = tqdm(loader, desc=f"train-{epoch:02d}", disable=args.quiet_progress)
        for batch in progress:
            (
                frame_features,
                quality_target,
                action_target,
                point_distance_target,
                start_target,
                end_target,
                delta_target,
                boundary_weight,
                batch_groups,
            ) = batch
            frame_features = frame_features.to(device, non_blocking=True)
            quality_target = quality_target.to(device, non_blocking=True)
            action_target = action_target.to(device, non_blocking=True)
            point_distance_target = point_distance_target.to(device, non_blocking=True)
            start_target = start_target.to(device, non_blocking=True)
            end_target = end_target.to(device, non_blocking=True)
            delta_target = delta_target.to(device, non_blocking=True)
            boundary_weight = boundary_weight.to(device, non_blocking=True)
            batch_groups = batch_groups.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(frame_features)
                sample_loss, components = dense_loss(
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
                trc_per_sample = torch.zeros_like(sample_loss)
                if blank_feature is not None:
                    corrupted_output = model(drop_one_event_frame(frame_features, blank_feature))
                    trc_per_sample = temporal_robust_consistency_loss(
                        output,
                        corrupted_output,
                        delta_target,
                        boundary_weight,
                        args.augment_factor,
                        args.trc_topk,
                    )
                    # The published TRC implementation sums over action-centric
                    # predictions instead of normalizing by the batch size.
                    sample_loss = (
                        sample_loss
                        + args.trc_weight * trc_per_sample * sample_loss.numel()
                    )
                components["trc_loss"] = float(trc_per_sample.sum().detach().cpu())
                loss = (
                    group_risk(sample_loss, batch_groups, group_weights, args.group_dro_eta)
                    if args.group_dro
                    else sample_loss.mean()
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            for key, value in components.items():
                component_sum[key] += value
            batches += 1
            progress.set_postfix(loss=f"{epoch_loss / batches:.4f}")
        scheduler.step()

        scored, metrics = evaluate_model(
            model,
            val_proposals,
            val_master_indices,
            cache_paths(cache_dir)["features"],
            logits,
            f"epoch{epoch:02d}",
            args,
            out_dir,
            device,
            variants=selection_variants,
        )
        metrics_by_variant = {
            (row["score_column"], row["boundary_mode"]): row for row in metrics
        }
        primary_selection = metrics_by_variant[
            (args.selection_score, args.selection_boundary)
        ]
        selection = float(primary_selection["mAP"])
        epoch_row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "selection_mAP": selection,
            **{key: value / max(batches, 1) for key, value in component_sum.items()},
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        pd.DataFrame(metrics).to_csv(out_dir / f"metrics_epoch{epoch:02d}.csv", index=False)
        scored.to_csv(out_dir / f"scored_epoch{epoch:02d}.csv", index=False)
        diagnostics = " ".join(
            f"{boundary}={float(row['mAP']):.4f}"
            for (score, boundary), row in metrics_by_variant.items()
            if score == args.selection_score
        )
        print(
            f"[EPOCH {epoch:02d}] loss={epoch_row['train_loss']:.4f} "
            f"{diagnostics} selection={selection:.4f}"
        )
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
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "group_weights": group_weights.detach().cpu(),
                "generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                "history": history,
                "best_selection": best_selection,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            },
            last_path,
        )
        if stale_epochs >= args.patience:
            print(f"[INFO] Early stopping en época {epoch}; mellor época {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": best_state,
        "args": vars(args),
        "feature_dim": int(metadata["feature_dim"]) + event_feature_dim,
        "best_epoch": best_epoch,
        "best_selection_mAP": best_selection,
        "recording_groups": recordings,
        "group_weights": group_weights.detach().cpu().tolist() if args.group_dro else None,
    }
    torch.save(checkpoint, out_dir / "best.pt")
    final_scored, final_metrics = evaluate_model(
        model,
        val_proposals,
        val_master_indices,
        cache_paths(cache_dir)["features"],
        logits,
        f"best_epoch{best_epoch:02d}",
        args,
        out_dir,
        device,
        variants=selection_variants,
    )
    final_scored.to_csv(out_dir / "scored_best.csv", index=False)
    pd.DataFrame(final_metrics).to_csv(out_dir / "metrics_best.csv", index=False)
    print(pd.DataFrame(final_metrics).to_string(index=False))


def evaluate_checkpoint(
    checkpoint_path: Path,
    master: pd.DataFrame,
    proposals: pd.DataFrame,
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    features, logits, metadata = load_cache(cache_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    for name in (
        "hidden_dim",
        "pyramid_levels",
        "dropout",
        "trident_bins",
        "event_features_only",
        "min_action_duration",
    ):
        if name in saved_args:
            setattr(args, name, saved_args[name])
    model = make_model(metadata, args).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    indices = map_to_master(master, proposals)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored, metrics = evaluate_model(
        model,
        proposals,
        indices,
        cache_paths(cache_dir)["features"],
        logits,
        args.eval_label,
        args,
        out_dir,
        device,
    )
    scored.to_csv(out_dir / f"scored_{args.eval_label}.csv", index=False)
    pd.DataFrame(metrics).to_csv(out_dir / f"metrics_{args.eval_label}.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.min_action_duration < 0:
        raise ValueError("--min-action-duration must be non-negative")
    set_seed(args.seed)
    device = choose_device(args.device)
    master_path = resolve(args.master_proposals)
    master = pd.read_csv(master_path).reset_index(drop=True)
    cache_dir = resolve(args.cache_dir)
    extract_representations(master, cache_dir, args, device)
    if args.extract_only:
        return
    if args.val_proposals is None:
        raise ValueError("--val-proposals is required unless --extract-only is used")
    val_proposals = pd.read_csv(resolve(args.val_proposals)).reset_index(drop=True)
    if args.eval_checkpoint:
        evaluate_checkpoint(
            resolve(args.eval_checkpoint),
            master,
            val_proposals,
            cache_dir,
            args,
            device,
        )
        return
    if args.train_proposals is None:
        raise ValueError("--train-proposals is required for training")
    train_proposals = pd.read_csv(resolve(args.train_proposals)).reset_index(drop=True)
    train(master, train_proposals, val_proposals, cache_dir, args, device)


if __name__ == "__main__":
    main()
