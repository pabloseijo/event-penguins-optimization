#!/usr/bin/env python3
"""Frozen-ATSN one-vs-rest transfer protocol for THUMOS14-E.

The source ATSN encoder is never updated. Proposal features are extracted once,
then a binary linear head is trained per THUMOS class with the reTAG recipe:
IoU > 0.7 positives, 10x negative downsampling, weighted cross entropy, SGD
(momentum 0.9, lr 1e-3, batch size 128), and ten fixed epochs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.prepare_thumos14_event_pilot import sha256_file  # noqa: E402
from src.augmented_tsn import AugmentedTsn  # noqa: E402
from src.classification import ProposalDataset  # noqa: E402
from src.utils import temporal_nms  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def common_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features-dir", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--data-path", type=Path, required=True)
    extract.add_argument("--proposals", type=Path, required=True)
    extract.add_argument("--source-model", type=Path, required=True)
    extract.add_argument("--out-dir", type=Path, required=True)
    extract.add_argument("--num-tsn-samples", type=int, default=7)
    extract.add_argument("--augment-factor", type=int, default=3)
    extract.add_argument("--sample-duration", type=float, default=1.0)
    extract.add_argument("--decay", type=float, default=5e-6)
    extract.add_argument("--batch-size", type=int, default=64)
    extract.add_argument("--num-workers", type=int, default=4)
    extract.add_argument("--timestamp-cache-dir", type=Path, default=None)
    extract.add_argument("--device", default=None)
    extract.add_argument("--force", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--train-features-dir", type=Path, required=True)
    train.add_argument(
        "--val-features-dir",
        type=Path,
        default=None,
        help="Defaults to --train-features-dir for video-disjoint CV over one shared cache.",
    )
    train.add_argument("--annotations", type=Path, required=True)
    train.add_argument("--source-model", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--positive-tiou", type=float, default=0.7)
    train.add_argument("--negative-keep-fraction", type=float, default=0.1)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--head-init", choices=("reset", "source"), default="reset")
    train.add_argument("--seed", type=int, default=1234567891)
    train.add_argument("--device", default=None)
    fold_mode = train.add_mutually_exclusive_group()
    fold_mode.add_argument(
        "--cv-fold",
        type=int,
        choices=range(5),
        default=None,
        help="Use cv_fold metadata to hold out one fifth of official validation videos.",
    )
    fold_mode.add_argument(
        "--train-all-validation",
        action="store_true",
        help="Final fit on all 200 official validation videos, with no validation selection.",
    )

    score = commands.add_parser("score")
    common_feature_args(score)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--out-dir", type=Path, required=True)
    score.add_argument("--min-positive-score", type=float, default=0.5)
    score.add_argument("--nms-threshold", type=float, default=0.6)
    score.add_argument("--batch-size", type=int, default=4096)
    score.add_argument("--device", default=None)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_source_model(
    source_path: Path,
    num_tsn_samples: int,
    augment_factor: int,
    device: torch.device,
) -> AugmentedTsn:
    model = AugmentedTsn(2, num_tsn_samples, augment_factor)
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(source_path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    model.load_state_dict(state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def consensus_features(model: AugmentedTsn, frame_features: torch.Tensor) -> torch.Tensor:
    augment = model.num_augment
    if frame_features.shape[1] <= 2 * augment:
        raise ValueError("Not enough temporal samples for ATSN start/main/end consensus")
    start = frame_features[:, :augment].mean(dim=1)
    main = frame_features[:, augment:-augment].mean(dim=1)
    end = frame_features[:, -augment:].mean(dim=1)
    return torch.cat((start, main, end), dim=1)


def canonical_proposals(path: Path) -> pd.DataFrame:
    proposals = pd.read_csv(path).copy()
    required = {"rec_name", "roi_id", "t_start", "t_end"}
    missing = required - set(proposals.columns)
    if missing:
        raise ValueError(f"Proposal CSV lacks columns {sorted(missing)}")
    proposals["source_row"] = np.arange(len(proposals), dtype=np.int64)
    return proposals.sort_values(
        ["rec_name", "roi_id", "t_start", "t_end", "source_row"],
        kind="mergesort",
    ).reset_index(drop=True)


def extraction_metadata(args: argparse.Namespace, proposals: pd.DataFrame) -> dict[str, object]:
    return {
        "protocol": "THUMOS14-E-frozen-ATSN-features-v1",
        "data_path": str(resolve(args.data_path)),
        "data_index_sha256": sha256_file(resolve(args.data_path)),
        "proposal_path": str(resolve(args.proposals)),
        "proposal_sha256": sha256_file(resolve(args.proposals)),
        "source_model": str(resolve(args.source_model)),
        "source_model_sha256": sha256_file(resolve(args.source_model)),
        "num_proposals": len(proposals),
        "num_tsn_samples": args.num_tsn_samples,
        "augment_factor": args.augment_factor,
        "expanded_samples": args.num_tsn_samples
        + 2 * int(np.ceil(args.num_tsn_samples / args.augment_factor)),
        "sample_duration_s": args.sample_duration,
        "decay_per_us": args.decay,
        "feature_dim": 1536,
    }


def extract(args: argparse.Namespace) -> None:
    data_path = resolve(args.data_path)
    proposal_path = resolve(args.proposals)
    source_model = resolve(args.source_model)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.npy"
    building_path = out_dir / "features.npy.building"
    proposals_out = out_dir / "proposals.csv"
    metadata_path = out_dir / "metadata.json"
    progress_path = out_dir / "progress.json"

    proposals = canonical_proposals(proposal_path)
    metadata = extraction_metadata(args, proposals)
    if features_path.exists() and metadata_path.exists() and not args.force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(f"Stale feature cache {out_dir}; use --force")
        print(f"[SKIP] frozen features already complete: {features_path}")
        return
    if args.force:
        for path in (features_path, building_path, proposals_out, metadata_path, progress_path):
            if path.exists():
                path.unlink()

    proposals.to_csv(proposals_out, index=False)
    completed = 0
    if progress_path.exists() and building_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("metadata") != metadata:
            raise RuntimeError(f"Stale interrupted extraction in {out_dir}; use --force")
        completed = int(progress["completed"])
        feature_array = np.lib.format.open_memmap(building_path, mode="r+")
    else:
        feature_array = np.lib.format.open_memmap(
            building_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(proposals), int(metadata["feature_dim"])),
        )
    if not 0 <= completed <= len(proposals):
        raise ValueError(f"Invalid extraction progress {completed}/{len(proposals)}")

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = load_source_model(
        source_model,
        args.num_tsn_samples,
        args.augment_factor,
        device,
    )
    remaining = proposals.iloc[completed:].reset_index(drop=True)
    dataset = ProposalDataset(
        remaining,
        augment_fraction=1.0 / args.augment_factor,
        data_path=str(data_path),
        num_tsn_samples=int(metadata["expanded_samples"]),
        sample_duration=args.sample_duration * 1e6,
        decay=args.decay,
        cache_full_events=False,
        timestamp_cache_dir=(
            str(resolve(args.timestamp_cache_dir))
            if args.timestamp_cache_dir is not None
            else str(out_dir / "timestamp_cache")
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    cursor = completed
    with torch.inference_mode():
        for batch in tqdm(loader, desc="frozen ATSN features"):
            images = batch[0].to(device, non_blocking=True)
            frame_features = model.encode_frames(images)
            values = consensus_features(model, frame_features).cpu().numpy()
            feature_array[cursor : cursor + len(values)] = values
            cursor += len(values)
            feature_array.flush()
            atomic_write_json(
                progress_path,
                {"completed": cursor, "metadata": metadata},
            )
    if cursor != len(proposals):
        raise RuntimeError(f"Incomplete extraction {cursor}/{len(proposals)}")
    del feature_array
    os.replace(building_path, features_path)
    if progress_path.exists():
        progress_path.unlink()
    atomic_write_json(metadata_path, metadata)
    print(f"[OK] features={features_path} shape=({len(proposals)}, 1536)")


def split_recordings(
    annotation_path: Path,
    split: str,
    cv_fold: int | None = None,
    all_validation: bool = False,
) -> set[str]:
    info_path = annotation_path.parent / "recording_info.csv"
    with info_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if all_validation:
        if split != "train":
            raise ValueError("all_validation is valid only for the training selection")
        return {
            row["timestamp"]
            for row in rows
            if row.get("official_subset") == "validation"
        }
    if cv_fold is not None:
        if split not in {"train", "val"}:
            raise ValueError("cv_fold supports only train/val selections")
        return {
            row["timestamp"]
            for row in rows
            if row.get("official_subset") == "validation"
            and bool(int(row["cv_fold"]) == cv_fold) == (split == "val")
        }
    return {row["timestamp"] for row in rows if row["split"] == split}


def annotation_index(
    annotation_path: Path,
    split: str,
    cv_fold: int | None = None,
    all_validation: bool = False,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    recordings = split_recordings(annotation_path, split, cv_fold, all_validation)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    targets: dict[tuple[str, str], np.ndarray] = {}
    ambiguous: dict[tuple[str, str], np.ndarray] = {}
    for recording, data in payload["database"].items():
        if recording not in recordings:
            continue
        for roi_id, items in data.get("annotations", {}).items():
            if roi_id == "null":
                continue
            targets[(recording, roi_id)] = np.asarray(
                [item["segment"] for item in items if item["label"] == "ed"],
                dtype=np.float64,
            ).reshape(-1, 2)
            ambiguous[(recording, roi_id)] = np.asarray(
                [item["segment"] for item in items if item["label"] == "ambiguous"],
                dtype=np.float64,
            ).reshape(-1, 2)
    return targets, ambiguous


def proposal_roi_key(value: object) -> str:
    text = str(value)
    return str(int(text[1:])) if text.startswith("N") else str(int(float(text)))


def temporal_iou(segment: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if len(candidates) == 0:
        return np.empty(0, dtype=np.float64)
    intersection = np.maximum(
        0.0,
        np.minimum(segment[1], candidates[:, 1])
        - np.maximum(segment[0], candidates[:, 0]),
    )
    union = (segment[1] - segment[0]) + (candidates[:, 1] - candidates[:, 0]) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def temporal_intersection(segment: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if len(candidates) == 0:
        return np.empty(0, dtype=np.float64)
    return np.maximum(
        0.0,
        np.minimum(segment[1], candidates[:, 1])
        - np.maximum(segment[0], candidates[:, 0]),
    )


def label_proposals(
    proposals: pd.DataFrame,
    annotation_path: Path,
    split: str,
    positive_tiou: float,
    cv_fold: int | None = None,
    all_validation: bool = False,
) -> pd.DataFrame:
    targets, ambiguous = annotation_index(
        annotation_path, split, cv_fold, all_validation
    )
    valid_recordings = split_recordings(
        annotation_path, split, cv_fold, all_validation
    )
    rows = []
    for index, row in proposals.reset_index(drop=True).iterrows():
        recording = str(row["rec_name"])
        if recording not in valid_recordings:
            continue
        key = (recording, proposal_roi_key(row["roi_id"]))
        segment = np.asarray(
            [float(row["t_start"]) / 1e6, float(row["t_end"]) / 1e6],
            dtype=np.float64,
        )
        ious = temporal_iou(segment, targets.get(key, np.empty((0, 2))))
        best_tiou = float(ious.max()) if len(ious) else 0.0
        overlaps_ambiguous = bool(
            np.any(
                temporal_intersection(
                    segment,
                    ambiguous.get(key, np.empty((0, 2))),
                )
                > 0
            )
        )
        label = 1 if best_tiou > positive_tiou else (0 if not overlaps_ambiguous else -1)
        rows.append(
            {
                "feature_index": index,
                "label": label,
                "best_target_tiou": best_tiou,
                "overlaps_ambiguous": overlaps_ambiguous,
                **row.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def stable_fraction_mask(rows: pd.DataFrame, fraction: float, seed: int) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("negative keep fraction must be in (0, 1]")
    limit = int(fraction * (1 << 64))
    values = []
    for _, row in rows.iterrows():
        identity = (
            f"{seed}:{row['rec_name']}:{row['roi_id']}:"
            f"{float(row['t_start']):.3f}:{float(row['t_end']):.3f}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") < limit)
    return np.asarray(values, dtype=bool)


def downsample_training_rows(
    labeled: pd.DataFrame,
    negative_fraction: float,
    seed: int,
) -> pd.DataFrame:
    positives = labeled[labeled["label"] == 1]
    negatives = labeled[labeled["label"] == 0]
    kept_negatives = negatives[stable_fraction_mask(negatives, negative_fraction, seed)]
    result = pd.concat((positives, kept_negatives), ignore_index=True)
    return result.sort_values("feature_index", kind="mergesort").reset_index(drop=True)


class MemmapFeatureDataset(Dataset):
    def __init__(self, feature_path: Path, rows: pd.DataFrame) -> None:
        self.feature_path = feature_path
        self.indices = rows["feature_index"].to_numpy(dtype=np.int64)
        self.labels = rows["label"].to_numpy(dtype=np.int64)
        self.features = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.features is None:
            self.features = np.load(self.feature_path, mmap_mode="r")
        values = np.array(self.features[self.indices[index]], dtype=np.float32, copy=True)
        return torch.from_numpy(values), torch.tensor(self.labels[index], dtype=torch.long)


def load_feature_cache(path: Path) -> tuple[Path, pd.DataFrame, dict[str, object]]:
    root = resolve(path)
    features = root / "features.npy"
    proposals = root / "proposals.csv"
    metadata = root / "metadata.json"
    for required in (features, proposals, metadata):
        if not required.exists():
            raise FileNotFoundError(required)
    return features, pd.read_csv(proposals), json.loads(metadata.read_text(encoding="utf-8"))


def class_weights(rows: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = np.bincount(rows["label"].to_numpy(dtype=np.int64), minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required, got counts={counts.tolist()}")
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.inference_mode()
def evaluate_head(
    model: nn.Linear,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss_sum += float(criterion(logits, labels)) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += len(labels)
    return {"loss": loss_sum / max(count, 1), "accuracy": correct / max(count, 1)}


def train(args: argparse.Namespace) -> None:
    if args.epochs != 10:
        raise ValueError("The literature-backed primary protocol uses exactly 10 epochs")
    set_seed(args.seed)
    annotation_path = resolve(args.annotations)
    source_model = resolve(args.source_model)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_features, train_proposals, train_meta = load_feature_cache(args.train_features_dir)
    val_cache_dir = args.val_features_dir or args.train_features_dir
    val_features = val_proposals = val_meta = None
    if not args.train_all_validation:
        val_features, val_proposals, val_meta = load_feature_cache(val_cache_dir)
        if train_meta["source_model_sha256"] != val_meta["source_model_sha256"]:
            raise ValueError("Train and validation features use different ATSN encoders")
    if train_meta["source_model_sha256"] != sha256_file(source_model):
        raise ValueError("Feature cache and --source-model differ")

    train_labeled = label_proposals(
        train_proposals,
        annotation_path,
        "train",
        args.positive_tiou,
        cv_fold=args.cv_fold,
        all_validation=args.train_all_validation,
    )
    val_labeled = None
    if not args.train_all_validation:
        val_labeled = label_proposals(
            val_proposals,
            annotation_path,
            "val",
            args.positive_tiou,
            cv_fold=args.cv_fold,
        )
    train_rows = downsample_training_rows(
        train_labeled,
        args.negative_keep_fraction,
        args.seed,
    )
    val_rows = (
        val_labeled[val_labeled["label"] >= 0].reset_index(drop=True)
        if val_labeled is not None
        else None
    )
    if not len(train_rows) or (val_rows is not None and not len(val_rows)):
        raise ValueError("Empty train or validation proposal set")
    train_labeled.to_csv(out_dir / "train_labels_all.csv", index=False)
    train_rows.to_csv(out_dir / "train_labels_sampled.csv", index=False)
    if val_labeled is not None:
        val_labeled.to_csv(out_dir / "val_labels_all.csv", index=False)

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = nn.Linear(int(train_meta["feature_dim"]), 2).to(device)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    if args.head_init == "source":
        source = load_source_model(
            source_model,
            int(train_meta["num_tsn_samples"]),
            int(train_meta["augment_factor"]),
            torch.device("cpu"),
        )
        model.load_state_dict(source.fc_cls.state_dict())

    weights = class_weights(train_rows, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(
        MemmapFeatureDataset(train_features, train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = None
    if val_rows is not None:
        val_loader = DataLoader(
            MemmapFeatureDataset(val_features, val_rows),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
        val_metrics = (
            evaluate_head(model, val_loader, criterion, device)
            if val_loader is not None
            else {"loss": None, "accuracy": None}
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "train_accuracy": correct / max(count, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"[EPOCH {epoch:02d}] train_loss={row['train_loss']:.5f} "
            + (
                f"val_loss={row['val_loss']:.5f} val_acc={row['val_accuracy']:.4f}"
                if row["val_loss"] is not None
                else "final_fit_no_validation"
            )
        )

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    target_class = json.loads(annotation_path.read_text(encoding="utf-8")).get("target_class")
    saved_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint = {
        "protocol": "THUMOS14-E-frozen-ATSN-OVR-v2",
        "target_class": target_class,
        "cv_fold": args.cv_fold,
        "train_all_validation": args.train_all_validation,
        "model_state_dict": model.state_dict(),
        "feature_dim": int(train_meta["feature_dim"]),
        "source_model_sha256": train_meta["source_model_sha256"],
        "args": saved_args,
    }
    torch.save(checkpoint, out_dir / "final.pt")
    summary = {
        "protocol": checkpoint["protocol"],
        "target_class": target_class,
        "train_proposals_all": len(train_labeled),
        "train_ambiguous_ignored": int((train_labeled["label"] < 0).sum()),
        "train_positive": int((train_rows["label"] == 1).sum()),
        "train_negative_after_10x_downsampling": int((train_rows["label"] == 0).sum()),
        "val_positive": (
            int((val_rows["label"] == 1).sum()) if val_rows is not None else None
        ),
        "val_negative": (
            int((val_rows["label"] == 0).sum()) if val_rows is not None else None
        ),
        "final_epoch": history[-1],
        "checkpoint_sha256": sha256_file(out_dir / "final.pt"),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    print(f"[OK] checkpoint={out_dir / 'final.pt'}")


@torch.inference_mode()
def score(args: argparse.Namespace) -> None:
    feature_path, proposals, metadata = load_feature_cache(args.features_dir)
    checkpoint_path = resolve(args.checkpoint)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["source_model_sha256"] != metadata["source_model_sha256"]:
        raise ValueError("Checkpoint and features use different frozen encoders")
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = nn.Linear(int(checkpoint["feature_dim"]), 2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    features = np.load(feature_path, mmap_mode="r")
    probabilities = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), args.batch_size):
        values = torch.from_numpy(
            np.array(features[start : start + args.batch_size], dtype=np.float32, copy=True)
        ).to(device)
        probabilities[start : start + len(values)] = (
            torch.softmax(model(values), dim=1)[:, 1].cpu().numpy()
        )
    scored = proposals.copy()
    scored["cnn_score"] = probabilities
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "proposal_scores.csv", index=False)

    results: dict[str, dict[str, list[dict[str, object]]]] = {}
    kept_rows = []
    selected = scored[scored["cnn_score"] >= args.min_positive_score]
    for (recording, roi_id), group in selected.groupby(["rec_name", "roi_id"], sort=True):
        values = group[["t_start", "t_end", "cnn_score"]].to_numpy(dtype=np.float64)
        kept = temporal_nms(values, args.nms_threshold) if len(values) else values
        results.setdefault(str(recording), {})[proposal_roi_key(roi_id)] = [
            {
                "label": "ed",
                "source_label": checkpoint.get("target_class"),
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(value),
            }
            for start, end, value in kept
        ]
        kept_rows.extend(
            {
                "rec_name": recording,
                "roi_id": roi_id,
                "t_start": start,
                "t_end": end,
                "score": value,
            }
            for start, end, value in kept
        )
    pd.DataFrame(kept_rows).to_csv(out_dir / "detections.csv", index=False)
    prediction = {
        "version": "THUMOS14-E-reTAG-OVR-v1",
        "target_class": checkpoint.get("target_class"),
        "results": results,
    }
    atomic_write_json(out_dir / "predictions.json", prediction)
    atomic_write_json(
        out_dir / "score_summary.json",
        {
            "target_class": checkpoint.get("target_class"),
            "proposals": len(scored),
            "above_threshold": len(selected),
            "after_nms": len(kept_rows),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "feature_metadata": metadata,
        },
    )
    print(f"[OK] predictions={out_dir / 'predictions.json'} detections={len(kept_rows)}")


def main() -> None:
    args = parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
