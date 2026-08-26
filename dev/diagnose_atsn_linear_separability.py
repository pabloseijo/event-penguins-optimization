"""Measure whether frozen ATSN features linearly separate ED from background.

This is a diagnostic, not a detector. Source metrics use recording-disjoint
folds. The target model is fitted once on every source recording and evaluated
per target recording without using target labels during fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def load_annotations(
    path: Path, min_duration_s: float
) -> dict[tuple[str, int], list[tuple[float, float]]]:
    database = json.loads(path.read_text(encoding="utf-8"))["database"]
    annotations: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for recording, recording_data in database.items():
        for roi_id, values in recording_data.get("annotations", {}).items():
            if roi_id == "null":
                continue
            annotations[(recording, int(roi_id))] = [
                (float(value["segment"][0]), float(value["segment"][1]))
                for value in values
                if value["label"] == "ed"
                and float(value["segment"][1]) - float(value["segment"][0])
                >= min_duration_s
            ]
    return annotations


def point_labels(
    length: int, stride_s: float, segments: list[tuple[float, float]]
) -> np.ndarray:
    centers = (np.arange(length, dtype=np.float64) + 0.5) * stride_s
    labels = np.zeros(length, dtype=np.uint8)
    for start, end in segments:
        labels[(centers >= start) & (centers <= end)] = 1
    return labels


def load_recordings(
    feature_dir: Path,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    stride_s = float(metadata["grid_stride_s"])
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    features = np.load(feature_dir / "frame_features.npy", mmap_mode="r")
    recordings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for recording, rows in sequences.groupby("rec_name", sort=True):
        feature_parts = []
        label_parts = []
        for row in rows.itertuples(index=False):
            offset = int(row.offset)
            length = int(row.length)
            feature_parts.append(np.asarray(features[offset : offset + length], dtype=np.float32))
            label_parts.append(
                point_labels(
                    length,
                    stride_s,
                    annotations.get((str(recording), int(row.roi_id)), []),
                )
            )
        recordings[str(recording)] = (
            np.concatenate(feature_parts),
            np.concatenate(label_parts),
        )
    return recordings


def balanced_training_sample(
    recordings: dict[str, tuple[np.ndarray, np.ndarray]],
    names: list[str],
    max_points_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_parts = []
    label_parts = []
    weight_parts = []
    for name in names:
        features, labels = recordings[name]
        present_labels = [label for label in (0, 1) if np.any(labels == label)]
        chosen_parts = []
        chosen_labels = []
        chosen_weights = []
        for label in present_labels:
            indices = np.flatnonzero(labels == label)
            if len(indices) > max_points_per_class:
                indices = rng.choice(indices, max_points_per_class, replace=False)
            chosen_parts.append(features[indices])
            chosen_labels.append(np.full(len(indices), label, dtype=np.uint8))
            # Every recording has equal total weight; present classes divide it equally.
            chosen_weights.append(
                np.full(
                    len(indices),
                    1.0 / (len(present_labels) * len(indices)),
                    dtype=np.float64,
                )
            )
        feature_parts.extend(chosen_parts)
        label_parts.extend(chosen_labels)
        weight_parts.extend(chosen_weights)
    return (
        np.concatenate(feature_parts),
        np.concatenate(label_parts),
        np.concatenate(weight_parts),
    )


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(np.float64)
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    positive_rank_sum = float(ranks[positive].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_positive = labels[order] == 1
    precision = np.cumsum(sorted_positive) / np.arange(1, len(labels) + 1)
    return float(precision[sorted_positive].sum() / positive_count)


def weighted_mean_std(
    features: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized_weights = weights / weights.sum()
    mean = np.sum(features * normalized_weights[:, None], axis=0)
    variance = np.sum(
        np.square(features - mean) * normalized_weights[:, None], axis=0
    )
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)


def fit_linear_probe(
    recordings: dict[str, tuple[np.ndarray, np.ndarray]],
    train_names: list[str],
    max_points_per_class: int,
    seed: int,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    features, labels, weights = balanced_training_sample(
        recordings, train_names, max_points_per_class, seed
    )
    mean, std = weighted_mean_std(features, weights)
    standardized = (features - mean) / std

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(labels), generator=generator).numpy()
    train_features = torch.from_numpy(standardized[permutation]).to(device)
    train_labels = torch.from_numpy(labels[permutation].astype(np.float32)).to(device)
    train_weights = torch.from_numpy(weights[permutation].astype(np.float32)).to(device)
    linear = torch.nn.Linear(features.shape[1], 1).to(device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = linear(train_features).squeeze(1)
        loss = (
            F.binary_cross_entropy_with_logits(logits, train_labels, reduction="none")
            * train_weights
        ).sum() / train_weights.sum()
        loss.backward()
        optimizer.step()
    linear_weight = linear.weight.detach().float().cpu().numpy().reshape(-1)
    linear_bias = float(linear.bias.detach().float().cpu())

    negative_center = np.average(
        standardized[labels == 0], axis=0, weights=weights[labels == 0]
    )
    positive_center = np.average(
        standardized[labels == 1], axis=0, weights=weights[labels == 1]
    )
    direction = positive_center - negative_center
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    midpoint = 0.5 * (positive_center + negative_center)
    return mean, std, linear_weight, linear_bias, direction, midpoint


def evaluate_recording(
    split: str,
    fold: int | str,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    linear_weight: np.ndarray,
    linear_bias: float,
    direction: np.ndarray,
    midpoint: np.ndarray,
) -> dict:
    standardized = (features - mean) / std
    logistic_score = standardized @ linear_weight + linear_bias
    centroid_score = (standardized - midpoint) @ direction
    return {
        "split": split,
        "fold": fold,
        "rec_name": name,
        "points": len(labels),
        "positive_points": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "linear_roc_auc": binary_roc_auc(labels, logistic_score),
        "linear_average_precision": binary_average_precision(labels, logistic_score),
        "centroid_roc_auc": binary_roc_auc(labels, centroid_score),
        "centroid_average_precision": binary_average_precision(labels, centroid_score),
    }


def summarize(rows: pd.DataFrame) -> dict:
    result = {}
    for split, values in rows.groupby("split", sort=False):
        result[split] = {
            "recordings": int(len(values)),
            "macro_linear_roc_auc": float(values["linear_roc_auc"].mean()),
            "worst_linear_roc_auc": float(values["linear_roc_auc"].min()),
            "macro_linear_average_precision": float(
                values["linear_average_precision"].mean()
            ),
            "macro_centroid_roc_auc": float(values["centroid_roc_auc"].mean()),
            "worst_centroid_roc_auc": float(values["centroid_roc_auc"].min()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1"
    )
    parser.add_argument(
        "--target-feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1"
    )
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-points-per-class", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/atsn_linear_separability_v1"
    )
    args = parser.parse_args()

    annotations = load_annotations(resolve(args.ann_path), args.min_duration)
    source = load_recordings(resolve(args.source_feature_dir), annotations)
    target = load_recordings(resolve(args.target_feature_dir), annotations)
    manifest = pd.read_csv(resolve(args.fold_manifest))
    device = torch.device(args.device)
    rows = []
    for fold_row in manifest.itertuples(index=False):
        validation_names = str(fold_row.val_record_names).split()
        train_names = sorted(set(source) - set(validation_names))
        probe = fit_linear_probe(
            source,
            train_names,
            args.max_points_per_class,
            args.seed + int(fold_row.fold),
            device,
            args.steps,
            args.learning_rate,
        )
        for name in validation_names:
            rows.append(
                evaluate_recording(
                    "source_oof",
                    int(fold_row.fold),
                    name,
                    *source[name],
                    *probe,
                )
            )

    probe = fit_linear_probe(
        source,
        sorted(source),
        args.max_points_per_class,
        args.seed,
        device,
        args.steps,
        args.learning_rate,
    )
    for name in sorted(target):
        rows.append(
            evaluate_recording(
                "target_diagnostic",
                "all_source",
                name,
                *target[name],
                *probe,
            )
        )

    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "recording_metrics.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
