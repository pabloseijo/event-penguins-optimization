#!/usr/bin/env python3
"""Cross-fit a conservative per-video/class negative MIL gate on OOF outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionformer-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--prediction-variant", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234567891)
    return parser.parse_args()


def load_prediction(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def load_video_targets(
    annotation_path: Path,
    split: str,
    num_classes: int,
) -> Dict[Tuple[str, int], int]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    targets = {}
    for video_id, video in data["database"].items():
        if str(video.get("subset", "")).lower() != split.lower():
            continue
        labels = {
            int(annotation["label_id"])
            for annotation in video.get("annotations", [])
        }
        for label in range(num_classes):
            targets[(video_id, label)] = int(label in labels)
    return targets


def top_mean(values: np.ndarray, count: int) -> float:
    if len(values) == 0:
        return 0.0
    selected = np.sort(values)[-count:]
    return float(selected.mean())


def build_bags(
    prediction: Mapping[str, np.ndarray],
    targets: Mapping[Tuple[str, int], int],
    num_classes: int,
) -> Tuple[List[Tuple[str, int]], np.ndarray, np.ndarray]:
    keys = sorted(targets)
    video_ids = np.asarray(prediction["video_id"]).astype(str)
    labels = prediction["label"].astype(np.int64)
    scores = prediction["score"].astype(np.float32)
    durations = (
        prediction["t_end"].astype(np.float32)
        - prediction["t_start"].astype(np.float32)
    )
    rows = []
    output_targets = []
    for video_id, label in keys:
        indices = np.flatnonzero(
            (video_ids == video_id) & (labels == label)
        )
        class_scores = scores[indices]
        class_durations = durations[indices]
        rows.append(
            [
                np.log1p(len(indices)),
                float(class_scores.max()) if len(indices) else 0.0,
                top_mean(class_scores, 3),
                top_mean(class_scores, 10),
                float(class_scores.sum()) if len(indices) else 0.0,
                float(class_durations.mean()) if len(indices) else 0.0,
                float(class_durations.max()) if len(indices) else 0.0,
            ]
            + np.eye(num_classes, dtype=np.float32)[label].tolist()
        )
        output_targets.append(targets[(video_id, label)])
    return (
        keys,
        np.asarray(rows, dtype=np.float32),
        np.asarray(output_targets, dtype=np.float32),
    )


def fit_logistic_gate(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> Dict[str, np.ndarray | float]:
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-4).astype(np.float32)
    inputs = torch.from_numpy((features - mean) / std).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    positives = max(float(targets.sum()), 1.0)
    negatives = max(float(len(targets) - targets.sum()), 1.0)
    sample_weights = np.where(
        targets > 0,
        negatives / positives,
        1.0,
    ).astype(np.float32)
    weight_tensor = torch.from_numpy(sample_weights).to(device)
    model = torch.nn.Linear(features.shape[1], 1).to(device)
    torch.manual_seed(seed)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(
            logits, target_tensor, reduction="none"
        )
        loss = (loss * weight_tensor).mean()
        loss.backward()
        optimizer.step()
    return {
        "mean": mean,
        "std": std,
        "weight": model.weight.detach().cpu().numpy().reshape(-1),
        "bias": float(model.bias.detach().cpu()),
    }


def predict_gate(
    features: np.ndarray,
    model: Mapping[str, np.ndarray | float],
) -> np.ndarray:
    logits = (
        ((features - np.asarray(model["mean"])) / np.asarray(model["std"]))
        @ np.asarray(model["weight"])
        + float(model["bias"])
    )
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def conservative_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    candidates: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> float:
    positive_probabilities = probabilities[targets > 0]
    if len(positive_probabilities) == 0:
        return 0.0
    minimum_positive = float(positive_probabilities.min())
    valid = [
        threshold
        for threshold in candidates
        if threshold < minimum_positive
    ]
    return max(valid, default=0.0)


def apply_gate(
    prediction: Mapping[str, np.ndarray],
    probabilities: Mapping[Tuple[str, int], float],
    threshold: float,
) -> Dict[str, np.ndarray]:
    video_ids = np.asarray(prediction["video_id"]).astype(str)
    labels = prediction["label"].astype(np.int64)
    keep = np.asarray(
        [
            probabilities[(video_id, int(label))] >= threshold
            for video_id, label in zip(video_ids, labels)
        ],
        dtype=bool,
    )
    return {
        key: np.asarray(value)[keep]
        for key, value in prediction.items()
    }


def evaluator_prediction(
    prediction: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    return {
        "video-id": np.asarray(prediction["video_id"]).astype(str).tolist(),
        "t-start": prediction["t_start"],
        "t-end": prediction["t_end"],
        "label": prediction["label"],
        "score": prediction["score"],
    }


def evaluate(
    prediction: Mapping[str, np.ndarray],
    annotation_path: Path,
    split: str,
    evaluator_class,
) -> Dict[str, float]:
    evaluator = evaluator_class(
        str(annotation_path),
        split,
        tiou_thresholds=np.linspace(0.3, 0.7, 5),
    )
    maps, average, _ = evaluator.evaluate(
        evaluator_prediction(prediction),
        verbose=False,
    )
    return {
        "mAP": float(average),
        "AP@0.3": float(maps[0]),
        "AP@0.4": float(maps[1]),
        "AP@0.5": float(maps[2]),
        "AP@0.6": float(maps[3]),
        "AP@0.7": float(maps[4]),
    }


def main() -> None:
    args = parse_args()
    actionformer_root = args.actionformer_root.expanduser().resolve()
    sys.path.insert(0, str(actionformer_root))
    os.chdir(actionformer_root)
    from libs.utils import ANETdetection

    predictions = []
    bag_data = []
    num_classes = 10
    for fold in range(args.folds):
        prediction = load_prediction(
            args.prediction_root
            / f"fold_{fold}_{args.prediction_variant}.npz"
        )
        annotation_path = (
            args.annotation_root
            / f"thumos14_10classes_transfer_fold{fold}.json"
        )
        targets = load_video_targets(
            annotation_path,
            f"transfer_val_{fold}",
            num_classes,
        )
        keys, features, labels = build_bags(
            prediction, targets, num_classes
        )
        predictions.append(prediction)
        bag_data.append((keys, features, labels))

    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        train_features = np.concatenate(
            [
                features
                for other_fold, (_, features, _) in enumerate(bag_data)
                if other_fold != fold
            ]
        )
        train_targets = np.concatenate(
            [
                targets
                for other_fold, (_, _, targets) in enumerate(bag_data)
                if other_fold != fold
            ]
        )
        model = fit_logistic_gate(
            train_features,
            train_targets,
            steps=args.steps,
            learning_rate=args.learning_rate,
            device=torch.device(args.device),
            seed=args.seed + fold,
        )
        train_probabilities = predict_gate(train_features, model)
        threshold = conservative_threshold(
            train_probabilities, train_targets
        )
        keys, validation_features, validation_targets = bag_data[fold]
        validation_probabilities = predict_gate(validation_features, model)
        probabilities_by_key = dict(zip(keys, validation_probabilities))
        gated = apply_gate(
            predictions[fold], probabilities_by_key, threshold
        )
        annotation_path = (
            args.annotation_root
            / f"thumos14_10classes_transfer_fold{fold}.json"
        )
        base_metrics = evaluate(
            predictions[fold],
            annotation_path,
            f"transfer_val_{fold}",
            ANETdetection,
        )
        gated_metrics = evaluate(
            gated,
            annotation_path,
            f"transfer_val_{fold}",
            ANETdetection,
        )
        np.savez_compressed(
            args.output_dir / f"fold_{fold}_gated.npz", **gated
        )
        np.savez_compressed(
            args.output_dir / f"fold_{fold}_model.npz",
            **{key: np.asarray(value) for key, value in model.items()},
        )
        suppressed_bags = sum(
            probability < threshold
            for probability in validation_probabilities
        )
        suppressed_positive_bags = sum(
            probability < threshold and target > 0
            for probability, target in zip(
                validation_probabilities, validation_targets
            )
        )
        rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "suppressed_bags": suppressed_bags,
                "suppressed_positive_bags": suppressed_positive_bags,
                "base_mAP": base_metrics["mAP"],
                "gated_mAP": gated_metrics["mAP"],
                "delta_mAP": gated_metrics["mAP"] - base_metrics["mAP"],
                "base_AP@0.7": base_metrics["AP@0.7"],
                "gated_AP@0.7": gated_metrics["AP@0.7"],
            }
        )
        print(json.dumps(rows[-1], indent=2))

    with (args.output_dir / "fold_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "base_mean_mAP": float(np.mean([row["base_mAP"] for row in rows])),
        "gated_mean_mAP": float(
            np.mean([row["gated_mAP"] for row in rows])
        ),
        "delta_mean_mAP": float(
            np.mean([row["delta_mAP"] for row in rows])
        ),
        "base_mean_AP@0.7": float(
            np.mean([row["base_AP@0.7"] for row in rows])
        ),
        "gated_mean_AP@0.7": float(
            np.mean([row["gated_AP@0.7"] for row in rows])
        ),
        "suppressed_positive_bags": int(
            sum(row["suppressed_positive_bags"] for row in rows)
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
