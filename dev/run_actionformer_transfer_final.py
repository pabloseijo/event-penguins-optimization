#!/usr/bin/env python3
"""Fit the frozen OOF QFL recipe and emit target-free final predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from eval_actionformer_qfl_cv import (
    build_design,
    collect_training_data,
    fit_qfl,
    geometric_quality_scores,
    predict_qfl,
)
from eval_actionformer_transfer_cv import (
    fit_classwise_ecdf,
    load_all_folds,
    postprocess_video,
    prediction_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actionformer-root", type=Path, required=True)
    parser.add_argument("--prepared-oof-root", type=Path, required=True)
    parser.add_argument("--prepared-inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--topk-per-class-video", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234567891)
    return parser.parse_args()


def load_inference_videos(path: Path) -> List[Dict[str, np.ndarray]]:
    videos = []
    for candidate_path in sorted(path.glob("*.npz")):
        with np.load(candidate_path) as candidate:
            video = {name: candidate[name] for name in candidate.files}
        if "target_tiou" in video:
            raise ValueError(
                f"Inference candidate contains target_tiou: {candidate_path}"
            )
        videos.append(video)
    if not videos:
        raise ValueError(f"No inference candidates found in {path}")
    return videos


def create_empty_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite final transfer output: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    actionformer_root = args.actionformer_root.expanduser().resolve()
    sys.path.insert(0, str(actionformer_root))
    os.chdir(actionformer_root)
    from libs.utils import batched_nms
    from libs.utils.nms import seg_voting

    folds = load_all_folds(args.prepared_oof_root, args.folds)
    training_videos = [video for fold in folds for video in fold]
    inference_videos = load_inference_videos(
        args.prepared_inference_dir
    )
    num_classes = int(
        max(int(video["labels"].max()) for video in training_videos) + 1
    )
    raw_references = fit_classwise_ecdf(
        training_videos,
        [video["scores"] for video in training_videos],
        num_classes,
    )
    training_designs = build_design(
        training_videos, num_classes, raw_references
    )
    inference_designs = build_design(
        inference_videos, num_classes, raw_references
    )
    features, targets, weights = collect_training_data(
        training_videos,
        training_designs,
        args.topk_per_class_video,
    )
    model = fit_qfl(
        features,
        targets,
        weights,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=torch.device(args.device),
        seed=args.seed,
    )
    quality_logits = predict_qfl(inference_designs, model)
    scores = geometric_quality_scores(inference_videos, quality_logits)
    outputs = [
        postprocess_video(
            video,
            video_scores,
            batched_nms=batched_nms,
            segment_voting=seg_voting,
            nms_method="soft",
            sigma=0.5,
            voting_threshold=0.5,
        )
        for video, video_scores in zip(inference_videos, scores)
    ]
    prediction = prediction_dict(outputs)

    create_empty_output_dir(args.output_dir)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        video_id=np.asarray(prediction["video-id"]),
        t_start=prediction["t-start"],
        t_end=prediction["t-end"],
        label=prediction["label"],
        score=prediction["score"],
    )
    model_artifact = {
        key: np.asarray(value) for key, value in model.items()
    }
    for label, reference in enumerate(raw_references):
        model_artifact[f"raw_score_ecdf_class_{label}"] = reference
    np.savez_compressed(
        args.output_dir / "qfl_model.npz", **model_artifact
    )
    report = {
        "recipe": "qfl_geometric_voting0.5",
        "training_source": "five OOF validation folds",
        "training_videos": len(training_videos),
        "inference_videos": len(inference_videos),
        "predictions": len(prediction["score"]),
        "targets_read_from_inference": False,
        "qfl_steps": args.steps,
        "qfl_batch_size": args.batch_size,
        "qfl_topk_per_class_video": args.topk_per_class_video,
        "qfl_learning_rate": args.learning_rate,
        "soft_nms_sigma": 0.5,
        "classwise_voting_threshold": 0.5,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
