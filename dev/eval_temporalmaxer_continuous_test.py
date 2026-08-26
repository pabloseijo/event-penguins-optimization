"""Single fixed test evaluation of the source-approved continuous/proposal fusion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_proposal_fusion_cv import (
    build_fused_prediction,
    evaluate,
    prediction_rows,
)
from dev.train_temporalmaxer_continuous import ContinuousSequenceDataset, collate_sequences
from src.temporalmaxer_continuous import TemporalMaxerContinuous
from src.utils import temporal_soft_nms


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1")
    parser.add_argument("--feature-array-name", default="frame_features.npy")
    parser.add_argument("--sequences-path", default=None)
    parser.add_argument("--standardize-features", action="store_true")
    parser.add_argument(
        "--feature-normalization",
        choices=("none", "temporal-center", "temporal-zscore"),
        default="none",
    )
    parser.add_argument("--checkpoint-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Optional explicit checkpoints; otherwise use five CV best.pt files.",
    )
    parser.add_argument(
        "--proposal-prediction",
        default=(
            "tmp/temporalmaxer_dense/multi_expert_boundary_voting_test/"
            "predictions/multi_median_blend050.json"
        ),
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument("--out-dir", default="tmp/temporalmaxer_continuous/test_fusion_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--continuous-only",
        action="store_true",
        help="Write the continuous ensemble without fusing or evaluating it.",
    )
    parser.add_argument(
        "--temporal-reversal-tta",
        action="store_true",
        help="Average original and temporally reversed predictions after inverse alignment.",
    )
    return parser.parse_args()


def average_outputs(outputs: list[dict[str, list[torch.Tensor]]]) -> dict[str, list[torch.Tensor]]:
    averaged = {
        "classification_logits": [
            torch.stack([output["classification_logits"][level] for output in outputs]).mean(0)
            for level in range(len(outputs[0]["classification_logits"]))
        ],
        "quality_logits": [
            torch.stack([output["quality_logits"][level] for output in outputs]).mean(0)
            for level in range(len(outputs[0]["quality_logits"]))
        ],
        "offsets": [
            torch.stack([output["offsets"][level] for output in outputs]).mean(0)
            for level in range(len(outputs[0]["offsets"]))
        ],
        "masks": outputs[0]["masks"],
    }
    if outputs[0].get("start_boundary_logits"):
        averaged["start_boundary_logits"] = [
            torch.stack([output["start_boundary_logits"][level] for output in outputs]).mean(0)
            for level in range(len(outputs[0]["start_boundary_logits"]))
        ]
        averaged["end_boundary_logits"] = [
            torch.stack([output["end_boundary_logits"][level] for output in outputs]).mean(0)
            for level in range(len(outputs[0]["end_boundary_logits"]))
        ]
    return averaged


def reverse_valid_features(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse each unpadded sequence while keeping right padding in place."""
    reversed_features = features.clone()
    for index, length in enumerate(mask.sum(dim=1).tolist()):
        reversed_features[index, :length] = features[index, :length].flip(0)
    return reversed_features


def reverse_valid_level(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Map a reversed pyramid tensor back to the original temporal coordinates."""
    aligned = values.clone()
    for index, length in enumerate(mask.sum(dim=1).tolist()):
        aligned[index, ..., :length] = values[index, ..., :length].flip(-1)
    return aligned


def align_reversed_offsets(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse point order and swap left/right distances for [B,T,2] offsets."""
    aligned = values.clone()
    for index, length in enumerate(mask.sum(dim=1).tolist()):
        aligned[index, :length] = values[index, :length].flip(0)[:, [1, 0]]
    return aligned


def align_reversed_output(output: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
    """Undo temporal reversal and exchange the semantic start/end channels."""
    aligned = {
        "classification_logits": [
            reverse_valid_level(values, mask)
            for values, mask in zip(output["classification_logits"], output["masks"])
        ],
        "quality_logits": [
            reverse_valid_level(values, mask)
            for values, mask in zip(output["quality_logits"], output["masks"])
        ],
        "offsets": [
            align_reversed_offsets(values, mask)
            for values, mask in zip(output["offsets"], output["masks"])
        ],
        "masks": output["masks"],
    }
    if output.get("start_boundary_logits"):
        aligned["start_boundary_logits"] = [
            reverse_valid_level(values, mask)
            for values, mask in zip(output["end_boundary_logits"], output["masks"])
        ]
        aligned["end_boundary_logits"] = [
            reverse_valid_level(values, mask)
            for values, mask in zip(output["start_boundary_logits"], output["masks"])
        ]
    return aligned


def load_models(
    checkpoint_root: Path,
    input_dim: int,
    device: torch.device,
    checkpoint_paths: list[Path] | None = None,
) -> list[TemporalMaxerContinuous]:
    models = []
    paths = checkpoint_paths or [
        checkpoint_root / f"fold_{fold:02d}" / "best.pt" for fold in range(5)
    ]
    for checkpoint_path in paths:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        saved_args = checkpoint["args"]
        model = TemporalMaxerContinuous(
            input_dim=input_dim,
            hidden_dim=int(saved_args["hidden_dim"]),
            pyramid_levels=int(saved_args["pyramid_levels"]),
            head_layers=int(saved_args["head_layers"]),
            dropout=float(saved_args["dropout"]),
            use_quality=not bool(saved_args.get("disable_quality", False)),
            reg_max=int(saved_args.get("reg_max", 0)),
            trident_bins=int(saved_args.get("trident_bins", 0)),
            center_sampling_radius=float(saved_args.get("center_sampling_radius", 0.0)),
            use_boundary_heads=bool(saved_args.get("use_boundary_heads", False)),
            boundary_refine_radius_seconds=float(
                saved_args.get("boundary_refine_radius", 0.0)
                if saved_args.get("use_boundary_heads", False)
                else 0.0
            ),
            boundary_refine_blend=float(saved_args.get("boundary_refine_blend", 0.5)),
            use_temporal_order=bool(saved_args.get("temporal_order", False)),
            temporal_order_chunks=int(saved_args.get("temporal_order_chunks", 3)),
            classification_input_dim=(
                int(checkpoint["metadata"]["feature_dim"])
                if saved_args.get("cross_layer_task_decoupling", False)
                else None
            ),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        models.append(model.eval())
    return models


@torch.no_grad()
def continuous_prediction(
    models: list[TemporalMaxerContinuous],
    loader: DataLoader,
    sequences: pd.DataFrame,
    grid_stride_s: float,
    device: torch.device,
    temporal_reversal_tta: bool = False,
    min_action_duration: float = 2.0,
    target_class: str | None = None,
) -> dict:
    results = {
        recording: {str(int(roi)): [] for roi in group["roi_id"].unique()}
        for recording, group in sequences.groupby("rec_name")
    }
    decoder = models[0]
    for batch in tqdm(loader, desc="test-ensemble"):
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = [model(features, mask) for model in models]
            if temporal_reversal_tta:
                reversed_features = reverse_valid_features(features, mask)
                outputs.extend(
                    align_reversed_output(model(reversed_features, mask))
                    for model in models
                )
            output = average_outputs(outputs)
        candidates = decoder.decode(
            output,
            grid_stride_seconds=grid_stride_s,
            durations_seconds=batch["duration_s"],
            score_threshold=0.005,
            pre_nms_topk=200,
            quality_power=1.0,
            min_duration_seconds=min_action_duration,
        )
        for recording, roi_id, values in zip(
            batch["rec_name"], batch["roi_id"], candidates
        ):
            detections = temporal_soft_nms(
                values.float().cpu().numpy(), sigma=0.25, score_threshold=0.001
            )[:200]
            results[recording][str(int(roi_id))] = [
                {
                    "label": "ed",
                    "segment": [float(start), float(end)],
                    "score": float(score),
                }
                for start, end, score in detections
                if end - start >= min_action_duration
            ]
    return {
        "version": "temporalmaxer-continuous-cv-logit-ensemble-v1",
        "target_class": target_class,
        "minimum_action_duration_s": min_action_duration,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences_path = (
        resolve(args.sequences_path) if args.sequences_path else feature_dir / "sequences.csv"
    )
    sequences = pd.read_csv(sequences_path)
    feature_mean = None
    feature_std = None
    if args.standardize_features:
        feature_mean = np.asarray(metadata["mean"], dtype=np.float32)
        feature_std = np.asarray(metadata["std"], dtype=np.float32)
    dataset = ContinuousSequenceDataset(
        feature_dir / args.feature_array_name,
        sequences,
        annotations={},
        feature_normalization=args.feature_normalization,
        feature_channel_mean=feature_mean,
        feature_channel_std=feature_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )
    explicit_checkpoints = (
        [resolve(path) for path in args.checkpoints] if args.checkpoints else None
    )
    models = load_models(
        resolve(args.checkpoint_root),
        int(metadata["feature_dim"]),
        device,
        explicit_checkpoints,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    continuous_path = out_dir / "continuous_ensemble.json"
    if continuous_path.exists():
        continuous = json.loads(continuous_path.read_text(encoding="utf-8"))
    else:
        continuous = continuous_prediction(
            models,
            loader,
            sequences,
            float(metadata["grid_stride_s"]),
            device,
            args.temporal_reversal_tta,
            args.min_action_duration,
            args.target_class,
        )
        continuous_path.write_text(json.dumps(continuous), encoding="utf-8")
    if args.continuous_only:
        print(json.dumps({"continuous_prediction": str(continuous_path)}, indent=2))
        return
    proposal = json.loads(resolve(args.proposal_prediction).read_text(encoding="utf-8"))
    fused = build_fused_prediction(
        prediction_rows(continuous, "continuous"),
        prediction_rows(proposal, "proposal"),
        continuous_weight=0.5,
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    fused["version"] = "source-approved-continuous-proposal-fusion-v1"
    metrics = evaluate(
        fused,
        sorted(sequences["rec_name"].unique().tolist()),
        resolve(args.ann_path),
        out_dir / "predictions.json",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
