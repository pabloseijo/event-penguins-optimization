"""Single fixed test evaluation of the source-selected three-expert fusion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import (
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_temporalmaxer_continuous_test import continuous_prediction, load_models
from dev.train_temporalmaxer_continuous import ContinuousSequenceDataset, collate_sequences


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1")
    parser.add_argument(
        "--auxiliary-feature-dir",
        default="tmp/temporalmaxer_continuous/test_event_stats_v1",
    )
    parser.add_argument(
        "--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1"
    )
    parser.add_argument(
        "--feature-normalization",
        choices=("none", "temporal-center", "temporal-zscore"),
        default="none",
    )
    parser.add_argument(
        "--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1"
    )
    parser.add_argument(
        "--proposal-prediction",
        default=(
            "tmp/temporalmaxer_dense/multi_expert_boundary_voting_test/"
            "predictions/multi_median_blend050.json"
        ),
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-class", default=None)
    parser.add_argument(
        "--recording-manifest",
        default=None,
        help="Optional CSV defining the exact inference recording universe.",
    )
    parser.add_argument("--recording-subset", default=None)
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Generate the two ensemble caches without evaluating a fused test prediction.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=(0.2, 0.4, 0.4),
        metavar=("CONTINUOUS", "EVENT", "PROPOSAL"),
        help="Frozen fusion recipe is (0.2, 0.4, 0.4); override only for a gated re-evaluation.",
    )
    return parser.parse_args()


def checkpoint_paths(root: Path) -> list[Path]:
    return [root / f"fold_{fold:02d}" / "best.pt" for fold in range(5)]


def select_inference_sequences(
    sequences: pd.DataFrame,
    manifest_path: str | None,
    subset: str | None,
) -> pd.DataFrame:
    if manifest_path is None:
        if subset is not None:
            raise ValueError("--recording-subset requires --recording-manifest")
        return sequences.reset_index(drop=True)
    manifest = pd.read_csv(resolve(manifest_path), keep_default_na=False)
    name_column = "video_id" if "video_id" in manifest else "rec_name"
    if name_column not in manifest:
        raise ValueError("Recording manifest requires video_id or rec_name")
    if subset is not None:
        if "official_subset" not in manifest:
            raise ValueError("Recording subset selection requires official_subset")
        manifest = manifest.loc[
            manifest["official_subset"].astype(str).str.lower() == subset.lower()
        ]
    selected = set(manifest[name_column].astype(str))
    available = set(sequences["rec_name"].astype(str))
    missing = selected - available
    if missing:
        raise ValueError(f"Inference recordings missing from feature cache: {sorted(missing)}")
    filtered = sequences.loc[sequences["rec_name"].astype(str).isin(selected)].copy()
    actual = set(filtered["rec_name"].astype(str))
    if actual != selected:
        raise ValueError("Inference sequence universe differs from the recording manifest")
    return filtered.reset_index(drop=True)


def auxiliary_normalization(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    means = []
    stds = []
    dimensions = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        means.append(np.asarray(metadata["auxiliary_mean"], dtype=np.float32))
        stds.append(np.asarray(metadata["auxiliary_std"], dtype=np.float32))
        dimensions.append(int(metadata["auxiliary_feature_dim"]))
    if len(set(dimensions)) != 1:
        raise ValueError("Event checkpoints disagree on auxiliary feature dimensionality")
    if not all(np.allclose(means[0], value) for value in means[1:]):
        raise ValueError("Event checkpoints disagree on source auxiliary means")
    if not all(np.allclose(stds[0], value) for value in stds[1:]):
        raise ValueError("Event checkpoints disagree on source auxiliary standard deviations")
    return means[0], stds[0], dimensions[0]


def make_loader(
    feature_dir: Path,
    sequences: pd.DataFrame,
    args: argparse.Namespace,
    auxiliary_path: Path | None = None,
    auxiliary_mean: np.ndarray | None = None,
    auxiliary_std: np.ndarray | None = None,
    feature_normalization: str = "none",
) -> DataLoader:
    dataset = ContinuousSequenceDataset(
        feature_dir / "frame_features.npy",
        sequences,
        annotations={},
        auxiliary_feature_path=auxiliary_path,
        auxiliary_mean=auxiliary_mean,
        auxiliary_std=auxiliary_std,
        feature_normalization=feature_normalization,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=collate_sequences,
    )


def cached_ensemble(
    path: Path,
    models,
    loader: DataLoader,
    sequences: pd.DataFrame,
    grid_stride_s: float,
    device: torch.device,
    min_action_duration: float = 2.0,
    target_class: str | None = None,
) -> dict:
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        expected = set(sequences["rec_name"].astype(str))
        if set(cached.get("results", {})) != expected:
            raise ValueError(f"Cached ensemble {path} has a foreign recording universe")
        if target_class is not None and cached.get("target_class") != target_class:
            raise ValueError(f"Cached ensemble {path} has a foreign target class")
        return cached
    prediction = continuous_prediction(
        models,
        loader,
        sequences,
        grid_stride_s,
        device,
        min_action_duration=min_action_duration,
        target_class=target_class,
    )
    path.write_text(json.dumps(prediction), encoding="utf-8")
    return prediction


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    sequences = select_inference_sequences(
        sequences, args.recording_manifest, args.recording_subset
    )
    continuous_paths = checkpoint_paths(resolve(args.continuous_root))
    event_paths = checkpoint_paths(resolve(args.event_root))
    auxiliary_mean, auxiliary_std, auxiliary_dim = auxiliary_normalization(event_paths)
    auxiliary_path = resolve(args.auxiliary_feature_dir) / "event_stats.npy"
    expected_points = int(metadata["num_points"])
    auxiliary_rows = int(np.load(auxiliary_path, mmap_mode="r").shape[0])
    if auxiliary_rows != expected_points:
        raise ValueError("Test base and auxiliary feature caches are not aligned")

    continuous_loader = make_loader(
        feature_dir,
        sequences,
        args,
        feature_normalization=args.feature_normalization,
    )
    event_loader = make_loader(
        feature_dir,
        sequences,
        args,
        auxiliary_path,
        auxiliary_mean,
        auxiliary_std,
        args.feature_normalization,
    )
    continuous_models = load_models(
        resolve(args.continuous_root),
        int(metadata["feature_dim"]),
        device,
        continuous_paths,
    )
    event_models = load_models(
        resolve(args.event_root),
        int(metadata["feature_dim"]) + auxiliary_dim,
        device,
        event_paths,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    continuous = cached_ensemble(
        out_dir / "continuous_ensemble.json",
        continuous_models,
        continuous_loader,
        sequences,
        float(metadata["grid_stride_s"]),
        device,
        args.min_action_duration,
        args.target_class,
    )
    event = cached_ensemble(
        out_dir / "event_ensemble.json",
        event_models,
        event_loader,
        sequences,
        float(metadata["grid_stride_s"]),
        device,
        args.min_action_duration,
        args.target_class,
    )
    if args.ensemble_only:
        print(
            json.dumps(
                {
                    "continuous_ensemble": str(out_dir / "continuous_ensemble.json"),
                    "event_ensemble": str(out_dir / "event_ensemble.json"),
                },
                indent=2,
            )
        )
        return
    proposal = json.loads(resolve(args.proposal_prediction).read_text(encoding="utf-8"))
    fused = build_prediction(
        [
            prediction_rows(continuous, "continuous"),
            prediction_rows(event, "event"),
            prediction_rows(proposal, "proposal"),
        ],
        {
            "continuous": args.weights[0],
            "event": args.weights[1],
            "proposal": args.weights[2],
        },
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
        min_action_duration=args.min_action_duration,
    )
    fused["version"] = "source-approved-continuous-multi-representation-fusion-v1"
    metrics = evaluate(
        fused,
        sorted(sequences["rec_name"].unique().tolist()),
        resolve(args.ann_path),
        out_dir / "predictions.json",
        args.tiou,
        args.min_action_duration,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "recipe.json").write_text(
        json.dumps(
            {
                "source_cv": "multi_rep_fusion_cv_v1",
                "weights": {
                    "continuous": args.weights[0],
                    "event": args.weights[1],
                    "proposal": args.weights[2],
                },
                "nms_sigma": 0.5,
                "minimum_action_duration_s": args.min_action_duration,
                "tiou_thresholds": args.tiou,
                "normalization": "source checkpoint mean/std",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
