"""Audit pointwise action/background ranking inside continuous TemporalMaxer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.diagnose_atsn_linear_separability import (  # noqa: E402
    binary_average_precision,
    binary_roc_auc,
    load_annotations,
    point_labels,
)
from dev.eval_temporalmaxer_continuous_test import (  # noqa: E402
    average_outputs,
    load_models,
)
from dev.train_temporalmaxer_continuous import (  # noqa: E402
    ContinuousSequenceDataset,
    collate_sequences,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def pyramid_point_score(
    output: dict[str, list[torch.Tensor]], include_quality: bool
) -> torch.Tensor:
    base_length = output["classification_logits"][0].shape[1]
    levels = []
    for class_logits, quality_logits in zip(
        output["classification_logits"], output["quality_logits"]
    ):
        score = class_logits.sigmoid()
        if include_quality:
            score = score * quality_logits.sigmoid()
        if score.shape[1] != base_length:
            score = F.interpolate(
                score[:, None], size=base_length, mode="linear", align_corners=False
            ).squeeze(1)
        levels.append(score)
    return torch.stack(levels).amax(0)


@torch.no_grad()
def score_sequences(
    models,
    feature_dir: Path,
    sequences: pd.DataFrame,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    stride_s = float(metadata["grid_stride_s"])
    dataset = ContinuousSequenceDataset(
        feature_dir / "frame_features.npy",
        sequences.reset_index(drop=True),
        annotations={},
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )
    parts: dict[str, dict[str, list[np.ndarray]]] = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = average_outputs([model(features, mask) for model in models])
            class_score = output["classification_logits"][0].sigmoid()
            quality_score = output["quality_logits"][0].sigmoid()
            combined_score = class_score * quality_score
            pyramid_classification_score = pyramid_point_score(
                output, include_quality=False
            )
            pyramid_score = pyramid_point_score(output, include_quality=True)
        score_arrays = {
            "classification": class_score.float().cpu().numpy(),
            "quality": quality_score.float().cpu().numpy(),
            "combined": combined_score.float().cpu().numpy(),
            "pyramid_classification": pyramid_classification_score.float().cpu().numpy(),
            "pyramid": pyramid_score.float().cpu().numpy(),
        }
        for index, (recording, roi_id) in enumerate(
            zip(batch["rec_name"], batch["roi_id"])
        ):
            valid_length = int(batch["mask"][index].sum())
            labels = point_labels(
                valid_length,
                stride_s,
                annotations.get((str(recording), int(roi_id)), []),
            )
            storage = parts.setdefault(
                str(recording),
                {"labels": [], **{name: [] for name in score_arrays}},
            )
            storage["labels"].append(labels)
            for name, values in score_arrays.items():
                storage[name].append(values[index, :valid_length])

    rows = []
    for recording, values in sorted(parts.items()):
        labels = np.concatenate(values["labels"])
        row = {
            "rec_name": recording,
            "points": len(labels),
            "positive_points": int(labels.sum()),
            "positive_fraction": float(labels.mean()),
        }
        for name in (
            "classification",
            "quality",
            "combined",
            "pyramid_classification",
            "pyramid",
        ):
            score = np.concatenate(values[name])
            row[f"{name}_roc_auc"] = binary_roc_auc(labels, score)
            row[f"{name}_average_precision"] = binary_average_precision(labels, score)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1"
    )
    parser.add_argument(
        "--target-feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1"
    )
    parser.add_argument("--checkpoint-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/point_score_diagnostic_v1"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    source_dir = resolve(args.source_feature_dir)
    target_dir = resolve(args.target_feature_dir)
    checkpoint_root = resolve(args.checkpoint_root)
    annotations = load_annotations(resolve(args.ann_path), args.min_duration)
    manifest = pd.read_csv(resolve(args.fold_manifest))
    source_sequences = pd.read_csv(source_dir / "sequences.csv")
    target_sequences = pd.read_csv(target_dir / "sequences.csv")
    input_dim = int(json.loads((source_dir / "metadata.json").read_text())["feature_dim"])

    source_parts = []
    for fold_row in manifest.itertuples(index=False):
        names = str(fold_row.val_record_names).split()
        selected = source_sequences[source_sequences["rec_name"].isin(names)].copy()
        models = load_models(
            checkpoint_root,
            input_dim,
            device,
            [checkpoint_root / f"fold_{int(fold_row.fold):02d}" / "best.pt"],
        )
        scored = score_sequences(
            models,
            source_dir,
            selected,
            annotations,
            device,
            args.batch_size,
            args.num_workers,
        )
        scored.insert(0, "fold", int(fold_row.fold))
        scored.insert(0, "split", "source_oof")
        source_parts.append(scored)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    models = load_models(checkpoint_root, input_dim, device)
    target = score_sequences(
        models,
        target_dir,
        target_sequences,
        annotations,
        device,
        args.batch_size,
        args.num_workers,
    )
    target.insert(0, "fold", "all_source")
    target.insert(0, "split", "target_diagnostic")
    frame = pd.concat([*source_parts, target], ignore_index=True)

    summary = {}
    for split, values in frame.groupby("split", sort=False):
        summary[split] = {"recordings": int(len(values))}
        for name in (
            "classification",
            "quality",
            "combined",
            "pyramid_classification",
            "pyramid",
        ):
            summary[split][f"macro_{name}_roc_auc"] = float(
                values[f"{name}_roc_auc"].mean()
            )
            summary[split][f"worst_{name}_roc_auc"] = float(
                values[f"{name}_roc_auc"].min()
            )
            summary[split][f"macro_{name}_average_precision"] = float(
                values[f"{name}_average_precision"].mean()
            )

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "recording_metrics.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
