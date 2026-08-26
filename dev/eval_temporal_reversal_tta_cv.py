"""Evaluate temporally equivariant reversal TTA on source recording folds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_test import (  # noqa: E402
    auxiliary_normalization,
    checkpoint_paths,
    make_loader,
)
from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import (  # noqa: E402
    continuous_prediction,
    load_models,
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1")
    parser.add_argument("--auxiliary-feature-dir", default=None)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--feature-normalization",
        choices=("none", "temporal-center", "temporal-zscore"),
        default="none",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    manifest = pd.read_csv(resolve(args.fold_manifest)).set_index("fold")
    root = resolve(args.checkpoint_root)
    paths = checkpoint_paths(root)
    auxiliary_path = None
    auxiliary_mean = None
    auxiliary_std = None
    auxiliary_dim = 0
    if args.auxiliary_feature_dir:
        auxiliary_mean, auxiliary_std, auxiliary_dim = auxiliary_normalization(paths)
        auxiliary_path = resolve(args.auxiliary_feature_dir) / "event_stats.npy"
        auxiliary_rows = int(np.load(auxiliary_path, mmap_mode="r").shape[0])
        if auxiliary_rows != int(metadata["num_points"]):
            raise ValueError("Base and auxiliary feature caches are not aligned")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, checkpoint_path in enumerate(paths):
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        val_sequences = sequences[sequences["rec_name"].isin(recordings)].copy()
        loader = make_loader(
            feature_dir,
            val_sequences,
            args,
            auxiliary_path,
            auxiliary_mean,
            auxiliary_std,
            args.feature_normalization,
        )
        models = load_models(
            root,
            int(metadata["feature_dim"]) + auxiliary_dim,
            device,
            [checkpoint_path],
        )
        prediction = continuous_prediction(
            models,
            loader,
            val_sequences,
            float(metadata["grid_stride_s"]),
            device,
            temporal_reversal_tta=True,
        )
        rows.append(
            {
                "fold": fold,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **evaluate(
                    prediction,
                    recordings,
                    resolve(args.ann_path),
                    out_dir / "predictions" / f"fold_{fold:02d}.json",
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    instance_weights = metrics["val_ed_instances"].to_numpy(np.float64)
    summary = {
        "mean_mAP": float(metrics["mAP"].mean()),
        "weighted_mAP": float(np.average(metrics["mAP"], weights=instance_weights)),
        "worst_mAP": float(metrics["mAP"].min()),
        **{
            f"mean_{column}": float(metrics[column].mean())
            for column in ("AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7")
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
