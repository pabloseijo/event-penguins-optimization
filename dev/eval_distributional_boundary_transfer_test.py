"""Fixed test transfer of source-approved event-DFL boundaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import evaluate
from dev.eval_continuous_multi_rep_fusion_test import (
    auxiliary_normalization,
    cached_ensemble,
    checkpoint_paths,
    make_loader,
)
from dev.eval_distributional_boundary_transfer_cv import rows, transfer
from dev.eval_temporalmaxer_continuous_test import load_models


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
        "--checkpoint-root", default="tmp/temporalmaxer_continuous/cv_eventstats_dfl_v1"
    )
    parser.add_argument(
        "--seed-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/predictions.json",
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/dfl_boundary_transfer_test_v1"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    paths = checkpoint_paths(resolve(args.checkpoint_root))
    auxiliary_mean, auxiliary_std, auxiliary_dim = auxiliary_normalization(paths)
    loader = make_loader(
        feature_dir,
        sequences,
        args,
        resolve(args.auxiliary_feature_dir) / "event_stats.npy",
        auxiliary_mean,
        auxiliary_std,
    )
    models = load_models(
        resolve(args.checkpoint_root),
        int(metadata["feature_dim"]) + auxiliary_dim,
        device,
        paths,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voter_prediction = cached_ensemble(
        out_dir / "event_dfl_ensemble.json",
        models,
        loader,
        sequences,
        float(metadata["grid_stride_s"]),
        device,
    )
    seed = json.loads(resolve(args.seed_prediction).read_text(encoding="utf-8"))
    prediction = transfer(
        seed,
        rows(voter_prediction, "event_dfl"),
        tiou=0.5,
        blend=0.5,
        topk=20,
    )
    prediction["version"] = "source-approved-event-dfl-boundary-transfer-v1"
    metrics = evaluate(
        prediction,
        sorted(sequences["rec_name"].unique().tolist()),
        resolve(args.ann_path),
        out_dir / "predictions.json",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "recipe.json").write_text(
        json.dumps(
            {
                "source_cv": "dfl_boundary_transfer_cv_v1",
                "seed_prediction": str(args.seed_prediction),
                "voter": "event-DFL CV logit ensemble",
                "vote_tiou": 0.5,
                "vote_blend": 0.5,
                "vote_topk": 20,
                "scores_changed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
