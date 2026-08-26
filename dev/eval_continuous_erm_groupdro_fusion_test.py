"""Single test of the source-selected ERM/GroupDRO four-expert fusion."""

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

from dev.eval_continuous_multi_rep_fusion_cv import (
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_multi_rep_fusion_test import (
    auxiliary_normalization,
    cached_ensemble,
    checkpoint_paths,
    make_loader,
)
from dev.eval_temporalmaxer_continuous_test import load_models


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1")
    parser.add_argument(
        "--auxiliary-feature-dir", default="tmp/temporalmaxer_continuous/test_event_stats_v1"
    )
    parser.add_argument("--erm-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--groupdro-root", default="tmp/temporalmaxer_continuous/cv_groupdro_continuous_v1"
    )
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--proposal-prediction",
        default=(
            "tmp/temporalmaxer_dense/multi_expert_boundary_voting_test/"
            "predictions/multi_median_blend050.json"
        ),
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/erm_groupdro_fusion_test_v1"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    erm_paths = checkpoint_paths(resolve(args.erm_root))
    groupdro_paths = checkpoint_paths(resolve(args.groupdro_root))
    event_paths = checkpoint_paths(resolve(args.event_root))
    auxiliary_mean, auxiliary_std, auxiliary_dim = auxiliary_normalization(event_paths)
    base_loader = make_loader(feature_dir, sequences, args)
    event_loader = make_loader(
        feature_dir,
        sequences,
        args,
        resolve(args.auxiliary_feature_dir) / "event_stats.npy",
        auxiliary_mean,
        auxiliary_std,
    )
    erm_models = load_models(
        resolve(args.erm_root), int(metadata["feature_dim"]), device, erm_paths
    )
    groupdro_models = load_models(
        resolve(args.groupdro_root),
        int(metadata["feature_dim"]),
        device,
        groupdro_paths,
    )
    event_models = load_models(
        resolve(args.event_root),
        int(metadata["feature_dim"]) + auxiliary_dim,
        device,
        event_paths,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_stride_s = float(metadata["grid_stride_s"])
    erm = cached_ensemble(
        out_dir / "erm_ensemble.json",
        erm_models,
        base_loader,
        sequences,
        grid_stride_s,
        device,
    )
    groupdro = cached_ensemble(
        out_dir / "groupdro_ensemble.json",
        groupdro_models,
        base_loader,
        sequences,
        grid_stride_s,
        device,
    )
    event = cached_ensemble(
        out_dir / "event_ensemble.json",
        event_models,
        event_loader,
        sequences,
        grid_stride_s,
        device,
    )
    proposal = json.loads(resolve(args.proposal_prediction).read_text(encoding="utf-8"))
    weights = {"erm": 0.05, "groupdro": 0.15, "event": 0.4, "proposal": 0.4}
    prediction = build_prediction(
        [
            prediction_rows(erm, "erm"),
            prediction_rows(groupdro, "groupdro"),
            prediction_rows(event, "event"),
            prediction_rows(proposal, "proposal"),
        ],
        weights,
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction["version"] = "source-selected-erm-groupdro-four-expert-fusion-v1"
    metrics = evaluate(
        prediction,
        sorted(sequences["rec_name"].unique()),
        resolve(args.ann_path),
        out_dir / "predictions.json",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "recipe.json").write_text(
        json.dumps(
            {
                "source_cv": "erm_groupdro_fusion_cv_v1",
                "weights": weights,
                "nms_sigma": 0.5,
                "per_model_topk": 100,
                "max_predictions": 200,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
