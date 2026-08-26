"""Single test evaluation of the source-approved four-expert fusion."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1")
    parser.add_argument(
        "--event-v2-feature-dir",
        default="tmp/temporalmaxer_continuous/test_event_features_v2",
    )
    parser.add_argument("--event-v2-root", default="tmp/temporalmaxer_continuous/cv_eventv2_v1")
    parser.add_argument(
        "--continuous-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/continuous_ensemble.json",
    )
    parser.add_argument(
        "--event-v1-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/event_ensemble.json",
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
        "--out-dir", default="tmp/temporalmaxer_continuous/four_rep_fusion_test_v1"
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
    paths = checkpoint_paths(resolve(args.event_v2_root))
    mean, std, auxiliary_dim = auxiliary_normalization(paths)
    loader = make_loader(
        feature_dir,
        sequences,
        args,
        resolve(args.event_v2_feature_dir) / "event_stats.npy",
        mean,
        std,
    )
    models = load_models(
        resolve(args.event_v2_root),
        int(metadata["feature_dim"]) + auxiliary_dim,
        device,
        paths,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_v2 = cached_ensemble(
        out_dir / "event_v2_ensemble.json",
        models,
        loader,
        sequences,
        float(metadata["grid_stride_s"]),
        device,
    )
    continuous = json.loads(resolve(args.continuous_prediction).read_text(encoding="utf-8"))
    event_v1 = json.loads(resolve(args.event_v1_prediction).read_text(encoding="utf-8"))
    proposal = json.loads(resolve(args.proposal_prediction).read_text(encoding="utf-8"))
    prediction = build_prediction(
        [
            prediction_rows(continuous, "continuous"),
            prediction_rows(event_v1, "event_v1"),
            prediction_rows(event_v2, "event_v2"),
            prediction_rows(proposal, "proposal"),
        ],
        {"continuous": 0.2, "event_v1": 0.4, "event_v2": 0.4, "proposal": 0.4},
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction["version"] = "source-approved-continuous-four-representation-fusion-v1"
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
                "source_cv": "four_rep_fusion_cv_v2",
                "weights": {
                    "continuous": 0.2,
                    "event_v1": 0.4,
                    "event_v2": 0.4,
                    "proposal": 0.4,
                },
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
