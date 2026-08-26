"""Single frozen test evaluation of the source-selected linear QFL head."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    FEATURE_COLUMNS,
    candidate_features,
    fit_linear_qfl,
    score_quality_head,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_multi_rep_fusion_test import (  # noqa: E402
    select_inference_sequences,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import extract_actionness  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1")
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--source-features",
        default="tmp/temporalmaxer_continuous/actionness_qfl_cv_v1/candidate_features.csv",
    )
    parser.add_argument(
        "--target-features",
        default=None,
        help=(
            "Optional cached test candidate features when proposals and actionness "
            "are unchanged."
        ),
    )
    parser.add_argument(
        "--continuous-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/continuous_ensemble.json",
    )
    parser.add_argument(
        "--secondary-continuous-prediction",
        default=None,
        help="Optional second source-approved continuous ensemble.",
    )
    parser.add_argument(
        "--secondary-continuous-weight",
        type=float,
        default=0.0,
        help="Fusion weight for --secondary-continuous-prediction.",
    )
    parser.add_argument(
        "--event-prediction",
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
        "--out-dir", default="tmp/temporalmaxer_continuous/actionness_qfl_test_v1"
    )
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
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
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="Write frozen predictions without loading test annotations.",
    )
    parser.add_argument(
        "--training-scope",
        default="source OOF only",
        help="Provenance label for the OOF candidate features used to fit QFL.",
    )
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=(0.2, 0.4, 0.4),
        metavar=("CONTINUOUS", "EVENT", "PROPOSAL"),
        help="Frozen source-CV weights for the three experts.",
    )
    args = parser.parse_args()
    total_weight = sum(args.weights) + args.secondary_continuous_weight
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Fusion weights must sum to one, got {total_weight:g}")
    if (args.secondary_continuous_prediction is None) != (
        args.secondary_continuous_weight == 0.0
    ):
        raise ValueError(
            "A secondary continuous prediction and a non-zero weight must be provided together"
        )

    device = torch.device(args.device)
    source = pd.read_csv(resolve(args.source_features))
    model = fit_linear_qfl(source, device, args.steps, args.learning_rate)
    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    sequences = select_inference_sequences(
        sequences, args.recording_manifest, args.recording_subset
    )
    if args.target_features:
        target_features = pd.read_csv(resolve(args.target_features))
    else:
        models = load_models(
            resolve(args.continuous_root),
            int(metadata["feature_dim"]),
            device,
        )
        actionness = extract_actionness(
            models,
            make_loader(feature_dir, sequences, args.batch_size, args.num_workers, device),
            device,
        )
        proposal = json.loads(resolve(args.proposal_prediction).read_text(encoding="utf-8"))
        target_features = candidate_features(
            proposal,
            actionness,
            float(metadata["grid_stride_s"]),
            annotations={},
            fold=-1,
        )
    proposal_frame = score_quality_head(target_features, model, args.score_blend)
    continuous = json.loads(
        resolve(args.continuous_prediction).read_text(encoding="utf-8")
    )
    event = json.loads(resolve(args.event_prediction).read_text(encoding="utf-8"))
    frames = [
        prediction_rows(continuous, "continuous"),
        prediction_rows(event, "event"),
        proposal_frame,
    ]
    fusion_weights = {
        "continuous": args.weights[0],
        "event": args.weights[1],
        "proposal": args.weights[2],
    }
    if args.secondary_continuous_prediction is not None:
        secondary_continuous = json.loads(
            resolve(args.secondary_continuous_prediction).read_text(encoding="utf-8")
        )
        frames.insert(
            1,
            prediction_rows(secondary_continuous, "secondary_continuous"),
        )
        fusion_weights["secondary_continuous"] = args.secondary_continuous_weight
    prediction = build_prediction(
        frames,
        fusion_weights,
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
        min_action_duration=args.min_action_duration,
    )
    prediction["version"] = "cross-fit-actionness-qfl-test-v1"
    prediction["target_class"] = args.target_class
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_features.to_csv(out_dir / "candidate_features.csv", index=False)
    proposal_frame.to_csv(out_dir / "proposal_frame.csv", index=False)
    prediction_path = out_dir / "predictions.json"
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    mean, std, weight, bias = model
    (out_dir / "model.json").write_text(
        json.dumps(
            {
                "feature_columns": list(FEATURE_COLUMNS),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "weight": weight.tolist(),
                "bias": bias,
                "training": {
                    "split": args.training_scope,
                    "loss": "quality focal loss beta=2",
                    "steps": args.steps,
                    "learning_rate": args.learning_rate,
                    "score_blend": args.score_blend,
                    "minimum_action_duration_s": args.min_action_duration,
                    "tiou_thresholds": args.tiou,
                    "fusion_weights": {
                        "continuous": args.weights[0],
                        "secondary_continuous": args.secondary_continuous_weight,
                        "event": args.weights[1],
                        "proposal": args.weights[2],
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.prediction_only:
        print(json.dumps({"prediction": str(prediction_path)}, indent=2))
        return
    recordings = sorted(sequences["rec_name"].unique().tolist())
    metrics = evaluate(
        prediction,
        recordings,
        resolve(args.ann_path),
        prediction_path,
        args.tiou,
        args.min_action_duration,
    )
    per_recording = []
    for recording in recordings:
        subset = {
            "version": prediction["version"],
            "results": {recording: prediction["results"][recording]},
        }
        per_recording.append(
            {
                "rec_name": recording,
                **evaluate(
                    subset,
                    [recording],
                    resolve(args.ann_path),
                    out_dir / "per_recording" / f"{recording}.json",
                    args.tiou,
                    args.min_action_duration,
                ),
            }
        )
    pd.DataFrame(per_recording).to_csv(out_dir / "per_recording_metrics.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(pd.DataFrame(per_recording).to_string(index=False))


if __name__ == "__main__":
    main()
