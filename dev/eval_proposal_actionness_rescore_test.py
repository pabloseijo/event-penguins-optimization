"""Single frozen test evaluation of source-selected completeness rescoring."""

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

from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import (  # noqa: E402
    blended_score_frame,
    extract_actionness,
)
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/test_features_v1")
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--continuous-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/continuous_ensemble.json",
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
        "--out-dir",
        default="tmp/temporalmaxer_continuous/proposal_actionness_completeness_test_v1",
    )
    parser.add_argument("--actionness-weight", type=float, default=0.25)
    parser.add_argument("--context-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    feature_dir = resolve(args.feature_dir)
    continuous_root = resolve(args.continuous_root)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    device = torch.device(args.device)
    models = load_models(
        continuous_root,
        int(metadata["feature_dim"]),
        device,
    )
    actionness = extract_actionness(
        models,
        make_loader(feature_dir, sequences, args.batch_size, args.num_workers, device),
        device,
    )
    proposal = json.loads(resolve(args.proposal_prediction).read_text(encoding="utf-8"))
    proposal_frame = blended_score_frame(
        proposal,
        actionness,
        float(metadata["grid_stride_s"]),
        args.actionness_weight,
        "completeness",
        args.context_ratio,
    )
    continuous = json.loads(
        resolve(args.continuous_prediction).read_text(encoding="utf-8")
    )
    event = json.loads(resolve(args.event_prediction).read_text(encoding="utf-8"))
    prediction = build_prediction(
        [
            prediction_rows(continuous, "continuous"),
            prediction_rows(event, "event"),
            proposal_frame,
        ],
        {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction["version"] = "source-selected-ssn-completeness-rescore-test-v1"
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "predictions.json"
    recordings = sorted(sequences["rec_name"].unique().tolist())
    metrics = evaluate(
        prediction,
        recordings,
        resolve(args.ann_path),
        prediction_path,
    )
    per_recording = []
    for recording in recordings:
        per_recording.append(
            {
                "rec_name": recording,
                **evaluate(
                    prediction,
                    [recording],
                    resolve(args.ann_path),
                    out_dir / "per_recording" / f"{recording}.json",
                ),
            }
        )
    pd.DataFrame(per_recording).to_csv(out_dir / "per_recording_metrics.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "recipe.json").write_text(
        json.dumps(
            {
                "selection_split": "recording-disjoint source CV",
                "source_cv_mAP": 0.840723044794521,
                "source_cv_control_mAP": 0.8317680634809035,
                "score_mode": "inside actionness minus adjacent context actionness",
                "actionness_weight": args.actionness_weight,
                "context_ratio": args.context_ratio,
                "expert_weights": {
                    "continuous": 0.2,
                    "event": 0.4,
                    "proposal": 0.4,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(pd.DataFrame(per_recording).to_string(index=False))


if __name__ == "__main__":
    main()
