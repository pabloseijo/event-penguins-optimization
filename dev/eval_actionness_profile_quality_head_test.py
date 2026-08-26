"""Frozen test evaluation of the source-selected ordered-actionness QFL blend."""

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

from dev.eval_actionness_profile_quality_head_cv import (  # noqa: E402
    PROFILE_FEATURE_COLUMNS,
    add_profiles,
)
from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    fit_linear_qfl,
    score_quality_head,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import extract_actionness  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-profiles",
        default="tmp/temporalmaxer_continuous/actionness_profile_qfl_cv_v1/candidate_profiles.csv",
    )
    parser.add_argument(
        "--source-summary",
        default="tmp/temporalmaxer_continuous/actionness_profile_qfl_cv_v1/summary.csv",
    )
    parser.add_argument(
        "--target-candidates",
        default="tmp/temporalmaxer_continuous/actionness_qfl_test_v1/candidate_features.csv",
    )
    parser.add_argument(
        "--summary-frame",
        default="tmp/temporalmaxer_continuous/actionness_qfl_test_v1/proposal_frame.csv",
    )
    parser.add_argument(
        "--feature-dir",
        default="tmp/temporalmaxer_continuous/test_features_v1",
    )
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--continuous-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/continuous_ensemble.json",
    )
    parser.add_argument(
        "--event-prediction",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_v1/event_ensemble.json",
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/actionness_profile_qfl_test_v1",
    )
    parser.add_argument("--profile-blend", type=float, default=0.5)
    parser.add_argument("--context-ratio", type=float, default=0.5)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def subset_prediction(prediction: dict, recording: str) -> dict:
    return {
        "version": prediction["version"],
        "results": {recording: prediction["results"][recording]},
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    source = pd.read_csv(resolve(args.source_profiles))
    profile_model = fit_linear_qfl(
        source,
        device,
        args.steps,
        args.learning_rate,
        feature_columns=PROFILE_FEATURE_COLUMNS,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_cache = out_dir / "candidate_profiles.csv"
    if profile_cache.exists():
        target = pd.read_csv(profile_cache)
    else:
        target_candidates = pd.read_csv(resolve(args.target_candidates))
        feature_dir = resolve(args.feature_dir)
        metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
        sequences = pd.read_csv(feature_dir / "sequences.csv")
        models = load_models(
            resolve(args.continuous_root),
            int(metadata["feature_dim"]),
            device,
        )
        actionness = extract_actionness(
            models,
            make_loader(
                feature_dir,
                sequences,
                args.batch_size,
                args.num_workers,
                device,
            ),
            device,
        )
        target = add_profiles(
            target_candidates,
            actionness,
            float(metadata["grid_stride_s"]),
            args.context_ratio,
        )
        target.to_csv(profile_cache, index=False)

    summary_frame = pd.read_csv(resolve(args.summary_frame))
    profile_frame = score_quality_head(
        target,
        profile_model,
        args.score_blend,
        feature_columns=PROFILE_FEATURE_COLUMNS,
    )
    blended_frame = summary_frame.copy()
    blended_frame["raw_score"] = (
        (1.0 - args.profile_blend) * summary_frame["raw_score"].to_numpy(np.float64)
        + args.profile_blend * profile_frame["raw_score"].to_numpy(np.float64)
    )
    blended_frame["rank_score"] = blended_frame["raw_score"].rank(
        method="average",
        pct=True,
    )

    continuous = json.loads(
        resolve(args.continuous_prediction).read_text(encoding="utf-8")
    )
    event = json.loads(resolve(args.event_prediction).read_text(encoding="utf-8"))
    continuous_frame = prediction_rows(continuous, "continuous")
    event_frame = prediction_rows(event, "event")
    control = build_prediction(
        [continuous_frame, event_frame, summary_frame],
        {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction = build_prediction(
        [continuous_frame, event_frame, blended_frame],
        {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    prediction["version"] = "source-selected-ordered-actionness-qfl-test-v1"
    recordings = sorted(continuous["results"])
    control_metrics = evaluate(
        control,
        recordings,
        resolve(args.ann_path),
        out_dir / "control_predictions.json",
    )
    metrics = evaluate(
        prediction,
        recordings,
        resolve(args.ann_path),
        out_dir / "predictions.json",
    )
    per_recording = []
    for recording in recordings:
        per_recording.append(
            {
                "rec_name": recording,
                **evaluate(
                    subset_prediction(prediction, recording),
                    [recording],
                    resolve(args.ann_path),
                    out_dir / "per_recording" / f"{recording}.json",
                ),
            }
        )
    pd.DataFrame(per_recording).to_csv(
        out_dir / "per_recording_metrics.csv",
        index=False,
    )
    source_summary = pd.read_csv(resolve(args.source_summary))
    source_variant = f"profile_blend_w{int(round(args.profile_blend * 100)):03d}"
    source_row = source_summary[source_summary["variant"] == source_variant].iloc[0]
    result = {
        "source_cv": {
            "variant": source_variant,
            "mean_mAP": float(source_row["mean_mAP"]),
            "weighted_mAP": float(source_row["weighted_mAP"]),
            "worst_mAP": float(source_row["worst_mAP"]),
            "mean_AP@0.7": float(source_row["mean_AP@0.7"]),
            "profile_blend": args.profile_blend,
        },
        "control_metrics": control_metrics,
        "metrics": metrics,
        "delta_mAP": metrics["mAP"] - control_metrics["mAP"],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(per_recording).to_string(index=False))


if __name__ == "__main__":
    main()
