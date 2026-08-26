"""Cross-fit a boundary-sensitive ordered-actionness quality head."""

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

from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    FEATURE_COLUMNS,
    fit_linear_qfl,
    score_quality_head,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import extract_actionness  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402


PROFILE_COLUMNS = tuple(f"profile_{index:02d}" for index in range(32))
SHAPE_COLUMNS = tuple(f"shape_{index:02d}" for index in range(32))
PROFILE_FEATURE_COLUMNS = FEATURE_COLUMNS + PROFILE_COLUMNS
SHAPE_FEATURE_COLUMNS = FEATURE_COLUMNS + SHAPE_COLUMNS


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features",
        default="tmp/temporalmaxer_continuous/actionness_qfl_cv_v1/candidate_features.csv",
    )
    parser.add_argument(
        "--feature-dir",
        default="tmp/temporalmaxer_continuous/source_features_v1",
    )
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/actionness_profile_qfl_cv_v1",
    )
    parser.add_argument("--context-ratio", type=float, default=0.5)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument(
        "--profile-blends",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "--shape-blends",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def region_centers(start: float, end: float, count: int) -> np.ndarray:
    width = end - start
    return start + (np.arange(count, dtype=np.float64) + 0.5) * width / count


def sample_actionness_profile(
    sequence: np.ndarray,
    t_start: float,
    t_end: float,
    stride_s: float,
    context_ratio: float,
) -> np.ndarray:
    duration = max(t_end - t_start, stride_s)
    context = context_ratio * duration
    positions = np.concatenate(
        (
            region_centers(t_start - context, t_start, 8),
            region_centers(t_start, t_end, 16),
            region_centers(t_end, t_end + context, 8),
        )
    )
    sample_indices = positions / stride_s - 0.5
    return np.interp(
        sample_indices,
        np.arange(len(sequence), dtype=np.float64),
        sequence,
        left=float(sequence[0]),
        right=float(sequence[-1]),
    ).astype(np.float32)


def add_profiles(
    frame: pd.DataFrame,
    actionness: dict[tuple[str, int], np.ndarray],
    stride_s: float,
    context_ratio: float,
) -> pd.DataFrame:
    profiles = np.stack(
        [
            sample_actionness_profile(
                actionness[(str(row.rec_name), int(row.roi_id))],
                float(row.t_start),
                float(row.t_end),
                stride_s,
                context_ratio,
            )
            for row in frame.itertuples(index=False)
        ]
    )
    output = frame.copy()
    output[list(PROFILE_COLUMNS)] = profiles
    return output


def add_shape_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    profiles = frame[list(PROFILE_COLUMNS)].to_numpy(np.float32)
    context = np.concatenate((profiles[:, :8], profiles[:, 24:]), axis=1)
    centered = profiles - context.mean(axis=1, keepdims=True)
    scale = np.maximum(centered.std(axis=1, keepdims=True), 1e-3)
    output = frame.copy()
    output[list(SHAPE_COLUMNS)] = np.clip(centered / scale, -5.0, 5.0)
    return output


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_cache = out_dir / "candidate_profiles.csv"
    feature_dir = resolve(args.feature_dir)
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    candidates = pd.read_csv(resolve(args.candidate_features))
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    input_dim = int(metadata["feature_dim"])
    stride_s = float(metadata["grid_stride_s"])
    device = torch.device(args.device)

    if profile_cache.exists():
        profiled = pd.read_csv(profile_cache)
    else:
        parts = []
        for fold in range(5):
            recordings = str(manifest.loc[fold, "val_record_names"]).split()
            selected_sequences = sequences[
                sequences["rec_name"].isin(recordings)
            ].copy()
            models = load_models(
                continuous_root,
                input_dim,
                device,
                [continuous_root / f"fold_{fold:02d}" / "best.pt"],
            )
            actionness = extract_actionness(
                models,
                make_loader(
                    feature_dir,
                    selected_sequences,
                    args.batch_size,
                    args.num_workers,
                    device,
                ),
                device,
            )
            parts.append(
                add_profiles(
                    candidates[candidates["fold"] == fold].copy(),
                    actionness,
                    stride_s,
                    args.context_ratio,
                )
            )
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
        profiled = pd.concat(parts, ignore_index=True)
        profiled.to_csv(profile_cache, index=False)
    profiled = add_shape_profiles(profiled)

    rows = []
    variants = {
        "summary_qfl": FEATURE_COLUMNS,
        "profile_qfl": PROFILE_FEATURE_COLUMNS,
        "shape_qfl": SHAPE_FEATURE_COLUMNS,
    }
    for fold in range(5):
        train = profiled[profiled["fold"] != fold].copy()
        validation = profiled[profiled["fold"] == fold].copy()
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        continuous_frame = prediction_rows(
            best_prediction(continuous_root, fold),
            "continuous",
        )
        event_frame = prediction_rows(best_prediction(event_root, fold), "event")
        proposal_frames = {}
        for variant, feature_columns in variants.items():
            model = fit_linear_qfl(
                train,
                device,
                args.steps,
                args.learning_rate,
                feature_columns=feature_columns,
            )
            proposal_frame = score_quality_head(
                validation,
                model,
                args.score_blend,
                feature_columns=feature_columns,
            )
            proposal_frames[variant] = proposal_frame
            prediction = build_prediction(
                [continuous_frame, event_frame, proposal_frame],
                {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_{variant}.json",
                    ),
                }
            )
        for profile_blend in args.profile_blends:
            summary_frame = proposal_frames["summary_qfl"]
            profile_frame = proposal_frames["profile_qfl"]
            proposal_frame = summary_frame.copy()
            proposal_frame["raw_score"] = (
                (1.0 - profile_blend)
                * summary_frame["raw_score"].to_numpy(np.float64)
                + profile_blend * profile_frame["raw_score"].to_numpy(np.float64)
            )
            proposal_frame["rank_score"] = proposal_frame["raw_score"].rank(
                method="average",
                pct=True,
            )
            variant = f"profile_blend_w{int(round(profile_blend * 100)):03d}"
            prediction = build_prediction(
                [continuous_frame, event_frame, proposal_frame],
                {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_{variant}.json",
                    ),
                }
            )
        for shape_blend in args.shape_blends:
            summary_frame = proposal_frames["summary_qfl"]
            shape_frame = proposal_frames["shape_qfl"]
            proposal_frame = summary_frame.copy()
            proposal_frame["raw_score"] = (
                (1.0 - shape_blend)
                * summary_frame["raw_score"].to_numpy(np.float64)
                + shape_blend * shape_frame["raw_score"].to_numpy(np.float64)
            )
            proposal_frame["rank_score"] = proposal_frame["raw_score"].rank(
                method="average",
                pct=True,
            )
            variant = f"shape_blend_w{int(round(shape_blend * 100)):03d}"
            prediction = build_prediction(
                [continuous_frame, event_frame, proposal_frame],
                {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_{variant}.json",
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
