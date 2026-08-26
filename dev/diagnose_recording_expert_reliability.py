"""Diagnose label-free recording signals for adaptive expert fusion."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.diagnose_atsn_linear_separability import load_annotations  # noqa: E402
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    prediction_rows,
    simplex_weights,
)
from src.evaluation import compute_average_precision_detection  # noqa: E402
from src.utils.detection import temporal_iou  # noqa: E402


MODELS = ("continuous", "event", "proposal")
THRESHOLDS = np.asarray([0.1, 0.3, 0.5, 0.7], dtype=np.float64)
CANONICAL_WEIGHTS = (0.2, 0.4, 0.4)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_cv_v1/predictions",
    )
    parser.add_argument(
        "--weight-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/recording_expert_reliability_v1",
    )
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--agreement-topk", type=int, default=25)
    return parser.parse_args()


def prediction_frame(prediction: dict, recording: str) -> pd.DataFrame:
    rows = []
    for roi_id, detections in prediction["results"].get(recording, {}).items():
        for detection in detections:
            if detection["label"] != "ed":
                continue
            rows.append(
                {
                    "video-id": f"{recording}_{roi_id}",
                    "t-start": float(detection["segment"][0]),
                    "t-end": float(detection["segment"][1]),
                    "score": float(detection["score"]),
                }
            )
    return pd.DataFrame(rows, columns=["video-id", "t-start", "t-end", "score"])


def ground_truth_frame(
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
    recording: str,
) -> pd.DataFrame:
    rows = [
        {
            "video-id": f"{recording}_{roi_id}",
            "t-start": start,
            "t-end": end,
        }
        for (rec_name, roi_id), segments in annotations.items()
        if rec_name == recording
        for start, end in segments
    ]
    return pd.DataFrame(rows, columns=["video-id", "t-start", "t-end"])


def evaluate_recording(
    prediction: dict,
    recording: str,
    ground_truth: pd.DataFrame,
) -> dict[str, float | int]:
    if ground_truth.empty:
        return {
            "gt_instances": 0,
            "mAP": np.nan,
            **{f"AP@{threshold:.1f}": np.nan for threshold in THRESHOLDS},
        }
    detections = prediction_frame(prediction, recording)
    ap = compute_average_precision_detection(
        ground_truth,
        detections,
        tiou_thresholds=THRESHOLDS,
    )
    return {
        "gt_instances": len(ground_truth),
        "mAP": float(ap.mean()),
        **{
            f"AP@{threshold:.1f}": float(value)
            for threshold, value in zip(THRESHOLDS, ap)
        },
    }


def model_features(frame: pd.DataFrame, model: str) -> dict[str, float | int]:
    local = frame[frame["model"] == model]
    if local.empty:
        return {
            f"{model}_count": 0,
            f"{model}_roi_coverage": 0,
            f"{model}_duration_median": 0.0,
            f"{model}_duration_iqr": 0.0,
            f"{model}_raw_q90": 0.0,
            f"{model}_raw_top10_mean": 0.0,
            f"{model}_rank_top10_mean": 0.0,
        }
    durations = (
        local["t_end"].to_numpy(np.float64)
        - local["t_start"].to_numpy(np.float64)
    )
    top = local.nlargest(min(10, len(local)), "rank_score")
    return {
        f"{model}_count": len(local),
        f"{model}_roi_coverage": int(local["roi_id"].nunique()),
        f"{model}_duration_median": float(np.median(durations)),
        f"{model}_duration_iqr": float(
            np.quantile(durations, 0.75) - np.quantile(durations, 0.25)
        ),
        f"{model}_raw_q90": float(local["raw_score"].quantile(0.90)),
        f"{model}_raw_top10_mean": float(top["raw_score"].mean()),
        f"{model}_rank_top10_mean": float(top["rank_score"].mean()),
    }


def directional_agreement(
    source: pd.DataFrame,
    target: pd.DataFrame,
    topk: int,
) -> tuple[float, float, float]:
    overlap_parts = []
    weight_parts = []
    for roi_id, source_roi in source.groupby("roi_id"):
        target_roi = target[target["roi_id"] == roi_id]
        if target_roi.empty:
            selected = source_roi.nlargest(min(topk, len(source_roi)), "rank_score")
            overlap_parts.append(np.zeros(len(selected), dtype=np.float64))
            weight_parts.append(selected["rank_score"].to_numpy(np.float64))
            continue
        selected_source = source_roi.nlargest(min(topk, len(source_roi)), "rank_score")
        selected_target = target_roi.nlargest(min(topk, len(target_roi)), "rank_score")
        target_starts = selected_target["t_start"].to_numpy(np.float64)
        target_ends = selected_target["t_end"].to_numpy(np.float64)
        maxima = [
            float(
                temporal_iou(
                    target_starts,
                    target_ends,
                    float(row.t_start),
                    float(row.t_end),
                ).max()
            )
            for row in selected_source.itertuples(index=False)
        ]
        overlap_parts.append(np.asarray(maxima, dtype=np.float64))
        weight_parts.append(selected_source["rank_score"].to_numpy(np.float64))
    if not overlap_parts:
        return 0.0, 0.0, 0.0
    overlaps = np.concatenate(overlap_parts)
    weights = np.concatenate(weight_parts)
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights)
    return (
        float(np.average(overlaps, weights=weights)),
        float(np.average(overlaps >= 0.5, weights=weights)),
        float(np.average(overlaps >= 0.7, weights=weights)),
    )


def pair_agreement(
    frame: pd.DataFrame,
    first: str,
    second: str,
    topk: int,
) -> dict[str, float]:
    first_frame = frame[frame["model"] == first]
    second_frame = frame[frame["model"] == second]
    forward = directional_agreement(first_frame, second_frame, topk)
    backward = directional_agreement(second_frame, first_frame, topk)
    values = tuple(0.5 * (a + b) for a, b in zip(forward, backward))
    prefix = f"agreement_{first}_{second}"
    return {
        f"{prefix}_mean": values[0],
        f"{prefix}_frac05": values[1],
        f"{prefix}_frac07": values[2],
    }


def recording_features(
    frames: list[pd.DataFrame],
    recording: str,
    topk: int,
) -> dict[str, float | int | str]:
    frame = pd.concat(
        [part[part["rec_name"] == recording] for part in frames],
        ignore_index=True,
    )
    row: dict[str, float | int | str] = {"rec_name": recording}
    for model in MODELS:
        row.update(model_features(frame, model))
    pairs = (
        ("continuous", "event"),
        ("continuous", "proposal"),
        ("event", "proposal"),
    )
    for first, second in pairs:
        row.update(pair_agreement(frame, first, second, topk))
    mean_values = [
        float(row[f"agreement_{first}_{second}_mean"])
        for first, second in pairs
    ]
    frac05_values = [
        float(row[f"agreement_{first}_{second}_frac05"])
        for first, second in pairs
    ]
    frac07_values = [
        float(row[f"agreement_{first}_{second}_frac07"])
        for first, second in pairs
    ]
    counts = np.asarray([float(row[f"{model}_count"]) for model in MODELS])
    row.update(
        {
            "agreement_mean": float(np.mean(mean_values)),
            "agreement_min": float(np.min(mean_values)),
            "agreement_frac05_mean": float(np.mean(frac05_values)),
            "agreement_frac07_mean": float(np.mean(frac07_values)),
            "expert_count_cv": float(
                counts.std() / counts.mean() if counts.mean() > 0.0 else 0.0
            ),
        }
    )
    return row


def weight_label(weights: tuple[float, float, float]) -> str:
    return f"c{weights[0]:g}_e{weights[1]:g}_p{weights[2]:g}"


def weight_prediction_path(
    root: Path,
    fold: int,
    weights: tuple[float, float, float],
) -> Path:
    return (
        root
        / "predictions"
        / (
            f"fold{fold:02d}_cw{weights[0]:g}_ew{weights[1]:g}"
            f"_pw{weights[2]:g}.json"
        )
    )


def load_or_build_weight_prediction(
    root: Path,
    fold: int,
    weights: tuple[float, float, float],
    frames: list[pd.DataFrame],
) -> dict:
    path = weight_prediction_path(root, fold, weights)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_prediction(
        frames,
        dict(zip(MODELS, weights)),
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3:
        return np.nan
    left_rank = left[valid].rank(method="average").to_numpy(np.float64)
    right_rank = right[valid].rank(method="average").to_numpy(np.float64)
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return np.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def summarize_correlations(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    joined = features.merge(targets, on=["fold", "rec_name"], how="inner")
    feature_columns = [
        column
        for column in features.columns
        if column not in {"fold", "rec_name", "gt_instances"}
    ]
    target_columns = [
        "canonical_mAP",
        "oracle_gain",
        "continuous_delta",
        "event_delta",
        "proposal_delta",
        "event_heavy_delta",
        "proposal_heavy_delta",
    ]
    rows = []
    for feature in feature_columns:
        for target in target_columns:
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "spearman": rank_correlation(joined[feature], joined[target]),
                    "n": int((joined[feature].notna() & joined[target].notna()).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "spearman", key=lambda values: values.abs(), ascending=False
    )


def main() -> None:
    args = parse_args()
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    proposal_root = resolve(args.proposal_root)
    weight_root = resolve(args.weight_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    annotations = load_annotations(resolve(args.ann_path), min_duration_s=2.0)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_grid = simplex_weights(args.weight_step)

    feature_rows = []
    expert_rows = []
    weight_rows = []
    for fold in range(5):
        continuous_prediction = best_prediction(continuous_root, fold)
        event_prediction = best_prediction(event_root, fold)
        proposal_prediction = json.loads(
            (
                proposal_root / f"fold_{fold:02d}_proposal_blend050.json"
            ).read_text(encoding="utf-8")
        )
        frames = [
            prediction_rows(continuous_prediction, "continuous"),
            prediction_rows(event_prediction, "event"),
            prediction_rows(proposal_prediction, "proposal"),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        ground_truth = {
            recording: ground_truth_frame(annotations, recording)
            for recording in recordings
        }
        for recording in recordings:
            feature_rows.append(
                {
                    "fold": fold,
                    "gt_instances": len(ground_truth[recording]),
                    **recording_features(frames, recording, args.agreement_topk),
                }
            )
        for model, prediction in zip(
            MODELS,
            (continuous_prediction, event_prediction, proposal_prediction),
        ):
            for recording in recordings:
                expert_rows.append(
                    {
                        "fold": fold,
                        "rec_name": recording,
                        "model": model,
                        **evaluate_recording(
                            prediction,
                            recording,
                            ground_truth[recording],
                        ),
                    }
                )
        for weights in weights_grid:
            prediction = load_or_build_weight_prediction(
                weight_root,
                fold,
                weights,
                frames,
            )
            for recording in recordings:
                weight_rows.append(
                    {
                        "fold": fold,
                        "rec_name": recording,
                        "weight_label": weight_label(weights),
                        "continuous_weight": weights[0],
                        "event_weight": weights[1],
                        "proposal_weight": weights[2],
                        **evaluate_recording(
                            prediction,
                            recording,
                            ground_truth[recording],
                        ),
                    }
                )

    features = pd.DataFrame(feature_rows)
    experts = pd.DataFrame(expert_rows)
    weight_metrics = pd.DataFrame(weight_rows)
    features.to_csv(out_dir / "recording_features.csv", index=False)
    experts.to_csv(out_dir / "recording_expert_metrics.csv", index=False)
    weight_metrics.to_csv(out_dir / "recording_weight_metrics.csv", index=False)

    valid_weights = weight_metrics[weight_metrics["gt_instances"] > 0].copy()
    canonical = valid_weights[
        np.isclose(valid_weights["continuous_weight"], CANONICAL_WEIGHTS[0])
        & np.isclose(valid_weights["event_weight"], CANONICAL_WEIGHTS[1])
        & np.isclose(valid_weights["proposal_weight"], CANONICAL_WEIGHTS[2])
    ][["fold", "rec_name", "gt_instances", "mAP"]].rename(
        columns={"mAP": "canonical_mAP"}
    )
    best = (
        valid_weights.sort_values("mAP", ascending=False)
        .groupby(["fold", "rec_name"], as_index=False)
        .first()
        .rename(columns={"mAP": "oracle_mAP"})
    )
    expert_pivot = experts.pivot(
        index=["fold", "rec_name"],
        columns="model",
        values="mAP",
    ).reset_index()
    targets = canonical.merge(
        best[
            [
                "fold",
                "rec_name",
                "oracle_mAP",
                "weight_label",
                "continuous_weight",
                "event_weight",
                "proposal_weight",
            ]
        ],
        on=["fold", "rec_name"],
    ).merge(expert_pivot, on=["fold", "rec_name"])
    targets["oracle_gain"] = targets["oracle_mAP"] - targets["canonical_mAP"]
    for model in MODELS:
        targets[f"{model}_delta"] = targets[model] - targets["canonical_mAP"]
    event_heavy = (
        valid_weights[valid_weights["event_weight"] >= 0.6]
        .groupby(["fold", "rec_name"])["mAP"]
        .max()
        .rename("event_heavy_mAP")
        .reset_index()
    )
    proposal_heavy = (
        valid_weights[valid_weights["proposal_weight"] >= 0.6]
        .groupby(["fold", "rec_name"])["mAP"]
        .max()
        .rename("proposal_heavy_mAP")
        .reset_index()
    )
    targets = targets.merge(event_heavy, on=["fold", "rec_name"]).merge(
        proposal_heavy,
        on=["fold", "rec_name"],
    )
    targets["event_heavy_delta"] = (
        targets["event_heavy_mAP"] - targets["canonical_mAP"]
    )
    targets["proposal_heavy_delta"] = (
        targets["proposal_heavy_mAP"] - targets["canonical_mAP"]
    )
    targets.to_csv(out_dir / "recording_targets.csv", index=False)
    correlations = summarize_correlations(features, targets)
    correlations.to_csv(out_dir / "feature_correlations.csv", index=False)

    expert_best = (
        experts[experts["gt_instances"] > 0]
        .sort_values("mAP", ascending=False)
        .groupby(["fold", "rec_name"], as_index=False)
        .first()
    )
    summary = {
        "recordings": int(len(features)),
        "positive_recordings": int(len(targets)),
        "canonical_macro_mAP": float(targets["canonical_mAP"].mean()),
        "canonical_instance_weighted_mAP": float(
            np.average(targets["canonical_mAP"], weights=targets["gt_instances"])
        ),
        "weight_oracle_macro_mAP": float(targets["oracle_mAP"].mean()),
        "weight_oracle_instance_weighted_mAP": float(
            np.average(targets["oracle_mAP"], weights=targets["gt_instances"])
        ),
        "weight_oracle_mean_gain": float(targets["oracle_gain"].mean()),
        "best_weight_counts": dict(Counter(targets["weight_label"])),
        "expert_oracle_macro_mAP": float(expert_best["mAP"].mean()),
        "best_expert_counts": dict(Counter(expert_best["model"])),
        "largest_absolute_correlations": correlations.head(20).to_dict("records"),
        "interpretation": (
            "The weight and expert oracles use recording GT and are diagnostic ceilings, "
            "not operational results."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print("\nTop label-free correlations")
    print(correlations.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
