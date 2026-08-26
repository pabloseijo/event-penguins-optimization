"""Relate label-free boundary-head statistics to per-recording CV gains.

The diagnostic consumes only out-of-fold boundary outputs and predictions.
It does not train a router or read the official test split. Its CSV artifacts
make the subsequent nested routing decision auditable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_heteroscedastic_boundary_cv import load_source_frames  # noqa: E402
from src.evaluation import DetectionsEvaluator  # noqa: E402


DEFAULT_VARIANTS = (
    "control",
    "reliable_w025",
    "reliable_w050",
    "reliable_w075",
    "reliable_w100",
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument(
        "--boundary-root",
        default=(
            "tmp/temporalmaxer_continuous/"
            "heteroscedastic_event_boundary_fold4_v1"
        ),
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default=(
            "tmp/temporalmaxer_continuous/"
            "event_boundary_reliability_diagnostic_v1"
        ),
    )
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    return parser.parse_args()


def summarize_boundary_outputs(
    frame: pd.DataFrame,
    mean: np.ndarray,
    variance: np.ndarray,
    quality: np.ndarray,
) -> pd.DataFrame:
    """Aggregate mechanistic, label-free head outputs by recording."""
    if not (len(frame) == len(mean) == len(variance) == len(quality)):
        raise ValueError("Frame and boundary outputs must have equal length")
    uncertainty = np.sqrt(np.maximum(variance, 1e-8)).mean(axis=1)
    reliability = np.clip(quality, 0.0, 1.0) * np.exp(-2.0 * uncertainty)
    correction_abs = np.abs(mean).mean(axis=1)
    effective_shift = reliability * correction_abs
    center_shift = np.abs(0.5 * (mean[:, 0] + mean[:, 1]))
    scale_shift = np.abs(mean[:, 1] - mean[:, 0])
    values = frame[["fold", "rec_name", "score"]].copy()
    values["quality"] = quality
    values["uncertainty"] = uncertainty
    values["reliability"] = reliability
    values["correction_abs"] = correction_abs
    values["effective_shift"] = effective_shift
    values["center_shift"] = center_shift
    values["scale_shift"] = scale_shift
    values["score_rank_recording"] = values.groupby("rec_name")["score"].rank(
        method="average",
        pct=True,
    )

    rows = []
    for (fold, recording), group in values.groupby(["fold", "rec_name"]):
        high_score = group[group["score_rank_recording"] >= 0.9]
        row: dict[str, float | int | str] = {
            "fold": int(fold),
            "rec_name": str(recording),
            "detections": len(group),
        }
        for column in (
            "quality",
            "uncertainty",
            "reliability",
            "correction_abs",
            "effective_shift",
            "center_shift",
            "scale_shift",
        ):
            array = group[column].to_numpy(np.float64)
            row[f"{column}_mean"] = float(array.mean())
            row[f"{column}_median"] = float(np.median(array))
            row[f"{column}_q90"] = float(np.quantile(array, 0.9))
        for column in ("quality", "uncertainty", "reliability", "effective_shift"):
            array = high_score[column].to_numpy(np.float64)
            row[f"top10_{column}_mean"] = float(array.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fold", "rec_name"]).reset_index(
        drop=True
    )


def evaluate_recording(
    prediction_path: Path,
    recording: str,
    ann_path: Path,
) -> dict[str, float]:
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(ann_path),
        prediction_filename=str(prediction_path),
        tiou_thresholds=np.asarray([0.1, 0.3, 0.5, 0.7]),
        valid_sequences=[recording],
        valid_labels=["ed"],
        min_duration=2.0,
    )
    result = {"mAP": float(evaluator.run())}
    for threshold, value in zip((0.1, 0.3, 0.5, 0.7), evaluator.mAP):
        result[f"AP@{threshold:.1f}"] = float(value)
    return result


def recording_gt_counts(ann_path: Path) -> dict[str, int]:
    database = json.loads(ann_path.read_text(encoding="utf-8"))["database"]
    counts = {}
    for recording, entry in database.items():
        counts[str(recording)] = sum(
            1
            for roi_id, annotations in entry.get("annotations", {}).items()
            if roi_id != "null"
            for annotation in annotations
            if annotation.get("label") == "ed"
            and float(annotation["segment"][1])
            - float(annotation["segment"][0])
            >= 2.0
        )
    return counts


def metric_correlations(
    features: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    joined = metrics.merge(features, on=["fold", "rec_name"], validate="many_to_one")
    feature_columns = [
        column
        for column in features.columns
        if column not in {"fold", "rec_name", "detections"}
    ]
    rows = []
    for variant in sorted(set(joined["variant"]) - {"control"}):
        subset = joined[joined["variant"] == variant]
        for feature in feature_columns:
            rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "spearman_delta_mAP": float(
                        subset[feature].corr(subset["delta_mAP"], method="spearman")
                    ),
                    "pearson_delta_mAP": float(
                        subset[feature].corr(subset["delta_mAP"], method="pearson")
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["variant", "spearman_delta_mAP"],
        ascending=[True, False],
    )


def main() -> None:
    args = parse_args()
    source_root = resolve(args.source_root)
    boundary_root = resolve(args.boundary_root)
    ann_path = resolve(args.ann_path)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = load_source_frames(source_root)
    feature_parts = []
    metric_rows = []
    gt_counts = recording_gt_counts(ann_path)
    for fold in range(5):
        validation = source[source["fold"] == fold].reset_index(drop=True)
        outputs = np.load(boundary_root / f"fold_{fold:02d}" / "boundary_outputs.npz")
        feature_parts.append(
            summarize_boundary_outputs(
                validation,
                outputs["mean"],
                outputs["variance"],
                outputs["quality"],
            )
        )
        recordings = [
            recording
            for recording in sorted(validation["rec_name"].astype(str).unique())
            if gt_counts.get(recording, 0) > 0
        ]
        control_by_recording = {}
        fold_rows = []
        for variant in args.variants:
            prediction_path = (
                boundary_root
                / f"fold_{fold:02d}"
                / "predictions"
                / f"{variant}.json"
            )
            for recording in recordings:
                result = evaluate_recording(prediction_path, recording, ann_path)
                if variant == "control":
                    control_by_recording[recording] = result
                fold_rows.append(
                    {
                        "fold": fold,
                        "rec_name": recording,
                        "gt_instances": gt_counts.get(recording, 0),
                        "variant": variant,
                        **result,
                    }
                )
        for row in fold_rows:
            control = control_by_recording[row["rec_name"]]
            row["delta_mAP"] = row["mAP"] - control["mAP"]
            row["delta_AP@0.7"] = row["AP@0.7"] - control["AP@0.7"]
        metric_rows.extend(fold_rows)

    features = pd.concat(feature_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    correlations = metric_correlations(features, metrics)
    features.to_csv(out_dir / "recording_features.csv", index=False)
    metrics.to_csv(out_dir / "recording_metrics.csv", index=False)
    correlations.to_csv(out_dir / "correlations.csv", index=False)
    print("Top absolute Spearman correlations with per-recording mAP gain:")
    ranked = correlations.assign(
        absolute=correlations["spearman_delta_mAP"].abs()
    ).sort_values(["variant", "absolute"], ascending=[True, False])
    print(ranked.groupby("variant", sort=False).head(8).to_string(index=False))
    print("\nPer-recording metrics:")
    print(
        metrics[metrics["variant"] != "control"]
        .sort_values(["variant", "delta_mAP"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
