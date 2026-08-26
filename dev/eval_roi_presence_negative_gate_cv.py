"""Evaluate a nested high-confidence negative ROI gate on source OOF predictions."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_pal_consistency_blend_cv_v1/predictions"
        ),
    )
    parser.add_argument(
        "--prediction-template",
        default="fold{fold:02d}_cltdr0.1.json",
    )
    parser.add_argument(
        "--probabilities",
        default=(
            "tmp/temporalmaxer_continuous/roi_presence_gate_cv_v1/"
            "oof_presence_probabilities.csv"
        ),
    )
    parser.add_argument(
        "--control-metrics",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_pal_consistency_blend_cv_v1/metrics.csv"
        ),
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default=(
            "tmp/temporalmaxer_continuous/"
            "roi_presence_nested_negative_gate_cv_v1"
        ),
    )
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=(0.02, 0.05, 0.1, 0.2),
    )
    parser.add_argument(
        "--suppression-factors",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75),
    )
    return parser.parse_args()


def aggregate_metrics(group: pd.DataFrame) -> dict[str, float]:
    weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
    return {
        "mean_mAP": float(group["mAP"].mean()),
        "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
        "worst_mAP": float(group["mAP"].min()),
        "mean_AP@0.7": float(group["AP@0.7"].mean()),
    }


def select_nested_configuration(
    calibration_metrics: pd.DataFrame,
    control_metrics: pd.DataFrame,
) -> dict[str, float]:
    """Select the best configuration passing all source gates."""
    control = aggregate_metrics(control_metrics)
    rows = []
    for (threshold, factor), group in calibration_metrics.groupby(
        ["threshold", "suppression_factor"]
    ):
        if int(group["suppressed_active_rois"].sum()) > 0:
            continue
        aggregate = aggregate_metrics(group)
        if any(
            aggregate[key] + 1e-12 < control[key]
            for key in (
                "mean_mAP",
                "weighted_mAP",
                "worst_mAP",
                "mean_AP@0.7",
            )
        ):
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "suppression_factor": float(factor),
                **{
                    f"calibration_{key}": value
                    for key, value in aggregate.items()
                },
            }
        )
    if not rows:
        raise ValueError("No safe negative-gate configuration on calibration folds")
    return sorted(
        rows,
        key=lambda row: (
            row["calibration_mean_mAP"],
            row["calibration_weighted_mAP"],
            row["calibration_worst_mAP"],
            row["calibration_mean_AP@0.7"],
            -row["threshold"],
            row["suppression_factor"],
        ),
        reverse=True,
    )[0]


def apply_negative_gate(
    prediction: dict,
    probabilities: pd.DataFrame,
    threshold: float,
    suppression_factor: float,
) -> dict:
    if threshold < 0 or not 0.0 <= suppression_factor <= 1.0:
        raise ValueError("Invalid negative-gate threshold or suppression factor")
    output = copy.deepcopy(prediction)
    probability_by_roi = {
        (str(row.rec_name), int(row.roi_id)): float(row.presence_probability)
        for row in probabilities.itertuples(index=False)
    }
    for recording, rois in output["results"].items():
        for roi, detections in rois.items():
            if probability_by_roi[(recording, int(roi))] >= threshold:
                continue
            for detection in detections:
                detection["score"] = (
                    float(detection["score"]) * suppression_factor
                )
    output["version"] = (
        f"nested-negative-roi-gate-t{threshold:g}-s{suppression_factor:g}"
    )
    return output


def main() -> None:
    args = parse_args()
    candidates = [float(value) for value in args.threshold_candidates]
    suppression_factors = [float(value) for value in args.suppression_factors]
    if any(value <= 0 or value >= 1 for value in candidates):
        raise ValueError("Threshold candidates must lie strictly inside (0,1)")
    if any(value < 0 or value > 1 for value in suppression_factors):
        raise ValueError("Suppression factors must lie in [0,1]")

    probabilities = pd.read_csv(resolve(args.probabilities))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    control_metrics = pd.read_csv(resolve(args.control_metrics))
    control_metrics = control_metrics[
        control_metrics["old_event_weight"].round(8).eq(0.1)
        & control_metrics["cltdr_event_weight"].round(8).eq(0.1)
    ].copy()
    if set(control_metrics["fold"]) != set(range(5)):
        raise ValueError("Expected one control metric row for each fold")
    annotation_path = resolve(args.ann_path)
    prediction_root = resolve(args.prediction_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(5):
        validation = probabilities[probabilities["fold"] == fold]
        prediction = json.loads(
            (
                prediction_root
                / args.prediction_template.format(fold=fold)
            ).read_text(encoding="utf-8")
        )
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for threshold in candidates:
            for factor in suppression_factors:
                adjusted = apply_negative_gate(
                    prediction,
                    validation,
                    threshold,
                    factor,
                )
                suppressed = int(
                    (validation["presence_probability"] < threshold).sum()
                )
                suppressed_active = int(
                    (
                        (validation["presence_probability"] < threshold)
                        & (validation["target_present"] > 0.5)
                    ).sum()
                )
                rows.append(
                    {
                        "fold": fold,
                        "threshold": threshold,
                        "suppression_factor": factor,
                        "suppressed_rois": suppressed,
                        "suppressed_active_rois": suppressed_active,
                        "val_ed_instances": int(
                            manifest.loc[fold, "val_ed_instances"]
                        ),
                        **evaluate(
                            adjusted,
                            recordings,
                            annotation_path,
                            out_dir
                            / "predictions"
                            / (
                                f"fold{fold:02d}_threshold{threshold:g}"
                                f"_factor{factor:g}.json"
                            ),
                        ),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    grid_rows = []
    for (threshold, factor), group in metrics.groupby(
        ["threshold", "suppression_factor"]
    ):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        grid_rows.append(
            {
                "threshold": float(threshold),
                "suppression_factor": float(factor),
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(
                    np.average(group["mAP"], weights=weights)
                ),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
                "suppressed_rois": int(group["suppressed_rois"].sum()),
                "suppressed_active_rois": int(
                    group["suppressed_active_rois"].sum()
                ),
            }
        )
    grid_summary = pd.DataFrame(grid_rows).sort_values(
        "mean_mAP", ascending=False
    )
    grid_summary.to_csv(out_dir / "grid_summary.csv", index=False)

    selected_rows = []
    for fold in range(5):
        selected = select_nested_configuration(
            metrics[metrics["fold"] != fold],
            control_metrics[control_metrics["fold"] != fold],
        )
        match = metrics[
            (metrics["fold"] == fold)
            & (metrics["threshold"] == selected["threshold"])
            & (
                metrics["suppression_factor"]
                == selected["suppression_factor"]
            )
        ]
        if len(match) != 1:
            raise RuntimeError("Expected exactly one outer-fold metric row")
        selected_rows.append({**match.iloc[0].to_dict(), **selected})
    selected_metrics = pd.DataFrame(selected_rows)
    selected_metrics.to_csv(out_dir / "nested_selected_metrics.csv", index=False)
    weights = selected_metrics["val_ed_instances"].to_numpy(dtype=np.float64)
    nested_summary = {
        "mean_mAP": float(selected_metrics["mAP"].mean()),
        "weighted_mAP": float(
            np.average(selected_metrics["mAP"], weights=weights)
        ),
        "worst_mAP": float(selected_metrics["mAP"].min()),
        "mean_AP@0.1": float(selected_metrics["AP@0.1"].mean()),
        "mean_AP@0.3": float(selected_metrics["AP@0.3"].mean()),
        "mean_AP@0.5": float(selected_metrics["AP@0.5"].mean()),
        "mean_AP@0.7": float(selected_metrics["AP@0.7"].mean()),
        "suppressed_rois": int(selected_metrics["suppressed_rois"].sum()),
        "suppressed_active_rois": int(
            selected_metrics["suppressed_active_rois"].sum()
        ),
    }
    final_configuration = select_nested_configuration(metrics, control_metrics)
    (out_dir / "nested_summary.json").write_text(
        json.dumps(nested_summary, indent=2), encoding="utf-8"
    )
    (out_dir / "final_configuration.json").write_text(
        json.dumps(final_configuration, indent=2), encoding="utf-8"
    )
    print(selected_metrics.to_string(index=False))
    print(json.dumps(nested_summary, indent=2))
    print(json.dumps({"final_configuration": final_configuration}, indent=2))


if __name__ == "__main__":
    main()
