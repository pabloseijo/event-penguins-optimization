"""Nested-CV evaluation of a one-feature recording reliability router."""

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

from dev.diagnose_recording_expert_reliability import (  # noqa: E402
    CANONICAL_WEIGHTS,
    weight_prediction_path,
)
from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402


FEATURE = "agreement_continuous_event_frac07"


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        default="tmp/temporalmaxer_continuous/recording_expert_reliability_v1",
    )
    parser.add_argument(
        "--weight-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/recording_reliability_router_cv_v1",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6],
    )
    parser.add_argument("--min-train-routed", type=int, default=2)
    parser.add_argument("--min-proposal-weight", type=float, default=0.5)
    return parser.parse_args()


def load_weight_prediction(
    root: Path,
    fold: int,
    weights: tuple[float, float, float],
) -> dict:
    return json.loads(
        weight_prediction_path(root, fold, weights).read_text(encoding="utf-8")
    )


def select_rule(
    features: pd.DataFrame,
    weight_metrics: pd.DataFrame,
    outer_fold: int,
    thresholds: list[float],
    min_train_routed: int,
    min_proposal_weight: float,
) -> dict[str, float | int]:
    canonical = weight_metrics[
        np.isclose(weight_metrics["continuous_weight"], CANONICAL_WEIGHTS[0])
        & np.isclose(weight_metrics["event_weight"], CANONICAL_WEIGHTS[1])
        & np.isclose(weight_metrics["proposal_weight"], CANONICAL_WEIGHTS[2])
        & (weight_metrics["gt_instances"] > 0)
    ][["fold", "rec_name", "gt_instances", "mAP"]].rename(
        columns={"mAP": "canonical_mAP"}
    )
    alternatives = weight_metrics[
        (weight_metrics["proposal_weight"] >= min_proposal_weight)
        & (weight_metrics["gt_instances"] > 0)
    ].merge(
        canonical,
        on=["fold", "rec_name", "gt_instances"],
    ).merge(
        features[["fold", "rec_name", FEATURE]],
        on=["fold", "rec_name"],
    )
    training = alternatives[alternatives["fold"] != outer_fold]
    best_key = None
    best_row = None
    for weights, group in training.groupby(
        ["continuous_weight", "event_weight", "proposal_weight"]
    ):
        for threshold in thresholds:
            routed = group[FEATURE] <= threshold
            routed_recordings = int(
                group.loc[routed, ["fold", "rec_name"]].drop_duplicates().shape[0]
            )
            if routed_recordings < min_train_routed:
                continue
            score = np.where(routed, group["mAP"], group["canonical_mAP"])
            weighted = float(np.average(score, weights=group["gt_instances"]))
            macro = float(np.mean(score))
            candidate_key = (
                weighted,
                macro,
                -threshold,
                -float(weights[2]),
                -float(weights[1]),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_row = {
                    "continuous_weight": float(weights[0]),
                    "event_weight": float(weights[1]),
                    "proposal_weight": float(weights[2]),
                    "threshold": float(threshold),
                    "train_weighted_mAP": weighted,
                    "train_macro_mAP": macro,
                    "train_routed_recordings": routed_recordings,
                }
    if best_row is None:
        raise RuntimeError(f"No valid routing rule for outer fold {outer_fold}")
    return best_row


def routed_prediction(
    canonical: dict,
    alternative: dict,
    feature_values: dict[str, float],
    threshold: float,
    alternative_weights: tuple[float, float, float],
) -> tuple[dict, list[str]]:
    scale = max(CANONICAL_WEIGHTS) / max(alternative_weights)
    routed = sorted(
        recording
        for recording, value in feature_values.items()
        if value <= threshold
    )
    routed_set = set(routed)
    results = {}
    for recording in canonical["results"]:
        source = (
            alternative["results"].get(recording, {})
            if recording in routed_set
            else canonical["results"].get(recording, {})
        )
        results[recording] = {
            roi_id: [
                {
                    **detection,
                    "score": float(detection["score"]) * (
                        scale if recording in routed_set else 1.0
                    ),
                }
                for detection in detections
            ]
            for roi_id, detections in source.items()
        }
    return (
        {
            "version": (
                "nested-recording-reliability-router:"
                f"{FEATURE}:threshold={threshold:g}"
            ),
            "results": results,
        },
        routed,
    )


def summarize(metrics: pd.DataFrame, prefix: str) -> dict[str, float]:
    weights = metrics["val_ed_instances"].to_numpy(np.float64)
    return {
        f"{prefix}_mean_mAP": float(metrics[f"{prefix}_mAP"].mean()),
        f"{prefix}_weighted_mAP": float(
            np.average(metrics[f"{prefix}_mAP"], weights=weights)
        ),
        f"{prefix}_worst_mAP": float(metrics[f"{prefix}_mAP"].min()),
        f"{prefix}_mean_AP@0.1": float(metrics[f"{prefix}_AP@0.1"].mean()),
        f"{prefix}_mean_AP@0.3": float(metrics[f"{prefix}_AP@0.3"].mean()),
        f"{prefix}_mean_AP@0.5": float(metrics[f"{prefix}_AP@0.5"].mean()),
        f"{prefix}_mean_AP@0.7": float(metrics[f"{prefix}_AP@0.7"].mean()),
    }


def main() -> None:
    args = parse_args()
    diagnostic_root = resolve(args.diagnostic_root)
    weight_root = resolve(args.weight_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(diagnostic_root / "recording_features.csv")
    weight_metrics = pd.read_csv(diagnostic_root / "recording_weight_metrics.csv")
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")

    rows = []
    routes = []
    for fold in range(5):
        rule = select_rule(
            features,
            weight_metrics,
            outer_fold=fold,
            thresholds=args.thresholds,
            min_train_routed=args.min_train_routed,
            min_proposal_weight=args.min_proposal_weight,
        )
        alternative_weights = (
            float(rule["continuous_weight"]),
            float(rule["event_weight"]),
            float(rule["proposal_weight"]),
        )
        canonical = load_weight_prediction(weight_root, fold, CANONICAL_WEIGHTS)
        alternative = load_weight_prediction(weight_root, fold, alternative_weights)
        fold_features = features[features["fold"] == fold]
        feature_values = dict(
            zip(
                fold_features["rec_name"].astype(str),
                fold_features[FEATURE].astype(float),
            )
        )
        prediction, routed = routed_prediction(
            canonical,
            alternative,
            feature_values,
            float(rule["threshold"]),
            alternative_weights,
        )
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        control_metrics = evaluate(
            canonical,
            recordings,
            resolve(args.ann_path),
            out_dir / "predictions" / f"fold_{fold:02d}_control.json",
        )
        router_metrics = evaluate(
            prediction,
            recordings,
            resolve(args.ann_path),
            out_dir / "predictions" / f"fold_{fold:02d}_router.json",
        )
        rows.append(
            {
                "fold": fold,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **rule,
                "routed_recordings": len(routed),
                "control_mAP": control_metrics["mAP"],
                "control_AP@0.1": control_metrics["AP@0.1"],
                "control_AP@0.3": control_metrics["AP@0.3"],
                "control_AP@0.5": control_metrics["AP@0.5"],
                "control_AP@0.7": control_metrics["AP@0.7"],
                "router_mAP": router_metrics["mAP"],
                "router_AP@0.1": router_metrics["AP@0.1"],
                "router_AP@0.3": router_metrics["AP@0.3"],
                "router_AP@0.5": router_metrics["AP@0.5"],
                "router_AP@0.7": router_metrics["AP@0.7"],
            }
        )
        routes.extend(
            {
                "fold": fold,
                "rec_name": recording,
                "feature": feature_values[recording],
                "threshold": rule["threshold"],
                "routed": recording in routed,
            }
            for recording in recordings
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(routes).to_csv(out_dir / "routes.csv", index=False)
    summary = {
        "feature": FEATURE,
        "threshold_candidates": args.thresholds,
        "selection": (
            "outer-fold source only; maximize instance-weighted per-recording mAP, "
            "then macro mAP"
        ),
        **summarize(metrics, "control"),
        **summarize(metrics, "router"),
    }
    for metric in ("mean_mAP", "weighted_mAP", "worst_mAP", "mean_AP@0.7"):
        summary[f"delta_{metric}"] = (
            summary[f"router_{metric}"] - summary[f"control_{metric}"]
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    print(pd.DataFrame(routes).to_string(index=False))


if __name__ == "__main__":
    main()
