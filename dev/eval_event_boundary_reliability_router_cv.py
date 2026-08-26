"""Nested recording-disjoint CV for a conservative boundary-correction router.

Only three literature-grounded, label-free gates are considered: low predicted
boundary uncertainty, high predicted localization quality, and high combined
reliability. The alternative is fixed to the 25% boundary correction. Every
outer fold selects its rule using the other four OOF folds, and a no-op remains
available whenever no rule improves all training safeguards.
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

from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402


FEATURE_DIRECTIONS = {
    "uncertainty_mean": "low",
    "quality_q90": "high",
    "reliability_q90": "high",
}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary-root",
        default=(
            "tmp/temporalmaxer_continuous/"
            "heteroscedastic_event_boundary_fold4_v1"
        ),
    )
    parser.add_argument(
        "--diagnostic-root",
        default=(
            "tmp/temporalmaxer_continuous/"
            "event_boundary_reliability_diagnostic_v1"
        ),
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default=(
            "tmp/temporalmaxer_continuous/"
            "event_boundary_reliability_router_cv_v1"
        ),
    )
    parser.add_argument("--alternative", default="reliable_w025")
    parser.add_argument(
        "--threshold-quantiles",
        type=float,
        nargs="+",
        default=[0.25, 0.4, 0.5, 0.6, 0.75],
    )
    parser.add_argument("--min-routed-recordings", type=int, default=2)
    return parser.parse_args()


def load_prediction(root: Path, fold: int, variant: str) -> dict:
    return json.loads(
        (
            root
            / f"fold_{fold:02d}"
            / "predictions"
            / f"{variant}.json"
        ).read_text(encoding="utf-8")
    )


def routed_recordings(
    features: pd.DataFrame,
    feature: str,
    direction: str,
    threshold: float,
) -> list[str]:
    if direction == "low":
        selected = features[feature] <= threshold
    elif direction == "high":
        selected = features[feature] >= threshold
    else:
        raise ValueError(f"Unknown routing direction: {direction}")
    return sorted(features.loc[selected, "rec_name"].astype(str).unique())


def merge_recording_predictions(
    control: dict,
    alternative: dict,
    routed: list[str],
    version: str,
) -> dict:
    """Swap complete recording predictions while preserving scores and order."""
    routed_set = set(routed)
    recordings = sorted(set(control["results"]) | set(alternative["results"]))
    return {
        "version": version,
        "results": {
            recording: (
                alternative["results"].get(recording, {})
                if recording in routed_set
                else control["results"].get(recording, {})
            )
            for recording in recordings
        },
    }


def summarize_fold_rows(rows: list[dict]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    weights = frame["val_ed_instances"].to_numpy(np.float64)
    return {
        "mean_mAP": float(frame["mAP"].mean()),
        "weighted_mAP": float(np.average(frame["mAP"], weights=weights)),
        "worst_mAP": float(frame["mAP"].min()),
        "mean_AP@0.1": float(frame["AP@0.1"].mean()),
        "mean_AP@0.3": float(frame["AP@0.3"].mean()),
        "mean_AP@0.5": float(frame["AP@0.5"].mean()),
        "mean_AP@0.7": float(frame["AP@0.7"].mean()),
    }


def safeguards_pass(
    candidate: dict[str, float],
    control: dict[str, float],
    tolerance: float = 1e-12,
) -> bool:
    guarded = ("mean_mAP", "weighted_mAP", "worst_mAP", "mean_AP@0.7")
    return all(candidate[key] >= control[key] - tolerance for key in guarded) and any(
        candidate[key] > control[key] + tolerance for key in guarded
    )


def main() -> None:
    args = parse_args()
    boundary_root = resolve(args.boundary_root)
    diagnostic_root = resolve(args.diagnostic_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    ann_path = resolve(args.ann_path)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(diagnostic_root / "recording_features.csv")
    controls = {
        fold: load_prediction(boundary_root, fold, "control") for fold in range(5)
    }
    alternatives = {
        fold: load_prediction(boundary_root, fold, args.alternative)
        for fold in range(5)
    }
    fold_recordings = {
        fold: str(manifest.loc[fold, "val_record_names"]).split()
        for fold in range(5)
    }
    fold_instances = {
        fold: int(manifest.loc[fold, "val_ed_instances"]) for fold in range(5)
    }

    control_metrics = {}
    for fold in range(5):
        control_metrics[fold] = evaluate(
            controls[fold],
            fold_recordings[fold],
            ann_path,
            out_dir / "selection" / "control" / f"fold_{fold:02d}.json",
        )

    outer_rows = []
    route_rows = []
    selection_rows = []
    for outer_fold in range(5):
        train_folds = [fold for fold in range(5) if fold != outer_fold]
        train_control = summarize_fold_rows(
            [
                {
                    "val_ed_instances": fold_instances[fold],
                    **control_metrics[fold],
                }
                for fold in train_folds
            ]
        )
        training_features = features[features["fold"].isin(train_folds)]
        best = None
        best_key = None
        for feature, direction in FEATURE_DIRECTIONS.items():
            thresholds = sorted(
                set(
                    float(value)
                    for value in np.quantile(
                        training_features[feature].to_numpy(np.float64),
                        args.threshold_quantiles,
                    )
                )
            )
            for threshold in thresholds:
                routed_train = routed_recordings(
                    training_features,
                    feature,
                    direction,
                    threshold,
                )
                if len(routed_train) < args.min_routed_recordings:
                    continue
                candidate_rows = []
                for fold in train_folds:
                    fold_features = features[features["fold"] == fold]
                    routed = routed_recordings(
                        fold_features,
                        feature,
                        direction,
                        threshold,
                    )
                    prediction = merge_recording_predictions(
                        controls[fold],
                        alternatives[fold],
                        routed,
                        (
                            "nested-event-boundary-router-selection:"
                            f"{feature}:{direction}:{threshold:.8g}"
                        ),
                    )
                    metrics = evaluate(
                        prediction,
                        fold_recordings[fold],
                        ann_path,
                        (
                            out_dir
                            / "selection"
                            / f"outer_{outer_fold:02d}"
                            / (
                                f"{feature}_{direction}_"
                                f"{threshold:.8g}_fold_{fold:02d}.json"
                            )
                        ),
                    )
                    candidate_rows.append(
                        {
                            "val_ed_instances": fold_instances[fold],
                            **metrics,
                        }
                    )
                candidate = summarize_fold_rows(candidate_rows)
                passed = safeguards_pass(candidate, train_control)
                selection_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "train_routed_recordings": len(routed_train),
                        "safeguards_pass": passed,
                        **{f"train_{key}": value for key, value in candidate.items()},
                    }
                )
                key = (
                    candidate["mean_mAP"],
                    candidate["weighted_mAP"],
                    candidate["mean_AP@0.7"],
                    candidate["worst_mAP"],
                )
                if passed and (best_key is None or key > best_key):
                    best_key = key
                    best = {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "train_routed_recordings": len(routed_train),
                        **{f"train_{key}": value for key, value in candidate.items()},
                    }

        outer_features = features[features["fold"] == outer_fold]
        if best is None:
            routed = []
            prediction = controls[outer_fold]
            best = {
                "feature": "no_op",
                "direction": "none",
                "threshold": np.nan,
                "train_routed_recordings": 0,
                **{f"train_{key}": value for key, value in train_control.items()},
            }
        else:
            routed = routed_recordings(
                outer_features,
                str(best["feature"]),
                str(best["direction"]),
                float(best["threshold"]),
            )
            prediction = merge_recording_predictions(
                controls[outer_fold],
                alternatives[outer_fold],
                routed,
                (
                    "nested-event-boundary-router:"
                    f"{best['feature']}:{best['direction']}:{best['threshold']:.8g}"
                ),
            )
        metrics = evaluate(
            prediction,
            fold_recordings[outer_fold],
            ann_path,
            out_dir / "predictions" / f"fold_{outer_fold:02d}_router.json",
        )
        outer_rows.append(
            {
                "fold": outer_fold,
                "val_ed_instances": fold_instances[outer_fold],
                **best,
                "routed_recordings": len(routed),
                **{f"control_{key}": value for key, value in control_metrics[outer_fold].items()},
                **{f"router_{key}": value for key, value in metrics.items()},
            }
        )
        route_rows.extend(
            {
                "fold": outer_fold,
                "rec_name": str(row.rec_name),
                "feature": best["feature"],
                "feature_value": (
                    float(getattr(row, str(best["feature"])))
                    if best["feature"] != "no_op"
                    else np.nan
                ),
                "threshold": best["threshold"],
                "routed": str(row.rec_name) in routed,
            }
            for row in outer_features.itertuples(index=False)
        )

    outer = pd.DataFrame(outer_rows)
    control_summary = summarize_fold_rows(
        [
            {
                "val_ed_instances": int(row["val_ed_instances"]),
                "mAP": float(row["control_mAP"]),
                "AP@0.1": float(row["control_AP@0.1"]),
                "AP@0.3": float(row["control_AP@0.3"]),
                "AP@0.5": float(row["control_AP@0.5"]),
                "AP@0.7": float(row["control_AP@0.7"]),
            }
            for _, row in outer.iterrows()
        ]
    )
    router_summary = summarize_fold_rows(
        [
            {
                "val_ed_instances": int(row["val_ed_instances"]),
                "mAP": float(row["router_mAP"]),
                "AP@0.1": float(row["router_AP@0.1"]),
                "AP@0.3": float(row["router_AP@0.3"]),
                "AP@0.5": float(row["router_AP@0.5"]),
                "AP@0.7": float(row["router_AP@0.7"]),
            }
            for _, row in outer.iterrows()
        ]
    )
    summary = {
        "alternative": args.alternative,
        "features": FEATURE_DIRECTIONS,
        "protocol": (
            "outer recording-fold CV; rule selected on four OOF folds; "
            "no-op unless mean, weighted, worst and AP@0.7 safeguards pass"
        ),
        **{f"control_{key}": value for key, value in control_summary.items()},
        **{f"router_{key}": value for key, value in router_summary.items()},
    }
    for key in ("mean_mAP", "weighted_mAP", "worst_mAP", "mean_AP@0.7"):
        summary[f"delta_{key}"] = (
            router_summary[key] - control_summary[key]
        )
    summary["accept"] = all(
        summary[f"delta_{key}"] > 0.0
        for key in ("mean_mAP", "weighted_mAP", "worst_mAP", "mean_AP@0.7")
    )
    outer.to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out_dir / "routes.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        out_dir / "selection_metrics.csv",
        index=False,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(outer.to_string(index=False))


if __name__ == "__main__":
    main()
