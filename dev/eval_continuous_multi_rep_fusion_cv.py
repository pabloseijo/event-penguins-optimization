"""Recording-disjoint CV fusion of continuous, event-stat, and proposal experts."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.evaluation import DetectionsEvaluator
from src.utils import temporal_soft_nms


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1"
    )
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/multi_rep_fusion_cv_v1"
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.1,
        help="Simplex grid step; zero weights are excluded so every expert contributes.",
    )
    parser.add_argument(
        "--fixed-weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("CONTINUOUS", "EVENT", "PROPOSAL"),
    )
    parser.add_argument(
        "--rank-scope", choices=("global", "recording", "roi"), default="global"
    )
    parser.add_argument("--nms-sigma", type=float, default=0.5)
    parser.add_argument("--per-model-topk", type=int, default=100)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    return parser.parse_args()


def prediction_rows(
    prediction: dict, model: str, rank_scope: str = "global"
) -> pd.DataFrame:
    rows = []
    for recording, rois in prediction["results"].items():
        for roi_id, detections in rois.items():
            for detection in detections:
                rows.append(
                    {
                        "rec_name": recording,
                        "roi_id": int(roi_id),
                        "t_start": float(detection["segment"][0]),
                        "t_end": float(detection["segment"][1]),
                        "raw_score": float(detection["score"]),
                        "model": model,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        group_columns = {
            "global": [],
            "recording": ["rec_name"],
            "roi": ["rec_name", "roi_id"],
        }[rank_scope]
        if group_columns:
            frame["rank_score"] = frame.groupby(group_columns)["raw_score"].rank(
                method="average", pct=True
            )
        else:
            frame["rank_score"] = frame["raw_score"].rank(method="average", pct=True)
    return frame


def simplex_weights(step: float) -> list[tuple[float, float, float]]:
    units = round(1.0 / step)
    if units < 3 or not np.isclose(units * step, 1.0):
        raise ValueError("weight-step must divide 1.0 and leave room for three experts")
    return [
        (i / units, j / units, (units - i - j) / units)
        for i, j in itertools.product(range(1, units), repeat=2)
        if units - i - j >= 1
    ]


def build_prediction(
    frames: list[pd.DataFrame],
    weights: dict[str, float],
    sigma: float,
    per_model_topk: int,
    max_predictions: int,
    min_action_duration: float = 2.0,
) -> dict:
    if min_action_duration < 0:
        raise ValueError("min_action_duration must be non-negative")
    candidates = pd.concat(frames, ignore_index=True)
    candidates["fusion_score"] = candidates["rank_score"] * candidates["model"].map(weights)
    recordings = sorted(candidates["rec_name"].unique())
    results: dict[str, dict[str, list]] = {recording: {} for recording in recordings}
    for (recording, roi_id), group in candidates.groupby(["rec_name", "roi_id"]):
        selected = pd.concat(
            [part.nlargest(per_model_topk, "fusion_score") for _, part in group.groupby("model")],
            ignore_index=True,
        )
        values = selected[["t_start", "t_end", "fusion_score"]].to_numpy(np.float64)
        detections = temporal_soft_nms(values, sigma=sigma, score_threshold=1e-5)
        results[str(recording)][str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(score),
            }
            for start, end, score in detections[:max_predictions]
            if end - start >= min_action_duration
        ]
    return {"version": "continuous-multi-representation-rank-fusion-v1", "results": results}


def evaluate(
    prediction: dict,
    recordings: list[str],
    ann_path: Path,
    path: Path,
    tiou_thresholds: tuple[float, ...] | list[float] = (0.1, 0.3, 0.5, 0.7),
    min_action_duration: float = 2.0,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prediction), encoding="utf-8")
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(ann_path),
        prediction_filename=str(path),
        tiou_thresholds=np.asarray(tiou_thresholds, dtype=np.float64),
        valid_sequences=recordings,
        valid_labels=["ed"],
        min_duration=min_action_duration,
    )
    row = {
        "mAP": float(evaluator.run()),
        "n_predictions": sum(
            len(detections)
            for rois in prediction["results"].values()
            for detections in rois.values()
        ),
    }
    for threshold, value in zip(tiou_thresholds, evaluator.mAP):
        row[f"AP@{threshold:.1f}"] = float(value)
    return row


def best_prediction(root: Path, fold: int) -> dict:
    fold_dir = root / f"fold_{fold:02d}"
    epoch = int(json.loads((fold_dir / "metrics_best.json").read_text())["epoch"])
    return json.loads((fold_dir / "predictions" / f"epoch_{epoch:03d}.json").read_text())


def main() -> None:
    args = parse_args()
    if args.min_action_duration < 0:
        raise ValueError("--min-action-duration must be non-negative")
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    proposal_root = resolve(args.proposal_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    if args.fixed_weights is not None:
        if any(weight <= 0.0 for weight in args.fixed_weights) or not np.isclose(
            sum(args.fixed_weights), 1.0
        ):
            raise ValueError("fixed-weights must be positive and sum to 1.0")
        weights_grid = [tuple(args.fixed_weights)]
    else:
        weights_grid = simplex_weights(args.weight_step)
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {
        (int(row["fold"]), float(row["continuous_weight"]), float(row["event_weight"]))
        for row in rows
    }

    for fold in range(5):
        proposal_path = (
            proposal_root
            / f"fold_{fold:02d}"
            / "predictions"
            / f"{args.proposal_variant}.json"
        )
        frames = [
            prediction_rows(
                best_prediction(continuous_root, fold), "continuous", args.rank_scope
            ),
            prediction_rows(best_prediction(event_root, fold), "event", args.rank_scope),
            prediction_rows(
                json.loads(proposal_path.read_text()), "proposal", args.rank_scope
            ),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for continuous_weight, event_weight, proposal_weight in weights_grid:
            key = (fold, continuous_weight, event_weight)
            if key in completed:
                continue
            label = (
                f"fold{fold:02d}_cw{continuous_weight:g}_ew{event_weight:g}"
                f"_pw{proposal_weight:g}"
            )
            prediction = build_prediction(
                frames,
                {
                    "continuous": continuous_weight,
                    "event": event_weight,
                    "proposal": proposal_weight,
                },
                args.nms_sigma,
                args.per_model_topk,
                args.max_predictions,
                args.min_action_duration,
            )
            rows.append(
                {
                    "fold": fold,
                    "continuous_weight": continuous_weight,
                    "event_weight": event_weight,
                    "proposal_weight": proposal_weight,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"{label}.json",
                        args.tiou,
                        args.min_action_duration,
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            completed.add(key)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for key, group in metrics.groupby(
        ["continuous_weight", "event_weight", "proposal_weight"]
    ):
        if len(group) != 5:
            continue
        instance_weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "continuous_weight": key[0],
                "event_weight": key[1],
                "proposal_weight": key[2],
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=instance_weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
