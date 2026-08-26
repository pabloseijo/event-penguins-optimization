"""Late-fusion CV for complementary continuous and proposal-local detectors."""

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

from src.evaluation import DetectionsEvaluator
from src.utils import temporal_soft_nms


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", default="tmp/temporalmaxer_continuous/fusion_cv_v1")
    parser.add_argument("--continuous-weights", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--nms-sigmas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--per-model-topk", type=int, default=100)
    parser.add_argument("--max-predictions", type=int, default=200)
    return parser.parse_args()


def prediction_rows(prediction: dict, model: str) -> pd.DataFrame:
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
        frame["rank_score"] = frame["raw_score"].rank(method="average", pct=True)
    return frame


def build_fused_prediction(
    continuous: pd.DataFrame,
    proposal: pd.DataFrame,
    continuous_weight: float,
    sigma: float,
    per_model_topk: int,
    max_predictions: int,
) -> dict:
    candidates = pd.concat((continuous, proposal), ignore_index=True)
    model_weight = np.where(
        candidates["model"].to_numpy() == "continuous",
        continuous_weight,
        1.0 - continuous_weight,
    )
    candidates["fusion_score"] = candidates["rank_score"].to_numpy() * model_weight
    recordings = sorted(set(continuous["rec_name"]) | set(proposal["rec_name"]))
    results: dict[str, dict[str, list]] = {recording: {} for recording in recordings}
    roi_pairs = set(
        zip(candidates["rec_name"].astype(str), candidates["roi_id"].astype(int))
    )
    for recording, roi_id in sorted(roi_pairs):
        group = candidates[
            (candidates["rec_name"] == recording) & (candidates["roi_id"] == roi_id)
        ]
        selected_parts = []
        for _, model_group in group.groupby("model"):
            selected_parts.append(model_group.nlargest(per_model_topk, "fusion_score"))
        selected = pd.concat(selected_parts, ignore_index=True)
        values = selected[["t_start", "t_end", "fusion_score"]].to_numpy(dtype=np.float64)
        detections = temporal_soft_nms(values, sigma=sigma, score_threshold=1e-5)
        results[recording][str(roi_id)] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(score),
            }
            for start, end, score in detections[:max_predictions]
            if end - start >= 2.0
        ]
    return {"version": "continuous-proposal-rank-fusion-v1", "results": results}


def evaluate(
    prediction: dict,
    recordings: list[str],
    ann_path: Path,
    prediction_path: Path,
) -> dict[str, float | int]:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(ann_path),
        prediction_filename=str(prediction_path),
        tiou_thresholds=np.asarray([0.1, 0.3, 0.5, 0.7]),
        valid_sequences=recordings,
        valid_labels=["ed"],
        min_duration=2.0,
    )
    row: dict[str, float | int] = {
        "mAP": float(evaluator.run()),
        "n_predictions": sum(
            len(detections)
            for rois in prediction["results"].values()
            for detections in rois.values()
        ),
    }
    for threshold, value in zip((0.1, 0.3, 0.5, 0.7), evaluator.mAP):
        row[f"AP@{threshold:.1f}"] = float(value)
    return row


def main() -> None:
    args = parse_args()
    continuous_root = resolve(args.continuous_root)
    proposal_root = resolve(args.proposal_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {
        (int(row["fold"]), float(row["continuous_weight"]), float(row["nms_sigma"]))
        for row in rows
    }
    for fold in range(5):
        best_metrics = json.loads(
            (continuous_root / f"fold_{fold:02d}" / "metrics_best.json").read_text(
                encoding="utf-8"
            )
        )
        epoch = int(best_metrics["epoch"])
        continuous_prediction = json.loads(
            (
                continuous_root
                / f"fold_{fold:02d}"
                / "predictions"
                / f"epoch_{epoch:03d}.json"
            ).read_text(encoding="utf-8")
        )
        proposal_prediction = json.loads(
            (
                proposal_root
                / f"fold_{fold:02d}"
                / "predictions"
                / f"{args.proposal_variant}.json"
            ).read_text(encoding="utf-8")
        )
        continuous = prediction_rows(continuous_prediction, "continuous")
        proposal = prediction_rows(proposal_prediction, "proposal")
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for weight in args.continuous_weights:
            for sigma in args.nms_sigmas:
                key = (fold, float(weight), float(sigma))
                if key in completed:
                    continue
                label = f"fold{fold:02d}_cw{weight:g}_sigma{sigma:g}"
                prediction = build_fused_prediction(
                    continuous,
                    proposal,
                    weight,
                    sigma,
                    args.per_model_topk,
                    args.max_predictions,
                )
                rows.append(
                    {
                        "fold": fold,
                        "continuous_weight": weight,
                        "nms_sigma": sigma,
                        "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                        **evaluate(
                            prediction,
                            recordings,
                            resolve(args.ann_path),
                            out_dir / "predictions" / f"{label}.json",
                        ),
                    }
                )
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                completed.add(key)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for (weight, sigma), group in metrics.groupby(["continuous_weight", "nms_sigma"]):
        if len(group) != 5:
            continue
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "continuous_weight": weight,
                "nms_sigma": sigma,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
