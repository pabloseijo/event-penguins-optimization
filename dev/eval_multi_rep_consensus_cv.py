"""Cross-model agreement scoring for the three source-validated TAD experts."""

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

from dev.eval_continuous_multi_rep_fusion_cv import (
    best_prediction,
    evaluate,
    prediction_rows,
    resolve,
)
from src.utils import temporal_soft_nms
from src.utils.detection import temporal_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/multi_rep_consensus_cv_v1"
    )
    parser.add_argument("--agreement-tious", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--agreement-blends", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--boundary-blends", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--per-model-topk", type=int, default=100)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--nms-sigma", type=float, default=0.5)
    return parser.parse_args()


def consensus_candidates(
    group: pd.DataFrame,
    model_weights: dict[str, float],
    agreement_tiou: float,
    agreement_blend: float,
    boundary_blend: float,
    per_model_topk: int,
) -> np.ndarray:
    selected = pd.concat(
        [part.nlargest(per_model_topk, "rank_score") for _, part in group.groupby("model")],
        ignore_index=True,
    )
    models = sorted(model_weights)
    by_model = {model: selected[selected["model"] == model] for model in models}
    output = []
    for row in selected.itertuples(index=False):
        support = []
        boundaries = []
        boundary_weights = []
        for model in models:
            if model == row.model:
                continue
            candidates = by_model[model]
            if candidates.empty:
                support.append(0.0)
                continue
            overlaps = temporal_iou(
                candidates["t_start"].to_numpy(np.float64),
                candidates["t_end"].to_numpy(np.float64),
                float(row.t_start),
                float(row.t_end),
            )
            eligible = overlaps >= agreement_tiou
            if not eligible.any():
                support.append(0.0)
                continue
            scores = candidates["rank_score"].to_numpy(np.float64) * overlaps
            index = int(np.argmax(np.where(eligible, scores, -np.inf)))
            support.append(float(scores[index]))
            boundaries.append(
                candidates.iloc[index][["t_start", "t_end"]].to_numpy(np.float64)
            )
            boundary_weights.append(float(scores[index]))
        agreement = float(np.mean(support)) if support else 0.0
        base_score = float(row.rank_score) * model_weights[str(row.model)]
        score = base_score * ((1.0 - agreement_blend) + agreement_blend * agreement)
        start, end = float(row.t_start), float(row.t_end)
        if boundary_blend > 0.0 and boundaries:
            voted = np.average(np.stack(boundaries), axis=0, weights=boundary_weights)
            start = (1.0 - boundary_blend) * start + boundary_blend * float(voted[0])
            end = (1.0 - boundary_blend) * end + boundary_blend * float(voted[1])
        if end - start >= 2.0:
            output.append([start, end, score])
    return np.asarray(output, dtype=np.float64).reshape(-1, 3)


def build_prediction(
    frames: list[pd.DataFrame],
    agreement_tiou: float,
    agreement_blend: float,
    boundary_blend: float,
    args: argparse.Namespace,
) -> dict:
    candidates = pd.concat(frames, ignore_index=True)
    results = {
        recording: {} for recording in sorted(candidates["rec_name"].astype(str).unique())
    }
    weights = {"continuous": 0.2, "event": 0.4, "proposal": 0.4}
    for (recording, roi_id), group in candidates.groupby(["rec_name", "roi_id"]):
        values = consensus_candidates(
            group,
            weights,
            agreement_tiou,
            agreement_blend,
            boundary_blend,
            args.per_model_topk,
        )
        detections = temporal_soft_nms(values, sigma=args.nms_sigma, score_threshold=1e-5)
        results[str(recording)][str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(score),
            }
            for start, end, score in detections[: args.max_predictions]
            if end - start >= 2.0
        ]
    return {"version": "cross-model-consensus-rank-fusion-v1", "results": results}


def main() -> None:
    args = parse_args()
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    proposal_root = resolve(args.proposal_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(5):
        proposal_path = (
            proposal_root / f"fold_{fold:02d}" / "predictions" / f"{args.proposal_variant}.json"
        )
        frames = [
            prediction_rows(best_prediction(continuous_root, fold), "continuous"),
            prediction_rows(best_prediction(event_root, fold), "event"),
            prediction_rows(json.loads(proposal_path.read_text()), "proposal"),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for tiou in args.agreement_tious:
            for blend in args.agreement_blends:
                for boundary_blend in args.boundary_blends:
                    label = f"fold{fold:02d}_t{tiou:g}_s{blend:g}_b{boundary_blend:g}"
                    prediction = build_prediction(frames, tiou, blend, boundary_blend, args)
                    rows.append(
                        {
                            "fold": fold,
                            "agreement_tiou": tiou,
                            "agreement_blend": blend,
                            "boundary_blend": boundary_blend,
                            "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                            **evaluate(
                                prediction,
                                recordings,
                                resolve(args.ann_path),
                                out_dir / "predictions" / f"{label}.json",
                            ),
                        }
                    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for key, group in metrics.groupby(
        ["agreement_tiou", "agreement_blend", "boundary_blend"]
    ):
        instance_weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "agreement_tiou": key[0],
                "agreement_blend": key[1],
                "boundary_blend": key[2],
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=instance_weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
