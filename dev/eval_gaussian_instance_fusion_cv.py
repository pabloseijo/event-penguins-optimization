"""Evaluate Gaussian weighted fusion on the current source-OOF QFL experts.

Zhou et al. (CVPR 2023) replace NMS with a confidence-temperature weighted
estimate of the score and temporal boundaries of overlapping instances. This
script applies that fixed operator to the same continuous, event-statistic and
QFL proposal experts used by the canonical 0.842171 source recipe. It never
reads the official test split.
"""

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
    fit_linear_qfl,
    score_quality_head,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    evaluate,
    prediction_rows,
)
from src.utils.detection import temporal_iou  # noqa: E402


MODEL_WEIGHTS = {"continuous": 0.2, "event": 0.4, "proposal": 0.4}


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
        "--continuous-root",
        default="tmp/temporalmaxer_continuous/cv_recipe_v1",
    )
    parser.add_argument(
        "--event-root",
        default="tmp/temporalmaxer_continuous/cv_eventstats_v1",
    )
    parser.add_argument(
        "--control-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/gaussian_instance_fusion_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument(
        "--overlap-thresholds", type=float, nargs="+", default=[0.25, 0.5]
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.03, 0.1, 0.2]
    )
    parser.add_argument(
        "--overlap-modes",
        choices=("iou", "suppressed"),
        nargs="+",
        default=["iou", "suppressed"],
        help="'suppressed' reproduces the asymmetric overlap in the official code.",
    )
    parser.add_argument("--per-model-topk", type=int, default=100)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument(
        "--refine-overlap-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7],
    )
    parser.add_argument(
        "--refine-temperatures", type=float, nargs="+", default=[0.03, 0.1]
    )
    parser.add_argument(
        "--refine-blends", type=float, nargs="+", default=[0.25, 0.5]
    )
    parser.add_argument("--refine-min-models", type=int, default=2)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--qfl-steps", type=int, default=500)
    parser.add_argument("--qfl-learning-rate", type=float, default=0.03)
    return parser.parse_args()


def stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive")
    shifted = (values - float(values.max())) / temperature
    weights = np.exp(np.clip(shifted, -80.0, 0.0))
    return weights / weights.sum()


def overlap_with_anchor(
    starts: np.ndarray,
    ends: np.ndarray,
    anchor_start: float,
    anchor_end: float,
    mode: str,
) -> np.ndarray:
    if mode == "iou":
        return temporal_iou(starts, ends, anchor_start, anchor_end)
    if mode != "suppressed":
        raise ValueError(f"Unknown overlap mode: {mode}")
    intersection = np.maximum(
        0.0,
        np.minimum(ends, anchor_end) - np.maximum(starts, anchor_start),
    )
    return intersection / np.maximum(ends - starts, 1e-8)


def gaussian_fuse_candidates(
    group: pd.DataFrame,
    overlap_threshold: float,
    temperature: float,
    overlap_mode: str,
    per_model_topk: int,
) -> np.ndarray:
    selected = pd.concat(
        [
            part.nlargest(per_model_topk, "fusion_score")
            for _, part in group.groupby("model")
        ],
        ignore_index=True,
    )
    values = selected[
        ["t_start", "t_end", "fusion_score"]
    ].to_numpy(np.float64)
    remaining = np.arange(len(values), dtype=np.int64)
    fused = []
    while len(remaining):
        scores = values[remaining, 2]
        anchor_position = int(np.argmax(scores))
        anchor_index = int(remaining[anchor_position])
        overlaps = overlap_with_anchor(
            values[remaining, 0],
            values[remaining, 1],
            float(values[anchor_index, 0]),
            float(values[anchor_index, 1]),
            overlap_mode,
        )
        members_mask = overlaps > overlap_threshold
        members_mask[anchor_position] = True
        members = remaining[members_mask]
        weights = stable_softmax(values[members, 2], temperature)
        fused.append(
            [
                float(np.sum(values[members, 0] * weights)),
                float(np.sum(values[members, 1] * weights)),
                float(np.sum(values[members, 2] * weights)),
            ]
        )
        remaining = remaining[~members_mask]
    return np.asarray(fused, dtype=np.float64)


def build_gaussian_prediction(
    frames: list[pd.DataFrame],
    overlap_threshold: float,
    temperature: float,
    overlap_mode: str,
    per_model_topk: int,
    max_predictions: int,
) -> dict:
    candidates = pd.concat(frames, ignore_index=True)
    candidates["fusion_score"] = (
        candidates["rank_score"] * candidates["model"].map(MODEL_WEIGHTS)
    )
    results = {
        str(recording): {}
        for recording in sorted(candidates["rec_name"].astype(str).unique())
    }
    for (recording, roi_id), group in candidates.groupby(["rec_name", "roi_id"]):
        fused = gaussian_fuse_candidates(
            group,
            overlap_threshold,
            temperature,
            overlap_mode,
            per_model_topk,
        )
        fused = fused[np.argsort(-fused[:, 2])][:max_predictions]
        results[str(recording)][str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(score),
            }
            for start, end, score in fused
            if end - start >= 2.0
        ]
    return {
        "version": (
            "gaussian-instance-fusion-qfl-source-"
            f"{overlap_mode}-h{overlap_threshold:g}-t{temperature:g}"
        ),
        "results": results,
    }


def selected_fusion_candidates(
    group: pd.DataFrame,
    per_model_topk: int,
) -> pd.DataFrame:
    selected = pd.concat(
        [
            part.nlargest(per_model_topk, "fusion_score")
            for _, part in group.groupby("model")
        ],
        ignore_index=True,
    )
    return selected


def refine_control_prediction(
    control: dict,
    frames: list[pd.DataFrame],
    overlap_threshold: float,
    temperature: float,
    blend: float,
    per_model_topk: int,
    min_models: int,
) -> dict:
    candidates = pd.concat(frames, ignore_index=True)
    candidates["fusion_score"] = (
        candidates["rank_score"] * candidates["model"].map(MODEL_WEIGHTS)
    )
    by_roi = {
        (str(recording), int(roi_id)): selected_fusion_candidates(
            group, per_model_topk
        )
        for (recording, roi_id), group in candidates.groupby(
            ["rec_name", "roi_id"]
        )
    }
    results: dict[str, dict[str, list[dict]]] = {}
    for recording, rois in control["results"].items():
        results[str(recording)] = {}
        for roi_id, detections in rois.items():
            support = by_roi.get((str(recording), int(roi_id)))
            refined = []
            for detection in detections:
                start, end = map(float, detection["segment"])
                new_start, new_end = start, end
                if support is not None and not support.empty:
                    overlaps = temporal_iou(
                        support["t_start"].to_numpy(np.float64),
                        support["t_end"].to_numpy(np.float64),
                        start,
                        end,
                    )
                    eligible = overlaps > overlap_threshold
                    matched = support[eligible]
                    if matched["model"].nunique() >= min_models:
                        weights = stable_softmax(
                            matched["fusion_score"].to_numpy(np.float64),
                            temperature,
                        )
                        fused_start = float(
                            np.sum(
                                matched["t_start"].to_numpy(np.float64) * weights
                            )
                        )
                        fused_end = float(
                            np.sum(
                                matched["t_end"].to_numpy(np.float64) * weights
                            )
                        )
                        candidate_start = (
                            (1.0 - blend) * start + blend * fused_start
                        )
                        candidate_end = (1.0 - blend) * end + blend * fused_end
                        if candidate_end - candidate_start >= 2.0:
                            new_start, new_end = candidate_start, candidate_end
                refined.append(
                    {
                        "label": "ed",
                        "segment": [new_start, new_end],
                        "score": float(detection["score"]),
                    }
                )
            results[str(recording)][str(roi_id)] = refined
    return {
        "version": (
            "gaussian-conservative-boundary-refinement-"
            f"h{overlap_threshold:g}-t{temperature:g}-w{blend:g}"
        ),
        "results": results,
    }


def control_prediction_path(root: Path, fold: int) -> Path:
    return root / "predictions" / f"fold{fold:02d}_cw0.2_ew0.4_pw0.4.json"


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in metrics.groupby(
        ["overlap_mode", "overlap_threshold", "temperature", "blend"],
        dropna=False,
        sort=False,
    ):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        rows.append(
            {
                "overlap_mode": key[0],
                "overlap_threshold": key[1],
                "temperature": key[2],
                "blend": key[3],
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mAP", ascending=False)


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(resolve(args.candidate_features))
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    control_root = resolve(args.control_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    device = torch.device("cpu")

    for fold in args.folds:
        train = candidates[candidates["fold"] != fold].copy()
        validation = candidates[candidates["fold"] == fold].copy()
        qfl_model = fit_linear_qfl(
            train,
            device,
            args.qfl_steps,
            args.qfl_learning_rate,
        )
        proposal = score_quality_head(
            validation,
            qfl_model,
            args.score_blend,
        )
        frames = [
            prediction_rows(best_prediction(continuous_root, fold), "continuous"),
            prediction_rows(best_prediction(event_root, fold), "event"),
            proposal,
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        control = json.loads(
            control_prediction_path(control_root, fold).read_text(encoding="utf-8")
        )
        control_metrics = evaluate(
            control,
            recordings,
            resolve(args.ann_path),
            out_dir / "predictions" / f"fold_{fold:02d}_control.json",
        )
        rows.append(
            {
                "fold": fold,
                "variant": "control",
                "overlap_mode": "control",
                "overlap_threshold": -1.0,
                "temperature": -1.0,
                "blend": -1.0,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **control_metrics,
            }
        )
        for threshold in args.refine_overlap_thresholds:
            for temperature in args.refine_temperatures:
                for blend in args.refine_blends:
                    variant = (
                        f"refine_h{threshold:g}_t{temperature:g}_w{blend:g}"
                    )
                    prediction = refine_control_prediction(
                        control,
                        frames,
                        threshold,
                        temperature,
                        blend,
                        args.per_model_topk,
                        args.refine_min_models,
                    )
                    rows.append(
                        {
                            "fold": fold,
                            "variant": variant,
                            "overlap_mode": "refine",
                            "overlap_threshold": threshold,
                            "temperature": temperature,
                            "blend": blend,
                            "val_ed_instances": int(
                                manifest.loc[fold, "val_ed_instances"]
                            ),
                            **evaluate(
                                prediction,
                                recordings,
                                resolve(args.ann_path),
                                out_dir
                                / "predictions"
                                / f"fold_{fold:02d}_{variant}.json",
                            ),
                        }
                    )
        for mode in args.overlap_modes:
            for threshold in args.overlap_thresholds:
                for temperature in args.temperatures:
                    variant = f"{mode}_h{threshold:g}_t{temperature:g}"
                    prediction = build_gaussian_prediction(
                        frames,
                        threshold,
                        temperature,
                        mode,
                        args.per_model_topk,
                        args.max_predictions,
                    )
                    rows.append(
                        {
                            "fold": fold,
                            "variant": variant,
                            "overlap_mode": mode,
                            "overlap_threshold": threshold,
                            "temperature": temperature,
                            "blend": 1.0,
                            "val_ed_instances": int(
                                manifest.loc[fold, "val_ed_instances"]
                            ),
                            **evaluate(
                                prediction,
                                recordings,
                                resolve(args.ann_path),
                                out_dir
                                / "predictions"
                                / f"fold_{fold:02d}_{variant}.json",
                            ),
                        }
                    )
        pd.DataFrame(rows).to_csv(out_dir / "metrics_partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
