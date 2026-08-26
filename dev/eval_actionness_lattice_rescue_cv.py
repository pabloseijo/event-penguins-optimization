"""Rescue high-recall lattice candidates with OOF actionness completeness."""

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

from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import extract_actionness  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402
from src.utils import temporal_soft_nms  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def interval_means(prefix: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    lengths = np.maximum(ends - starts, 1)
    return (prefix[ends] - prefix[starts]) / lengths


def score_lattice(
    frame: pd.DataFrame,
    actionness: dict[tuple[str, int], np.ndarray],
    stride_s: float,
    completeness_weight: float,
    context_ratio: float,
) -> pd.DataFrame:
    output = frame.copy()
    output["roi_id"] = (
        output["roi_id"].astype(str).str.lstrip("N").astype(int)
    )
    output["t_start"] = output["t_start"].to_numpy(np.float64) / 1e6
    output["t_end"] = output["t_end"].to_numpy(np.float64) / 1e6
    completeness = np.zeros(len(output), dtype=np.float64)
    for (recording, roi_id), positions in output.groupby(
        ["rec_name", "roi_id"], sort=False
    ).groups.items():
        positions = np.asarray(list(positions), dtype=np.int64)
        sequence = actionness[(str(recording), int(roi_id))].astype(np.float64)
        prefix = np.concatenate(([0.0], np.cumsum(sequence)))
        starts = np.floor(
            output.loc[positions, "t_start"].to_numpy(np.float64) / stride_s
        ).astype(np.int64)
        ends = np.ceil(
            output.loc[positions, "t_end"].to_numpy(np.float64) / stride_s
        ).astype(np.int64)
        starts = np.clip(starts, 0, len(sequence) - 1)
        ends = np.clip(ends, starts + 1, len(sequence))
        inside = interval_means(prefix, starts, ends)
        context = np.maximum(1, np.rint((ends - starts) * context_ratio).astype(np.int64))
        left_starts = np.maximum(0, starts - context)
        right_ends = np.minimum(len(sequence), ends + context)
        left = interval_means(prefix, left_starts, starts)
        right = interval_means(prefix, ends, right_ends)
        # At ROI edges, the unavailable side is neutral rather than artificial background.
        left[starts == 0] = inside[starts == 0]
        right[ends == len(sequence)] = inside[ends == len(sequence)]
        completeness[positions] = inside - 0.5 * (left + right)
    output["completeness"] = completeness
    quality_rank = output["quality_score"].rank(method="average", pct=True)
    completeness_rank = output["completeness"].rank(method="average", pct=True)
    output["lattice_score"] = (
        (1.0 - completeness_weight) * quality_rank
        + completeness_weight * completeness_rank
    )
    return output


def lattice_prediction(
    scored: pd.DataFrame,
    pre_nms_topk: int,
    max_predictions: int,
    sigma: float,
) -> dict:
    results: dict[str, dict[str, list]] = {}
    for (recording, roi_id), values in scored.groupby(
        ["rec_name", "roi_id"], sort=False
    ):
        if pre_nms_topk > 0 and len(values) > pre_nms_topk:
            values = values.nlargest(pre_nms_topk, "lattice_score")
        detections = temporal_soft_nms(
            values[["t_start", "t_end", "lattice_score"]].to_numpy(np.float64),
            sigma=sigma,
            score_threshold=1e-5,
        )[:max_predictions]
        results.setdefault(str(recording), {})[str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(start), float(end)],
                "score": float(score),
            }
            for start, end, score in detections
            if end - start >= 2.0
        ]
    return {"version": "actionness-completeness-lattice-rescue-v1", "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1")
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--qhead-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/actionness_lattice_rescue_cv_v1"
    )
    parser.add_argument("--completeness-weight", type=float, default=0.25)
    parser.add_argument("--context-ratio", type=float, default=0.5)
    parser.add_argument("--lattice-weights", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--pre-nms-topk", type=int, default=500)
    parser.add_argument("--max-predictions", type=int, default=200)
    parser.add_argument("--nms-sigma", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    feature_dir = resolve(args.feature_dir)
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    proposal_root = resolve(args.proposal_root)
    qhead_root = resolve(args.qhead_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    input_dim = int(metadata["feature_dim"])
    stride_s = float(metadata["grid_stride_s"])
    device = torch.device(args.device)
    rows = []

    for fold in range(5):
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        selected = sequences[sequences["rec_name"].isin(recordings)].copy()
        models = load_models(
            continuous_root,
            input_dim,
            device,
            [continuous_root / f"fold_{fold:02d}" / "best.pt"],
        )
        actionness = extract_actionness(
            models,
            make_loader(feature_dir, selected, args.batch_size, args.num_workers, device),
            device,
        )
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
        lattice = pd.read_csv(
            qhead_root
            / f"fold_{fold:02d}"
            / "cache"
            / "val_scores_qhead_qfl_only.csv",
            usecols=[
                "rec_name",
                "roi_id",
                "t_start",
                "t_end",
                "quality_score",
            ],
        )
        scored = score_lattice(
            lattice,
            actionness,
            stride_s,
            args.completeness_weight,
            args.context_ratio,
        )
        lattice_json = lattice_prediction(
            scored, args.pre_nms_topk, args.max_predictions, args.nms_sigma
        )
        lattice_path = out_dir / "predictions" / f"fold_{fold:02d}_lattice.json"
        rows.append(
            {
                "fold": fold,
                "variant": "lattice_standalone",
                "lattice_weight": 1.0,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **evaluate(
                    lattice_json,
                    recordings,
                    resolve(args.ann_path),
                    lattice_path,
                ),
            }
        )
        frames = [
            prediction_rows(best_prediction(continuous_root, fold), "continuous"),
            prediction_rows(best_prediction(event_root, fold), "event"),
            prediction_rows(
                json.loads(
                    (
                        proposal_root
                        / f"fold_{fold:02d}"
                        / "predictions"
                        / f"{args.proposal_variant}.json"
                    ).read_text(encoding="utf-8")
                ),
                "proposal",
            ),
            prediction_rows(lattice_json, "lattice"),
        ]
        for lattice_weight in args.lattice_weights:
            old = 1.0 - lattice_weight
            prediction = build_prediction(
                frames,
                {
                    "continuous": 0.2 * old,
                    "event": 0.4 * old,
                    "proposal": 0.4 * old,
                    "lattice": lattice_weight,
                },
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            suffix = f"w{int(round(lattice_weight * 100)):03d}"
            rows.append(
                {
                    "fold": fold,
                    "variant": f"fusion_lattice_{suffix}",
                    "lattice_weight": lattice_weight,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_fusion_{suffix}.json",
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, values in metrics.groupby("variant", sort=False):
        instance_weights = values["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "mean_mAP": float(values["mAP"].mean()),
                "weighted_mAP": float(np.average(values["mAP"], weights=instance_weights)),
                "worst_mAP": float(values["mAP"].min()),
                "mean_AP@0.1": float(values["AP@0.1"].mean()),
                "mean_AP@0.3": float(values["AP@0.3"].mean()),
                "mean_AP@0.5": float(values["AP@0.5"].mean()),
                "mean_AP@0.7": float(values["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
