"""Rescore final detections by label-free recording background novelty."""

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

from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_final_boundary_gradient_cv import (  # noqa: E402
    frame_prediction,
    prediction_frame,
)
from dev.eval_proposal_actionness_rescore_cv import extract_actionness  # noqa: E402
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402
from dev.diagnose_final_prediction_oracles import source_prediction_path  # noqa: E402


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
        "--feature-dir",
        default="tmp/temporalmaxer_continuous/source_features_v1",
    )
    parser.add_argument(
        "--continuous-root",
        default="tmp/temporalmaxer_continuous/cv_recipe_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/recording_background_prototype_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--background-quantile", type=float, default=0.5)
    parser.add_argument(
        "--novelty-modes",
        nargs="+",
        default=["roi", "recording", "mean"],
    )
    parser.add_argument("--novelty-weights", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalized_rows(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def unit_centroid(features: np.ndarray) -> np.ndarray:
    centroid = normalized_rows(features).mean(axis=0)
    return centroid / max(float(np.linalg.norm(centroid)), 1e-8)


def cosine_novelty(vector: np.ndarray, prototype: np.ndarray) -> float:
    normalized = vector / max(float(np.linalg.norm(vector)), 1e-8)
    return float(1.0 - np.clip(normalized @ prototype, -1.0, 1.0))


def add_background_novelty(
    frame: pd.DataFrame,
    features: np.ndarray,
    sequences: pd.DataFrame,
    actionness: dict[tuple[str, int], np.ndarray],
    stride_seconds: float,
    background_quantile: float,
) -> pd.DataFrame:
    sequence_features = {}
    roi_prototypes = {}
    recording_background = {}
    for row in sequences.itertuples(index=False):
        key = (str(row.rec_name), int(row.roi_id))
        values = np.asarray(
            features[int(row.offset) : int(row.offset) + int(row.length)],
            dtype=np.float32,
        )
        sequence_features[key] = values
        scores = actionness[key][: len(values)]
        threshold = float(np.quantile(scores, background_quantile))
        selected = values[scores <= threshold]
        if len(selected) == 0:
            selected = values
        roi_prototypes[key] = unit_centroid(selected)
        recording_background.setdefault(str(row.rec_name), []).append(selected)
    recording_prototypes = {
        recording: unit_centroid(np.concatenate(parts, axis=0))
        for recording, parts in recording_background.items()
    }

    roi_novelty = np.zeros(len(frame), dtype=np.float64)
    recording_novelty = np.zeros(len(frame), dtype=np.float64)
    for index, row in enumerate(frame.itertuples(index=False)):
        key = (str(row.rec_name), int(row.roi_id))
        values = sequence_features[key]
        start = max(0, int(np.floor(float(row.t_start) / stride_seconds)))
        end = min(len(values), int(np.ceil(float(row.t_end) / stride_seconds)))
        if end <= start:
            center = min(
                len(values) - 1,
                max(
                    0,
                    int(
                        round(
                            0.5
                            * (float(row.t_start) + float(row.t_end))
                            / stride_seconds
                        )
                    ),
                ),
            )
            candidate = values[center]
        else:
            candidate = normalized_rows(values[start:end]).mean(axis=0)
        roi_novelty[index] = cosine_novelty(candidate, roi_prototypes[key])
        recording_novelty[index] = cosine_novelty(
            candidate,
            recording_prototypes[str(row.rec_name)],
        )
    output = frame.copy()
    output["roi_background_novelty"] = roi_novelty
    output["recording_background_novelty"] = recording_novelty
    output["mean_background_novelty"] = 0.5 * (
        roi_novelty + recording_novelty
    )
    return output


def novelty_rescore(
    frame: pd.DataFrame,
    novelty_column: str,
    weight: float,
) -> pd.DataFrame:
    original_rank = frame["score"].rank(
        method="average",
        pct=True,
    ).to_numpy(np.float64)
    novelty_rank = frame[novelty_column].rank(
        method="average",
        pct=True,
    ).to_numpy(np.float64)
    output = frame.copy()
    output["score"] = (1.0 - weight) * original_rank + weight * novelty_rank
    return output


def main() -> None:
    args = parse_args()
    if not 0.0 < args.background_quantile < 1.0:
        raise ValueError("background-quantile must be in (0, 1)")
    feature_dir = resolve(args.feature_dir)
    continuous_root = resolve(args.continuous_root)
    source_root = resolve(args.source_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    features = np.load(feature_dir / "frame_features.npy", mmap_mode="r")
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    device = torch.device(args.device)
    rows = []

    for fold in args.folds:
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        selected_sequences = sequences[
            sequences["rec_name"].isin(recordings)
        ].copy()
        models = load_models(
            continuous_root,
            int(metadata["feature_dim"]),
            device,
            [continuous_root / f"fold_{fold:02d}" / "best.pt"],
        )
        actionness = extract_actionness(
            models,
            make_loader(
                feature_dir,
                selected_sequences,
                args.batch_size,
                args.num_workers,
                device,
            ),
            device,
        )
        control = json.loads(
            source_prediction_path(source_root, fold).read_text(encoding="utf-8")
        )
        control_frame = prediction_frame(control, fold)
        cached = out_dir / f"fold_{fold:02d}_novelty.csv"
        if cached.exists():
            novelty_frame = pd.read_csv(cached)
        else:
            novelty_frame = add_background_novelty(
                control_frame,
                features,
                selected_sequences,
                actionness,
                float(metadata["grid_stride_s"]),
                args.background_quantile,
            )
            novelty_frame.to_csv(cached, index=False)
        variants = [("control", control_frame)]
        for mode in args.novelty_modes:
            column = f"{mode}_background_novelty"
            for weight in args.novelty_weights:
                variant = f"{mode}_novelty_w{int(round(100 * weight)):03d}"
                variants.append(
                    (
                        variant,
                        novelty_rescore(novelty_frame, column, weight),
                    )
                )
        for variant, frame in variants:
            prediction = frame_prediction(
                frame,
                f"source-recording-background-{variant}",
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_{variant}.json",
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / "fold_metrics_partial.csv", index=False)
        print(f"Completed fold {fold}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "folds": len(group),
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
