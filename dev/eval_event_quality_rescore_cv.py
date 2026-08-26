"""OOF evaluation of label-free event quality on fused temporal detections."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from dev.eval_continuous_multi_rep_fusion_cv import evaluate


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_cv_v1/predictions",
    )
    parser.add_argument(
        "--event-dir", default="tmp/temporalmaxer_continuous/source_event_stats_v1"
    )
    parser.add_argument(
        "--sequence-path",
        default="tmp/temporalmaxer_continuous/source_features_v1/sequences.csv",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/event_quality_rescore_cv_v1",
    )
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(np.float64)


def quality_by_roi(
    event_dir: Path, sequence_path: Path
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], float]:
    metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
    names = list(metadata["feature_names"])
    count_index = names.index("log_event_count")
    spectral_index = names.index("spectral_energy_ratio")
    matrix = np.load(event_dir / "event_stats.npy", mmap_mode="r")
    sequences = pd.read_csv(sequence_path)
    result = {}
    for row in sequences.itertuples(index=False):
        start = int(row.offset)
        values = np.asarray(matrix[start : start + int(row.length)], dtype=np.float64)
        count = percentile_rank(values[:, count_index])
        low_spectral = 1.0 - percentile_rank(values[:, spectral_index])
        result[(str(row.rec_name), int(row.roi_id))] = (count, low_spectral)
    return result, float(metadata["grid_stride_s"])


def segment_quality(
    point_quality: np.ndarray, start: float, end: float, stride_s: float
) -> float:
    centers = (np.arange(len(point_quality), dtype=np.float64) + 0.5) * stride_s
    selected = (centers >= start) & (centers <= end)
    if not selected.any():
        center_index = int(np.clip(round((0.5 * (start + end)) / stride_s - 0.5), 0, len(centers) - 1))
        return float(point_quality[center_index])
    return float(point_quality[selected].mean())


def rescore_prediction(
    prediction: dict,
    qualities: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    stride_s: float,
    variant: str,
    weight: float,
) -> dict:
    result = copy.deepcopy(prediction)
    for recording, rois in result["results"].items():
        for roi_id, detections in rois.items():
            count, low_spectral = qualities[(str(recording), int(roi_id))]
            point_quality = count if variant == "count" else 0.5 * (count + low_spectral)
            for detection in detections:
                start, end = map(float, detection["segment"])
                quality = segment_quality(point_quality, start, end, stride_s)
                detection["score"] = float(detection["score"]) * (
                    (1.0 - weight) + weight * quality
                )
    result["version"] = f"event-quality-{variant}-weight-{weight:g}"
    return result


def main() -> None:
    args = parse_args()
    if any(not 0.0 <= weight <= 1.0 for weight in args.weights):
        raise ValueError("weights must be in [0,1]")
    qualities, stride_s = quality_by_roi(resolve(args.event_dir), resolve(args.sequence_path))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(5):
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        source_path = (
            resolve(args.prediction_root)
            / f"fold{fold:02d}_cw0.2_ew0.4_pw0.4.json"
        )
        prediction = json.loads(source_path.read_text(encoding="utf-8"))
        for variant in ("count", "count_spectral"):
            for weight in args.weights:
                rescored = rescore_prediction(
                    prediction, qualities, stride_s, variant, weight
                )
                metrics = evaluate(
                    rescored,
                    recordings,
                    resolve(args.ann_path),
                    out_dir / "predictions" / f"fold{fold:02d}_{variant}_w{weight:g}.json",
                )
                rows.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "weight": weight,
                        "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                        **metrics,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for (variant, weight), group in frame.groupby(["variant", "weight"]):
        instance_weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "weight": weight,
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
