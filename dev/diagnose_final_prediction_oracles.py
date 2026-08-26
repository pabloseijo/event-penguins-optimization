"""Measure ranking and boundary ceilings of the final selected detections."""

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

from dev.diagnose_atsn_linear_separability import load_annotations  # noqa: E402
from dev.eval_continuous_multi_rep_fusion_cv import evaluate  # noqa: E402
from src.evaluation import segment_iou  # noqa: E402


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
        "--test-prediction",
        default="tmp/temporalmaxer_continuous/actionness_qfl_test_v1/predictions.json",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/final_prediction_oracles_v1",
    )
    parser.add_argument("--localization-min-tiou", type=float, default=0.1)
    return parser.parse_args()


def best_match(
    detection: dict,
    targets: list[tuple[float, float]],
) -> tuple[float, tuple[float, float] | None]:
    if not targets:
        return 0.0, None
    segments = np.asarray(targets, dtype=np.float64)
    overlaps = segment_iou(
        np.asarray(detection["segment"], dtype=np.float64),
        segments,
    )
    index = int(np.argmax(overlaps))
    return float(overlaps[index]), targets[index]


def oracle_prediction(
    prediction: dict,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
    mode: str,
    localization_min_tiou: float,
) -> dict:
    if mode not in {"control", "boundary", "ranking", "joint"}:
        raise ValueError(f"Unknown oracle mode: {mode}")
    output = copy.deepcopy(prediction)
    output["version"] = f"diagnostic-{mode}-oracle"
    for recording, rois in output["results"].items():
        for roi_id, detections in rois.items():
            targets = annotations.get((recording, int(roi_id)), [])
            for detection in detections:
                overlap, match = best_match(detection, targets)
                original_score = float(detection["score"])
                if mode in {"boundary", "joint"} and overlap >= localization_min_tiou:
                    detection["segment"] = [float(match[0]), float(match[1])]
                if mode in {"ranking", "joint"}:
                    detection["score"] = overlap + 1e-9 * original_score
    return output


def source_prediction_path(root: Path, fold: int) -> Path:
    return root / "predictions" / f"fold{fold:02d}_cw0.2_ew0.4_pw0.4.json"


def main() -> None:
    args = parse_args()
    annotations = load_annotations(resolve(args.ann_path), min_duration_s=2.0)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    source_root = resolve(args.source_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ("control", "boundary", "ranking", "joint")

    source_rows = []
    for fold in range(5):
        prediction = json.loads(
            source_prediction_path(source_root, fold).read_text(encoding="utf-8")
        )
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for mode in modes:
            transformed = oracle_prediction(
                prediction,
                annotations,
                mode,
                args.localization_min_tiou,
            )
            source_rows.append(
                {
                    "fold": fold,
                    "mode": mode,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        transformed,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "source" / f"fold_{fold:02d}_{mode}.json",
                    ),
                }
            )
    source_metrics = pd.DataFrame(source_rows)
    source_metrics.to_csv(out_dir / "source_fold_metrics.csv", index=False)
    source_summary = []
    for mode, group in source_metrics.groupby("mode", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        source_summary.append(
            {
                "mode": mode,
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    source_summary_frame = pd.DataFrame(source_summary)
    source_summary_frame.to_csv(out_dir / "source_summary.csv", index=False)

    test_prediction = json.loads(
        resolve(args.test_prediction).read_text(encoding="utf-8")
    )
    test_recordings = sorted(test_prediction["results"])
    test_rows = []
    for mode in modes:
        transformed = oracle_prediction(
            test_prediction,
            annotations,
            mode,
            args.localization_min_tiou,
        )
        test_rows.append(
            {
                "mode": mode,
                **evaluate(
                    transformed,
                    test_recordings,
                    resolve(args.ann_path),
                    out_dir / "test" / f"{mode}.json",
                ),
            }
        )
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    print("Source")
    print(source_summary_frame.to_string(index=False))
    print("\nHistorical test diagnostic")
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
