"""Diagnose label-free reliability signals for DFL boundary transfer."""

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

from dev.diagnose_atsn_linear_separability import load_annotations  # noqa: E402
from dev.diagnose_recording_expert_reliability import (  # noqa: E402
    evaluate_recording,
    ground_truth_frame,
    rank_correlation,
)
from dev.eval_distributional_boundary_transfer_cv import transfer  # noqa: E402
from dev.diagnose_final_prediction_oracles import source_prediction_path  # noqa: E402
from dev.eval_distributional_boundary_transfer_cv import rows as voter_rows  # noqa: E402
from src.utils.detection import temporal_iou  # noqa: E402


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
        "--dfl-root",
        default="tmp/temporalmaxer_continuous/cv_eventstats_dfl_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--test-root",
        default="tmp/temporalmaxer_continuous/current_qfl_dfl_boundary_test_v1",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/dfl_transfer_reliability_v1",
    )
    parser.add_argument("--topk", type=int, default=25)
    return parser.parse_args()


def matched_detection_values(
    control: dict,
    transferred: dict,
    recording: str,
) -> pd.DataFrame:
    values = []
    for roi_id, detections in control["results"].get(recording, {}).items():
        refined = transferred["results"].get(recording, {}).get(roi_id, [])
        if len(detections) != len(refined):
            raise ValueError(f"Detection count changed for {recording}/{roi_id}")
        for original, updated in zip(detections, refined):
            start, end = map(float, original["segment"])
            new_start, new_end = map(float, updated["segment"])
            duration = max(end - start, 1e-6)
            agreement = float(
                temporal_iou(
                    np.asarray([new_start]),
                    np.asarray([new_end]),
                    start,
                    end,
                )[0]
            )
            values.append(
                {
                    "roi_id": int(roi_id),
                    "score": float(original["score"]),
                    "start_shift": (new_start - start) / duration,
                    "end_shift": (new_end - end) / duration,
                    "absolute_shift": (
                        abs(new_start - start) + abs(new_end - end)
                    )
                    / (2.0 * duration),
                    "duration_ratio": (new_end - new_start) / duration,
                    "agreement": agreement,
                }
            )
    return pd.DataFrame(values)


def boundary_transfer_features(
    control: dict,
    transferred: dict,
    recording: str,
    topk: int,
) -> dict[str, float | int | str]:
    frame = matched_detection_values(control, transferred, recording)
    if frame.empty:
        return {"rec_name": recording, "count": 0}
    frame["score_rank"] = frame["score"].rank(method="average", pct=True)
    top = frame.nlargest(min(topk, len(frame)), "score")
    changed = frame["absolute_shift"].to_numpy(np.float64) > 1e-9
    weights = frame["score_rank"].to_numpy(np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    return {
        "rec_name": recording,
        "count": len(frame),
        "roi_count": int(frame["roi_id"].nunique()),
        "changed_fraction": float(changed.mean()),
        "changed_score_weighted": float(np.average(changed, weights=weights)),
        "start_shift_mean": float(frame["start_shift"].mean()),
        "end_shift_mean": float(frame["end_shift"].mean()),
        "absolute_shift_mean": float(frame["absolute_shift"].mean()),
        "absolute_shift_q90": float(frame["absolute_shift"].quantile(0.90)),
        "duration_ratio_mean": float(frame["duration_ratio"].mean()),
        "agreement_mean": float(frame["agreement"].mean()),
        "agreement_frac07": float((frame["agreement"] >= 0.7).mean()),
        "top_absolute_shift_mean": float(top["absolute_shift"].mean()),
        "top_start_shift_mean": float(top["start_shift"].mean()),
        "top_end_shift_mean": float(top["end_shift"].mean()),
        "top_duration_ratio_mean": float(top["duration_ratio"].mean()),
        "top_agreement_mean": float(top["agreement"].mean()),
        "top_agreement_frac07": float((top["agreement"] >= 0.7).mean()),
    }


def recording_rows(
    control: dict,
    transferred: dict,
    fold: int,
    annotations,
    topk: int,
) -> list[dict]:
    result = []
    for recording in sorted(control["results"]):
        ground_truth = ground_truth_frame(annotations, recording)
        if ground_truth.empty:
            continue
        control_metrics = evaluate_recording(control, recording, ground_truth)
        transfer_metrics = evaluate_recording(transferred, recording, ground_truth)
        result.append(
            {
                "fold": fold,
                **boundary_transfer_features(
                    control,
                    transferred,
                    recording,
                    topk,
                ),
                "gt_instances": int(control_metrics["gt_instances"]),
                "control_mAP": float(control_metrics["mAP"]),
                "transfer_mAP": float(transfer_metrics["mAP"]),
                "delta_mAP": float(
                    transfer_metrics["mAP"] - control_metrics["mAP"]
                ),
                "control_AP@0.7": float(control_metrics["AP@0.7"]),
                "transfer_AP@0.7": float(transfer_metrics["AP@0.7"]),
                "delta_AP@0.7": float(
                    transfer_metrics["AP@0.7"] - control_metrics["AP@0.7"]
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    annotations = load_annotations(resolve(args.ann_path), min_duration_s=2.0)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    source_root = resolve(args.source_root)
    dfl_root = resolve(args.dfl_root)
    source_rows = []
    for fold in range(5):
        control = json.loads(
            source_prediction_path(source_root, fold).read_text(encoding="utf-8")
        )
        epoch = int(
            json.loads(
                (dfl_root / f"fold_{fold:02d}" / "metrics_best.json").read_text()
            )["epoch"]
        )
        voter = json.loads(
            (
                dfl_root
                / f"fold_{fold:02d}"
                / "predictions"
                / f"epoch_{epoch:03d}.json"
            ).read_text(encoding="utf-8")
        )
        transferred = transfer(
            control,
            voter_rows(voter, "event_dfl"),
            tiou=0.5,
            blend=0.5,
            topk=20,
        )
        recordings = set(str(manifest.loc[fold, "val_record_names"]).split())
        control["results"] = {
            key: value for key, value in control["results"].items() if key in recordings
        }
        transferred["results"] = {
            key: value
            for key, value in transferred["results"].items()
            if key in recordings
        }
        source_rows.extend(
            recording_rows(
                control,
                transferred,
                fold,
                annotations,
                args.topk,
            )
        )

    test_root = resolve(args.test_root)
    test_control = json.loads(
        (test_root / "control_predictions.json").read_text(encoding="utf-8")
    )
    test_transfer = json.loads(
        (test_root / "predictions.json").read_text(encoding="utf-8")
    )
    test_rows = recording_rows(
        test_control,
        test_transfer,
        -1,
        annotations,
        args.topk,
    )
    source = pd.DataFrame(source_rows)
    test = pd.DataFrame(test_rows)
    feature_columns = [
        column
        for column in source.columns
        if column
        not in {
            "fold",
            "rec_name",
            "gt_instances",
            "control_mAP",
            "transfer_mAP",
            "delta_mAP",
            "control_AP@0.7",
            "transfer_AP@0.7",
            "delta_AP@0.7",
        }
    ]
    correlations = pd.DataFrame(
        [
            {
                "feature": feature,
                "spearman_delta_mAP": rank_correlation(
                    source[feature],
                    source["delta_mAP"],
                ),
                "spearman_delta_AP@0.7": rank_correlation(
                    source[feature],
                    source["delta_AP@0.7"],
                ),
            }
            for feature in feature_columns
        ]
    ).sort_values(
        "spearman_delta_mAP",
        key=lambda values: values.abs(),
        ascending=False,
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(out_dir / "source_recordings.csv", index=False)
    test.to_csv(out_dir / "test_recordings_diagnostic.csv", index=False)
    correlations.to_csv(out_dir / "correlations.csv", index=False)
    print("Source correlations")
    print(correlations.to_string(index=False))
    print("\nSource recordings")
    print(source.to_string(index=False))
    print("\nHistorical test diagnostic")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
