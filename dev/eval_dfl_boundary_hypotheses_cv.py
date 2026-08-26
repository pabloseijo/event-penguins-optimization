"""Evaluate conservative DFL boundary hypotheses on canonical predictions.

Each selected canonical detection is preserved. A lower-scored DFL-refined
copy is added as a second localization hypothesis, following the motivation of
distributional/multi-hypothesis boundary decoders such as TriDet.
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
        "--transfer-root",
        default="tmp/temporalmaxer_continuous/current_qfl_boundary_transfer_cv_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/dfl_boundary_hypotheses_cv_v1",
    )
    parser.add_argument("--topk-per-roi", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--score-scales", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    return parser.parse_args()


def add_boundary_hypotheses(
    control: dict,
    refined: dict,
    topk_per_roi: int,
    score_scale: float,
) -> dict:
    if topk_per_roi < 1:
        raise ValueError("topk_per_roi must be positive")
    if not 0.0 < score_scale < 1.0:
        raise ValueError("score_scale must be in (0, 1)")
    output = json.loads(json.dumps(control))
    added = 0
    for recording, rois in output["results"].items():
        for roi_id, detections in rois.items():
            alternatives = refined["results"].get(recording, {}).get(roi_id, [])
            if len(detections) != len(alternatives):
                raise ValueError(f"Detection count changed for {recording}/{roi_id}")
            order = np.argsort(
                -np.asarray([float(item["score"]) for item in detections]),
                kind="stable",
            )[:topk_per_roi]
            hypotheses = []
            for index in order:
                original = detections[int(index)]
                alternative = alternatives[int(index)]
                if np.allclose(
                    original["segment"],
                    alternative["segment"],
                    rtol=0.0,
                    atol=1e-9,
                ):
                    continue
                hypotheses.append(
                    {
                        **alternative,
                        "score": float(original["score"]) * score_scale,
                    }
                )
            detections.extend(hypotheses)
            detections.sort(key=lambda item: float(item["score"]), reverse=True)
            added += len(hypotheses)
    output["version"] = (
        "source-dfl-boundary-hypotheses:"
        f"topk={topk_per_roi}:score-scale={score_scale:g}:added={added}"
    )
    return output


def main() -> None:
    args = parse_args()
    source_root = resolve(args.source_root)
    transfer_root = resolve(args.transfer_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    rows = []
    for fold in range(5):
        control = json.loads(
            source_prediction_path(source_root, fold).read_text(encoding="utf-8")
        )
        refined = json.loads(
            (
                transfer_root
                / f"fold_{fold:02d}"
                / "predictions"
                / "event_dfl_tiou0.5_blend0.5.json"
            ).read_text(encoding="utf-8")
        )
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        variants = [("control", control)]
        for topk in args.topk_per_roi:
            for scale in args.score_scales:
                variant = f"dfl_hyp_top{topk}_s{int(round(100 * scale)):02d}"
                variants.append(
                    (
                        variant,
                        add_boundary_hypotheses(
                            control,
                            refined,
                            topk,
                            scale,
                        ),
                    )
                )
        for variant, prediction in variants:
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

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
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
