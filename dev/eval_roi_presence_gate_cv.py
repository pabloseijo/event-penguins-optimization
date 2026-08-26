"""Cross-fit MIL-style ROI presence calibration for a frozen TAD prediction."""

from __future__ import annotations

import argparse
import copy
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


FEATURE_COLUMNS = [
    "log_count",
    "score_max",
    "score_mean",
    "score_std",
    "score_q50",
    "score_q75",
    "score_q90",
    "score_q95",
    "score_q99",
    "top3_mean",
    "top10_mean",
    "top25_mean",
    "top10_mass_fraction",
    "count_above_005",
    "count_above_010",
    "count_above_020",
    "top_duration",
    "score_weighted_duration",
]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_pal_consistency_blend_cv_v1/predictions"
        ),
    )
    parser.add_argument(
        "--prediction-template",
        default="fold{fold:02d}_cltdr0.1.json",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/roi_presence_gate_cv_v1",
    )
    parser.add_argument(
        "--gate-strengths",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--l2", type=float, default=0.1)
    return parser.parse_args()


def top_mean(values: np.ndarray, count: int) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.sort(values)[-count:].mean())


def roi_bag_features(detections: list[dict]) -> dict[str, float]:
    """Summarize a temporal detection bag without using labels or identity."""
    if not detections:
        return {name: 0.0 for name in FEATURE_COLUMNS}
    scores = np.asarray(
        [float(item["score"]) for item in detections], dtype=np.float64
    )
    durations = np.asarray(
        [
            float(item["segment"][1]) - float(item["segment"][0])
            for item in detections
        ],
        dtype=np.float64,
    )
    order = np.argsort(scores)
    score_mass = float(scores.sum())
    return {
        "log_count": float(np.log1p(len(scores))),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_q50": float(np.quantile(scores, 0.50)),
        "score_q75": float(np.quantile(scores, 0.75)),
        "score_q90": float(np.quantile(scores, 0.90)),
        "score_q95": float(np.quantile(scores, 0.95)),
        "score_q99": float(np.quantile(scores, 0.99)),
        "top3_mean": top_mean(scores, 3),
        "top10_mean": top_mean(scores, 10),
        "top25_mean": top_mean(scores, 25),
        "top10_mass_fraction": (
            float(np.sort(scores)[-10:].sum() / score_mass)
            if score_mass > 0
            else 0.0
        ),
        "count_above_005": float(np.mean(scores >= 0.05)),
        "count_above_010": float(np.mean(scores >= 0.10)),
        "count_above_020": float(np.mean(scores >= 0.20)),
        "top_duration": float(durations[order[-1]]),
        "score_weighted_duration": (
            float(np.average(durations, weights=scores))
            if score_mass > 0
            else float(durations.mean())
        ),
    }


def has_ed(annotations: list[dict], min_duration: float = 2.0) -> bool:
    return any(
        item["label"] == "ed"
        and float(item["segment"][1]) - float(item["segment"][0]) >= min_duration
        for item in annotations
    )


def prediction_bags(
    prediction: dict,
    annotations: dict,
    fold: int,
) -> pd.DataFrame:
    rows = []
    for recording, rois in prediction["results"].items():
        recording_annotations = annotations[recording].get("annotations", {})
        for roi, detections in rois.items():
            rows.append(
                {
                    "fold": fold,
                    "rec_name": recording,
                    "roi_id": int(roi),
                    "target_present": float(
                        has_ed(recording_annotations.get(str(roi), []))
                    ),
                    **roi_bag_features(detections),
                }
            )
    return pd.DataFrame(rows)


def fit_presence_model(
    frame: pd.DataFrame,
    l2: float,
) -> dict[str, np.ndarray | float]:
    if l2 < 0:
        raise ValueError("Presence-model L2 must be non-negative")
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    targets = torch.from_numpy(
        frame["target_present"].to_numpy(dtype=np.float32)
    )
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = torch.from_numpy(((features - mean) / scale).astype(np.float32))
    weights = torch.zeros(normalized.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        max_iter=100,
        tolerance_grad=1e-8,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = normalized @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        ) + l2 * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights.detach().numpy().astype(np.float64),
        "bias": float(bias.detach()),
    }


def presence_probabilities(frame: pd.DataFrame, model: dict) -> np.ndarray:
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    normalized = (features - model["mean"]) / model["scale"]
    logits = normalized @ model["weights"] + float(model["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    positives = targets > 0.5
    negative_count = int((~positives).sum())
    positive_count = int(positives.sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def apply_presence_gate(
    prediction: dict,
    probabilities: pd.DataFrame,
    strength: float,
) -> dict:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Presence gate strength must lie in [0,1]")
    output = copy.deepcopy(prediction)
    probability_by_roi = {
        (str(row.rec_name), int(row.roi_id)): float(row.presence_probability)
        for row in probabilities.itertuples(index=False)
    }
    for recording, rois in output["results"].items():
        for roi, detections in rois.items():
            probability = probability_by_roi[(recording, int(roi))]
            gate = (1.0 - strength) + strength * probability
            for detection in detections:
                detection["score"] = float(detection["score"]) * gate
    output["version"] = f"cross-fit-roi-presence-gate-strength-{strength:g}"
    return output


def main() -> None:
    args = parse_args()
    if any(value < 0 or value > 1 for value in args.gate_strengths):
        raise ValueError("Every gate strength must lie in [0,1]")
    annotation_path = resolve(args.ann_path)
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))[
        "database"
    ]
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    root = resolve(args.prediction_root)
    predictions = {}
    bags = []
    for fold in range(5):
        prediction = json.loads(
            (root / args.prediction_template.format(fold=fold)).read_text(
                encoding="utf-8"
            )
        )
        predictions[fold] = prediction
        bags.append(prediction_bags(prediction, annotations, fold))
    bags = pd.concat(bags, ignore_index=True)

    probability_frames = []
    fold_auc = []
    for fold in range(5):
        train = bags[bags["fold"] != fold]
        validation = bags[bags["fold"] == fold].copy()
        model = fit_presence_model(train, args.l2)
        validation["presence_probability"] = presence_probabilities(
            validation, model
        )
        probability_frames.append(validation)
        fold_auc.append(
            {
                "fold": fold,
                "presence_auc": roc_auc(
                    validation["target_present"].to_numpy(),
                    validation["presence_probability"].to_numpy(),
                ),
            }
        )
    probabilities = pd.concat(probability_frames, ignore_index=True)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(out_dir / "oof_presence_probabilities.csv", index=False)
    pd.DataFrame(fold_auc).to_csv(out_dir / "presence_auc.csv", index=False)

    rows = []
    for fold in range(5):
        fold_probabilities = probabilities[probabilities["fold"] == fold]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        for strength in args.gate_strengths:
            adjusted = apply_presence_gate(
                predictions[fold],
                fold_probabilities,
                float(strength),
            )
            rows.append(
                {
                    "fold": fold,
                    "gate_strength": float(strength),
                    "presence_auc": fold_auc[fold]["presence_auc"],
                    "val_ed_instances": int(
                        manifest.loc[fold, "val_ed_instances"]
                    ),
                    **evaluate(
                        adjusted,
                        recordings,
                        annotation_path,
                        out_dir
                        / "predictions"
                        / f"fold{fold:02d}_strength{strength:g}.json",
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for strength, group in metrics.groupby("gate_strength"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "gate_strength": float(strength),
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(
                    np.average(group["mAP"], weights=weights)
                ),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
                "mean_presence_auc": float(group["presence_auc"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        "mean_mAP", ascending=False
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
