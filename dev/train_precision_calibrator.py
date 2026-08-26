"""Precision recalibrator trained on a broad OOF hard-negative bank (all source recordings).

Diagnosis this targets: union recall across the 3 fusion experts is already ~82% on the
hard day-15 test domain, so the bottleneck is not coverage but ranking/precision — many
false positives outscore or crowd true positives. Earlier hard-negative oversampling used
only one source exemplar of that domain, which was not enough signal. This pools OOF false
positives from every source recording (all 5 CV folds = all 19 recordings) to fit a small
logistic recalibrator per expert, then applies it uniformly before fusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    l2: float = 1e-3,
    lr: float = 0.1,
    steps: int = 5000,
) -> tuple[np.ndarray, float]:
    """Balanced-class logistic regression via full-batch gradient descent (no sklearn on remote)."""
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    x = (features - mean) / std
    y = labels.astype(np.float64)
    n_pos = max(y.sum(), 1.0)
    n_neg = max(len(y) - y.sum(), 1.0)
    sample_weight = np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(steps):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = sample_weight * (probs - y)
        grad_w = x.T @ grad / len(y) + l2 * weights
        grad_b = grad.mean()
        weights -= lr * grad_w
        bias -= lr * grad_b
    # Fold standardization into raw-feature-space weights: (x-mean)/std @ w + b
    raw_weights = weights / std
    raw_bias = bias - float((mean / std) @ weights)
    return raw_weights, raw_bias


def predict_proba(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = features @ weights + bias
    return 1.0 / (1.0 + np.exp(-logits))


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def load_oof_predictions(cv_root: Path) -> dict:
    """Pool best-epoch (OOF) predictions across all 5 folds into one {rec: {roi: dets}}."""
    merged: dict[str, dict[str, list]] = {}
    for fold in range(5):
        fold_dir = cv_root / f"fold_{fold:02d}"
        best_epoch = int(json.loads((fold_dir / "metrics_best.json").read_text())["epoch"])
        pred = json.loads((fold_dir / "predictions" / f"epoch_{best_epoch:03d}.json").read_text())
        for rec, rois in pred["results"].items():
            merged.setdefault(rec, {}).update(rois)
    return merged


def build_tp_fp_dataset(
    predictions: dict, annotations: dict, iou_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for rec, rois in predictions.items():
        ann = annotations.get(rec, {}).get("annotations", {})
        for roi, dets in rois.items():
            gt = [
                tuple(item["segment"])
                for item in ann.get(roi, [])
                if item["label"] == "ed" and item["segment"][1] - item["segment"][0] >= 2.0
            ]
            matched_gt: set[int] = set()
            dets_sorted = sorted(dets, key=lambda d: -d["score"])
            for det in dets_sorted:
                seg = (float(det["segment"][0]), float(det["segment"][1]))
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gt):
                    if j in matched_gt:
                        continue
                    iou = temporal_iou(seg, g)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                is_tp = best_iou >= iou_threshold
                if is_tp:
                    matched_gt.add(best_j)
                duration = seg[1] - seg[0]
                features.append([float(det["score"]), np.log1p(duration)])
                labels.append(1 if is_tp else 0)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-cv-root", default="tmp/temporalmaxer_continuous/cv_recipe_hardneg_v1")
    parser.add_argument("--event-cv-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/precision_calibrator_v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = json.loads(resolve(args.ann_path).read_text(encoding="utf-8"))["database"]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cv_root in [
        ("continuous", resolve(args.continuous_cv_root)),
        ("event", resolve(args.event_cv_root)),
    ]:
        oof_predictions = load_oof_predictions(cv_root)
        features, labels = build_tp_fp_dataset(oof_predictions, annotations)
        n_tp = int(labels.sum())
        n_fp = int(len(labels) - n_tp)
        weights, bias = fit_logistic_regression(features, labels)
        probs = predict_proba(features, weights, bias)
        preds = (probs >= 0.5).astype(int)
        train_acc = float((preds == labels).mean())
        (out_dir / f"{name}_calibrator.json").write_text(
            json.dumps({"weights": weights.tolist(), "bias": float(bias)}, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{name}] pooled OOF from {len(oof_predictions)} recordings: "
            f"n_tp={n_tp} n_fp={n_fp} train_acc={train_acc:.4f} "
            f"weights={weights.tolist()} bias={bias:.4f}"
        )


if __name__ == "__main__":
    main()
