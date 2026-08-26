"""Train a lightweight quality predictor using FFT/event features (no GPU needed).

Uses pre-computed window_features.csv from fft_phase_analysis experiments.
Trains on val split, evaluates on test split.

Run from event_penguins/:
    python dev/train_fft_quality.py \
        --val-features tmp/fft_phase_analysis/val/window_features.csv \
        --test-features tmp/fft_phase_analysis/test/window_features.csv \
        --test-proposals tmp/deep_diagnosis/fixed_r5_single_remote/proposals.csv \
        --out-dir tmp/fft_quality
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import DetectionsEvaluator, segment_iou
from src.utils import temporal_nms, temporal_soft_nms

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "config/annotations/annotations.json"

FEATURE_COLS = [
    "proposal_score",
    "duration_s",
    "event_count",
    "unsigned_rate_mean",
    "signed_rate_abs_mean",
    "signed_rate_std",
    "band_power",
    "band_energy_frac",
    "dom_freq_hz",
    "dom_power_frac_total",
    "dom_power_frac_band",
    "dom_phase_sin",
    "dom_phase_cos",
    "band_entropy",
    "phase_n_windows",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val-features", required=True)
    p.add_argument("--test-features", required=True)
    p.add_argument("--test-proposals", required=True)
    p.add_argument("--out-dir", default="tmp/fft_quality")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--min-score", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2])
    p.add_argument("--ann-path", default="config/annotations/annotations.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--also-train-features", default=None,
                   help="Optional path to train split features CSV to augment training")
    return p.parse_args()


class QualityMLP(nn.Module):
    def __init__(self, n_feat: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def load_proposals_with_features(feat_path: str) -> pd.DataFrame:
    df = pd.read_csv(feat_path)
    props = df[df["source"] == "proposal"].copy()
    # Align columns
    missing = [c for c in FEATURE_COLS if c not in props.columns]
    if missing:
        print(f"[WARN] Missing feature columns: {missing}")
    for c in missing:
        props[c] = 0.0
    # Fill NaN with column median
    for c in FEATURE_COLS:
        med = props[c].median()
        props[c] = props[c].fillna(med if not np.isnan(med) else 0.0)
    return props.reset_index(drop=True)


def build_xy(props: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = props[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = props["max_tiou_ed"].to_numpy(dtype=np.float32)
    return X, y


def normalize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def quality_focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)
    # CE-like loss weighted by quality target
    loss = -target * (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    # Focal weight: reduce loss for easy examples
    pt = torch.where(target >= 0.5, pred, 1 - pred)
    focal_w = (1 - pt) ** gamma
    return (focal_w * loss).mean()


def train_model(X_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> QualityMLP:
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    n_feat = X_train.shape[1]
    model = QualityMLP(n_feat, args.hidden, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = quality_focal_loss(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"  epoch {epoch+1:4d}/{args.epochs}: loss={avg_loss:.4f}")

    return model


def predict(model: QualityMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X.astype(np.float32))
        return model(xt).numpy()


def build_predictions(
    props: pd.DataFrame,
    quality_scores: np.ndarray,
    min_score: float,
    soft_nms_sigma: float = 0.25,
    soft_nms_threshold: float = 0.001,
    duration_dmax: float = 60.0,
    duration_sigma: float = 20.0,
) -> dict:
    """Convert scored proposals to ActivityNet detection format using Soft-NMS pipeline."""
    result: dict = {}
    for (rec, roi), grp in props.groupby(["rec_name", "roi_id"]):
        # roi_id in window_features is "N09" but predictions expect "9" (numeric string)
        roi_key = str(int(roi.lstrip("N"))) if isinstance(roi, str) else str(roi)
        idx = grp.index
        raw_scores = quality_scores[idx].copy().astype(np.float64)
        # Duration penalty (same as train_quality_head.py)
        durations_s = (grp["t_end_s"].to_numpy() - grp["t_start_s"].to_numpy())
        excess = np.maximum(0.0, durations_s - duration_dmax)
        final_scores = raw_scores * np.exp(-excess / duration_sigma)
        mask = final_scores >= min_score
        grp_f = grp[mask]
        scores_f = final_scores[mask]
        if len(grp_f) == 0:
            result.setdefault(rec, {})[roi_key] = []
            continue
        arr = np.column_stack([
            grp_f["t_start_s"].to_numpy(),
            grp_f["t_end_s"].to_numpy(),
            scores_f,
        ])
        kept = temporal_soft_nms(arr, sigma=soft_nms_sigma, score_threshold=soft_nms_threshold)
        result.setdefault(rec, {})[roi_key] = [
            {"label": "ed", "segment": [float(r[0]), float(r[1])], "score": float(r[2])}
            for r in kept
            if float(r[1]) - float(r[0]) >= 2.0
        ]
    return {"version": "fft_quality", "results": result}


def evaluate_predictions(preds: dict, ann_path: str, tiou_thresholds: list[float], tmp_dir: Path) -> dict:
    pred_path = tmp_dir / "_tmp_predictions.json"
    pred_path.write_text(json.dumps(preds))
    valid_sequences = list(preds["results"].keys())
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(ann_path),
        prediction_filename=str(pred_path),
        tiou_thresholds=np.asarray(tiou_thresholds, dtype=np.float64),
        valid_labels="ed",
        valid_sequences=valid_sequences,
        min_duration=2.0,
    )
    mean_ap = evaluator.run()
    n_pred = sum(len(dets) for rois in preds["results"].values() for dets in rois.values())
    result = {"mAP": float(mean_ap), "n_pred": n_pred}
    for i, t in enumerate(tiou_thresholds):
        result[f"AP@{t}"] = float(evaluator.mAP[i])
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading val features (training data)...")
    val_props = load_proposals_with_features(args.val_features)
    print(f"  val proposals: {len(val_props)} (positives tIoU>=0.5: {(val_props['max_tiou_ed']>=0.5).sum()})")

    train_props = val_props.copy()

    # Optional: add train features
    if args.also_train_features:
        print(f"[INFO] Loading additional train features: {args.also_train_features}")
        extra = load_proposals_with_features(args.also_train_features)
        train_props = pd.concat([train_props, extra], ignore_index=True)
        print(f"  combined training: {len(train_props)} proposals")

    print("[INFO] Loading test features (evaluation data)...")
    test_props = load_proposals_with_features(args.test_features)
    print(f"  test proposals: {len(test_props)}")

    X_train, y_train = build_xy(train_props)
    X_test, y_test = build_xy(test_props)

    # Normalize
    X_train_n, X_test_n, feat_mean, feat_std = normalize(X_train, X_test)

    print(f"[INFO] Training MLP on CPU: {len(X_train)} examples, {X_train.shape[1]} features...")
    model = train_model(X_train_n, y_train, args)

    print("[INFO] Predicting on test...")
    quality_scores = predict(model, X_test_n)
    test_props["fft_quality"] = quality_scores

    print(f"  quality score: mean={quality_scores.mean():.3f} median={np.median(quality_scores):.3f} max={quality_scores.max():.3f}")

    # Save predictions and evaluate
    tiou_thresholds = [0.1, 0.3, 0.5, 0.7]
    rows = []
    for ms in args.min_score:
        preds = build_predictions(test_props, quality_scores, ms)
        metrics = evaluate_predictions(preds, args.ann_path, tiou_thresholds, out_dir)
        metrics["min_score"] = ms
        metrics["variant"] = "fft_quality_mlp"
        rows.append(metrics)
        print(f"  min_score={ms:.3f}: mAP={metrics['mAP']:.4f} AP@0.1={metrics['AP@0.1']:.4f} AP@0.5={metrics['AP@0.5']:.4f} AP@0.7={metrics['AP@0.7']:.4f} n_pred={metrics['n_pred']}")

    # Baseline: cnn_score_t2 (ATSN softmax — the true CNN baseline, ~0.68 mAP)
    cnn_t2_scores = test_props["cnn_score_t2"].to_numpy(dtype=np.float64)
    cnn_has_t2 = not np.isnan(cnn_t2_scores).all()
    if cnn_has_t2:
        for ms in [0.001, 0.005, 0.01, 0.05]:
            preds = build_predictions(test_props, cnn_t2_scores, ms)
            metrics = evaluate_predictions(preds, args.ann_path, tiou_thresholds, out_dir)
            metrics["min_score"] = ms
            metrics["variant"] = "base_cnn_t2"
            rows.append(metrics)
        best_t2 = max((r for r in rows if r["variant"] == "base_cnn_t2"), key=lambda r: r["mAP"])
        print(f"  [base CNN-T2] best mAP={best_t2['mAP']:.4f} AP@0.7={best_t2['AP@0.7']:.4f} min_score={best_t2['min_score']}")

    # Combined: fft_quality * cnn_score_t2
    if cnn_has_t2:
        combined_t2 = quality_scores * cnn_t2_scores
        for ms in args.min_score:
            preds = build_predictions(test_props, combined_t2, ms)
            metrics = evaluate_predictions(preds, args.ann_path, tiou_thresholds, out_dir)
            metrics["min_score"] = ms
            metrics["variant"] = "fft_x_cnn_t2"
            rows.append(metrics)
        best_comb = max((r for r in rows if r["variant"] == "fft_x_cnn_t2"), key=lambda r: r["mAP"])
        print(f"  [fft_x_cnn_t2 best] mAP={best_comb['mAP']:.4f} AP@0.7={best_comb['AP@0.7']:.4f}")

    # Also try weighted avg: alpha*fft + (1-alpha)*cnn_t2
    if cnn_has_t2:
        for alpha in [0.2, 0.4, 0.6]:
            blend = alpha * quality_scores + (1 - alpha) * cnn_t2_scores
            for ms in [0.001, 0.005]:
                preds = build_predictions(test_props, blend, ms)
                metrics = evaluate_predictions(preds, args.ann_path, tiou_thresholds, out_dir)
                metrics["min_score"] = ms
                metrics["variant"] = f"blend_fft{alpha:.1f}_cnn{1-alpha:.1f}"
                rows.append(metrics)
        best_blend = max((r for r in rows if r["variant"].startswith("blend_")), key=lambda r: r["mAP"], default=None)
        if best_blend:
            print(f"  [best blend] mAP={best_blend['mAP']:.4f} AP@0.7={best_blend['AP@0.7']:.4f} ({best_blend['variant']})")

    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[INFO] Summary saved to {summary_path}")

    best = summary.iloc[0].to_dict()
    best_path = out_dir / "best.json"
    best_path.write_text(json.dumps(best, indent=2))
    print("\n[RESULTADO FINAL]")
    print(summary.head(10).to_string(index=False))
    print(f"\nBest mAP: {best['mAP']:.4f} (variant={best['variant']}, min_score={best['min_score']})")


if __name__ == "__main__":
    main()
