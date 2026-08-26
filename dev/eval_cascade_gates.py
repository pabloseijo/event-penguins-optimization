"""Evaluate label-free gates for trained temporal cascade checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dev.train_cascade_refiner import apply_residual, load_or_prepare, predict
from dev.train_quality_head import ROOT, TemporalCompletenessHead, evaluate_score, resolve_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate label-free cascade residual gates.")
    parser.add_argument("--stage1-csv", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-col", default="quality_score")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25])
    parser.add_argument(
        "--gates",
        nargs="+",
        choices=["constant", "score", "sqrt_score", "cnn_gain", "cnn_gain_score"],
        default=["constant", "score", "sqrt_score", "cnn_gain", "cnn_gain_score"],
    )
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repr-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--boundary-min-tiou", type=float, default=0.3)
    parser.add_argument("--max-boundary-delta", type=float, default=0.5)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--use-boundary-refinement", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def gate_values(frame: pd.DataFrame, gate: str) -> np.ndarray:
    score = np.clip(frame["cascade_score"].to_numpy(dtype=np.float64), 0.0, 1.0)
    cnn_gain = (frame["cascade_cnn_score_delta"].to_numpy(dtype=np.float64) >= 0.0).astype(np.float64)
    if gate == "constant":
        return np.ones(len(frame), dtype=np.float64)
    if gate == "score":
        return score
    if gate == "sqrt_score":
        return np.sqrt(score)
    if gate == "cnn_gain":
        return cnn_gain
    if gate == "cnn_gain_score":
        return cnn_gain * score
    raise ValueError(f"Unknown gate: {gate}")


def load_offsets(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    columns = checkpoint["numeric_columns"]
    numeric = frame[columns].to_numpy(dtype=np.float32)
    raw = np.concatenate((embeddings.astype(np.float32), numeric), axis=1)
    mean = np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["scaler"]["std"], dtype=np.float32)
    values = (raw - mean) / std
    checkpoint_args = checkpoint["args"]
    model = TemporalCompletenessHead(
        embedding_dim=int(checkpoint["embedding_dim"]),
        numeric_dim=len(columns),
        hidden=int(checkpoint_args["hidden"]),
        dropout=float(checkpoint_args["dropout"]),
        out_dim=2,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return predict(model, values.astype(np.float32), device)


def main() -> None:
    args = parse_args()
    if len(args.stage1_csv) != len(args.checkpoints):
        raise ValueError("--stage1-csv and --checkpoints must have the same length")
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out_dir = resolve_path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(args.min_score)
    rows = []

    for fold, (csv_arg, checkpoint_arg) in enumerate(zip(args.stage1_csv, args.checkpoints)):
        frame, embeddings = load_or_prepare(
            resolve_path(csv_arg),
            args.score_col,
            resolve_path(args.cache_dir),
            args,
            device,
        )
        offsets = load_offsets(frame, embeddings, resolve_path(checkpoint_arg), device)
        args.min_score = [threshold]
        baseline = evaluate_score(
            frame, "cascade_score", f"fold_{fold:02d}_stage1", args, pred_dir, "stage1"
        )[0]
        baseline.update({"fold": fold, "gate": "baseline", "alpha": 0.0})
        rows.append(baseline)
        for gate in args.gates:
            gain = gate_values(frame, gate)
            for alpha in args.alpha:
                scored = apply_residual(frame, offsets, alpha, args, gain=gain)
                metric = evaluate_score(
                    scored,
                    "cascade_score",
                    f"fold_{fold:02d}_{gate}",
                    args,
                    pred_dir,
                    f"alpha{alpha:g}",
                )[0]
                metric.update({"fold": fold, "gate": gate, "alpha": float(alpha)})
                rows.append(metric)
        args.min_score = threshold

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "detail.csv", index=False)
    metrics = ["mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7", "recall@0.7"]
    summary = (
        detail.groupby(["gate", "alpha"], as_index=False)[metrics]
        .mean()
        .sort_values(["mAP", "AP@0.7"], ascending=False)
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
