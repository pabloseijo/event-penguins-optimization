"""Train a second-stage temporal refiner on recomputed ATSN features.

The first quality head predicts proposal scores and temporal offsets from the
original interval. This experiment crops the event stream again at those
predicted boundaries, recomputes the frozen ATSN representation, and learns a
residual start/end correction. Proposal scores remain untouched so the
experiment isolates temporal localization from ranking.

The cross-validation mode expects out-of-fold first-stage score files. It is a
screening protocol for the cascade; a positive result must still be confirmed
with nested cross-fitting before evaluating the held-out test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dev.train_quality_head import (
    NUMERIC_COLUMNS,
    ROOT,
    TemporalCompletenessHead,
    collect_or_load_representations,
    evaluate_score,
    resolve_path,
    set_seed,
    softmax_ed,
    standardize_for_head,
)


CASCADE_COLUMNS = [
    "cascade_score",
    "cascade_initial_delta_start",
    "cascade_initial_delta_end",
    "cascade_cnn_score",
    "cascade_cnn_margin",
    "cascade_cnn_score_delta",
    "cascade_cnn_margin_delta",
    "cascade_duration_log",
    "cascade_duration_ratio_log",
    "cascade_center_shift",
    "cascade_span_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recomputed-feature temporal cascade refiner.")
    parser.add_argument("--train-stage1-csv", nargs="+", required=True)
    parser.add_argument("--train-score-col", default="quality_score")
    parser.add_argument("--eval-stage1-csv", default=None)
    parser.add_argument("--eval-score-col", default="cv_mean_score")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--out-dir", default="tmp/quality_head/cascade_refiner")
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repr-batch-size", type=int, default=16)
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
    parser.add_argument(
        "--iou-loss-weight",
        type=float,
        default=0.0,
        help="Weight of a direct one-dimensional GIoU loss on supervised boundaries.",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-epochs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--residual-gate",
        choices=["constant", "score", "sqrt_score", "cnn_gain", "cnn_gain_score"],
        default="constant",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--use-boundary-refinement", action="store_true", default=True)
    return parser.parse_args()


def cache_tag(path: Path, score_col: str, min_score: float) -> str:
    payload = f"cascade-v2|{path.resolve()}|{score_col}|{min_score:.6f}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def cascade_fingerprint(frame: pd.DataFrame, score_col: str) -> str:
    columns = [
        "rec_name",
        "roi_id",
        "t_start",
        "t_end",
        "t_start_original",
        "t_end_original",
        score_col,
        "cnn_score",
        "cnn_margin",
        "boundary_delta_start",
        "boundary_delta_end",
    ]
    available = [column for column in columns if column in frame]
    hashed = pd.util.hash_pandas_object(frame[available], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def refined_proposals(stage1: pd.DataFrame, score_col: str, min_score: float) -> pd.DataFrame:
    required = {
        "rec_name",
        "roi_id",
        "t_start",
        "t_end",
        "refined_t_start",
        "refined_t_end",
        score_col,
    }
    missing = sorted(required - set(stage1.columns))
    if missing:
        raise ValueError(f"Missing first-stage columns: {missing}")
    selected = stage1.loc[stage1[score_col] >= min_score].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No proposals survive {score_col} >= {min_score}")
    selected["cascade_score"] = selected[score_col].to_numpy(dtype=np.float64)
    selected["first_stage_t_start"] = selected["refined_t_start"].to_numpy(dtype=np.float64)
    selected["first_stage_t_end"] = selected["refined_t_end"].to_numpy(dtype=np.float64)
    selected["t_start_original"] = selected["t_start"].to_numpy(dtype=np.float64)
    selected["t_end_original"] = selected["t_end"].to_numpy(dtype=np.float64)
    original_duration = np.maximum(
        selected["t_end_original"].to_numpy(dtype=np.float64)
        - selected["t_start_original"].to_numpy(dtype=np.float64),
        1.0,
    )
    selected["cascade_initial_delta_start"] = (
        selected["first_stage_t_start"].to_numpy(dtype=np.float64)
        - selected["t_start_original"].to_numpy(dtype=np.float64)
    ) / original_duration
    selected["cascade_initial_delta_end"] = (
        selected["first_stage_t_end"].to_numpy(dtype=np.float64)
        - selected["t_end_original"].to_numpy(dtype=np.float64)
    ) / original_duration
    selected["t_start"] = selected["first_stage_t_start"]
    selected["t_end"] = selected["first_stage_t_end"]
    return selected


def temporal_iou(
    start: np.ndarray,
    end: np.ndarray,
    gt_start: np.ndarray,
    gt_end: np.ndarray,
) -> np.ndarray:
    intersection = np.maximum(0.0, np.minimum(end, gt_end) - np.maximum(start, gt_start))
    union = (end - start) + (gt_end - gt_start) - intersection
    valid = np.isfinite(gt_start) & np.isfinite(gt_end) & (union > 0)
    out = np.zeros(len(start), dtype=np.float64)
    out[valid] = intersection[valid] / union[valid]
    return out


def add_cascade_features(
    selected: pd.DataFrame,
    logits: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = selected.copy()
    if len(logits) != len(out):
        raise ValueError("Recomputed logits and selected proposals are not aligned")

    original_start = out["t_start_original"].to_numpy(dtype=np.float64)
    original_end = out["t_end_original"].to_numpy(dtype=np.float64)
    first_start = out["first_stage_t_start"].to_numpy(dtype=np.float64)
    first_end = out["first_stage_t_end"].to_numpy(dtype=np.float64)
    original_duration = np.maximum(original_end - original_start, 1.0)
    first_duration = np.maximum(first_end - first_start, 1.0)
    original_center = 0.5 * (original_start + original_end)
    first_center = 0.5 * (first_start + first_end)

    recomputed_score = softmax_ed(logits, args.temperature)
    recomputed_margin = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
    original_score = out["cnn_score"].to_numpy(dtype=np.float64)
    original_margin = out["cnn_margin"].to_numpy(dtype=np.float64)
    out["cascade_cnn_score"] = recomputed_score
    out["cascade_cnn_margin"] = recomputed_margin
    out["cascade_cnn_score_delta"] = recomputed_score - original_score
    out["cascade_cnn_margin_delta"] = recomputed_margin - original_margin
    out["cascade_duration_log"] = np.log1p(first_duration / 1e6)
    out["cascade_duration_ratio_log"] = np.log(first_duration / original_duration)
    out["cascade_center_shift"] = (first_center - original_center) / original_duration
    out["cascade_span_delta"] = (first_duration - original_duration) / original_duration

    gt_start = out.get("gt_start_s", pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64) * 1e6
    gt_end = out.get("gt_end_s", pd.Series(np.nan, index=out.index)).to_numpy(dtype=np.float64) * 1e6
    first_tiou = temporal_iou(first_start, first_end, gt_start, gt_end)
    out["cascade_tiou"] = first_tiou
    has_target = np.isfinite(gt_start) & np.isfinite(gt_end)
    start_target = np.zeros(len(out), dtype=np.float64)
    end_target = np.zeros(len(out), dtype=np.float64)
    start_target[has_target] = (gt_start[has_target] - first_start[has_target]) / first_duration[has_target]
    end_target[has_target] = (gt_end[has_target] - first_end[has_target]) / first_duration[has_target]
    out["cascade_start_target"] = np.clip(
        start_target, -args.max_boundary_delta, args.max_boundary_delta
    )
    out["cascade_end_target"] = np.clip(
        end_target, -args.max_boundary_delta, args.max_boundary_delta
    )
    original_tiou = out.get("best_ed_tiou", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    supervised = has_target & (
        (first_tiou >= args.boundary_min_tiou) | (original_tiou >= args.boundary_min_tiou)
    )
    out["cascade_boundary_weight"] = np.where(
        supervised,
        np.maximum(first_tiou, original_tiou),
        args.identity_weight,
    )
    out.loc[~supervised, ["cascade_start_target", "cascade_end_target"]] = 0.0
    out["cascade_has_target"] = supervised.astype(float)
    return out


def load_or_prepare(
    csv_path: Path,
    score_col: str,
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    tag = cache_tag(csv_path, score_col, args.min_score)
    frame_path = cache_dir / f"{tag}_frame.csv"
    repr_path = cache_dir / f"{tag}_repr.npz"
    stage1 = pd.read_csv(csv_path).reset_index(drop=True)
    selected = refined_proposals(stage1, score_col, args.min_score)
    fingerprint = cascade_fingerprint(selected, score_col)

    if frame_path.exists() and repr_path.exists():
        data = np.load(repr_path, allow_pickle=False)
        cached_fingerprint = str(data["fingerprint"].item()) if "fingerprint" in data else ""
        frame = pd.read_csv(frame_path).reset_index(drop=True)
        if (
            cached_fingerprint == fingerprint
            and len(frame) == len(selected)
            and len(data["embeddings"]) == len(selected)
        ):
            print(f"[INFO] Cascade cache loaded: {tag} n={len(frame)}")
            return frame, data["embeddings"]
        raise ValueError(f"Stale or misaligned cascade cache: {repr_path}")

    embeddings, logits = collect_or_load_representations(selected, repr_path, args, device)
    frame = add_cascade_features(selected, logits, args)
    np.savez(
        repr_path,
        embeddings=embeddings,
        logits=logits,
        fingerprint=np.asarray(fingerprint),
    )
    frame.to_csv(frame_path, index=False)
    print(f"[INFO] Cascade cache written: {tag} n={len(frame)}")
    return frame, embeddings


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    columns.extend(column for column in CASCADE_COLUMNS if column in frame.columns and column not in columns)
    missing = sorted(set(CASCADE_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Missing cascade numeric columns: {missing}")
    return columns


def recording_balanced_sample(frame: pd.DataFrame, max_samples: int, seed: int) -> np.ndarray:
    if len(frame) <= max_samples:
        return np.arange(len(frame), dtype=np.int64)
    groups = list(frame.groupby("rec_name", sort=True).groups.values())
    quota = max(1, int(math.ceil(max_samples / max(len(groups), 1))))
    rng = np.random.default_rng(seed)
    chosen = []
    for indices in groups:
        indices = np.asarray(indices, dtype=np.int64)
        take = min(len(indices), quota)
        chosen.append(rng.choice(indices, size=take, replace=False))
    out = np.concatenate(chosen)
    if len(out) > max_samples:
        out = rng.choice(out, size=max_samples, replace=False)
    rng.shuffle(out)
    return out


def make_training_arrays(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, list[str]]:
    columns = feature_columns(train_frame)
    if sorted(columns) != sorted(feature_columns(val_frame)):
        raise ValueError("Train and validation cascade feature sets differ")
    indices = recording_balanced_sample(train_frame, args.max_train_samples, args.seed)
    train_num = train_frame.loc[indices, columns].to_numpy(dtype=np.float32)
    val_num = val_frame[columns].to_numpy(dtype=np.float32)
    train_raw = np.concatenate((train_embeddings[indices].astype(np.float32), train_num), axis=1)
    val_raw = np.concatenate((val_embeddings.astype(np.float32), val_num), axis=1)
    train_x, val_x, scaler = standardize_for_head(
        train_raw,
        val_raw,
        embedding_dim=train_embeddings.shape[1],
        numeric_dim=len(columns),
        architecture="temporal_completeness",
    )
    targets = train_frame.loc[
        indices, ["cascade_start_target", "cascade_end_target"]
    ].to_numpy(dtype=np.float32)
    weights = train_frame.loc[indices, "cascade_boundary_weight"].to_numpy(dtype=np.float32)
    supervised = train_frame.loc[indices, "cascade_has_target"].to_numpy(dtype=np.float32)
    return train_x, val_x, targets, weights, supervised, indices, scaler, columns


def temporal_giou_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss for intervals represented by start/end residuals."""
    pred_start = prediction[:, 0]
    pred_end = torch.maximum(1.0 + prediction[:, 1], pred_start + 1e-4)
    target_start = target[:, 0]
    target_end = torch.maximum(1.0 + target[:, 1], target_start + 1e-4)

    intersection = (
        torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start)
    ).clamp_min(0.0)
    pred_duration = pred_end - pred_start
    target_duration = target_end - target_start
    union = (pred_duration + target_duration - intersection).clamp_min(1e-6)
    iou = intersection / union
    enclosure = (
        torch.maximum(pred_end, target_end) - torch.minimum(pred_start, target_start)
    ).clamp_min(1e-6)
    giou = iou - (enclosure - union) / enclosure
    return 1.0 - giou


def apply_residual(
    frame: pd.DataFrame,
    offsets: np.ndarray,
    alpha: float,
    args: argparse.Namespace,
    gain: np.ndarray | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    start = out["first_stage_t_start"].to_numpy(dtype=np.float64)
    end = out["first_stage_t_end"].to_numpy(dtype=np.float64)
    duration = np.maximum(end - start, 1.0)
    clipped = np.clip(offsets.astype(np.float64), -args.max_boundary_delta, args.max_boundary_delta)
    effective_gain = np.ones(len(out), dtype=np.float64) if gain is None else np.asarray(gain, dtype=np.float64)
    if effective_gain.shape != (len(out),):
        raise ValueError(f"Residual gain has shape {effective_gain.shape}, expected {(len(out),)}")
    effective_gain = np.clip(effective_gain, 0.0, 1.0)
    refined_start = start + alpha * effective_gain * clipped[:, 0] * duration
    refined_end = end + alpha * effective_gain * clipped[:, 1] * duration
    min_duration = 2e6
    center = 0.5 * (refined_start + refined_end)
    short = refined_end - refined_start < min_duration
    refined_start[short] = center[short] - 0.5 * min_duration
    refined_end[short] = center[short] + 0.5 * min_duration
    refined_start = np.maximum(0.0, refined_start)
    refined_end = np.maximum(refined_start + min_duration, refined_end)
    out["cascade_delta_start"] = clipped[:, 0]
    out["cascade_delta_end"] = clipped[:, 1]
    out["refined_t_start"] = refined_start
    out["refined_t_end"] = refined_end
    return out


def residual_gain(frame: pd.DataFrame, gate: str) -> np.ndarray:
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
    raise ValueError(f"Unknown residual gate: {gate}")


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(values), 8192):
        batch = torch.from_numpy(values[start:start + 8192]).to(device)
        outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_one(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    label: str,
) -> list[dict]:
    train_x, val_x, targets, weights, supervised, _, scaler, columns = make_training_arrays(
        train_frame, val_frame, train_embeddings, val_embeddings, args
    )
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(targets),
        torch.from_numpy(weights),
        torch.from_numpy(supervised),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = TemporalCompletenessHead(
        embedding_dim=train_embeddings.shape[1],
        numeric_dim=len(columns),
        hidden=args.hidden,
        dropout=args.dropout,
        out_dim=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pred_dir = out_dir / "predictions" / label
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    baseline = evaluate_score(
        val_frame,
        "cascade_score",
        f"{label}_stage1",
        args,
        pred_dir,
        "stage1",
    )[0]
    baseline.update({"fold": label, "epoch": 0, "alpha": 0.0})
    rows.append(baseline)

    eval_epochs = set(args.eval_epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for xb, yb, wb, supervised_b in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            supervised_b = supervised_b.to(device)
            output = model(xb)
            per_sample = F.smooth_l1_loss(output, yb, reduction="none").mean(dim=1)
            smooth_l1 = (per_sample * wb).sum() / wb.sum().clamp_min(1e-6)
            positive_weight = wb * supervised_b
            giou = (
                (temporal_giou_loss(output, yb) * positive_weight).sum()
                / positive_weight.sum().clamp_min(1e-6)
            )
            loss = smooth_l1 + args.iou_loss_weight * giou
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total_rows += len(xb)
        if epoch not in eval_epochs and epoch != args.epochs:
            continue
        offsets = predict(model, val_x.astype(np.float32), device)
        gain = residual_gain(val_frame, args.residual_gate)
        for alpha in args.alpha:
            scored = apply_residual(val_frame, offsets, alpha, args, gain=gain)
            metric = evaluate_score(
                scored,
                "cascade_score",
                f"{label}_cascade",
                args,
                pred_dir,
                f"epoch{epoch}_alpha{alpha:g}",
            )[0]
            metric.update(
                {
                    "fold": label,
                    "epoch": epoch,
                    "alpha": float(alpha),
                    "gate": args.residual_gate,
                    "train_loss": total_loss / max(total_rows, 1),
                }
            )
            rows.append(metric)
        print(
            f"[{label}] epoch={epoch} loss={total_loss / max(total_rows, 1):.5f} "
            f"offset_abs={np.abs(offsets).mean():.4f}"
        )

    checkpoint = {
        "state_dict": model.state_dict(),
        "scaler": scaler,
        "numeric_columns": columns,
        "embedding_dim": int(train_embeddings.shape[1]),
        "args": vars(args),
    }
    torch.save(checkpoint, out_dir / f"{label}_cascade.pt")
    return rows


def summarize_cv(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7", "recall@0.7"]
    grouped = rows.groupby(["epoch", "alpha"], as_index=False)[metrics].mean()
    counts = rows.groupby(["epoch", "alpha"], as_index=False).size().rename(columns={"size": "folds"})
    return grouped.merge(counts, on=["epoch", "alpha"]).sort_values(
        ["mAP", "AP@0.7"], ascending=False
    )


def main() -> None:
    args = parse_args()
    if args.cross_validate and args.eval_stage1_csv:
        raise ValueError("--cross-validate and --eval-stage1-csv are mutually exclusive")
    if not args.cross_validate and not args.eval_stage1_csv:
        raise ValueError("Choose --cross-validate or provide --eval-stage1-csv")
    args.eval_epochs = sorted(set(args.eval_epochs + [args.epochs]))
    threshold = float(args.min_score)
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out_dir = resolve_path(args.out_dir)
    cache_dir = resolve_path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device={device} threshold={threshold}")

    train_parts = []
    for path_arg in args.train_stage1_csv:
        path = resolve_path(path_arg)
        frame, embeddings = load_or_prepare(path, args.train_score_col, cache_dir, args, device)
        train_parts.append((frame, embeddings))

    rows = []
    if args.cross_validate:
        args.min_score = [threshold]
        if len(train_parts) < 2:
            raise ValueError("Cross-validation needs at least two first-stage folds")
        for fold_index, (val_frame, val_embeddings) in enumerate(train_parts):
            train_frame = pd.concat(
                [part[0] for index, part in enumerate(train_parts) if index != fold_index],
                ignore_index=True,
            )
            train_embeddings = np.concatenate(
                [part[1] for index, part in enumerate(train_parts) if index != fold_index], axis=0
            )
            rows.extend(
                train_one(
                    train_frame,
                    val_frame,
                    train_embeddings,
                    val_embeddings,
                    args,
                    device,
                    out_dir,
                    f"fold_{fold_index:02d}",
                )
            )
        detail = pd.DataFrame(rows)
        detail.to_csv(out_dir / "cv_detail.csv", index=False)
        summary = summarize_cv(detail)
        summary.to_csv(out_dir / "cv_summary.csv", index=False)
        print("\n[CV SUMMARY]")
        print(summary.head(20).to_string(index=False))
        return

    eval_path = resolve_path(args.eval_stage1_csv)
    eval_frame, eval_embeddings = load_or_prepare(
        eval_path, args.eval_score_col, cache_dir, args, device
    )
    args.min_score = [threshold]
    train_frame = pd.concat([part[0] for part in train_parts], ignore_index=True)
    train_embeddings = np.concatenate([part[1] for part in train_parts], axis=0)
    rows = train_one(
        train_frame,
        eval_frame,
        train_embeddings,
        eval_embeddings,
        args,
        device,
        out_dir,
        "eval",
    )
    summary = pd.DataFrame(rows).sort_values(["mAP", "AP@0.7"], ascending=False)
    summary.to_csv(out_dir / "eval_summary.csv", index=False)
    (out_dir / "best.json").write_text(
        json.dumps(summary.iloc[0].to_dict(), indent=2), encoding="utf-8"
    )
    print("\n[EVAL SUMMARY]")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
