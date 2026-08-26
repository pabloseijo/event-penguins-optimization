"""Evaluate instance prototypes over frozen ATSN proposal embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dev.train_quality_head import ROOT, evaluate_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ATSN embeddings with ED instance prototypes.")
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--train-repr", required=True)
    parser.add_argument("--extra-labels", default=None)
    parser.add_argument("--extra-repr", default=None)
    parser.add_argument("--eval-scored-proposals", required=True)
    parser.add_argument("--eval-repr", required=True)
    parser.add_argument("--base-score-col", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--positive-tiou", type=float, default=0.7)
    parser.add_argument("--positive-top-per-gt", type=int, default=16)
    parser.add_argument("--negative-top-per-group", type=int, default=16)
    parser.add_argument("--similarity-topk", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--fusion-weights", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--only-score-col", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default="embedding_prototypes")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-score", type=float, nargs="+", default=[0.1])
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--use-boundary-refinement", action="store_true", default=True)
    parser.add_argument("--no-boundary-refinement", dest="use_boundary_refinement", action="store_false")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def normalized_mean(rows: np.ndarray) -> np.ndarray:
    values = rows.astype(np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    prototype = values.mean(axis=0)
    return prototype / max(float(np.linalg.norm(prototype)), 1e-12)


def build_banks(labels: pd.DataFrame, embeddings: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    positive = labels[
        (labels["best_ed_tiou"] >= args.positive_tiou)
        & labels["gt_start_s"].notna()
        & labels["gt_end_s"].notna()
    ]
    positive_bank = []
    gt_columns = ["rec_name", "roi_id", "gt_start_s", "gt_end_s"]
    for _, group in positive.groupby(gt_columns, sort=False):
        selected = group.nlargest(args.positive_top_per_gt, "best_ed_tiou")
        positive_bank.append(normalized_mean(embeddings[selected.index.to_numpy()]))

    negative = labels[(labels["best_ed_tiou"] < 0.1)].copy()
    negative["negative_hardness"] = np.maximum(
        negative["cnn_score"].to_numpy(dtype=np.float64),
        negative["best_flap_tiou"].fillna(0.0).to_numpy(dtype=np.float64),
    )
    negative_bank = []
    for _, group in negative.groupby(["rec_name", "roi_id"], sort=False):
        selected = group.nlargest(args.negative_top_per_group, "negative_hardness")
        negative_bank.append(normalized_mean(embeddings[selected.index.to_numpy()]))

    if not positive_bank or not negative_bank:
        raise RuntimeError("Could not construct positive and negative prototype banks")
    return np.stack(positive_bank), np.stack(negative_bank)


@torch.no_grad()
def score_embeddings(
    embeddings: np.ndarray,
    positive_bank: np.ndarray,
    negative_bank: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    positive = torch.as_tensor(positive_bank, device=device, dtype=dtype)
    negative = torch.as_tensor(negative_bank, device=device, dtype=dtype)
    topk_pos = min(args.similarity_topk, len(positive_bank))
    topk_neg = min(args.similarity_topk, len(negative_bank))
    pos_max_parts = []
    pos_mean_parts = []
    neg_mean_parts = []
    for start in range(0, len(embeddings), args.chunk_size):
        chunk = torch.as_tensor(
            embeddings[start:start + args.chunk_size], device=device, dtype=dtype
        )
        chunk = torch.nn.functional.normalize(chunk, dim=1)
        pos_similarity = chunk @ positive.T
        neg_similarity = chunk @ negative.T
        pos_top = torch.topk(pos_similarity, topk_pos, dim=1).values.float()
        neg_top = torch.topk(neg_similarity, topk_neg, dim=1).values.float()
        pos_max_parts.append(pos_top[:, 0].cpu().numpy())
        pos_mean_parts.append(pos_top.mean(dim=1).cpu().numpy())
        neg_mean_parts.append(neg_top.mean(dim=1).cpu().numpy())
    return (
        np.concatenate(pos_max_parts),
        np.concatenate(pos_mean_parts),
        np.concatenate(neg_mean_parts),
    )


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    indices = np.load(resolve(args.train_indices))
    label_columns = [
        "rec_name", "roi_id", "best_ed_tiou", "best_flap_tiou", "cnn_score",
        "gt_start_s", "gt_end_s",
    ]
    full_labels = pd.read_csv(resolve(args.train_labels), usecols=label_columns)
    labels = full_labels.iloc[indices].reset_index(drop=True)
    train_embeddings = np.load(resolve(args.train_repr), mmap_mode="r")["embeddings"]
    if len(labels) != len(train_embeddings):
        raise ValueError("Training labels and embeddings are not aligned")
    if bool(args.extra_labels) != bool(args.extra_repr):
        raise ValueError("--extra-labels and --extra-repr must be provided together")
    if args.extra_labels:
        extra_labels = pd.read_csv(resolve(args.extra_labels), usecols=label_columns).reset_index(drop=True)
        extra_embeddings = np.load(resolve(args.extra_repr), mmap_mode="r")["embeddings"]
        if len(extra_labels) != len(extra_embeddings):
            raise ValueError("Extra labels and embeddings are not aligned")
        labels = pd.concat([labels, extra_labels], ignore_index=True)
        train_embeddings = np.concatenate([train_embeddings, extra_embeddings], axis=0)
    positive_bank, negative_bank = build_banks(labels, train_embeddings, args)
    print(f"[INFO] prototypes positive={len(positive_bank)} negative={len(negative_bank)}")

    scored = pd.read_csv(resolve(args.eval_scored_proposals)).reset_index(drop=True)
    eval_embeddings = np.load(resolve(args.eval_repr), mmap_mode="r")["embeddings"]
    if len(scored) != len(eval_embeddings):
        raise ValueError(f"Evaluation rows={len(scored)} embeddings={len(eval_embeddings)}")
    pos_max, pos_mean, neg_mean = score_embeddings(
        eval_embeddings, positive_bank, negative_bank, args
    )
    scored["prototype_pos_max"] = (pos_max + 1.0) / 2.0
    scored["prototype_pos_topk"] = (pos_mean + 1.0) / 2.0
    scored["prototype_margin"] = torch.sigmoid(torch.from_numpy(10.0 * (pos_mean - neg_mean))).numpy()
    group_columns = ["rec_name", "roi_id"]
    for column in ["prototype_pos_max", "prototype_pos_topk", "prototype_margin"]:
        scored[f"{column}_roi_rank"] = scored.groupby(group_columns)[column].rank(method="average", pct=True)

    score_columns = [
        "prototype_pos_max", "prototype_pos_topk", "prototype_margin",
        "prototype_pos_max_roi_rank", "prototype_pos_topk_roi_rank", "prototype_margin_roi_rank",
    ]
    base = scored[args.base_score_col].to_numpy(dtype=np.float64)
    for prototype_column in ["prototype_pos_topk", "prototype_margin", "prototype_pos_topk_roi_rank", "prototype_margin_roi_rank"]:
        prototype_score = scored[prototype_column].to_numpy(dtype=np.float64)
        for weight in args.fusion_weights:
            name = f"fusion_{prototype_column}_w{weight:g}"
            scored[name] = (1.0 - weight) * base + weight * prototype_score
            score_columns.append(name)

    scored_path = out_dir / "scored_prototypes.csv"
    scored.to_csv(scored_path, index=False)
    if args.only_score_col:
        if args.only_score_col not in score_columns:
            raise ValueError(f"Unknown score column: {args.only_score_col}")
        score_columns = [args.only_score_col]
    rows = []
    for score_column in score_columns:
        rows.extend(evaluate_score(scored, score_column, args.label, args, pred_dir, "prototype"))
    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "prototype_manifest.json").write_text(
        json.dumps(
            {
                "positive_prototypes": len(positive_bank),
                "negative_prototypes": len(negative_bank),
                "positive_tiou": args.positive_tiou,
                "positive_top_per_gt": args.positive_top_per_gt,
                "negative_top_per_group": args.negative_top_per_group,
                "similarity_topk": args.similarity_topk,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
