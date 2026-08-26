"""Train an external proposal score calibrator/ranker.

The ATSN weights are not modified. We collect/cached logits for train/val
proposals, build lightweight proposal features, and learn a residual score on
top of the original CNN logit. This targets ranking/calibration after the
head-only LP-FT experiment degraded AP.

Run from event_penguins/:
    python dev/train_score_calibrator.py \
        --train-proposals tmp/atsn_lpft/head_only_pilot/cache/proposals_train.csv \
        --val-proposals tmp/atsn_lpft/head_only_pilot/cache/proposals_val.csv \
        --out-dir tmp/score_calibrator/residual_ranker
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.classification import ProposalClassifier
from src.evaluation import DetectionsEvaluator, segment_iou
from src.utils import temporal_soft_nms


ROOT = Path(__file__).resolve().parents[1]


FEATURE_COLUMNS = [
    "cnn_score",
    "cnn_margin",
    "proposal_score_robust",
    "proposal_roi_rank",
    "proposal_roi_z",
    "cnn_roi_rank",
    "cnn_roi_z",
    "duration_log",
    "duration_penalty",
    "cnn_x_prop",
    "cnn_x_duration_penalty",
]


@dataclass(frozen=True)
class CalibratorConfig:
    name: str
    residual_scale: float
    rank_weight: float
    bce_weight: float
    distill_weight: float
    pos_frac: float
    hard_neg_frac: float
    easy_neg_frac: float


CONFIGS = [
    CalibratorConfig("resid_s010_rank", 0.10, 1.0, 0.05, 1.0, 0.25, 0.25, 0.50),
    CalibratorConfig("resid_s025_rank", 0.25, 1.0, 0.05, 1.0, 0.25, 0.25, 0.50),
    CalibratorConfig("resid_s050_rank", 0.50, 1.0, 0.05, 1.0, 0.25, 0.25, 0.50),
    CalibratorConfig("resid_s025_bce", 0.25, 0.5, 0.20, 1.0, 0.20, 0.20, 0.60),
    CalibratorConfig("resid_s010_distill", 0.10, 0.5, 0.05, 5.0, 0.20, 0.20, 0.60),
]


class ResidualRanker(nn.Module):
    def __init__(self, n_features: int, hidden: int = 0) -> None:
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Dropout(p=0.10),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual score calibrator.")
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--train-proposals", required=True)
    parser.add_argument("--val-proposals", required=True)
    parser.add_argument("--train-logits", default=None)
    parser.add_argument("--val-logits", default=None)
    parser.add_argument("--out-dir", default="tmp/score_calibrator/residual_ranker")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--flap-neg-tiou", type=float, default=0.3)
    parser.add_argument("--hard-score", type=float, default=0.3)
    parser.add_argument("--min-gt-duration", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--min-ed-score", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_recordings(ann_path: Path, split: str) -> set[str]:
    info_path = ann_path.parent / "recording_info.csv"
    recordings: set[str] = set()
    with open(info_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                recordings.add(row["timestamp"])
    return recordings


def split_from_proposals(proposals: pd.DataFrame, ann_path: Path) -> str:
    recs = set(proposals["rec_name"].unique())
    for split in ["train", "val", "test"]:
        if recs <= load_split_recordings(ann_path, split):
            return split
    raise ValueError("Could not infer split from proposal recording names.")


def collect_or_load_logits(
    proposals: pd.DataFrame,
    logits_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if logits_path.exists():
        data = np.load(logits_path, allow_pickle=True)
        logits = data["logits"]
        if len(logits) != len(proposals):
            raise ValueError(f"{logits_path} has {len(logits)} logits for {len(proposals)} proposals")
        print(f"[INFO] Logits cargados: {logits_path} {logits.shape}")
        return logits

    print(f"[INFO] Recollendo logits CNN: {logits_path}")
    clf = ProposalClassifier(
        device=device,
        model_path=str(resolve_path(args.model_path)),
        num_tsn_samples=args.num_tsn_samples,
        augment_factor=args.augment_factor,
        data_path=str(resolve_path(args.data_path)),
        sample_duration=args.sample_duration,
        decay=args.decay,
        nms_threshold=args.nms_threshold,
        batch_size=args.cnn_batch_size,
        use_soft_nms=True,
        soft_nms_sigma=args.soft_nms_sigma,
        min_ed_score=0.0,
    )
    logits, meta = clf.collect_logits(proposals)
    if len(meta) != len(proposals):
        raise RuntimeError("Classifier returned a different number of logits than proposals.")
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logits_path, logits=logits)
    print(f"[INFO] Logits gardados: {logits_path}")
    return logits


def softmax_ed(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp[:, 1] / exp.sum(axis=1)


def robust01(values: pd.Series | np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros(len(arr), dtype=np.float64)
    if finite.sum() == 0:
        return out
    qlo, qhi = np.nanpercentile(arr[finite], [lo, hi])
    if qhi <= qlo + 1e-12:
        return out
    out[finite] = np.clip((arr[finite] - qlo) / (qhi - qlo), 0.0, 1.0)
    return out


def max_tiou_seconds(t_start_us: float, t_end_us: float, segments_s: np.ndarray) -> float:
    if segments_s.size == 0:
        return 0.0
    t_start = float(t_start_us) / 1e6
    t_end = float(t_end_us) / 1e6
    inter = np.maximum(0.0, np.minimum(t_end, segments_s[:, 1]) - np.maximum(t_start, segments_s[:, 0]))
    union = (t_end - t_start) + (segments_s[:, 1] - segments_s[:, 0]) - inter
    valid = union > 0
    if not np.any(valid):
        return 0.0
    return float(np.max(inter[valid] / union[valid]))


def build_annotation_index(
    ann_path: Path,
    split: str,
    min_duration: float,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], set[str]]:
    split_recs = load_split_recordings(ann_path, split)
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    index: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    for rec_name, rec_data in ann["database"].items():
        if rec_name not in split_recs:
            continue
        rec_index: dict[str, dict[str, list[list[float]]]] = {}
        for roi_key, roi_anns in rec_data.get("annotations", {}).items():
            if roi_key == "null":
                continue
            ed_segments = []
            flap_segments = []
            for item in roi_anns:
                start, end = map(float, item["segment"])
                if end - start < min_duration:
                    continue
                if item["label"] == "ed":
                    ed_segments.append([start, end])
                elif item["label"] in {"adult_flap", "chick_flap"}:
                    flap_segments.append([start, end])
            rec_index[roi_key] = {"ed": ed_segments, "flap": flap_segments}
        index[rec_name] = rec_index

    np_index: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for rec_name, rec_data in index.items():
        np_index[rec_name] = {}
        for roi_key, values in rec_data.items():
            np_index[rec_name][roi_key] = {
                label: np.asarray(segments, dtype=np.float64).reshape(-1, 2)
                for label, segments in values.items()
            }
    return np_index, split_recs


def roi_to_ann_key(roi_id: str) -> str:
    return str(int(str(roi_id)[1:]))


def add_labels(
    df: pd.DataFrame,
    ann_path: Path,
    split: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    ann_index, split_recs = build_annotation_index(ann_path, split, args.min_gt_duration)
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        rec_name = row["rec_name"]
        if rec_name not in split_recs:
            continue
        roi_key = roi_to_ann_key(row["roi_id"])
        segments = ann_index.get(rec_name, {}).get(roi_key, {})
        best_ed = max_tiou_seconds(row["t_start"], row["t_end"], segments.get("ed", np.empty((0, 2))))
        best_flap = max_tiou_seconds(row["t_start"], row["t_end"], segments.get("flap", np.empty((0, 2))))
        label = -1
        sample_kind = "ignored"
        if best_ed >= args.pos_tiou:
            label = 1
            sample_kind = "positive"
        elif best_ed < args.neg_tiou:
            label = 0
            if best_flap >= args.flap_neg_tiou or float(row["cnn_score"]) >= args.hard_score:
                sample_kind = "hard_negative"
            else:
                sample_kind = "easy_negative"
        labeled = row.to_dict()
        labeled.update(
            {
                "proposal_index": idx,
                "label": label,
                "sample_kind": sample_kind,
                "best_ed_tiou": best_ed,
                "best_flap_tiou": best_flap,
            }
        )
        rows.append(labeled)
    return pd.DataFrame(rows).reset_index(drop=True)


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proposal_score_robust"] = robust01(df["score"])
    df["duration_s"] = (df["t_end"] - df["t_start"]) / 1e6
    df["duration_log"] = np.log1p(np.maximum(df["duration_s"], 0.0))
    excess = np.maximum(0.0, df["duration_s"].to_numpy(dtype=np.float64) - 60.0)
    df["duration_penalty"] = np.exp(-excess / 20.0)
    df["cnn_x_prop"] = df["cnn_score"] * df["proposal_score_robust"]
    df["cnn_x_duration_penalty"] = df["cnn_score"] * df["duration_penalty"]

    df["cnn_roi_rank"] = df.groupby(["rec_name", "roi_id"])["cnn_score"].rank(method="average", pct=True).fillna(0.0)
    df["proposal_roi_rank"] = df.groupby(["rec_name", "roi_id"])["score"].rank(method="average", pct=True).fillna(0.0)
    df["cnn_roi_z"] = 0.0
    df["proposal_roi_z"] = 0.0
    for _, idx in df.groupby(["rec_name", "roi_id"]).groups.items():
        for src, dst in [("cnn_score", "cnn_roi_z"), ("score", "proposal_roi_z")]:
            vals = df.loc[idx, src].to_numpy(dtype=np.float64)
            std = vals.std()
            if std > 1e-12:
                z = (vals - vals.mean()) / std
                df.loc[idx, dst] = 1.0 / (1.0 + np.exp(-z))
    return df


def prepare_frame(
    proposals: pd.DataFrame,
    logits: np.ndarray,
    split: str,
    ann_path: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    df = proposals.reset_index(drop=True).copy()
    df["logit_bg"] = logits[:, 0].astype(np.float64)
    df["logit_ed"] = logits[:, 1].astype(np.float64)
    df["cnn_margin"] = df["logit_ed"] - df["logit_bg"]
    df["cnn_score"] = softmax_ed(logits, args.temperature)
    df = add_rank_features(df)
    return add_labels(df, ann_path, split, args)


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (other_x - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def sample_train_indices(train_df: pd.DataFrame, cfg: CalibratorConfig, max_samples: int, seed: int) -> np.ndarray:
    usable = train_df[train_df["label"] >= 0].copy()
    if len(usable) <= max_samples:
        return usable.index.to_numpy(dtype=np.int64)

    rng = np.random.default_rng(seed)
    targets = {
        "positive": cfg.pos_frac,
        "hard_negative": cfg.hard_neg_frac,
        "easy_negative": cfg.easy_neg_frac,
    }
    chosen = []
    used = 0
    for kind, frac in targets.items():
        group = usable[usable["sample_kind"] == kind]
        if group.empty:
            continue
        take = min(len(group), max(1, int(round(max_samples * frac))))
        chosen.append(rng.choice(group.index.to_numpy(dtype=np.int64), size=take, replace=False))
        used += take
    if used < max_samples:
        selected = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        remaining = usable.drop(index=selected, errors="ignore")
        if not remaining.empty:
            take = min(len(remaining), max_samples - used)
            chosen.append(rng.choice(remaining.index.to_numpy(dtype=np.int64), size=take, replace=False))
    out = np.concatenate(chosen) if chosen else usable.index.to_numpy(dtype=np.int64)
    rng.shuffle(out)
    return out


def inverse_sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def train_one_config(
    cfg: CalibratorConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    train_idx = sample_train_indices(train_df, cfg, args.max_train_samples, args.seed)
    sub = train_df.loc[train_idx].copy()

    train_x_raw = sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    val_x_raw = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    train_x, val_x, scaler = standardize(train_x_raw, val_x_raw)

    y = sub["label"].to_numpy(dtype=np.float32)
    base_train = sub["cnn_score"].to_numpy(dtype=np.float32)
    base_logit_train = inverse_sigmoid(base_train).astype(np.float32)
    base_val = val_df["cnn_score"].to_numpy(dtype=np.float32)
    base_logit_val = inverse_sigmoid(base_val).astype(np.float32)

    model = ResidualRanker(train_x.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(y),
        torch.from_numpy(base_logit_train),
        torch.from_numpy(base_train),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    pos_pool = np.where(y == 1)[0]
    neg_pool = np.where(y == 0)[0]
    hard_pool = np.where((y == 0) & (sub["sample_kind"].to_numpy() == "hard_negative"))[0]
    if len(hard_pool) == 0:
        hard_pool = neg_pool
    rng = np.random.default_rng(args.seed)

    progress = tqdm(range(args.epochs), desc=cfg.name, disable=args.quiet_progress)
    for _ in progress:
        model.train()
        losses = []
        for xb, yb, base_logit_b, base_score_b in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            base_logit_b = base_logit_b.to(device)
            base_score_b = base_score_b.to(device)

            delta = model(xb)
            score = torch.sigmoid(base_logit_b + cfg.residual_scale * delta)
            bce = F.binary_cross_entropy(score, yb)
            distill = F.mse_loss(score, base_score_b)

            if len(pos_pool) > 0 and len(neg_pool) > 0:
                n_pairs = min(args.pair_batch_size, len(pos_pool), len(neg_pool))
                pos_idx = rng.choice(pos_pool, size=n_pairs, replace=len(pos_pool) < n_pairs)
                neg_source = hard_pool if rng.random() < 0.7 else neg_pool
                neg_idx = rng.choice(neg_source, size=n_pairs, replace=len(neg_source) < n_pairs)
                pair_idx = np.concatenate([pos_idx, neg_idx])
                pair_x = torch.from_numpy(train_x[pair_idx]).to(device)
                pair_base = torch.from_numpy(base_logit_train[pair_idx]).to(device)
                pair_delta = model(pair_x)
                pair_score = pair_base + cfg.residual_scale * pair_delta
                pos_score = pair_score[:n_pairs]
                neg_score = pair_score[n_pairs:]
                rank = F.softplus(-(pos_score - neg_score)).mean()
            else:
                rank = torch.tensor(0.0, device=device)

            loss = cfg.bce_weight * bce + cfg.distill_weight * distill + cfg.rank_weight * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        progress.set_postfix(loss=f"{np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        vx = torch.from_numpy(val_x).to(device)
        delta_val = model(vx).cpu().numpy()
    val_scored = val_df.copy()
    val_scored[f"{cfg.name}_delta"] = delta_val
    val_scored["calibrated_score"] = 1.0 / (1.0 + np.exp(-(base_logit_val + cfg.residual_scale * delta_val)))

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "features": FEATURE_COLUMNS,
        "scaler": scaler,
        "args": vars(args),
    }
    torch.save(checkpoint, out_dir / f"{cfg.name}.pt")
    return val_scored, checkpoint


def load_gt(valid_sequences: list[str], ann_path: Path) -> pd.DataFrame:
    with open(ann_path, encoding="utf-8") as f:
        db = json.load(f)["database"]
    rows = []
    for rec, value in db.items():
        if rec not in valid_sequences:
            continue
        for roi, annotations in value["annotations"].items():
            if roi == "null":
                continue
            for ann in annotations:
                if ann["label"] != "ed":
                    continue
                start, end = map(float, ann["segment"])
                if end - start < 2.0:
                    continue
                rows.append({"video-id": f"{rec}_{int(roi)}", "t-start": start, "t-end": end})
    return pd.DataFrame(rows)


def predictions_to_df(prediction: dict) -> pd.DataFrame:
    rows = []
    for rec, rois in prediction["results"].items():
        for roi, detections in rois.items():
            for det in detections:
                start, end = det["segment"]
                if end - start < 2:
                    continue
                rows.append(
                    {
                        "video-id": f"{rec}_{int(roi)}",
                        "t-start": float(start),
                        "t-end": float(end),
                        "score": float(det["score"]),
                    }
                )
    return pd.DataFrame(rows)


def best_iou_by_gt(gt: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
    if pred.empty:
        return np.zeros(len(gt))
    grouped = {key: grp.reset_index(drop=True) for key, grp in pred.groupby("video-id")}
    best = []
    for _, row in gt.iterrows():
        candidates = grouped.get(row["video-id"])
        if candidates is None or candidates.empty:
            best.append(0.0)
            continue
        iou = segment_iou(
            np.asarray([row["t-start"], row["t-end"]]),
            candidates[["t-start", "t-end"]].to_numpy(dtype=np.float64),
        )
        best.append(float(iou.max()))
    return np.asarray(best, dtype=np.float64)


def build_prediction(df: pd.DataFrame, score_col: str, min_ed_score: float, args: argparse.Namespace) -> dict:
    result: dict[str, dict[int, list[dict]]] = {
        rec: {int(roi[1:]): [] for roi in grp["roi_id"].unique()}
        for rec, grp in df.groupby("rec_name")
    }
    selected = df[df["cnn_score"] >= min_ed_score].copy()
    if selected.empty:
        return {"version": "VERSION 0.0", "results": result}

    scores = selected[score_col].to_numpy(dtype=np.float64).copy()
    durations_s = (selected["t_end"].to_numpy() - selected["t_start"].to_numpy()) / 1e6
    excess = np.maximum(0.0, durations_s - args.duration_dmax)
    scores *= np.exp(-excess / args.duration_sigma)
    selected["final_score"] = scores

    for (rec, roi_id), grp in selected.groupby(["rec_name", "roi_id"]):
        arr = grp[["t_start", "t_end", "final_score"]].to_numpy(dtype=np.float64)
        processed = temporal_soft_nms(arr, sigma=args.soft_nms_sigma, score_threshold=args.soft_nms_score_threshold)
        result[rec][int(roi_id[1:])] = [
            {
                "label": "ed",
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(score),
            }
            for start, end, score in processed
            if (float(end) - float(start)) / 1e6 >= 2.0
        ]
    return {"version": "VERSION 0.0", "results": result}


def evaluate_scored(
    df: pd.DataFrame,
    score_col: str,
    label: str,
    ann_path: Path,
    args: argparse.Namespace,
    pred_dir: Path,
) -> dict:
    valid_sequences = sorted(df["rec_name"].unique())
    gt = load_gt(valid_sequences, ann_path)
    best = None
    for min_score in args.min_ed_score:
        prediction = build_prediction(df, score_col, min_score, args)
        pred_path = pred_dir / f"{label}_min{min_score:.3f}.json"
        pred_path.write_text(json.dumps(prediction), encoding="utf-8")
        evaluator = DetectionsEvaluator(
            ground_truth_filename=str(ann_path),
            prediction_filename=str(pred_path),
            tiou_thresholds=np.asarray(args.tiou, dtype=np.float64),
            valid_labels="ed",
            valid_sequences=valid_sequences,
            min_duration=2.0,
        )
        mean_ap = evaluator.run()
        pred_df = predictions_to_df(prediction)
        best_iou = best_iou_by_gt(gt, pred_df)
        metrics = {
            "variant": label,
            "score_col": score_col,
            "min_ed_score": float(min_score),
            "n_pred": int(len(pred_df)),
            "mAP": float(mean_ap),
            "AP@0.1": float(evaluator.mAP[0]),
            "AP@0.3": float(evaluator.mAP[1]),
            "AP@0.5": float(evaluator.mAP[2]),
            "AP@0.7": float(evaluator.mAP[3]),
            "recall@0.1": float((best_iou >= 0.1).mean()) if len(best_iou) else float("nan"),
            "recall@0.3": float((best_iou >= 0.3).mean()) if len(best_iou) else float("nan"),
            "recall@0.5": float((best_iou >= 0.5).mean()) if len(best_iou) else float("nan"),
            "missed@0.1": int((best_iou < 0.1).sum()) if len(best_iou) else 0,
            "missed@0.5": int((best_iou < 0.5).sum()) if len(best_iou) else 0,
        }
        if best is None or metrics["mAP"] > best["mAP"]:
            best = metrics
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_train_samples = min(args.max_train_samples, 512)
        args.quiet_progress = True

    set_seed(args.seed)
    out_dir = resolve_path(args.out_dir)
    cache_dir = out_dir / "cache"
    pred_dir = out_dir / "predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ann_path = resolve_path(args.ann_path)
    train_props = pd.read_csv(resolve_path(args.train_proposals)).reset_index(drop=True)
    val_props = pd.read_csv(resolve_path(args.val_proposals)).reset_index(drop=True)
    train_split = split_from_proposals(train_props, ann_path)
    val_split = split_from_proposals(val_props, ann_path)

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Train proposals={len(train_props)} split={train_split}; val proposals={len(val_props)} split={val_split}")

    train_logits_path = resolve_path(args.train_logits) if args.train_logits else cache_dir / "train_logits.npz"
    val_logits_path = resolve_path(args.val_logits) if args.val_logits else cache_dir / "val_logits.npz"
    train_logits = collect_or_load_logits(train_props, train_logits_path, args, device)
    val_logits = collect_or_load_logits(val_props, val_logits_path, args, device)

    train_df = prepare_frame(train_props, train_logits, train_split, ann_path, args)
    val_df = prepare_frame(val_props, val_logits, val_split, ann_path, args)
    train_df.to_csv(cache_dir / "train_features_labels.csv", index=False)
    val_df.to_csv(cache_dir / "val_features_labels.csv", index=False)

    for name, df in [("train", train_df), ("val", val_df)]:
        counts = df["sample_kind"].value_counts().to_dict()
        print(
            f"[INFO] {name}: usable={(df['label'] >= 0).sum()} "
            f"pos={counts.get('positive', 0)} hard={counts.get('hard_negative', 0)} "
            f"easy={counts.get('easy_negative', 0)} ignored={counts.get('ignored', 0)}"
        )

    rows = []
    base_metrics = evaluate_scored(val_df, "cnn_score", "base_cnn", ann_path, args, pred_dir)
    rows.append(base_metrics)
    print(
        f"[BASE] mAP={base_metrics['mAP']:.4f} AP@0.5={base_metrics['AP@0.5']:.4f} "
        f"recall@0.5={base_metrics['recall@0.5']:.4f} n={base_metrics['n_pred']}"
    )

    best_metrics = base_metrics
    for cfg in CONFIGS:
        val_scored, _ = train_one_config(cfg, train_df, val_df, args, device, out_dir)
        val_scored.to_csv(cache_dir / f"val_scores_{cfg.name}.csv", index=False)
        metrics = evaluate_scored(val_scored, "calibrated_score", cfg.name, ann_path, args, pred_dir)
        rows.append(metrics)
        print(
            f"[{cfg.name}] mAP={metrics['mAP']:.4f} AP@0.5={metrics['AP@0.5']:.4f} "
            f"recall@0.5={metrics['recall@0.5']:.4f} n={metrics['n_pred']} min={metrics['min_ed_score']}"
        )
        if metrics["mAP"] > best_metrics["mAP"]:
            best_metrics = metrics

    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "best.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print("\n[RESULTADO]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
