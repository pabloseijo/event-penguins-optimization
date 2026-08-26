"""Evaluate an arbitrary proposal CSV with the frozen ATSN classifier.

This is the bridge between proposal-ceiling experiments and real mAP: it caches
CNN logits for a proposal CSV once, then evaluates several score/threshold
variants without rerunning the network.

Run from event_penguins/:
    python dev/eval_proposal_csv_cnn.py \
        --proposals tmp/proposal_lattice/fixed_plus_short_lattice_k12000_protected/proposals.csv \
        --out-dir tmp/proposal_lattice_eval/k12000_protected
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

from src.classification import ProposalClassifier
from src.evaluation import DetectionsEvaluator, segment_iou
from src.utils import temporal_nms, temporal_soft_nms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proposal CSV with cached CNN logits.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--logits", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--use-soft-nms", action="store_true")
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1500)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--min-score", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional score variants to evaluate. Defaults to all variants.",
    )
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


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


def add_score_columns(df: pd.DataFrame, logits: np.ndarray, temperature: float) -> pd.DataFrame:
    out = df.copy()
    out["cnn_score"] = softmax_ed(logits, temperature)
    out["cnn_margin"] = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
    out["proposal_score_robust"] = robust01(out["score"])
    out["duration_s"] = (out["t_end"].astype(float) - out["t_start"].astype(float)) / 1e6
    out["duration_log"] = np.log1p(np.maximum(out["duration_s"], 0.0))
    out["source_score_robust"] = robust01(out["source_score"] if "source_score" in out.columns else out["score"])
    out["is_lattice"] = (out.get("source", "") == "lattice").astype(float) if "source" in out.columns else 0.0
    if "variant" in out.columns:
        out["is_center_duration"] = out["variant"].astype(str).str.startswith("center_dur_").astype(float)
        out["is_protected_expand"] = out["variant"].astype(str).isin({"expand_both_1"}).astype(float)
    else:
        out["is_center_duration"] = 0.0
        out["is_protected_expand"] = 0.0

    out["cnn_roi_rank"] = (
        out.groupby(["rec_name", "roi_id"])["cnn_score"]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )
    out["prop_roi_rank"] = (
        out.groupby(["rec_name", "roi_id"])["score"]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )
    return out


def make_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    cnn = df["cnn_score"].to_numpy(dtype=np.float64)
    prop = df["proposal_score_robust"].to_numpy(dtype=np.float64)
    source = df["source_score_robust"].to_numpy(dtype=np.float64)
    roi = df["cnn_roi_rank"].to_numpy(dtype=np.float64)
    dur = df["duration_s"].to_numpy(dtype=np.float64)
    dur_pref = np.exp(-0.5 * ((np.log1p(dur) - np.log1p(8.0)) / 0.95) ** 2)
    lattice_pen = 1.0 - 0.10 * df["is_lattice"].to_numpy(dtype=np.float64)
    center_bonus = 1.0 + 0.08 * df["is_center_duration"].to_numpy(dtype=np.float64)
    expand_bonus = 1.0 + 0.05 * df["is_protected_expand"].to_numpy(dtype=np.float64)

    return {
        "cnn": cnn,
        "cnn_x_prop_w03": cnn * (1.0 + 0.3 * prop),
        "cnn_x_source_w03": cnn * (1.0 + 0.3 * source),
        "cnn_x_roi_rank_w05": cnn * (0.5 + 0.5 * roi),
        "cnn_x_duration_pref": cnn * (0.7 + 0.3 * dur_pref),
        "cnn_lattice_guard": cnn * lattice_pen * center_bonus * expand_bonus,
        "cnn_guard_x_prop": cnn * lattice_pen * center_bonus * expand_bonus * (1.0 + 0.2 * prop),
    }


def load_gt(ann_path: Path, valid_sequences: list[str]) -> pd.DataFrame:
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


def build_prediction(df: pd.DataFrame, scores: np.ndarray, min_score: float, args: argparse.Namespace) -> dict:
    selected = df[df["cnn_score"] >= min_score].copy()
    if selected.empty:
        return {"version": "proposal_csv_cnn", "results": {rec: {} for rec in df["rec_name"].unique()}}

    final = np.asarray(scores[selected.index], dtype=np.float64).copy()
    durations_s = (selected["t_end"].to_numpy(dtype=np.float64) - selected["t_start"].to_numpy(dtype=np.float64)) / 1e6
    excess = np.maximum(0.0, durations_s - args.duration_dmax)
    final *= np.exp(-excess / args.duration_sigma)
    selected["final_score"] = final

    result: dict[str, dict[int, list[dict]]] = {
        rec: {int(roi[1:]): [] for roi in grp["roi_id"].unique()}
        for rec, grp in df.groupby("rec_name")
    }
    for (rec, roi_id), grp in selected.groupby(["rec_name", "roi_id"], sort=False):
        if args.pre_nms_topk_per_roi > 0 and len(grp) > args.pre_nms_topk_per_roi:
            grp = grp.sort_values("final_score", ascending=False).head(args.pre_nms_topk_per_roi)
        arr = grp[["t_start", "t_end", "final_score"]].to_numpy(dtype=np.float64)
        processed = (
            temporal_soft_nms(arr, sigma=args.soft_nms_sigma, score_threshold=args.soft_nms_score_threshold)
            if args.use_soft_nms
            else temporal_nms(arr, args.nms_threshold)
        )
        result[rec][int(roi_id[1:])] = [
            {
                "label": "ed",
                "segment": [float(start) / 1e6, float(end) / 1e6],
                "score": float(score),
            }
            for start, end, score in processed
            if (float(end) - float(start)) / 1e6 >= 2.0
        ]
    return {"version": "proposal_csv_cnn", "results": result}


def evaluate_prediction(
    prediction: dict,
    valid_sequences: list[str],
    gt: pd.DataFrame,
    ann_path: Path,
    pred_path: Path,
    args: argparse.Namespace,
) -> dict:
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
    return {
        "n_pred": int(len(pred_df)),
        "mAP": float(mean_ap),
        "AP@0.1": float(evaluator.mAP[0]),
        "AP@0.3": float(evaluator.mAP[1]),
        "AP@0.5": float(evaluator.mAP[2]),
        "AP@0.7": float(evaluator.mAP[3]),
        "recall@0.1": float((best_iou >= 0.1).mean()),
        "recall@0.3": float((best_iou >= 0.3).mean()),
        "recall@0.5": float((best_iou >= 0.5).mean()),
        "missed@0.1": int((best_iou < 0.1).sum()),
        "missed@0.5": int((best_iou < 0.5).sum()),
    }


def collect_or_load_logits(proposals: pd.DataFrame, logits_path: Path, args: argparse.Namespace) -> np.ndarray:
    if logits_path.exists():
        data = np.load(logits_path, allow_pickle=True)
        logits = data["logits"]
        if len(logits) != len(proposals):
            raise ValueError(f"{logits_path} has {len(logits)} logits for {len(proposals)} proposals")
        print(f"[INFO] Logits cargados: {logits_path} {logits.shape}", flush=True)
        return logits

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Extraendo logits en {device}: n={len(proposals)}", flush=True)
    clf = ProposalClassifier(
        device=device,
        model_path=str(resolve(args.model_path)),
        num_tsn_samples=args.num_tsn_samples,
        augment_factor=args.augment_factor,
        data_path=str(resolve(args.data_path)),
        sample_duration=args.sample_duration,
        decay=args.decay,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size,
        min_ed_score=0.0,
        num_workers=args.num_workers,
    )
    logits, meta = clf.collect_logits(proposals)
    if len(meta) != len(proposals):
        raise RuntimeError("Classifier returned a different number of logits than proposals.")
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logits_path, logits=logits)
    print(f"[INFO] Logits gardados: {logits_path}", flush=True)
    return logits


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    if args.max_proposals is not None and len(proposals) > args.max_proposals:
        proposals = proposals.sample(n=args.max_proposals, random_state=args.seed).reset_index(drop=True)
    logits_path = resolve(args.logits) if args.logits else out_dir / "logits.npz"
    logits = collect_or_load_logits(proposals, logits_path, args)
    scored = add_score_columns(proposals, logits, args.temperature)
    scored.to_csv(out_dir / "scored_proposals.csv", index=False)

    ann_path = resolve(args.ann_path)
    valid_sequences = sorted(scored["rec_name"].unique())
    gt = load_gt(ann_path, valid_sequences)
    variants = make_scores(scored)
    if args.variants:
        unknown = sorted(set(args.variants) - set(variants))
        if unknown:
            raise ValueError(f"Unknown score variants: {unknown}. Available: {sorted(variants)}")
        variants = {name: variants[name] for name in args.variants}

    rows = []
    for variant_name, variant_scores in variants.items():
        for min_score in args.min_score:
            prediction = build_prediction(scored, variant_scores, min_score, args)
            metrics = evaluate_prediction(
                prediction,
                valid_sequences,
                gt,
                ann_path,
                pred_dir / f"{variant_name}_min{min_score:.3f}.json",
                args,
            )
            row = {
                "variant": variant_name,
                "min_score": float(min_score),
                **metrics,
            }
            rows.append(row)
            print(
                f"{variant_name:22s} min={min_score:.3f} mAP={row['mAP']:.4f} "
                f"AP05={row['AP@0.5']:.4f} AP07={row['AP@0.7']:.4f} n={row['n_pred']}",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n[RESULTADO]")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
