"""Evaluate post-hoc scoring variants from cached proposal logits.

This is a diagnostic script: it reuses proposals, CNN logits and optional
feature CSVs to test whether a non-trained re-ranking can reduce score
inversion. It does not run the CNN.

Run from event_penguins/:
    python dev/eval_scoring_variants.py \
        --proposals tmp/deep_diagnosis/fixed_r5_single_remote/proposals.csv \
        --logits tmp/deep_diagnosis/min_score_sweep/logits.npz \
        --features tmp/fft_phase_analysis/test/window_features.csv \
        --out-dir tmp/scoring_variants/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation import DetectionsEvaluator, segment_iou
from src.utils import temporal_soft_nms


ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "config/annotations/annotations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached scoring variants.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out-dir", default="tmp/scoring_variants")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--min-ed-score", type=float, default=0.02)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


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


def load_feature_columns(features_path: str | None, n_proposals: int) -> pd.DataFrame:
    base = pd.DataFrame(index=np.arange(n_proposals))
    if not features_path:
        return base

    features = pd.read_csv(features_path)
    features = features[features["source"] == "proposal"].copy()
    features["proposal_idx"] = features["window_id"].str.replace("prop:", "", regex=False).astype(int)
    features = features.sort_values("proposal_idx")
    features = features.drop_duplicates("proposal_idx", keep="first")
    features = features.set_index("proposal_idx")

    wanted = [
        "band_energy_frac",
        "dom_freq_hz",
        "dom_power_frac_total",
        "band_entropy",
        "phase_coherence",
        "phase_circ_std",
        "event_count",
        "signed_rate_std",
        "signed_rate_abs_mean",
        "duration_s",
    ]
    for col in wanted:
        if col in features.columns:
            base[col] = features[col]
    return base


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cnn_roi_rank"] = (
        df.groupby(["rec_name", "roi_id"])["cnn_score"]
        .rank(method="average", pct=True)
        .fillna(0.0)
    )
    df["cnn_roi_z"] = 0.0
    for _, idx in df.groupby(["rec_name", "roi_id"]).groups.items():
        vals = df.loc[idx, "cnn_score"].to_numpy(dtype=np.float64)
        std = vals.std()
        if std > 1e-12:
            z = (vals - vals.mean()) / std
            df.loc[idx, "cnn_roi_z"] = 1.0 / (1.0 + np.exp(-z))
    return df


def make_variant_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    cnn = df["cnn_score"].to_numpy(dtype=np.float64)
    prop = robust01(df["score"])
    energy = robust01(df.get("band_energy_frac", pd.Series(np.zeros(len(df)))))
    peak = robust01(df.get("dom_power_frac_total", pd.Series(np.zeros(len(df)))))
    entropy = robust01(df.get("band_entropy", pd.Series(np.zeros(len(df)))))
    phase = robust01(df.get("phase_coherence", pd.Series(np.zeros(len(df)))))
    freq = pd.to_numeric(df.get("dom_freq_hz", pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(dtype=np.float64)
    freq_match = np.nan_to_num(np.exp(-0.5 * ((freq - 2.1) / 0.55) ** 2), nan=0.0)
    roi_rank = df["cnn_roi_rank"].to_numpy(dtype=np.float64)
    roi_z = df["cnn_roi_z"].to_numpy(dtype=np.float64)

    variants = {
        "cnn": cnn,
        "cnn_x_prop_w03": cnn * (1.0 + 0.3 * prop),
        "cnn_x_prop_w10": cnn * (1.0 + 1.0 * prop),
        "cnn_x_energy_w05": cnn * (1.0 + 0.5 * energy),
        "cnn_x_energy_w10": cnn * (1.0 + 1.0 * energy),
        "cnn_x_peak_w05": cnn * (1.0 + 0.5 * peak),
        "cnn_x_freq_w05": cnn * (1.0 + 0.5 * freq_match),
        "cnn_x_freq_w10": cnn * (1.0 + 1.0 * freq_match),
        "cnn_pen_entropy_w03": cnn * (1.0 - 0.3 * entropy),
        "cnn_pen_entropy_w05": cnn * (1.0 - 0.5 * entropy),
        "cnn_pen_phase_w03": cnn * (1.0 - 0.3 * phase),
        "cnn_x_roi_rank_w05": cnn * (0.5 + 0.5 * roi_rank),
        "cnn_x_roi_rank_w10": cnn * roi_rank,
        "cnn_x_roi_z_w05": cnn * (0.5 + 0.5 * roi_z),
        "cnn_energy_entropy": cnn * (1.0 + 0.5 * energy) * (1.0 - 0.3 * entropy),
        "cnn_freq_entropy": cnn * (1.0 + 0.5 * freq_match) * (1.0 - 0.3 * entropy),
    }
    return variants


def load_gt(valid_sequences: list[str]) -> pd.DataFrame:
    with open(ANN) as f:
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
                if end - start < 2:
                    continue
                rows.append(
                    {
                        "video-id": f"{rec}_{int(roi)}",
                        "rec": rec,
                        "roi": int(roi),
                        "t-start": start,
                        "t-end": end,
                    }
                )
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
                        "rec": rec,
                        "roi": int(roi),
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
            np.array([row["t-start"], row["t-end"]]),
            candidates[["t-start", "t-end"]].values,
        )
        best.append(float(iou.max()))
    return np.array(best)


def build_prediction(
    df: pd.DataFrame,
    variant_score: np.ndarray,
    min_ed_score: float,
    args: argparse.Namespace,
) -> dict:
    selected = df[df["cnn_score"] >= min_ed_score].copy()
    if selected.empty:
        return {"version": "VERSION 0.0", "results": {rec: {} for rec in df["rec_name"].unique()}}

    scores = np.asarray(variant_score[selected.index], dtype=np.float64).copy()
    durations_s = (selected["t_end"].to_numpy() - selected["t_start"].to_numpy()) / 1e6
    excess = np.maximum(0.0, durations_s - args.duration_dmax)
    scores *= np.exp(-excess / args.duration_sigma)
    selected["final_score"] = scores

    result: dict[str, dict[int, list[dict]]] = {
        rec: {int(roi[1:]): [] for roi in grp["roi_id"].unique()}
        for rec, grp in df.groupby("rec_name")
    }

    for (rec, roi_id), grp in selected.groupby(["rec_name", "roi_id"]):
        arr = grp[["t_start", "t_end", "final_score"]].to_numpy(dtype=float)
        processed = temporal_soft_nms(
            arr,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
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

    return {"version": "VERSION 0.0", "results": result}


def evaluate_prediction(prediction: dict, valid_sequences: list[str], gt: pd.DataFrame, out_path: Path) -> dict:
    out_path.write_text(json.dumps(prediction), encoding="utf-8")
    evaluator = DetectionsEvaluator(
        ground_truth_filename=str(ANN),
        prediction_filename=str(out_path),
        tiou_thresholds=np.array([0.1, 0.3, 0.5, 0.7]),
        valid_labels="ed",
        valid_sequences=valid_sequences,
        min_duration=2.0,
    )
    mean_ap = evaluator.run()
    pred_df = predictions_to_df(prediction)
    best = best_iou_by_gt(gt, pred_df)
    return {
        "n_pred": len(pred_df),
        "mAP": float(mean_ap),
        "AP@0.1": float(evaluator.mAP[0]),
        "AP@0.3": float(evaluator.mAP[1]),
        "AP@0.5": float(evaluator.mAP[2]),
        "AP@0.7": float(evaluator.mAP[3]),
        "recall@0.1": float((best >= 0.1).mean()),
        "recall@0.3": float((best >= 0.3).mean()),
        "recall@0.5": float((best >= 0.5).mean()),
        "missed@0.1": int((best < 0.1).sum()),
        "missed@0.5": int((best < 0.5).sum()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    proposals = pd.read_csv(args.proposals).reset_index(drop=True)
    logits = np.load(args.logits, allow_pickle=True)["logits"]
    if len(proposals) != len(logits):
        raise ValueError(f"proposals ({len(proposals)}) and logits ({len(logits)}) differ")

    features = load_feature_columns(args.features, len(proposals))
    df = pd.concat([proposals, features], axis=1)
    df["cnn_score"] = softmax_ed(logits, args.temperature)
    df = add_rank_features(df)

    valid_sequences = sorted(df["rec_name"].unique())
    gt = load_gt(valid_sequences)

    rows = []
    variants = make_variant_scores(df)
    for name, scores in variants.items():
        prediction = build_prediction(df, scores, args.min_ed_score, args)
        metrics = evaluate_prediction(
            prediction,
            valid_sequences,
            gt,
            pred_dir / f"{name}.json",
        )
        row = {"variant": name, **metrics}
        rows.append(row)
        print(
            f"{name:24s} mAP={row['mAP']:.4f} "
            f"AP01={row['AP@0.1']:.4f} AP05={row['AP@0.5']:.4f} "
            f"miss01={row['missed@0.1']} n={row['n_pred']}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[INFO] Summary: {out_dir / 'summary.csv'}")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
