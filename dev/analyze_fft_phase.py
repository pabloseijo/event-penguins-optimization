"""Analyze FFT phase features for ED vs flap windows.

This script is diagnostic: it extracts Fourier magnitude/phase descriptors from
annotation windows and, optionally, from proposal windows. It does not change
the detector. The goal is to test whether phase-related features add
separation beyond the energy-band cue used by the Fourier paper.

Run from event_penguins/:
    python dev/analyze_fft_phase.py --split val
    python dev/analyze_fft_phase.py --split test --include-proposals \
        --proposals-csv tmp/deep_diagnosis/fixed_r5_single_remote/proposals.csv \
        --logits-npz tmp/deep_diagnosis/min_score_sweep/logits.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.proposals import ProposalGenerator
from src.prototype import build_ed_prototype


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract FFT energy/phase descriptors for ED-vs-flap analysis."
    )
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--proto-path", default="tmp/prototype/ed_prototype.npy")
    parser.add_argument("--out-dir", default="tmp/fft_phase_analysis")
    parser.add_argument("--split", default="val", help="train, val, test, all, or comma list")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--event-bin-s", type=float, default=0.01)
    parser.add_argument("--freq-min", type=float, default=0.5)
    parser.add_argument("--freq-max", type=float, default=3.0)
    parser.add_argument("--subwindow-s", type=float, default=2.0)
    parser.add_argument("--hop-s", type=float, default=0.5)
    parser.add_argument("--include-proposals", action="store_true")
    parser.add_argument("--proposals-csv", default=None)
    parser.add_argument("--logits-npz", default=None)
    parser.add_argument(
        "--high-score-threshold",
        type=float,
        default=0.3,
        help="CNN score threshold used to define high-score background proposals.",
    )
    parser.add_argument("--proposal-label-tiou", type=float, default=0.5)
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument(
        "--features-csv",
        default=None,
        help="Reuse a previously extracted window_features.csv and only rebuild summaries/report.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def selected_splits(split_arg: str) -> set[str] | None:
    if split_arg == "all":
        return None
    return {item.strip() for item in split_arg.split(",") if item.strip()}


def load_recording_splits(data_path: str) -> dict[str, str]:
    with h5py.File(data_path, "r") as hf:
        return {rec: str(hf[rec].attrs.get("split")) for rec in hf.keys()}


def label_group(label: str) -> str:
    if label == "ed":
        return "ed"
    if label in {"adult_flap", "chick_flap"}:
        return "flap"
    return label


def load_annotation_windows(
    ann_path: str,
    rec_splits: dict[str, str],
    split_arg: str,
    min_duration: float,
) -> pd.DataFrame:
    splits = selected_splits(split_arg)
    with open(ann_path) as f:
        db = json.load(f)["database"]

    rows = []
    for rec, rec_data in db.items():
        rec_split = rec_splits.get(rec)
        if splits is not None and rec_split not in splits:
            continue
        for roi_str, annotations in rec_data.get("annotations", {}).items():
            if roi_str == "null":
                continue
            roi_id = f"N{int(roi_str):02d}"
            for idx, ann in enumerate(annotations):
                start, end = map(float, ann["segment"])
                if end - start < min_duration:
                    continue
                rows.append(
                    {
                        "window_id": f"ann:{rec}:{roi_id}:{idx}",
                        "source": "annotation",
                        "rec_name": rec,
                        "roi_id": roi_id,
                        "split": rec_split,
                        "t_start_s": start,
                        "t_end_s": end,
                        "duration_s": end - start,
                        "label": ann["label"],
                        "label_group": label_group(ann["label"]),
                        "proposal_score": np.nan,
                        "cnn_score_t2": np.nan,
                        "max_tiou_ed": np.nan,
                        "max_tiou_flap": np.nan,
                    }
                )

    return pd.DataFrame(rows)


def temporal_iou_one_to_many(start: float, end: float, intervals: np.ndarray) -> np.ndarray:
    if len(intervals) == 0:
        return np.zeros(0, dtype=np.float64)
    inter = np.maximum(0.0, np.minimum(end, intervals[:, 1]) - np.maximum(start, intervals[:, 0]))
    union = (end - start) + (intervals[:, 1] - intervals[:, 0]) - inter
    return inter / np.maximum(union, 1e-9)


def load_proposals(
    args: argparse.Namespace,
    rec_splits: dict[str, str],
    out_dir: Path,
) -> pd.DataFrame:
    splits = selected_splits(args.split)

    if args.proposals_csv:
        proposals = pd.read_csv(args.proposals_csv)
        proposals["split"] = proposals["rec_name"].map(rec_splits)
        if splits is not None:
            proposals = proposals[proposals["split"].isin(splits)].copy()
        if proposals.empty:
            print("[WARN] Proposals CSV has no rows for requested split; generating proposals.")
        else:
            print(f"[INFO] Loaded proposals: {len(proposals)}")
            return proposals

    proto_path = Path(args.proto_path)
    if proto_path.exists():
        prototype = np.load(proto_path)
    else:
        prototype = build_ed_prototype(args.data_path, args.ann_path, split="train")
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proto_path, prototype)

    print("[INFO] Generating current-best proposals for phase analysis.")
    gen = ProposalGenerator(
        data_path=args.data_path,
        output_dir=str(out_dir / "proposal_logs"),
        bin_width=0.033,
        percentile=1.0,
        nms_threshold=0.95,
        use_adaptive_lambda=True,
        use_spatial_compactness=True,
        use_noise_penalization=True,
        use_dispersed_noise=True,
        use_periodicity=True,
        prototype=prototype,
    )
    split_for_generator = None if args.split == "all" or "," in args.split else args.split
    proposals = gen.run(split=split_for_generator)
    proposals["split"] = proposals["rec_name"].map(rec_splits)
    if splits is not None:
        proposals = proposals[proposals["split"].isin(splits)].copy()

    generated_path = out_dir / f"generated_proposals_{args.split.replace(',', '_')}.csv"
    proposals.to_csv(generated_path, index=False)
    print(f"[INFO] Saved generated proposals: {generated_path}")
    return proposals


def load_logits_scores(logits_path: str | None, n_rows: int) -> np.ndarray | None:
    if not logits_path:
        return None
    data = np.load(logits_path, allow_pickle=True)
    logits = data["logits"]
    if len(logits) != n_rows:
        raise ValueError(
            f"Logits rows ({len(logits)}) do not match proposal rows ({n_rows})."
        )
    scaled = logits / 2.0
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp[:, 1] / exp.sum(axis=1)


def annotation_index(annotation_windows: pd.DataFrame) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    index: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (rec, roi), grp in annotation_windows.groupby(["rec_name", "roi_id"]):
        item: dict[str, np.ndarray] = {}
        for group_name in ["ed", "flap"]:
            sub = grp[grp["label_group"] == group_name]
            item[group_name] = sub[["t_start_s", "t_end_s"]].to_numpy(dtype=np.float64)
        index[(rec, roi)] = item
    return index


def proposal_windows(
    proposals: pd.DataFrame,
    annotation_windows_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    props = proposals.copy().reset_index(drop=True)
    if args.logits_npz:
        props["cnn_score_t2"] = load_logits_scores(args.logits_npz, len(props))
    else:
        props["cnn_score_t2"] = np.nan

    props["t_start_s"] = props["t_start"] / 1e6
    props["t_end_s"] = props["t_end"] / 1e6
    props["duration_s"] = props["t_end_s"] - props["t_start_s"]
    props = props[props["duration_s"] >= args.min_duration].copy()

    if args.max_proposals is not None and len(props) > args.max_proposals:
        props = props.sample(args.max_proposals, random_state=args.sample_seed).copy()

    ann_idx = annotation_index(annotation_windows_df)
    rows = []
    for idx, row in props.iterrows():
        key = (row["rec_name"], row["roi_id"])
        intervals = ann_idx.get(key, {})
        ed_iou = temporal_iou_one_to_many(
            float(row["t_start_s"]), float(row["t_end_s"]), intervals.get("ed", np.empty((0, 2)))
        )
        flap_iou = temporal_iou_one_to_many(
            float(row["t_start_s"]), float(row["t_end_s"]), intervals.get("flap", np.empty((0, 2)))
        )
        max_ed = float(ed_iou.max()) if len(ed_iou) else 0.0
        max_flap = float(flap_iou.max()) if len(flap_iou) else 0.0
        if max_ed >= args.proposal_label_tiou and max_ed >= max_flap:
            group = "ed"
            label = "proposal_ed"
        elif max_flap >= args.proposal_label_tiou:
            group = "flap"
            label = "proposal_flap"
        else:
            group = "background"
            label = "proposal_background"

        rows.append(
            {
                "window_id": f"prop:{idx}",
                "source": "proposal",
                "rec_name": row["rec_name"],
                "roi_id": row["roi_id"],
                "split": row.get("split", np.nan),
                "t_start_s": float(row["t_start_s"]),
                "t_end_s": float(row["t_end_s"]),
                "duration_s": float(row["duration_s"]),
                "label": label,
                "label_group": group,
                "proposal_score": float(row["score"]),
                "cnn_score_t2": float(row["cnn_score_t2"]) if not pd.isna(row["cnn_score_t2"]) else np.nan,
                "max_tiou_ed": max_ed,
                "max_tiou_flap": max_flap,
            }
        )

    return pd.DataFrame(rows)


def spectral_entropy(power: np.ndarray) -> float:
    total = float(power.sum())
    if total <= 1e-12 or len(power) <= 1:
        return np.nan
    p = power / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(power)))


def phase_coherence(
    rate: np.ndarray,
    bin_s: float,
    freq: float,
    subwindow_s: float,
    hop_s: float,
) -> tuple[float, float, int]:
    if not np.isfinite(freq) or freq <= 0:
        return np.nan, np.nan, 0

    window_bins = max(4, int(round(subwindow_s / bin_s)))
    hop_bins = max(1, int(round(hop_s / bin_s)))
    if len(rate) < window_bins + hop_bins:
        return np.nan, np.nan, 0

    phases = []
    starts_s = []
    local_freqs = None
    for start in range(0, len(rate) - window_bins + 1, hop_bins):
        seg = rate[start : start + window_bins].astype(np.float64)
        seg = seg - seg.mean()
        if np.allclose(seg, 0):
            continue
        seg = seg * np.hanning(len(seg))
        coeff = np.fft.rfft(seg)
        if local_freqs is None:
            local_freqs = np.fft.rfftfreq(len(seg), d=bin_s)
        idx = int(np.argmin(np.abs(local_freqs - freq)))
        if idx <= 0:
            continue
        phases.append(float(np.angle(coeff[idx])))
        starts_s.append(start * bin_s)

    if len(phases) < 2:
        return np.nan, np.nan, len(phases)

    phases_arr = np.asarray(phases)
    starts_arr = np.asarray(starts_s)
    aligned = phases_arr - 2.0 * np.pi * freq * starts_arr
    vectors = np.exp(1j * aligned)
    plv = float(np.abs(vectors.mean()))
    circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(plv, 1e-12)))))
    return plv, circ_std, len(phases)


def extract_window_features(
    events: np.ndarray,
    start_s: float,
    end_s: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    start_us = start_s * 1e6
    end_us = end_s * 1e6
    left = int(np.searchsorted(events[:, 2], start_us, side="left"))
    right = int(np.searchsorted(events[:, 2], end_us, side="right"))
    subset = events[left:right]
    duration_s = max(end_s - start_s, 0.0)
    n_bins = max(2, int(math.ceil(duration_s / args.event_bin_s)))

    edges = start_us + np.arange(n_bins + 1, dtype=np.float64) * args.event_bin_s * 1e6
    edges[-1] = end_us
    if np.any(np.diff(edges) <= 0):
        return {"valid_fft": 0.0, "event_count": float(len(subset))}

    if len(subset) == 0:
        signed_rate = np.zeros(n_bins, dtype=np.float64)
        unsigned_rate = np.zeros(n_bins, dtype=np.float64)
    else:
        ts = subset[:, 2].astype(np.float64)
        polarity = subset[:, 3]
        weights = np.where(polarity > 0, 1.0, -1.0)
        signed_rate = np.histogram(ts, bins=edges, weights=weights)[0].astype(np.float64)
        unsigned_rate = np.histogram(ts, bins=edges)[0].astype(np.float64)

    centered = signed_rate - signed_rate.mean()
    if len(centered) > 3:
        signal = centered * np.hanning(len(centered))
    else:
        signal = centered

    coeff = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=args.event_bin_s)
    power = np.abs(coeff) ** 2
    pos_mask = freqs > 0
    band_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    total_power = float(power[pos_mask].sum())
    band_power = float(power[band_mask].sum())

    base = {
        "valid_fft": 1.0,
        "event_count": float(len(subset)),
        "unsigned_rate_mean": float(unsigned_rate.mean()) if len(unsigned_rate) else np.nan,
        "signed_rate_abs_mean": float(np.mean(np.abs(signed_rate))) if len(signed_rate) else np.nan,
        "signed_rate_std": float(np.std(signed_rate)) if len(signed_rate) else np.nan,
        "fft_total_power": total_power,
        "band_power": band_power,
        "band_energy_frac": band_power / total_power if total_power > 1e-12 else np.nan,
    }

    if band_power <= 1e-12 or not np.any(band_mask):
        base.update(
            {
                "dom_freq_hz": np.nan,
                "dom_power_frac_total": np.nan,
                "dom_power_frac_band": np.nan,
                "dom_phase_rad": np.nan,
                "dom_phase_sin": np.nan,
                "dom_phase_cos": np.nan,
                "band_entropy": np.nan,
                "phase_coherence": np.nan,
                "phase_circ_std": np.nan,
                "phase_n_windows": 0.0,
            }
        )
        return base

    band_indices = np.where(band_mask)[0]
    dom_idx = int(band_indices[np.argmax(power[band_indices])])
    dom_freq = float(freqs[dom_idx])
    dom_phase = float(np.angle(coeff[dom_idx]))
    plv, circ_std, n_phase = phase_coherence(
        signed_rate,
        args.event_bin_s,
        dom_freq,
        args.subwindow_s,
        args.hop_s,
    )

    base.update(
        {
            "dom_freq_hz": dom_freq,
            "dom_power_frac_total": float(power[dom_idx] / total_power) if total_power > 1e-12 else np.nan,
            "dom_power_frac_band": float(power[dom_idx] / band_power),
            "dom_phase_rad": dom_phase,
            "dom_phase_sin": float(np.sin(dom_phase)),
            "dom_phase_cos": float(np.cos(dom_phase)),
            "band_entropy": spectral_entropy(power[band_mask]),
            "phase_coherence": plv,
            "phase_circ_std": circ_std,
            "phase_n_windows": float(n_phase),
        }
    )
    return base


def extract_features(windows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if windows.empty:
        return windows

    rows = []
    total_groups = windows.groupby(["rec_name", "roi_id"]).ngroups
    print(f"[INFO] Extracting FFT features from {len(windows)} windows in {total_groups} ROI groups.")

    with h5py.File(args.data_path, "r") as hf:
        group_no = 0
        for (rec, roi), grp in windows.groupby(["rec_name", "roi_id"], sort=False):
            group_no += 1
            if group_no % 25 == 0 or group_no == 1:
                print(f"[INFO] ROI group {group_no}/{total_groups}: {rec}/{roi}", flush=True)
            if rec not in hf or roi not in hf[rec]:
                continue
            events = np.asarray(hf[rec][roi]["events"])
            for _, window in grp.iterrows():
                feats = extract_window_features(
                    events,
                    float(window["t_start_s"]),
                    float(window["t_end_s"]),
                    args,
                )
                item = window.to_dict()
                item.update(feats)
                rows.append(item)

    return pd.DataFrame(rows)


def auc_rank(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values)
    values = values[mask]
    labels = labels[mask].astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(values).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cohen_d(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    pooled = math.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    if pooled <= 1e-12:
        return np.nan
    return float((pos.mean() - neg.mean()) / pooled)


def summarize_features(features: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = feature_columns(features)

    summary_rows = []
    for (source, label), grp in features.groupby(["source", "label_group"], dropna=False):
        for col in feature_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if vals.empty:
                continue
            summary_rows.append(
                {
                    "source": source,
                    "label_group": label,
                    "feature": col,
                    "n": int(vals.size),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "median": float(vals.median()),
                    "q25": float(vals.quantile(0.25)),
                    "q75": float(vals.quantile(0.75)),
                }
            )
    summary = pd.DataFrame(summary_rows)

    sep_rows = []
    for source, grp in features.groupby("source"):
        binary = grp[grp["label_group"].isin(["ed", "flap"])].copy()
        if binary.empty:
            continue
        sep_rows.extend(binary_separability_rows(binary, source, feature_cols))
    separability = pd.DataFrame(sep_rows).sort_values(
        ["source", "auc_best_direction"], ascending=[True, False]
    )

    summary.to_csv(out_dir / "feature_summary_by_label.csv", index=False)
    separability.to_csv(out_dir / "feature_separability.csv", index=False)
    return summary, separability


def feature_columns(features: pd.DataFrame) -> list[str]:
    candidates = [
        "event_count",
        "duration_s",
        "unsigned_rate_mean",
        "signed_rate_abs_mean",
        "signed_rate_std",
        "band_energy_frac",
        "dom_freq_hz",
        "dom_power_frac_total",
        "dom_power_frac_band",
        "dom_phase_sin",
        "dom_phase_cos",
        "band_entropy",
        "phase_coherence",
        "phase_circ_std",
        "phase_n_windows",
        "proposal_score",
        "cnn_score_t2",
    ]
    return [col for col in candidates if col in features.columns]


def binary_separability_rows(
    binary: pd.DataFrame,
    source: str,
    feature_cols: list[str],
    pos_name: str = "ed",
    neg_name: str = "flap",
) -> list[dict]:
    rows = []
    y = (binary["label_group"] == pos_name).to_numpy()
    for col in feature_cols:
        vals = pd.to_numeric(binary[col], errors="coerce").to_numpy(dtype=np.float64)
        pos_vals = vals[y]
        neg_vals = vals[~y]
        auc = auc_rank(vals, y)
        rows.append(
            {
                "source": source,
                "feature": col,
                f"n_{pos_name}": int(np.isfinite(pos_vals).sum()),
                f"n_{neg_name}": int(np.isfinite(neg_vals).sum()),
                f"mean_{pos_name}": float(np.nanmean(pos_vals)) if np.isfinite(pos_vals).any() else np.nan,
                f"mean_{neg_name}": float(np.nanmean(neg_vals)) if np.isfinite(neg_vals).any() else np.nan,
                f"median_{pos_name}": float(np.nanmedian(pos_vals)) if np.isfinite(pos_vals).any() else np.nan,
                f"median_{neg_name}": float(np.nanmedian(neg_vals)) if np.isfinite(neg_vals).any() else np.nan,
                f"cohen_d_{pos_name}_minus_{neg_name}": cohen_d(pos_vals, neg_vals),
                f"auc_{pos_name}_high": auc,
                "auc_best_direction": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
            }
        )
    return rows


def score_error_separability(
    features: pd.DataFrame,
    out_dir: Path,
    high_score_threshold: float,
) -> pd.DataFrame:
    if "cnn_score_t2" not in features.columns:
        return pd.DataFrame()

    proposals = features[features["source"] == "proposal"].copy()
    if proposals.empty:
        return pd.DataFrame()

    feature_cols = feature_columns(features)
    comparisons = []
    specs = [
        (
            "proposal_ed_vs_background_all",
            proposals["label_group"] == "ed",
            proposals["label_group"] == "background",
        ),
        (
            f"proposal_ed_vs_background_cnn_ge_{high_score_threshold:g}",
            proposals["label_group"] == "ed",
            (proposals["label_group"] == "background")
            & (pd.to_numeric(proposals["cnn_score_t2"], errors="coerce") >= high_score_threshold),
        ),
        (
            f"proposal_ed_vs_flap_cnn_ge_{high_score_threshold:g}",
            proposals["label_group"] == "ed",
            (proposals["label_group"] == "flap")
            & (pd.to_numeric(proposals["cnn_score_t2"], errors="coerce") >= high_score_threshold),
        ),
    ]

    for name, pos_mask, neg_mask in specs:
        binary = pd.concat(
            [
                proposals[pos_mask].assign(label_group="ed"),
                proposals[neg_mask].assign(label_group="hard_negative"),
            ],
            ignore_index=True,
        )
        if binary["label_group"].nunique() < 2:
            continue
        rows = binary_separability_rows(
            binary,
            name,
            feature_cols,
            pos_name="ed",
            neg_name="hard_negative",
        )
        comparisons.extend(rows)

    result = pd.DataFrame(comparisons)
    if not result.empty:
        result = result.sort_values(["source", "auc_best_direction"], ascending=[True, False])
        result.to_csv(out_dir / "score_error_separability.csv", index=False)
    return result


def plot_top_features(features: pd.DataFrame, separability: pd.DataFrame, out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for source in separability["source"].dropna().unique():
        top = separability[separability["source"] == source].head(6)
        data = features[(features["source"] == source) & (features["label_group"].isin(["ed", "flap"]))]
        if data.empty:
            continue
        for feature in top["feature"]:
            vals_ed = pd.to_numeric(data[data["label_group"] == "ed"][feature], errors="coerce").dropna()
            vals_flap = pd.to_numeric(data[data["label_group"] == "flap"][feature], errors="coerce").dropna()
            if vals_ed.empty or vals_flap.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(vals_ed, bins=30, alpha=0.6, label="ED", density=True)
            ax.hist(vals_flap, bins=30, alpha=0.6, label="Flap", density=True)
            ax.set_title(f"{source}: {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("density")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{source}_{feature}.png", dpi=140)
            plt.close(fig)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_Sen datos._"

    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda value: "" if pd.isna(value) else format(float(value), floatfmt)
            )
        else:
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(
    features: pd.DataFrame,
    separability: pd.DataFrame,
    score_errors: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    counts = (
        features.groupby(["source", "label_group"])
        .size()
        .rename("n")
        .reset_index()
    )
    lines = [
        "# FFT phase analysis",
        "",
        f"Split: `{args.split}`",
        f"Event bin: `{args.event_bin_s}` s",
        f"Frequency band: `{args.freq_min}-{args.freq_max}` Hz",
        f"Subwindow/hop for phase coherence: `{args.subwindow_s}` s / `{args.hop_s}` s",
        "",
        "## Window counts",
        "",
        markdown_table(counts, floatfmt=".0f"),
        "",
        "## Best ED-vs-flap separability",
        "",
    ]

    for source in separability["source"].dropna().unique():
        top = separability[separability["source"] == source].head(10).copy()
        lines.extend(
            [
                f"### {source}",
                "",
                markdown_table(
                    top[
                        [
                            "feature",
                            "n_ed",
                            "n_flap",
                            "mean_ed",
                            "mean_flap",
                            "cohen_d_ed_minus_flap",
                            "auc_ed_high",
                            "auc_best_direction",
                        ]
                    ]
                ),
                "",
            ]
        )

    phase_rows = separability[
        separability["feature"].isin(
            ["dom_phase_sin", "dom_phase_cos", "phase_coherence", "phase_circ_std"]
        )
    ].copy()
    if not phase_rows.empty:
        lines.extend(
            [
                "## Phase-specific features",
                "",
                markdown_table(
                    phase_rows[
                        [
                            "source",
                            "feature",
                            "n_ed",
                            "n_flap",
                            "mean_ed",
                            "mean_flap",
                            "cohen_d_ed_minus_flap",
                            "auc_best_direction",
                        ]
                    ]
                ),
                "",
            ]
        )

    if not score_errors.empty:
        lines.extend(["## CNN score-error comparisons", ""])
        for source in score_errors["source"].dropna().unique():
            top = score_errors[score_errors["source"] == source].head(10).copy()
            lines.extend(
                [
                    f"### {source}",
                    "",
                    markdown_table(
                        top[
                            [
                                "feature",
                                "n_ed",
                                "n_hard_negative",
                                "mean_ed",
                                "mean_hard_negative",
                                "cohen_d_ed_minus_hard_negative",
                                "auc_ed_high",
                                "auc_best_direction",
                            ]
                        ]
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Files",
            "",
            "- `window_features.csv`",
            "- `feature_summary_by_label.csv`",
            "- `feature_separability.csv`",
            "- `score_error_separability.csv`",
            "- `plots/`",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.features_csv:
        features = pd.read_csv(args.features_csv)
        print(f"[INFO] Reusing features: {args.features_csv} ({len(features)} rows)")
    else:
        rec_splits = load_recording_splits(args.data_path)
        ann_windows = load_annotation_windows(
            args.ann_path,
            rec_splits,
            args.split,
            args.min_duration,
        )
        print(f"[INFO] Annotation windows: {len(ann_windows)}")

        all_windows = [ann_windows]
        if args.include_proposals:
            proposals = load_proposals(args, rec_splits, out_dir)
            prop_windows = proposal_windows(proposals, ann_windows, args)
            print(f"[INFO] Proposal windows: {len(prop_windows)}")
            all_windows.append(prop_windows)

        windows = pd.concat(all_windows, ignore_index=True)
        features = extract_features(windows, args)
        features.to_csv(out_dir / "window_features.csv", index=False)
        print(f"[INFO] Saved features: {out_dir / 'window_features.csv'}")

    _, separability = summarize_features(features, out_dir)
    score_errors = score_error_separability(features, out_dir, args.high_score_threshold)
    if not args.no_plots:
        plot_top_features(features, separability, out_dir)
    write_report(features, separability, score_errors, args, out_dir)

    print("\n=== Top separability ===")
    for source in separability["source"].dropna().unique():
        print(f"\n[{source}]")
        cols = ["feature", "n_ed", "n_flap", "mean_ed", "mean_flap", "auc_ed_high", "auc_best_direction"]
        print(separability[separability["source"] == source].head(8)[cols].to_string(index=False))

    if not score_errors.empty:
        print("\n=== CNN score-error comparisons ===")
        for source in score_errors["source"].dropna().unique():
            print(f"\n[{source}]")
            cols = [
                "feature",
                "n_ed",
                "n_hard_negative",
                "mean_ed",
                "mean_hard_negative",
                "auc_ed_high",
                "auc_best_direction",
            ]
            print(score_errors[score_errors["source"] == source].head(8)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
