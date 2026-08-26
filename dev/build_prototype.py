"""Build and evaluate the ED spatial prototype.

Constructs the prototype from training annotations, saves it to disk,
visualises it, and evaluates whether adding it improves AR on the val split
without losing recall.

Run from the event_penguins/ directory:
    python dev/build_prototype.py
    python dev/build_prototype.py --grid-h 32 --grid-w 32 --proto-weight 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

from src.prototype import build_ed_prototype, get_prototype_score
from src.proposals import ProposalGenerator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",     default="data/preprocessed.h5")
    p.add_argument("--ann-path",      default="config/annotations/annotations.json")
    p.add_argument("--out-dir",       default="tmp/prototype")
    p.add_argument("--proto-path",    default="tmp/prototype/ed_prototype.npy",
                   help="Where to save/load the prototype.")
    p.add_argument("--grid-h",        type=int,   default=16)
    p.add_argument("--grid-w",        type=int,   default=16)
    p.add_argument("--proto-weight",  type=float, default=0.3)
    p.add_argument("--bin-width",     type=float, default=0.033)
    p.add_argument("--percentile",    type=float, default=1.0)
    p.add_argument("--nms-threshold", type=float, default=0.95)
    p.add_argument("--min-duration",  type=float, default=2.0)
    p.add_argument("--eval-split",    default="val",
                   help="Split to evaluate on (never 'test' during development).")
    p.add_argument("--tiou",          type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.9])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ground truth (same logic as eval_proposals.py)
# ---------------------------------------------------------------------------

def load_ground_truth(ann_path, recordings, min_duration):
    with open(ann_path) as f:
        ann = json.load(f)
    rows = []
    for rec, v in ann["database"].items():
        if rec not in recordings:
            continue
        for roi_str, roi_anns in v["annotations"].items():
            if roi_str == "null":
                continue
            roi_id = f"N{int(roi_str):02d}"
            for a in roi_anns:
                if a["label"] != "ed":
                    continue
                t_start, t_end = a["segment"]
                if (t_end - t_start) >= min_duration:
                    rows.append({"rec_name": rec, "roi_id": roi_id,
                                 "t_start": t_start, "t_end": t_end})
    return pd.DataFrame(rows)


def compute_recall(proposals, ground_truth, tiou_thresholds):
    props = proposals.copy()
    props["t_start"] = props["t_start"] / 1e6
    props["t_end"]   = props["t_end"]   / 1e6
    grouped = {k: g for k, g in props.groupby(["rec_name", "roi_id"])}
    tp = np.zeros(len(tiou_thresholds))
    for _, gt in ground_truth.iterrows():
        key = (gt["rec_name"], gt["roi_id"])
        if key not in grouped:
            continue
        g = grouped[key]
        inter = np.maximum(np.minimum(gt["t_end"], g["t_end"].values)
                           - np.maximum(gt["t_start"], g["t_start"].values), 0.0)
        union = (g["t_end"].values - g["t_start"].values) + (gt["t_end"] - gt["t_start"]) - inter
        tp += tiou_thresholds <= (inter / np.maximum(union, 1e-9)).max()
    return tp / len(ground_truth)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_prototype(prototype: np.ndarray, out_path: Path, n_instances: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(prototype, cmap="hot", origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="Densidade relativa (L2-norm)")
    ax.set_title(f"Prototipo ED — {n_instances} instancias de train\n"
                 f"Grid {prototype.shape[0]}×{prototype.shape[1]}")
    ax.set_xlabel("x (columnas ROI →)")
    ax.set_ylabel("y (filas ROI ↓)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Prototipo visualizado en {out_path}")


def plot_score_distributions(
    baseline_df: pd.DataFrame,
    proto_df: pd.DataFrame,
    prototype: np.ndarray,
    data_path: str,
    ann_path: str,
    split: str,
    out_path: Path,
) -> None:
    """Distribution of per-proposal prototype scores for ED vs non-ED proposals."""
    with open(ann_path) as f:
        ann = json.load(f)
    with h5py.File(data_path, "r") as hf:
        split_recs = {r for r in hf if hf[r].attrs.get("split") == split}

    # Collect GT intervals
    gt_intervals: dict[tuple, list] = {}
    for rec, v in ann["database"].items():
        if rec not in split_recs:
            continue
        for roi_str, roi_anns in v["annotations"].items():
            if roi_str == "null":
                continue
            roi_id = f"N{int(roi_str):02d}"
            for a in roi_anns:
                if a["label"] == "ed":
                    gt_intervals.setdefault((rec, roi_id), []).append(a["segment"])

    # Compute prototype score for each proposal in baseline_df
    scores_ed, scores_non_ed = [], []

    with h5py.File(data_path, "r") as hf:
        for (rec, roi_id), grp in baseline_df.groupby(["rec_name", "roi_id"]):
            if rec not in hf or roi_id not in hf[rec]:
                continue
            roi_height = int(hf[rec][roi_id].attrs["height"])
            roi_width  = int(hf[rec][roi_id].attrs["width"])
            all_events = np.array(hf[rec][roi_id]["events"])
            intervals  = gt_intervals.get((rec, roi_id), [])

            for _, row in grp.iterrows():
                ts = row["t_start"] / 1e6
                te = row["t_end"]   / 1e6
                mask   = (all_events[:, 2] >= row["t_start"]) & (all_events[:, 2] <= row["t_end"])
                events = all_events[mask]
                if len(events) < 5:
                    continue

                # Compute mean prototype score for this proposal's events
                from src.prototype import _events_to_grid
                grid  = _events_to_grid(events, roi_height, roi_width,
                                        prototype.shape[0], prototype.shape[1])
                norm  = np.linalg.norm(grid.ravel())
                score = float(np.dot(grid.ravel() / norm, prototype.ravel())) if norm > 0 else 0.0
                score = max(0.0, score)

                # Is this proposal a true positive (IoU ≥ 0.5 with any GT)?
                is_ed = any(
                    (min(te, g[1]) - max(ts, g[0])) /
                    max((te - ts) + (g[1] - g[0]) - (min(te, g[1]) - max(ts, g[0])), 1e-9) >= 0.5
                    for g in intervals
                )
                (scores_ed if is_ed else scores_non_ed).append(score)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 30)
    ax.hist(scores_non_ed, bins=bins, alpha=0.6, color="#FF9800", label=f"Non-ED ({len(scores_non_ed)})")
    ax.hist(scores_ed,     bins=bins, alpha=0.7, color="#4CAF50", label=f"ED ({len(scores_ed)})")
    ax.set_xlabel("Similitude co prototipo ED")
    ax.set_ylabel("Nº propostas")
    ax.set_title("Distribución de scores — propostas ED vs non-ED")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Distribución de scores gardada en {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args    = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiou = np.array(args.tiou)

    # --- Build or load prototype ---
    proto_path = Path(args.proto_path)
    if proto_path.exists():
        prototype = np.load(proto_path)
        print(f"[INFO] Prototipo cargado dende {proto_path}  shape={prototype.shape}")
    else:
        print("[INFO] Construíndo prototipo dende o train split...")
        prototype = build_ed_prototype(
            args.data_path, args.ann_path,
            split="train",
            grid_h=args.grid_h, grid_w=args.grid_w,
        )
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proto_path, prototype)
        print(f"[INFO] Prototipo gardado en {proto_path}")

    # Count training instances used (re-read for display)
    with open(args.ann_path) as f:
        ann = json.load(f)
    with h5py.File(args.data_path, "r") as hf:
        train_recs = {r for r in hf if hf[r].attrs.get("split") == "train"}
    n_instances = sum(
        1 for rec, v in ann["database"].items() if rec in train_recs
        for roi_anns in v["annotations"].values()
        for a in roi_anns if a["label"] == "ed"
    )
    plot_prototype(prototype, out_dir / "prototype_heatmap.png", n_instances)

    # --- Evaluate on val split ---
    with h5py.File(args.data_path, "r") as hf:
        eval_recs = {r for r in hf if hf[r].attrs.get("split") == args.eval_split}

    gt = load_ground_truth(args.ann_path, eval_recs, args.min_duration)
    print(f"\n[INFO] GT instances ({args.eval_split}, ≥{args.min_duration}s): {len(gt)}")

    base_cfg = dict(
        data_path=args.data_path, bin_width=args.bin_width,
        percentile=args.percentile, nms_threshold=args.nms_threshold,
        output_dir=str(out_dir / "logs"),
        use_adaptive_lambda=True, use_spatial_compactness=True,
        use_noise_penalization=True, use_dispersed_noise=True,
    )

    variants = [
        ("Óptimo (sen prototipo)", {**base_cfg}),
        ("+ Prototipo ED",         {**base_cfg, "prototype": prototype,
                                    "prototype_weight": args.proto_weight}),
    ]

    results = []
    for name, cfg in variants:
        print(f"\n[RUN] {name}")
        gen = ProposalGenerator(**cfg)
        df  = gen.run(split=args.eval_split)
        ar  = compute_recall(df, gt, tiou)
        print(f"      n={len(df):,}  AR@0.5={ar[tiou==0.5][0]:.4f}  AR_mean={ar.mean():.4f}")
        results.append({"name": name, "df": df, "ar": ar, "n": len(df)})

    # --- AR comparison plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#555555", "#9C27B0"]
    for res, color in zip(results, colors):
        ax.plot(tiou, res["ar"], marker="o", label=res["name"], color=color, linewidth=2)
    ax.set_xlabel("tIoU")
    ax.set_ylabel("Action Recall")
    ax.set_title(f"Impacto do prototipo ED — {args.eval_split} split")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "ar_comparison.png", dpi=150)
    plt.close(fig)

    # --- Score distribution plot ---
    plot_score_distributions(
        results[0]["df"], results[1]["df"],
        prototype, args.data_path, args.ann_path, args.eval_split,
        out_dir / "score_distributions.png",
    )

    print("\n=== RESUMO ===")
    for res in results:
        ar = res["ar"]
        print(f"  {res['name']:<30}  n={res['n']:>8,}  "
              f"AR@0.5={ar[tiou==0.5][0]:.4f}  AR_mean={ar.mean():.4f}")
