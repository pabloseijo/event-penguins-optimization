"""Grid search over proposal hyperparameters on the validation split.

Sweeps adaptive-lambda and spatial-compactness parameters and ranks combinations
by Action Recall. Results are saved as CSV and visualised as heatmaps so that
the best configuration can be selected before touching the test split.

Run from the event_penguins/ directory:
    python dev/tune_proposals.py
    python dev/tune_proposals.py --metric AR@0.5 --out-dir tmp/tune
    python dev/tune_proposals.py --lp-values 65 70 75 80 85 --ld-values 0.05 0.10 0.15 0.20
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.proposals import ProposalGenerator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter grid search for reTAG proposals.")
    p.add_argument("--data-path",  default="data/preprocessed.h5")
    p.add_argument("--ann-path",   default="config/annotations/annotations.json")
    p.add_argument("--out-dir",    default="tmp/tune")
    p.add_argument("--split",      default="val",
                   help="HDF5 split to tune on (default: val). Never use 'test' here.")
    p.add_argument("--bin-width",  type=float, default=0.033)
    p.add_argument("--percentile", type=float, default=1.0)
    p.add_argument("--nms-threshold", type=float, default=0.95)
    p.add_argument("--min-duration",  type=float, default=2.0)
    p.add_argument("--tiou",          type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--metric",        default="AR@0.5",
                   help="Primary metric to maximise (e.g. AR@0.5, AR@0.9, AR_mean).")

    # Grid definitions — pass explicit lists to override defaults
    p.add_argument("--lp-values",  type=float, nargs="+",
                   default=[65, 70, 75, 80, 85],
                   help="lambda_percentile values to sweep.")
    p.add_argument("--ld-values",  type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.25],
                   help="lambda_delta values to sweep.")
    p.add_argument("--sw-values",  type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5],
                   help="spatial_weight values to sweep.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ground truth (shared with eval_proposals.py — kept local to avoid coupling)
# ---------------------------------------------------------------------------

def load_ground_truth(ann_path: str, recordings: set, min_duration: float) -> pd.DataFrame:
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


def compute_proposal_recall(
    proposals: pd.DataFrame,
    ground_truth: pd.DataFrame,
    tiou_thresholds: np.ndarray,
) -> np.ndarray:
    props = proposals.copy()
    props["t_start"] = props["t_start"] / 1e6
    props["t_end"]   = props["t_end"]   / 1e6
    grouped = {key: grp for key, grp in props.groupby(["rec_name", "roi_id"])}
    tp = np.zeros(len(tiou_thresholds))
    for _, gt in ground_truth.iterrows():
        key = (gt["rec_name"], gt["roi_id"])
        if key not in grouped:
            continue
        roi_props = grouped[key]
        starts = roi_props["t_start"].values
        ends   = roi_props["t_end"].values
        tt1 = np.maximum(gt["t_start"], starts)
        tt2 = np.minimum(gt["t_end"],   ends)
        inter = np.maximum(tt2 - tt1, 0.0)
        union = (ends - starts) + (gt["t_end"] - gt["t_start"]) - inter
        tp   += tiou_thresholds <= (inter / np.maximum(union, 1e-9)).max()
    return tp / len(ground_truth)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metric(ar: np.ndarray, tiou: np.ndarray, metric: str) -> float:
    """Pull a scalar from the AR array by metric name (e.g. 'AR@0.5', 'AR_mean')."""
    if metric == "AR_mean":
        return float(ar.mean())
    # Expect format "AR@<threshold>"
    try:
        t = float(metric.split("@")[1])
        idx = np.argmin(np.abs(tiou - t))
        return float(ar[idx])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Unknown metric '{metric}'. Use 'AR_mean' or 'AR@<tIoU>' "
            f"where tIoU is one of {tiou.tolist()}."
        ) from exc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _heatmap(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    fixed_col: str,
    fixed_val: float,
    out_path: Path,
) -> None:
    """Plot a heatmap of z_col for the slice where fixed_col == fixed_val."""
    subset = data[np.isclose(data[fixed_col], fixed_val)]
    pivot  = subset.pivot_table(index=y_col, columns=x_col, values=z_col)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn",
                   vmin=data[z_col].min(), vmax=data[z_col].max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{z_col}  [{fixed_col}={fixed_val:.2f}]")
    plt.colorbar(im, ax=ax)

    for i, row in enumerate(pivot.values):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                    color="black" if val < 0.9 * pivot.values.max() else "white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Heatmap gardado en {out_path}")


def plot_heatmaps(df: pd.DataFrame, metric: str, out_dir: Path, best: pd.Series) -> None:
    """Three heatmaps, each fixing one parameter at its best value."""
    pairs = [
        ("lambda_delta",    "spatial_weight",    "lambda_percentile", best["lambda_percentile"]),
        ("lambda_percentile","spatial_weight",   "lambda_delta",      best["lambda_delta"]),
        ("lambda_percentile","lambda_delta",     "spatial_weight",    best["spatial_weight"]),
    ]
    for x, y, fixed, val in pairs:
        fname = f"heatmap_{x}_vs_{y}.png"
        _heatmap(df, x, y, metric, fixed, val, out_dir / fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args     = _parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiou     = np.array(args.tiou)

    if args.split == "test":
        print("[WARN] You are tuning on the test split — this leaks information. "
              "Use --split val instead.")

    with h5py.File(args.data_path, "r") as hf:
        split_recs = {r for r in hf if hf[r].attrs.get("split") == args.split}

    if not split_recs:
        sys.exit(f"[ERROR] No recordings found for split='{args.split}' in {args.data_path}")

    gt = load_ground_truth(args.ann_path, split_recs, args.min_duration)
    print(f"[INFO] GT instances ({args.split} split, ≥{args.min_duration}s): {len(gt)}")

    if len(gt) == 0:
        sys.exit("[ERROR] No ground truth instances found. Check --ann-path and --split.")

    # Build the full grid
    grid = list(itertools.product(args.lp_values, args.ld_values, args.sw_values))
    print(f"[INFO] Grid size: {len(grid)} combinations "
          f"(lp×{len(args.lp_values)} × ld×{len(args.ld_values)} × sw×{len(args.sw_values)})")

    base_cfg = dict(
        data_path     = args.data_path,
        bin_width     = args.bin_width,
        percentile    = args.percentile,
        nms_threshold = args.nms_threshold,
        output_dir    = str(out_dir / "logs"),
    )

    records = []
    for i, (lp, ld, sw) in enumerate(grid, 1):
        print(f"[{i:3d}/{len(grid)}] lp={lp:.0f}  ld={ld:.2f}  sw={sw:.1f}", end="  ", flush=True)
        gen = ProposalGenerator(
            **base_cfg,
            use_adaptive_lambda     = True,
            lambda_percentile       = lp,
            lambda_delta            = ld,
            use_spatial_compactness = True,
            spatial_weight          = sw,
            use_noise_penalization  = True,
            use_dispersed_noise     = True,
        )
        df  = gen.run(split=args.split)
        ar  = compute_proposal_recall(df, gt, tiou)
        val = extract_metric(ar, tiou, args.metric)
        print(f"n={len(df):6d}  {args.metric}={val:.4f}")

        row = {"lambda_percentile": lp, "lambda_delta": ld, "spatial_weight": sw,
               "n_proposals": len(df), args.metric: val, "AR_mean": float(ar.mean())}
        for t, v in zip(tiou, ar):
            row[f"AR@{t}"] = float(v)
        records.append(row)

    # Primary sort: metric descending. Tiebreaker: fewer proposals (cheaper for CNN).
    results = pd.DataFrame(records).sort_values(
        [args.metric, "n_proposals"], ascending=[False, True]
    )

    csv_path = out_dir / "grid_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Resultados gardados en {csv_path}")

    best = results.iloc[0]
    print("\n=== MELLOR COMBINACIÓN ===")
    print(f"  lambda_percentile : {best['lambda_percentile']:.0f}")
    print(f"  lambda_delta      : {best['lambda_delta']:.2f}")
    print(f"  spatial_weight    : {best['spatial_weight']:.1f}")
    print(f"  n_proposals       : {best['n_proposals']:.0f}")
    print(f"  {args.metric}     : {best[args.metric]:.4f}")
    print(f"  AR_mean           : {best['AR_mean']:.4f}")

    print("\n=== TOP 10 ===")
    cols = ["lambda_percentile", "lambda_delta", "spatial_weight",
            "n_proposals", args.metric, "AR_mean"]
    print(results[cols].head(10).to_string(index=False))

    plot_heatmaps(results, args.metric, out_dir, best)

    # Write a short recommendation file
    rec_path = out_dir / "recomendacion.md"
    rec_path.write_text(
        f"# Recomendación de hiperparámetros\n\n"
        f"**Split de tuning:** {args.split}  \n"
        f"**Métrica:** {args.metric}  \n"
        f"**Gravacións:** {len(split_recs)} | GT instances: {len(gt)}\n\n"
        f"## Mellor combinación\n\n"
        f"| Parámetro | Valor |\n|---|---|\n"
        f"| `lambda_percentile` | {best['lambda_percentile']:.0f} |\n"
        f"| `lambda_delta` | {best['lambda_delta']:.2f} |\n"
        f"| `spatial_weight` | {best['spatial_weight']:.1f} |\n\n"
        f"| Métrica | Valor |\n|---|---|\n"
        f"| Propostas | {best['n_proposals']:.0f} |\n"
        f"| {args.metric} | {best[args.metric]:.4f} |\n"
        f"| AR medio | {best['AR_mean']:.4f} |\n\n"
        f"## Top 10\n\n"
        + results[cols].head(10).to_string(index=False)
        + f"\n\n*Xerado por `dev/tune_proposals.py`. Ver `grid_results.csv` para todos os resultados.*\n",
        encoding="utf-8",
    )
    print(f"[INFO] Recomendación gardada en {rec_path}")
