"""Debug runner for the reTAG proposal pipeline.

Executes variants over all recordings (train + val + test), saves per-variant
CSVs to tmp/, and generates distribution plots comparing baseline vs optimal.

Run from the event_penguins/ directory:
    python dev/run_retag_debug.py
    python dev/run_retag_debug.py --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.proposals import ProposalGenerator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",    default="data/preprocessed.h5")
    p.add_argument("--out-dir",      default="tmp/debug")
    p.add_argument("--split",        default=None,
                   help="Split to process (train/val/test). Default: all.")
    p.add_argument("--bin-width",    type=float, default=0.033)
    p.add_argument("--percentile",   type=float, default=1.0)
    p.add_argument("--nms-threshold",type=float, default=0.95)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Variants — rely on ProposalGenerator defaults for tuned values
# ---------------------------------------------------------------------------

_VARIANTS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("adaptive", {
        "use_adaptive_lambda": True,
    }),
    ("adaptive_spatial", {
        "use_adaptive_lambda":     True,
        "use_spatial_compactness": True,
    }),
    ("adaptive_spatial_noise", {
        "use_adaptive_lambda":     True,
        "use_spatial_compactness": True,
        "use_noise_penalization":  True,
    }),
    ("all", {
        "use_adaptive_lambda":     True,
        "use_spatial_compactness": True,
        "use_noise_penalization":  True,
        "use_dispersed_noise":     True,
    }),
]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reduction_distribution(
    baseline_df: pd.DataFrame,
    optimal_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Histogram of per-ROI proposal reduction (%) vs baseline."""
    def per_roi_counts(df: pd.DataFrame) -> pd.Series:
        return df.groupby(["rec_name", "roi_id"]).size()

    base_counts = per_roi_counts(baseline_df)
    opt_counts  = per_roi_counts(optimal_df).reindex(base_counts.index, fill_value=0)
    reduction   = (1 - opt_counts / base_counts) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, 91, 2)
    ax.hist(reduction, bins=bins, color="#9C27B0", alpha=0.8, label="Redución NMS (%)")
    ax.axvline(reduction.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Media = {reduction.mean():.1f} %")
    ax.set_xlabel("Redución vs baseline (%)")
    ax.set_ylabel("Número de ROIs")
    ax.set_title("Distribución da redución de propostas por ROI\n(Baseline → Óptimo, tras NMS)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Gráfica gardada en {out_path}")


def plot_totals(results: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Bar chart of total proposal count after NMS per variant."""
    names  = list(results.keys())
    counts = [len(df) for df in results.values()]
    colors = ["#555555", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    baseline = counts[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, counts, color=colors[:len(names)])
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1,
               label=f"Baseline = {baseline:,}")
    for bar, count in zip(bars, counts):
        pct   = (count - baseline) / baseline * 100
        label = f"{count:,}\n({pct:+.1f}%)" if count != baseline else f"{count:,}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + baseline * 0.005,
                label, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Propostas (tras NMS)")
    ax.set_title("Total de propostas por variante — todos os splits")
    ax.legend()
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Gráfica gardada en {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args    = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = dict(
        data_path     = args.data_path,
        bin_width     = args.bin_width,
        percentile    = args.percentile,
        nms_threshold = args.nms_threshold,
        output_dir    = str(out_dir / "logs"),
    )

    results: dict[str, pd.DataFrame] = {}

    for name, kwargs in _VARIANTS:
        print(f"\n=== {name.upper()} ===")
        gen = ProposalGenerator(**base_cfg, **kwargs)
        df  = gen.run(split=args.split)
        df.to_csv(out_dir / f"proposals_{name}.csv", index=False)
        results[name] = df

        dur = (df["t_end"] - df["t_start"]) / 1e6
        print(f"  n_proposals : {len(df):>10,}")
        print(f"  mean_dur_s  : {dur.mean():>10.1f}")
        print(f"  max_dur_s   : {dur.max():>10.1f}")

    print("\n=== RESUMO ===")
    summary = pd.DataFrame({
        name: {
            "n_proposals": len(df),
            "mean_dur_s":  round((df["t_end"] - df["t_start"]).mean() / 1e6, 1),
            "max_dur_s":   round((df["t_end"] - df["t_start"]).max()  / 1e6, 1),
        }
        for name, df in results.items()
    }).T
    print(summary.to_string())

    print("\n[INFO] Xerando gráficas...")
    plot_totals(results, out_dir / "totals_by_variant.png")
    plot_reduction_distribution(
        results["baseline"], results["all"],
        out_dir / "reduction_distribution.png",
    )
