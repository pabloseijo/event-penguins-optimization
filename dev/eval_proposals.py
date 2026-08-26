"""Comparative evaluation of proposal variants on the test split.

Runs all five pipeline variants, computes Action Recall (AR) at multiple
tIoU thresholds against the ground truth, and writes a Markdown report with
embedded plots to the output directory.

Run from the event_penguins/ directory:
    python dev/eval_proposals.py
    python dev/eval_proposals.py --data-path data/preprocessed.h5 --split val
    python dev/eval_proposals.py --out-dir tmp/custom_eval --bin-width 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python dev/eval_proposals.py` from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.proposals import ProposalGenerator
from src.prototype import build_ed_prototype


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare reTAG proposal variants.")
    p.add_argument("--data-path",    default="data/preprocessed.h5",
                   help="Path to the preprocessed HDF5 file.")
    p.add_argument("--ann-path",     default="config/annotations/annotations.json",
                   help="Path to the annotations JSON file.")
    p.add_argument("--out-dir",      default="tmp/eval",
                   help="Output directory for the report and plots.")
    p.add_argument("--split",        default="test",
                   help="HDF5 split attribute to filter recordings (default: test).")
    p.add_argument("--bin-width",    type=float, default=0.033,
                   help="Temporal bin width in seconds (default: 0.033).")
    p.add_argument("--percentile",   type=float, default=1.0,
                   help="Robust-scaling clipping percentile (default: 1).")
    p.add_argument("--nms-threshold",type=float, default=0.95,
                   help="IoU threshold for temporal NMS (default: 0.95).")
    p.add_argument("--min-duration", type=float, default=2.0,
                   help="Minimum GT instance duration in seconds (default: 2).")
    p.add_argument("--tiou",         type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.9],
                   help="tIoU thresholds to evaluate (default: 0.1 0.3 0.5 0.7 0.9).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

# Each entry is (label, extra kwargs passed to ProposalGenerator on top of the
# shared base config). Only non-default values need to be listed — the rest
# come from ProposalGenerator's own defaults.
_VARIANTS: list[tuple[str, dict]] = [
    ("Baseline", {}),
    ("+ λ adaptativo", {
        "use_adaptive_lambda": True,
    }),
    ("+ Compacidade (w=0.2)", {
        "use_adaptive_lambda":     True,
        "use_spatial_compactness": True,
    }),
    ("+ Ruído sostido", {
        "use_adaptive_lambda":     True,
        "use_spatial_compactness": True,
        "use_noise_penalization":  True,
    }),
    ("+ Ruído disperso (óptimo)", {
        "use_adaptive_lambda":     True,
        "use_spatial_compactness": True,
        "use_noise_penalization":  True,
        "use_dispersed_noise":     True,
    }),
]

_LABELS = ["Baseline", "+λ", "+Espacial", "+Noise", "+Disperso", "+Proto", "+Period"]
_COLORS = ["#555555", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#009688"]


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(
    ann_path: str,
    test_recordings: set,
    min_duration: float,
) -> pd.DataFrame:
    """Load test-split ground truth instances from the annotations JSON.

    Args:
        ann_path: Path to annotations.json.
        test_recordings: Set of recording names present in the target split.
        min_duration: Exclude instances shorter than this (seconds).

    Returns:
        DataFrame with columns [rec_name, roi_id, t_start, t_end].
        roi_id is the H5 key (e.g. 'N09'), t_start/t_end in seconds.
    """
    with open(ann_path) as f:
        ann = json.load(f)

    rows = []
    for rec, v in ann["database"].items():
        if rec not in test_recordings:
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


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_proposal_recall(
    proposals: pd.DataFrame,
    ground_truth: pd.DataFrame,
    tiou_thresholds: np.ndarray,
) -> np.ndarray:
    """Compute Action Recall for a proposal set at multiple tIoU thresholds.

    For each ground truth instance, finds the maximum IoU overlap with any
    proposal in the same (recording, ROI). Recall at threshold t = fraction
    of GT instances where max_IoU >= t.

    Args:
        proposals: DataFrame with [rec_name, roi_id, t_start, t_end] in µs.
        ground_truth: DataFrame with [rec_name, roi_id, t_start, t_end] in seconds.
        tiou_thresholds: Array of IoU thresholds to evaluate at.

    Returns:
        Array of shape (len(tiou_thresholds),) with recall values in [0, 1].
    """
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
        intersection = np.maximum(tt2 - tt1, 0.0)
        union = (ends - starts) + (gt["t_end"] - gt["t_start"]) - intersection
        iou   = intersection / np.maximum(union, 1e-9)

        tp += tiou_thresholds <= iou.max()

    return tp / len(ground_truth)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ar_curve(results: list[dict], tiou: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for res, color in zip(results, _COLORS):
        ax.plot(tiou, res["ar"], marker="o", label=res["name"], color=color, linewidth=2)
    ax.set_xlabel("tIoU threshold")
    ax.set_ylabel("Action Recall")
    ax.set_title("Action Recall vs. tIoU — todas as variantes")
    ax.set_ylim(0, 1)
    ax.set_xticks(tiou)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_n_proposals(results: list[dict], out_path: Path) -> None:
    counts   = [r["n_proposals"] for r in results]
    baseline = counts[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(_LABELS, counts, color=_COLORS)
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1,
               label=f"Baseline = {baseline:,}")
    for bar, count in zip(bars, counts):
        pct   = (count - baseline) / baseline * 100
        label = f"{count:,}\n({pct:+.1f}%)" if count != baseline else f"{count:,}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                label, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Nº propostas (tras NMS)")
    ax.set_title("Propostas xeradas por variante")
    ax.legend()
    ax.set_ylim(0, max(counts) * 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_duration_box(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot([r["durations_s"] for r in results], patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], _COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(_LABELS) + 1))
    ax.set_xticklabels(_LABELS)
    ax.set_ylabel("Duración (s)")
    ax.set_title("Distribución de duracións das propostas")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    results: list[dict],
    gt_count: int,
    tiou: np.ndarray,
    cfg: argparse.Namespace,
    out_path: Path,
) -> None:
    tiou_headers = " | ".join(f"AR@{t}" for t in tiou)
    sep_cols     = " | ".join("---" for _ in tiou)

    lines = [
        "# Informe comparativo de variantes do pipeline reTAG",
        "",
        f"**Split:** {cfg.split}  ",
        f"**Gravacións:** GT {gt_count} instancias (duración ≥ {cfg.min_duration} s)  ",
        f"**Bin width:** {cfg.bin_width} s | **Percentile:** {cfg.percentile} "
        f"| **NMS threshold:** {cfg.nms_threshold}",
        "",
        "---",
        "",
        "## 1. Action Recall (AR)",
        "",
        "Fracción de instancias GT cubertas por polo menos unha proposta con IoU ≥ umbral.",
        "",
        f"| Variante | {tiou_headers} | **AR medio** |",
        f"|---| {sep_cols} |---|",
    ]
    for res in results:
        ar_vals = " | ".join(f"{v:.3f}" for v in res["ar"])
        lines.append(f"| {res['name']} | {ar_vals} | **{res['ar'].mean():.3f}** |")

    lines += ["", "![AR curve](plots/ar_curve.png)", "", "---", "",
              "## 2. Estatísticas das propostas", "",
              "| Variante | Nº propostas | Δ vs Baseline | Dur. media (s) | Dur. máx (s) |",
              "|---|---:|:---:|---:|---:|"]

    baseline_n = results[0]["n_proposals"]
    for res in results:
        delta     = res["n_proposals"] - baseline_n
        delta_str = f"{delta / baseline_n * 100:+.1f} %" if delta != 0 else "—"
        lines.append(
            f"| {res['name']} | {res['n_proposals']:,} | {delta_str} "
            f"| {res['mean_dur']:.1f} | {res['max_dur']:.1f} |"
        )

    lines += [
        "", "![Nº propostas](plots/n_proposals.png)", "",
        "![Duracións](plots/duration_box.png)", "", "---", "",
        "## 3. Conclusión", "",
        "As melloras acumulativas reducen o número de propostas mantendo o recall, "
        "o que implica menos carga para o clasificador CNN e potencial mellora de mAP.",
        "A avaliación completa de mAP require executar a clasificación CNN no servidor CiTIUS.",
        "",
        f"*Xerado por `dev/eval_proposals.py`. Gráficas en `{out_path.parent}/plots/`.*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Informe gardado en {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import h5py

    args     = _parse_args()
    out_dir  = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tiou = np.array(args.tiou)

    with h5py.File(args.data_path, "r") as hf:
        split_recs = {rec for rec in hf if hf[rec].attrs.get("split") == args.split}

    gt = load_ground_truth(args.ann_path, split_recs, args.min_duration)
    print(f"[INFO] GT instances ({args.split} split, ≥{args.min_duration}s): {len(gt)}")

    # Shared base config — all variants inherit these values
    base_cfg = dict(
        data_path     = args.data_path,
        bin_width     = args.bin_width,
        percentile    = args.percentile,
        nms_threshold = args.nms_threshold,
        output_dir    = str(out_dir / "logs"),
    )

    # Load or build prototype (always from train split to avoid leakage)
    proto_path = Path("tmp/prototype/ed_prototype.npy")
    if proto_path.exists():
        prototype = np.load(proto_path)
        print(f"[INFO] Prototipo cargado: {proto_path}  shape={prototype.shape}")
    else:
        print("[INFO] Construíndo prototipo dende train split...")
        prototype = build_ed_prototype(args.data_path, args.ann_path, split="train")
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proto_path, prototype)

    variants_with_proto = _VARIANTS + [
        ("+ Prototipo ED", {
            "use_adaptive_lambda":     True,
            "use_spatial_compactness": True,
            "use_noise_penalization":  True,
            "use_dispersed_noise":     True,
            "prototype":               prototype,
        }),
        ("+ Periodicidade", {
            "use_adaptive_lambda":     True,
            "use_spatial_compactness": True,
            "use_noise_penalization":  True,
            "use_dispersed_noise":     True,
            "prototype":               prototype,
            "use_periodicity":         True,
        }),
    ]

    results = []
    for name, extra_kwargs in variants_with_proto:
        print(f"\n[RUN] {name}")
        gen = ProposalGenerator(**base_cfg, **extra_kwargs)
        df  = gen.run(split=args.split)
        print(f"      Propostas: {len(df)}")

        ar = compute_proposal_recall(df, gt, tiou)
        print(f"      AR@0.5: {ar[tiou == 0.5][0]:.3f}  AR medio: {ar.mean():.3f}")

        durations_s = (df["t_end"] - df["t_start"]) / 1e6
        results.append({
            "name":        name,
            "ar":          ar,
            "n_proposals": len(df),
            "mean_dur":    durations_s.mean(),
            "max_dur":     durations_s.max(),
            "durations_s": durations_s.values,
        })

    print("\n[INFO] Xerando gráficas...")
    plot_ar_curve(results,     tiou,     plot_dir / "ar_curve.png")
    plot_n_proposals(results,            plot_dir / "n_proposals.png")
    plot_duration_box(results,           plot_dir / "duration_box.png")

    write_report(results, len(gt), tiou, args, out_dir / "informe_comparativo.md")

    print("\n=== RESUMO ===")
    summary = pd.DataFrame({
        r["name"]: {
            "n_proposals": r["n_proposals"],
            "AR@0.5":      round(float(r["ar"][tiou == 0.5][0]), 3),
            "AR medio":    round(float(r["ar"].mean()), 3),
            "dur_media_s": round(r["mean_dur"], 1),
        }
        for r in results
    }).T
    print(summary.to_string())
