"""Build a boundary-rescue proposal lattice from an existing proposal CSV.

The script is intentionally label-free: it never reads ground truth. It expands
the proposal search space around plausible candidates so downstream scoring can
recover high-tIoU intervals that the original event-rate watershed missed.

Run from event_penguins/:
    python dev/build_proposal_lattice.py \
        --proposals tmp/deep_diagnosis/fixed_r5_single_remote/proposals.csv \
        --out-proposals tmp/proposal_lattice/fixed_r5_lattice/proposals.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd

from src.utils import temporal_nms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a proposal boundary lattice.")
    parser.add_argument("--proposals", required=True, nargs="+")
    parser.add_argument("--out-proposals", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--max-duration-s", type=float, default=60.0)
    parser.add_argument("--top-k-per-roi", type=int, default=300)
    parser.add_argument("--score-quantile", type=float, default=0.35)
    parser.add_argument("--max-per-roi", type=int, default=3500)
    parser.add_argument(
        "--max-per-variant-per-roi",
        type=int,
        default=0,
        help="Keep up to this many lattice proposals per variant and ROI before the global cap.",
    )
    parser.add_argument(
        "--protect-variants",
        nargs="+",
        default=[],
        help="Lattice variant names to keep before applying max-per-roi.",
    )
    parser.add_argument(
        "--include-lattice-variants",
        nargs="+",
        default=[],
        help="If set, keep only these exact lattice variant names. Original input proposals are always kept.",
    )
    parser.add_argument(
        "--include-lattice-prefixes",
        nargs="+",
        default=[],
        help="If set, keep lattice variants whose name starts with any of these prefixes.",
    )
    parser.add_argument("--nms-threshold", type=float, default=0.995)
    parser.add_argument("--keep-without-nms", action="store_true")
    parser.add_argument("--include-duration-grid", action="store_true", default=True)
    parser.add_argument("--no-duration-grid", dest="include_duration_grid", action="store_false")
    parser.add_argument(
        "--duration-grid-s",
        type=float,
        nargs="+",
        default=[2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 40.0],
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_roi_bounds(data_path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    bounds: dict[tuple[str, str], tuple[float, float]] = {}
    with h5py.File(data_path, "r") as hf:
        for rec in hf.keys():
            for roi in hf[rec].keys():
                events = np.asarray(hf[rec][roi]["events"])
                if len(events) == 0:
                    continue
                bounds[(rec, roi)] = (float(events[0, 2]), float(events[-1, 2]))
    return bounds


def clip_segment(
    start: float,
    end: float,
    lower: float,
    upper: float,
    min_duration_us: float,
    max_duration_us: float,
) -> tuple[float, float] | None:
    if upper <= lower:
        return None

    duration = end - start
    if duration <= 0:
        return None

    if duration > max_duration_us:
        center = 0.5 * (start + end)
        start = center - 0.5 * max_duration_us
        end = center + 0.5 * max_duration_us

    if end - start < min_duration_us:
        center = 0.5 * (start + end)
        start = center - 0.5 * min_duration_us
        end = center + 0.5 * min_duration_us

    if start < lower:
        end += lower - start
        start = lower
    if end > upper:
        start -= end - upper
        end = upper

    start = max(lower, start)
    end = min(upper, end)
    if end - start < min_duration_us:
        return None
    return float(start), float(end)


def add_variant(
    rows: list[dict],
    base: pd.Series,
    start: float,
    end: float,
    score_scale: float,
    variant: str,
    bounds: tuple[float, float],
    min_duration_us: float,
    max_duration_us: float,
) -> None:
    clipped = clip_segment(start, end, bounds[0], bounds[1], min_duration_us, max_duration_us)
    if clipped is None:
        return
    out_start, out_end = clipped
    rows.append(
        {
            "rec_name": base["rec_name"],
            "roi_id": base["roi_id"],
            "t_start": out_start,
            "t_end": out_end,
            "score": float(base["score"]) * score_scale,
            "source": "lattice",
            "variant": variant,
            "source_score": float(base["score"]),
        }
    )


def select_support(df: pd.DataFrame, top_k: int, score_quantile: float) -> pd.DataFrame:
    selected = []
    for _, grp in df.groupby(["rec_name", "roi_id"], sort=False):
        grp = grp.copy()
        q = float(grp["score"].quantile(score_quantile)) if len(grp) else 0.0
        support = grp[grp["score"] >= q]
        if top_k > 0:
            support = pd.concat(
                [support, grp.sort_values("score", ascending=False).head(top_k)],
                ignore_index=False,
            )
        support = support.drop_duplicates()
        selected.append(support)
    return pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()


def build_lattice_for_row(
    row: pd.Series,
    bounds: tuple[float, float],
    args: argparse.Namespace,
) -> list[dict]:
    start = float(row["t_start"])
    end = float(row["t_end"])
    duration = max(end - start, 1.0)
    center = 0.5 * (start + end)
    min_duration_us = args.min_duration_s * 1e6
    max_duration_us = args.max_duration_s * 1e6
    rows: list[dict] = []

    # A base is kept as an explicit lattice row so all later operations can be
    # traced to the same branch.
    add_variant(rows, row, start, end, 1.0, "base", bounds, min_duration_us, max_duration_us)

    for frac in [0.125, 0.25, 0.50, 0.75, 1.00]:
        delta = frac * duration
        scale = max(0.35, 1.0 - 0.18 * frac)
        add_variant(rows, row, start - delta, end, scale, f"expand_left_{frac:g}", bounds, min_duration_us, max_duration_us)
        add_variant(rows, row, start, end + delta, scale, f"expand_right_{frac:g}", bounds, min_duration_us, max_duration_us)
        add_variant(rows, row, start - delta, end + delta, scale * 0.95, f"expand_both_{frac:g}", bounds, min_duration_us, max_duration_us)

    for frac in [0.125, 0.25, 0.375]:
        delta = frac * duration
        scale = max(0.45, 0.90 - 0.20 * frac)
        add_variant(rows, row, start + delta, end, scale, f"trim_left_{frac:g}", bounds, min_duration_us, max_duration_us)
        add_variant(rows, row, start, end - delta, scale, f"trim_right_{frac:g}", bounds, min_duration_us, max_duration_us)
        add_variant(rows, row, start + delta, end - delta, scale * 0.95, f"trim_both_{frac:g}", bounds, min_duration_us, max_duration_us)

    for frac in [-0.75, -0.50, -0.25, 0.25, 0.50, 0.75]:
        delta = frac * duration
        scale = max(0.40, 0.82 - 0.12 * abs(frac))
        add_variant(rows, row, start + delta, end + delta, scale, f"shift_{frac:g}", bounds, min_duration_us, max_duration_us)

    if args.include_duration_grid:
        for dur_s in args.duration_grid_s:
            dur_us = dur_s * 1e6
            if dur_us < min_duration_us or dur_us > max_duration_us:
                continue
            # Grid windows are useful for short/fragmented events. Their score is
            # deliberately lower so a later ranker can prefer original proposals.
            scale = 0.70 if dur_us >= duration else 0.62
            add_variant(
                rows,
                row,
                center - 0.5 * dur_us,
                center + 0.5 * dur_us,
                scale,
                f"center_dur_{dur_s:g}s",
                bounds,
                min_duration_us,
                max_duration_us,
            )

    return rows


def dedupe_rows(df: pd.DataFrame, time_round_us: float = 1000.0) -> pd.DataFrame:
    out = df.copy()
    out["_start_bin"] = np.round(out["t_start"].astype(float) / time_round_us).astype(np.int64)
    out["_end_bin"] = np.round(out["t_end"].astype(float) / time_round_us).astype(np.int64)
    out = out.sort_values("score", ascending=False)
    out = out.drop_duplicates(["rec_name", "roi_id", "_start_bin", "_end_bin"], keep="first")
    return out.drop(columns=["_start_bin", "_end_bin"]).reset_index(drop=True)


def filter_lattice_variants(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.include_lattice_variants and not args.include_lattice_prefixes:
        return df

    is_base_input = df["source"].astype(str) != "lattice"
    variants = df["variant"].astype(str)
    keep_lattice = pd.Series(False, index=df.index)
    if args.include_lattice_variants:
        keep_lattice |= variants.isin(args.include_lattice_variants)
    for prefix in args.include_lattice_prefixes:
        keep_lattice |= variants.str.startswith(prefix)
    return df[is_base_input | keep_lattice].reset_index(drop=True)


def limit_with_nms(grp: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    grp = grp.sort_values("score", ascending=False).reset_index(drop=True)
    protected = grp[grp["source"].astype(str) != "lattice"].copy()
    lattice = grp[grp["source"].astype(str) == "lattice"].copy()
    if lattice.empty:
        return protected

    if args.protect_variants:
        protected_variants = lattice[lattice["variant"].isin(args.protect_variants)].copy()
        lattice = lattice.drop(index=protected_variants.index, errors="ignore")
        protected = pd.concat([protected, protected_variants], ignore_index=True)

    lattice_limit = args.max_per_roi
    if lattice_limit > 0:
        lattice_limit = max(0, lattice_limit - len(protected))

    if args.keep_without_nms:
        if args.max_per_variant_per_roi > 0:
            variant_kept = []
            for _, var_grp in lattice.groupby("variant", sort=False):
                variant_kept.append(var_grp.head(args.max_per_variant_per_roi))
            kept_lattice = pd.concat(variant_kept, ignore_index=False) if variant_kept else lattice.iloc[0:0]
            remaining = lattice.drop(index=kept_lattice.index, errors="ignore")
            if lattice_limit > 0 and len(kept_lattice) < lattice_limit and not remaining.empty:
                kept_lattice = pd.concat(
                    [kept_lattice, remaining.head(lattice_limit - len(kept_lattice))],
                    ignore_index=True,
                )
            elif lattice_limit > 0 and len(kept_lattice) > lattice_limit:
                kept_lattice = kept_lattice.sort_values("score", ascending=False).head(lattice_limit)
        else:
            kept_lattice = lattice.head(lattice_limit) if lattice_limit > 0 else lattice
        return pd.concat([protected, kept_lattice], ignore_index=True)

    arr = lattice[["t_start", "t_end", "score"]].to_numpy(dtype=np.float64)
    keep_arr = temporal_nms(arr, args.nms_threshold)
    keep = pd.DataFrame(keep_arr, columns=["t_start", "t_end", "score"])
    merged = keep.merge(lattice, on=["t_start", "t_end", "score"], how="left")
    merged = merged.drop_duplicates(["t_start", "t_end", "score"], keep="first")
    if lattice_limit > 0:
        merged = merged.head(lattice_limit)
    return pd.concat([protected, merged[grp.columns]], ignore_index=True)


def main() -> None:
    args = parse_args()
    proposal_paths = [resolve(path) for path in args.proposals]
    out_path = resolve(args.out_proposals)
    data_path = resolve(args.data_path)

    frames = []
    for idx, proposal_path in enumerate(proposal_paths):
        frame = pd.read_csv(proposal_path).reset_index(drop=True)
        frame["proposal_file"] = proposal_path.name if len(proposal_paths) == 1 else f"{idx}:{proposal_path.name}"
        frames.append(frame)
    base = pd.concat(frames, ignore_index=True, sort=False)
    base = base.copy()
    base["source"] = base.get("source", "base")
    base["variant"] = base.get("variant", "base")
    base["source_score"] = base.get("source_score", base["score"])

    bounds = load_roi_bounds(data_path)
    support = select_support(base, args.top_k_per_roi, args.score_quantile)

    rows: list[dict] = []
    for _, row in support.iterrows():
        key = (row["rec_name"], row["roi_id"])
        roi_bounds = bounds.get(key)
        if roi_bounds is None:
            roi_bounds = (float(base["t_start"].min()), float(base["t_end"].max()))
        rows.extend(build_lattice_for_row(row, roi_bounds, args))

    lattice = pd.DataFrame(rows)
    combined = pd.concat([base, lattice], ignore_index=True, sort=False)
    combined = dedupe_rows(combined)
    combined = filter_lattice_variants(combined, args)

    limited = []
    for _, grp in combined.groupby(["rec_name", "roi_id"], sort=False):
        limited.append(limit_with_nms(grp, args))
    out = pd.concat(limited, ignore_index=True) if limited else combined.iloc[0:0].copy()
    out = out.sort_values(["rec_name", "roi_id", "score"], ascending=[True, True, False]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[INFO] Base proposals: {len(base)}")
    print(f"[INFO] Support proposals: {len(support)}")
    print(f"[INFO] Lattice+base proposals after dedupe/limit: {len(out)}")
    print(f"[INFO] Written: {out_path}")


if __name__ == "__main__":
    main()
