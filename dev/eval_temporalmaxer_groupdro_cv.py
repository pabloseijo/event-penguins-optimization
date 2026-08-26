"""Evaluate GroupDRO scores with original or TemporalMaxer boundaries in CV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.train_temporalmaxer_dense import (  # noqa: E402
    evaluate_variant,
    load_cache,
    make_model,
    map_to_master,
    score_model,
)


RANKING_COLUMNS = (
    "quality_score",
    "dense_score",
    "brem_score",
    "qhead_dense_score",
    "qhead_brem_score",
    "qhead_brem_w020_score",
    "qhead_point_score",
)


def add_ranking_fusions(scored: pd.DataFrame) -> pd.DataFrame:
    """Add fixed quality fusions selected before external-CV evaluation."""
    required = {"quality_score", "dense_score", "brem_score", "dense_point"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Missing ranking columns: {missing}")
    output = scored.copy()
    qhead = np.clip(output["quality_score"].to_numpy(dtype=np.float64), 0.0, 1.0)
    dense = np.clip(output["dense_score"].to_numpy(dtype=np.float64), 0.0, 1.0)
    brem = np.clip(output["brem_score"].to_numpy(dtype=np.float64), 0.0, 1.0)
    point = np.clip(output["dense_point"].to_numpy(dtype=np.float64), 0.0, 1.0)
    output["qhead_dense_score"] = np.sqrt(qhead * dense)
    output["qhead_brem_score"] = np.sqrt(qhead * brem)
    output["qhead_brem_w020_score"] = np.power(qhead, 0.8) * np.power(brem, 0.2)
    output["qhead_point_score"] = np.sqrt(qhead * point)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-proposals", required=True)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--event-feature-cache-dir", default=None)
    parser.add_argument("--event-features-only", action="store_true")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--boundary-reference-root",
        default="tmp/temporalmaxer_dense/hybrid_cv_eval",
        help="Scored folds from the fixed best source-CV boundary model.",
    )
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--score-columns",
        nargs="+",
        choices=RANKING_COLUMNS,
        default=None,
        help="Optional pre-registered subset of ranking columns to evaluate.",
    )
    parser.add_argument(
        "--boundary-modes",
        nargs="+",
        choices=["qhead", "blend025", "blend050", "blend", "reference"],
        default=None,
        help="Optional pre-registered subset of boundary modes to evaluate.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
    parser.add_argument("--trident-bins", type=int, default=0)
    parser.add_argument("--trident-only", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    args = parse_args()
    selected_scores = tuple(args.score_columns or RANKING_COLUMNS)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, logits, metadata = load_cache(resolve(args.cache_dir))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(5):
        metrics_path = out_dir / f"metrics_fold_{fold:02d}.csv"
        if metrics_path.exists():
            existing = pd.read_csv(metrics_path)
            completed_pairs = set(
                zip(existing["score_column"], existing["boundary_mode"])
            )
            expected_modes = set(
                args.boundary_modes or ("qhead", "blend025", "blend050", "blend")
            )
            if args.boundary_reference_root:
                if args.boundary_modes is None or "reference" in args.boundary_modes:
                    expected_modes.add("reference")
            expected_pairs = {
                (score, mode)
                for score in selected_scores
                for mode in expected_modes
            }
        else:
            completed_pairs = set()
            expected_pairs = set()
        if expected_pairs and expected_pairs <= completed_pairs:
            print(f"[FOLD {fold:02d}] reutilizado")
            continue
        proposals = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        indices = map_to_master(master, proposals)
        scored_path = out_dir / f"scored_fold_{fold:02d}.csv"
        if scored_path.exists():
            scored = pd.read_csv(scored_path)
        else:
            checkpoint = torch.load(
                resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt",
                map_location=device,
                weights_only=False,
            )
            saved_args = checkpoint["args"]
            for name in (
                "hidden_dim",
                "pyramid_levels",
                "dropout",
                "trident_bins",
                "event_features_only",
            ):
                if name in saved_args:
                    setattr(args, name, saved_args[name])
            model = make_model(metadata, args).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            scored = score_model(
                model,
                proposals,
                indices,
                resolve(args.cache_dir) / "frame_features.npy",
                logits,
                args,
                device,
            )
            scored.to_csv(scored_path, index=False)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        groupdro = pd.read_csv(
            resolve(args.groupdro_root)
            / f"fold_{fold:02d}"
            / "cache"
            / "val_scores_qhead_qfl_only.csv"
        )
        positions = map_to_master(groupdro, proposals)
        selected = groupdro.iloc[positions].reset_index(drop=True)
        scored["quality_score"] = selected["quality_score"].to_numpy(dtype=np.float64)
        scored["qhead_t_start"] = selected["refined_t_start"].to_numpy(dtype=np.float64)
        scored["qhead_t_end"] = selected["refined_t_end"].to_numpy(dtype=np.float64)
        scored = add_ranking_fusions(scored)
        has_reference = False
        if args.boundary_reference_root:
            reference_path = (
                resolve(args.boundary_reference_root) / f"scored_fold_{fold:02d}.csv"
            )
            if reference_path.exists():
                reference = pd.read_csv(reference_path)
                reference_positions = map_to_master(reference, proposals)
                reference = reference.iloc[reference_positions].reset_index(drop=True)
                scored["reference_t_start"] = 0.5 * (
                    scored["t_start"].to_numpy(dtype=np.float64)
                    + reference["delta_t_start"].to_numpy(dtype=np.float64)
                )
                scored["reference_t_end"] = 0.5 * (
                    scored["t_end"].to_numpy(dtype=np.float64)
                    + reference["delta_t_end"].to_numpy(dtype=np.float64)
                )
                has_reference = True
        for alpha in (0.25, 0.50):
            mode = f"blend{int(alpha * 100):03d}"
            scored[f"{mode}_t_start"] = (
                (1.0 - alpha) * scored["t_start"] + alpha * scored["delta_t_start"]
            )
            scored[f"{mode}_t_end"] = (
                (1.0 - alpha) * scored["t_end"] + alpha * scored["delta_t_end"]
            )
        trident_modes = []
        if {"trident_t_start", "trident_t_end"} <= set(scored.columns):
            for alpha in (0.25, 0.50, 0.75):
                mode = f"trident{int(alpha * 100):03d}"
                scored[f"{mode}_t_start"] = (
                    (1.0 - alpha) * scored["t_start"] + alpha * scored["trident_t_start"]
                )
                scored[f"{mode}_t_end"] = (
                    (1.0 - alpha) * scored["t_end"] + alpha * scored["trident_t_end"]
                )
                trident_modes.append(mode)
            trident_modes.append("trident")
        partial_path = out_dir / f"metrics_fold_{fold:02d}_partial.csv"
        rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
        completed = {
            (str(row["score_column"]), str(row["boundary_mode"])) for row in rows
        }
        eval_modes = (
            ["qhead", *trident_modes]
            if args.trident_only
            else [
                "qhead",
                "blend025",
                "blend050",
                "blend",
                *(["reference"] if has_reference else []),
                *trident_modes,
            ]
        )
        if args.boundary_modes is not None:
            unavailable = set(args.boundary_modes) - set(eval_modes)
            if unavailable:
                raise ValueError(f"Unavailable boundary modes: {sorted(unavailable)}")
            eval_modes = list(args.boundary_modes)
        for score_column in selected_scores:
            for mode in eval_modes:
                if (score_column, mode) in completed:
                    continue
                rows.append(
                    evaluate_variant(
                        scored,
                        score_column,
                        mode,
                        f"hybrid_fold_{fold:02d}",
                        args,
                        out_dir / "predictions",
                    )
                )
                pd.DataFrame(rows).to_csv(partial_path, index=False)
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"[FOLD {fold:02d}] completo")

    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    frames = []
    for fold in range(5):
        frame = pd.read_csv(out_dir / f"metrics_fold_{fold:02d}.csv")
        frame.insert(0, "fold", fold)
        frame["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[metrics["score_column"].isin(selected_scores)]
    if args.boundary_modes is not None:
        metrics = metrics[metrics["boundary_mode"].isin(args.boundary_modes)]
    metrics.to_csv(out_dir / "all_metrics.csv", index=False)
    rows = []
    for (score_column, mode), group in metrics.groupby(
        ["score_column", "boundary_mode"]
    ):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row = {"score_column": score_column, "boundary_mode": mode}
        for column in ["mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"]:
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
