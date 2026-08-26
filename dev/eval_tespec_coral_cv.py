"""Evaluate label-free per-recording diagonal CORAL on TESPEC source CV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.eval_temporalmaxer_groupdro_cv import (  # noqa: E402
    RANKING_COLUMNS,
    add_ranking_fusions,
)
from dev.train_temporalmaxer_dense import (  # noqa: E402
    evaluate_variant,
    load_cache,
    load_event_cache,
    make_model,
    map_to_master,
    score_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-proposals",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument("--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source")
    parser.add_argument(
        "--cache-dir", default="tmp/temporalmaxer_dense/hybrid_source_cache"
    )
    parser.add_argument(
        "--event-feature-cache-dir",
        default="tmp/temporalmaxer_dense/tespec_hybrid_source",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="tmp/temporalmaxer_dense/screened_cv_tespec_combined",
    )
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/hybrid_cv_tespec_coral"
    )
    parser.add_argument("--scored-root", default=None)
    parser.add_argument(
        "--score-column", choices=RANKING_COLUMNS, default="qhead_brem_score"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
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


def feature_moments(
    features: np.ndarray,
    indices: np.ndarray,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(features.shape[-1], dtype=np.float64)
    total_square = np.zeros(features.shape[-1], dtype=np.float64)
    count = 0
    for offset in range(0, len(indices), chunk_size):
        batch = np.asarray(features[indices[offset : offset + chunk_size]], dtype=np.float32)
        flat = batch.reshape(-1, batch.shape[-1])
        total += flat.sum(axis=0, dtype=np.float64)
        total_square += np.square(flat, dtype=np.float64).sum(axis=0)
        count += len(flat)
    mean = total / max(count, 1)
    variance = np.maximum(total_square / max(count, 1) - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def recording_coral_affines(
    features: np.ndarray,
    source_indices: np.ndarray,
    target: pd.DataFrame,
    target_indices: np.ndarray,
    target_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if target_features is None:
        target_features = features
    source_mean, source_std = feature_moments(features, source_indices)
    scale = np.empty((len(target), target_features.shape[-1]), dtype=np.float32)
    bias = np.empty_like(scale)
    recordings = target["rec_name"].astype(str).to_numpy()
    for recording in sorted(set(recordings)):
        local = np.flatnonzero(recordings == recording)
        target_mean, target_std = feature_moments(
            target_features, target_indices[local]
        )
        local_scale = np.clip(source_std / np.maximum(target_std, 1e-4), 0.25, 4.0)
        scale[local] = local_scale
        bias[local] = source_mean - target_mean * local_scale
    return scale, bias


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, logits, metadata = load_cache(resolve(args.cache_dir))
    event_features, _ = load_event_cache(resolve(args.event_feature_cache_dir), metadata)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {int(row["fold"]) for row in rows}

    for fold in range(5):
        if fold in completed:
            continue
        fold_path = resolve(args.fold_dir) / f"fold_{fold:02d}"
        target = pd.read_csv(fold_path / "val_proposals.csv").reset_index(drop=True)
        target_recordings = set(target["rec_name"].astype(str))
        train = master[
            ~master["rec_name"].astype(str).isin(target_recordings)
        ].reset_index(drop=True)
        train_indices = map_to_master(master, train)
        target_indices = map_to_master(master, target)
        scored_path = out_dir / f"scored_fold_{fold:02d}.csv"
        reused_path = (
            resolve(args.scored_root) / f"scored_fold_{fold:02d}.csv"
            if args.scored_root
            else None
        )
        if reused_path is not None and reused_path.exists():
            scored = pd.read_csv(reused_path)
        elif scored_path.exists():
            scored = pd.read_csv(scored_path)
        else:
            scale, bias = recording_coral_affines(
                event_features, train_indices, target, target_indices
            )
            checkpoint = torch.load(
                resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt",
                map_location=device,
                weights_only=False,
            )
            for name in (
                "hidden_dim",
                "pyramid_levels",
                "dropout",
                "trident_bins",
                "event_features_only",
            ):
                if name in checkpoint["args"]:
                    setattr(args, name, checkpoint["args"][name])
            model = make_model(metadata, args).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            scored = score_model(
                model,
                target,
                target_indices,
                resolve(args.cache_dir) / "frame_features.npy",
                logits,
                args,
                device,
                event_scale=scale,
                event_bias=bias,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            groupdro = pd.read_csv(
                resolve(args.groupdro_root)
                / f"fold_{fold:02d}"
                / "cache"
                / "val_scores_qhead_qfl_only.csv"
            )
            selected = groupdro.iloc[map_to_master(groupdro, target)].reset_index(drop=True)
            scored["quality_score"] = selected["quality_score"].to_numpy(dtype=np.float64)
            scored = add_ranking_fusions(scored)
            scored["blend050_t_start"] = 0.5 * (
                scored["t_start"] + scored["delta_t_start"]
            )
            scored["blend050_t_end"] = 0.5 * (
                scored["t_end"] + scored["delta_t_end"]
            )
            scored.to_csv(scored_path, index=False)
        row = evaluate_variant(
            scored,
            args.score_column,
            "blend050",
            f"tespec_coral_fold_{fold:02d}",
            args,
            out_dir / "predictions",
        )
        row["fold"] = fold
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    weights = metrics["val_ed_instances"].to_numpy(dtype=np.float64)
    summary = {"score_column": args.score_column, "boundary_mode": "blend050"}
    for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
        values = metrics[column].to_numpy(dtype=np.float64)
        summary[f"mean_{column}"] = float(values.mean())
        summary[f"weighted_{column}"] = float(np.average(values, weights=weights))
        summary[f"worst_{column}"] = float(values.min())
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
