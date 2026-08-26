"""Evaluate the source-CV-approved label-free TESPEC CORAL candidate on test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.eval_temporalmaxer_groupdro_cv import add_ranking_fusions  # noqa: E402
from dev.eval_tespec_coral_cv import recording_coral_affines  # noqa: E402
from dev.train_temporalmaxer_dense import (  # noqa: E402
    evaluate_variant,
    load_cache,
    load_event_cache,
    make_model,
    map_to_master,
    score_model,
    stable_proposal_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-master",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument("--source-fold-dir", default="tmp/temporalmaxer_dense/hybrid_source")
    parser.add_argument(
        "--source-cache-dir", default="tmp/temporalmaxer_dense/hybrid_source_cache"
    )
    parser.add_argument(
        "--source-event-cache-dir",
        default="tmp/temporalmaxer_dense/tespec_hybrid_source",
    )
    parser.add_argument("--master-proposals", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--event-feature-cache-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--boundary-ensemble", required=True)
    parser.add_argument("--out-dir", required=True)
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


def assert_aligned(frame: pd.DataFrame, proposals: pd.DataFrame, label: str) -> None:
    if len(frame) != len(proposals) or not stable_proposal_index(frame).equals(
        stable_proposal_index(proposals)
    ):
        raise ValueError(f"{label} is not aligned with the proposal master")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source_master = pd.read_csv(resolve(args.source_master)).reset_index(drop=True)
    _, _, source_metadata = load_cache(resolve(args.source_cache_dir))
    source_features, _ = load_event_cache(
        resolve(args.source_event_cache_dir), source_metadata
    )
    target = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, logits, target_metadata = load_cache(resolve(args.cache_dir))
    target_features, _ = load_event_cache(
        resolve(args.event_feature_cache_dir), target_metadata
    )
    target_indices = np.arange(len(target), dtype=np.int64)
    boundary = pd.read_csv(resolve(args.boundary_ensemble)).reset_index(drop=True)
    assert_aligned(boundary, target, "Boundary ensemble")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for fold in range(5):
        scored_path = out_dir / f"scored_fold_{fold:02d}.csv"
        if scored_path.exists():
            scored = pd.read_csv(scored_path)
        else:
            validation = pd.read_csv(
                resolve(args.source_fold_dir)
                / f"fold_{fold:02d}"
                / "val_proposals.csv"
            )
            held_out = set(validation["rec_name"].astype(str))
            source_train = source_master[
                ~source_master["rec_name"].astype(str).isin(held_out)
            ].reset_index(drop=True)
            source_indices = map_to_master(source_master, source_train)
            scale, bias = recording_coral_affines(
                source_features,
                source_indices,
                target,
                target_indices,
                target_features=target_features,
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
            model = make_model(target_metadata, args).to(device)
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
            scored.to_csv(scored_path, index=False)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        assert_aligned(scored, target, f"CORAL fold {fold}")
        frames.append(scored)

    ensemble = boundary.copy()
    for column in ("dense_score", "brem_score", "dense_point"):
        ensemble[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )
    for column in ("delta_t_start", "delta_t_end"):
        ensemble[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )
    ensemble = add_ranking_fusions(ensemble)
    ensemble["blend050_t_start"] = 0.5 * (
        ensemble["t_start"] + ensemble["delta_t_start"]
    )
    ensemble["blend050_t_end"] = 0.5 * (
        ensemble["t_end"] + ensemble["delta_t_end"]
    )
    ensemble.to_csv(out_dir / "scored_ensemble.csv", index=False)
    row = evaluate_variant(
        ensemble,
        "qhead_brem_score",
        "blend050",
        "source_cv_approved_tespec_coral_test",
        args,
        out_dir / "predictions",
    )
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
