"""Evaluate a fixed ensemble of TemporalMaxer boundary checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.train_temporalmaxer_dense import (
    evaluate_variant,
    load_cache,
    make_model,
    map_to_master,
    score_model,
    stable_proposal_index,
)

KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-proposals", required=True)
    parser.add_argument("--test-proposals", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
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


def choose_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    proposals = pd.read_csv(resolve(args.test_proposals)).reset_index(drop=True)
    features, logits, metadata = load_cache(resolve(args.cache_dir))
    indices = map_to_master(master, proposals)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_frames = []

    for fold in args.folds:
        scored_path = out_dir / f"scored_fold_{fold:02d}.csv"
        if scored_path.exists():
            scored = pd.read_csv(scored_path)
        else:
            checkpoint_path = resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            saved_args = checkpoint.get("args", {})
            for name in ("hidden_dim", "pyramid_levels", "dropout"):
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
        if len(scored) != len(proposals) or not stable_proposal_index(scored).equals(
            stable_proposal_index(proposals)
        ):
            raise ValueError(f"Fold {fold} scored proposals are misaligned")
        scored_frames.append(scored)
        print(f"[FOLD {fold:02d}] puntuado")

    ensemble = scored_frames[0].copy()
    ensemble["ensemble_t_start"] = np.mean(
        [frame["blend_t_start"].to_numpy(dtype=np.float64) for frame in scored_frames],
        axis=0,
    )
    ensemble["ensemble_t_end"] = np.mean(
        [frame["blend_t_end"].to_numpy(dtype=np.float64) for frame in scored_frames],
        axis=0,
    )
    ensemble.to_csv(out_dir / "scored_ensemble.csv", index=False)
    partial_path = out_dir / "metrics_partial.csv"
    metrics = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed_labels = {str(row["label"]) for row in metrics}
    if "baseline" not in completed_labels:
        metrics.append(
            evaluate_variant(
                ensemble,
                "cnn_score",
                "raw",
                "baseline",
                args,
                out_dir / "predictions",
            )
        )
        pd.DataFrame(metrics).to_csv(partial_path, index=False)
    if "temporalmaxer_ensemble" not in completed_labels:
        metrics.append(evaluate_variant(
            ensemble,
            "cnn_score",
            "ensemble",
            "temporalmaxer_ensemble",
            args,
            out_dir / "predictions",
        ))
    summary = pd.DataFrame(metrics)
    summary.to_csv(partial_path, index=False)
    summary.to_csv(out_dir / "metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
