"""Evaluate fixed GroupDRO scores with TemporalMaxer ensemble boundaries on test."""

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
    stable_proposal_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-proposals", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hybrid-boundary-blend", type=float, default=0.5)
    parser.add_argument(
        "--boundary-modes",
        nargs="+",
        choices=["qhead", "hybrid"],
        default=["qhead", "hybrid"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
    parser.add_argument("--trident-bins", type=int, default=0)
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
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    proposals = master.copy()
    _, logits, metadata = load_cache(resolve(args.cache_dir))
    indices = np.arange(len(master), dtype=np.int64)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporal_frames = []

    for fold in range(5):
        path = out_dir / f"temporal_scored_fold_{fold:02d}.csv"
        if path.exists():
            scored = pd.read_csv(path)
        else:
            checkpoint = torch.load(
                resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt",
                map_location=device,
                weights_only=False,
            )
            for name in ("hidden_dim", "pyramid_levels", "dropout", "trident_bins"):
                if name in checkpoint["args"]:
                    setattr(args, name, checkpoint["args"][name])
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
            scored.to_csv(path, index=False)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not stable_proposal_index(scored).equals(stable_proposal_index(proposals)):
            raise ValueError(f"Temporal fold {fold} is misaligned")
        temporal_frames.append(scored)
        print(f"[TEMPORAL {fold:02d}] puntuado")

    groupdro_frames = []
    for fold in range(5):
        path = (
            resolve(args.groupdro_root)
            / f"fold_{fold:02d}"
            / "cache"
            / f"test_groupdro_fold_{fold:02d}_scores_qhead_qfl_only.csv"
        )
        frame = pd.read_csv(path)
        positions = map_to_master(frame, proposals)
        groupdro_frames.append(frame.iloc[positions].reset_index(drop=True))

    ensemble = proposals.copy()
    ensemble["quality_score"] = np.mean(
        [frame["quality_score"].to_numpy(dtype=np.float64) for frame in groupdro_frames],
        axis=0,
    )
    ensemble["qhead_t_start"] = np.mean(
        [frame["refined_t_start"].to_numpy(dtype=np.float64) for frame in groupdro_frames],
        axis=0,
    )
    ensemble["qhead_t_end"] = np.mean(
        [frame["refined_t_end"].to_numpy(dtype=np.float64) for frame in groupdro_frames],
        axis=0,
    )
    mean_delta_start = np.mean(
        [frame["delta_t_start"].to_numpy(dtype=np.float64) for frame in temporal_frames],
        axis=0,
    )
    mean_delta_end = np.mean(
        [frame["delta_t_end"].to_numpy(dtype=np.float64) for frame in temporal_frames],
        axis=0,
    )
    alpha = float(args.hybrid_boundary_blend)
    ensemble["hybrid_t_start"] = (1.0 - alpha) * ensemble["t_start"] + alpha * mean_delta_start
    ensemble["hybrid_t_end"] = (1.0 - alpha) * ensemble["t_end"] + alpha * mean_delta_end
    ensemble.to_csv(out_dir / "scored_ensemble.csv", index=False)

    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {str(row["boundary_mode"]) for row in rows}
    for mode in args.boundary_modes:
        if mode in completed:
            continue
        rows.append(
            evaluate_variant(
                ensemble,
                "quality_score",
                mode,
                "groupdro_temporalmaxer_test",
                args,
                out_dir / "predictions",
            )
        )
        pd.DataFrame(rows).to_csv(partial_path, index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
