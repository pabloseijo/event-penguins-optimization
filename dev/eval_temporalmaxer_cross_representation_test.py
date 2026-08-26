"""Evaluate the fixed source-approved TESPEC/TISM fusion once on test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.eval_temporalmaxer_cross_representation_cv import (  # noqa: E402
    geometric_trimodal_score,
)
from dev.train_temporalmaxer_dense import (  # noqa: E402
    evaluate_variant,
    load_cache,
    make_model,
    score_model,
    stable_proposal_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-proposals", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tism-feature-cache-dir", required=True)
    parser.add_argument("--tism-checkpoint-root", required=True)
    parser.add_argument("--tespec-ensemble", required=True)
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
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def assert_aligned(frame: pd.DataFrame, proposals: pd.DataFrame, label: str) -> None:
    if len(frame) != len(proposals) or not stable_proposal_index(frame).equals(
        stable_proposal_index(proposals)
    ):
        raise ValueError(f"{label} is not aligned with the test proposal master")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    proposals = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, logits, metadata = load_cache(resolve(args.cache_dir))
    indices = np.arange(len(proposals), dtype=np.int64)
    tespec = pd.read_csv(resolve(args.tespec_ensemble)).reset_index(drop=True)
    assert_aligned(tespec, proposals, "TESPEC ensemble")
    required = {
        "quality_score",
        "brem_score",
        "blend050_t_start",
        "blend050_t_end",
    }
    missing = sorted(required - set(tespec.columns))
    if missing:
        raise ValueError(f"TESPEC ensemble is missing fixed candidate columns: {missing}")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    args.event_feature_cache_dir = args.tism_feature_cache_dir
    for fold in range(5):
        scored_path = out_dir / f"tism_scored_fold_{fold:02d}.csv"
        if scored_path.exists():
            scored = pd.read_csv(scored_path)
        else:
            checkpoint = torch.load(
                resolve(args.tism_checkpoint_root) / f"fold_{fold:02d}" / "best.pt",
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
        assert_aligned(scored, proposals, f"TISM fold {fold}")
        frames.append(scored)

    ensemble = tespec.copy()
    ensemble["tism_brem_score"] = np.mean(
        [frame["brem_score"].to_numpy(dtype=np.float64) for frame in frames], axis=0
    )
    ensemble["trimodal_score"] = geometric_trimodal_score(
        ensemble["quality_score"].to_numpy(dtype=np.float64),
        ensemble["brem_score"].to_numpy(dtype=np.float64),
        ensemble["tism_brem_score"].to_numpy(dtype=np.float64),
    )
    ensemble.to_csv(out_dir / "scored_ensemble.csv", index=False)
    row = evaluate_variant(
        ensemble,
        "trimodal_score",
        "blend050",
        "fixed_cross_representation_test",
        args,
        out_dir / "predictions",
    )
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
