"""Evaluate one source-CV-approved representation/score pair on test.

The script deliberately accepts one score and one fixed boundary mode. It
ensembles five representation checkpoints, combines their score with the
already frozen GroupDRO test ensemble, and performs no test-time sweep.
"""

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
    make_model,
    score_model,
    stable_proposal_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-proposals", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--event-feature-cache-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--boundary-ensemble",
        default="tmp/temporalmaxer_dense/hybrid_test_eval/scored_ensemble.csv",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-column", choices=RANKING_COLUMNS, required=True)
    parser.add_argument(
        "--boundary-mode",
        choices=["qhead", "reference", "blend025", "blend050", "blend"],
        required=True,
    )
    parser.add_argument(
        "--boundary-source",
        choices=["representation", "ensemble"],
        default="representation",
        help="Build boundaries from the evaluated checkpoints or preserve the supplied ensemble.",
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


def assert_aligned(frame: pd.DataFrame, proposals: pd.DataFrame, label: str) -> None:
    if len(frame) != len(proposals) or not stable_proposal_index(frame).equals(
        stable_proposal_index(proposals)
    ):
        raise ValueError(f"{label} is not aligned with the test proposal master")


def add_boundary_modes(
    ensemble: pd.DataFrame,
    frames: list[pd.DataFrame],
    boundary_blend: float,
) -> pd.DataFrame:
    """Reproduce the boundary modes fixed during source external-CV."""
    output = ensemble.copy()
    for column in ("delta_t_start", "delta_t_end"):
        output[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )
    for alpha in (0.25, 0.50):
        mode = f"blend{int(alpha * 100):03d}"
        output[f"{mode}_t_start"] = (
            (1.0 - alpha) * output["t_start"] + alpha * output["delta_t_start"]
        )
        output[f"{mode}_t_end"] = (
            (1.0 - alpha) * output["t_end"] + alpha * output["delta_t_end"]
        )
    output["blend_t_start"] = (
        (1.0 - boundary_blend) * output["t_start"]
        + boundary_blend * output["delta_t_start"]
    )
    output["blend_t_end"] = (
        (1.0 - boundary_blend) * output["t_end"]
        + boundary_blend * output["delta_t_end"]
    )
    if {"hybrid_t_start", "hybrid_t_end"} <= set(output.columns):
        output["reference_t_start"] = output["hybrid_t_start"]
        output["reference_t_end"] = output["hybrid_t_end"]
    return output


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    proposals = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, logits, metadata = load_cache(resolve(args.cache_dir))
    indices = np.arange(len(proposals), dtype=np.int64)
    boundary = pd.read_csv(resolve(args.boundary_ensemble)).reset_index(drop=True)
    assert_aligned(boundary, proposals, "Boundary ensemble")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for fold in range(5):
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
        assert_aligned(scored, proposals, f"Representation fold {fold}")
        frames.append(scored)

    ensemble = boundary.copy()
    for column in ("dense_score", "brem_score", "dense_point"):
        ensemble[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )
    ensemble = add_ranking_fusions(ensemble)
    if args.boundary_source == "representation":
        ensemble = add_boundary_modes(ensemble, frames, args.boundary_blend)
    required_boundary = {
        f"{args.boundary_mode}_t_start",
        f"{args.boundary_mode}_t_end",
    }
    missing_boundary = sorted(required_boundary - set(ensemble.columns))
    if missing_boundary:
        raise ValueError(
            f"Boundary mode {args.boundary_mode!r} is unavailable: {missing_boundary}"
        )
    ensemble.to_csv(out_dir / "scored_ensemble.csv", index=False)
    row = evaluate_variant(
        ensemble,
        args.score_column,
        args.boundary_mode,
        "source_cv_approved_test",
        args,
        out_dir / "predictions",
    )
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
