"""Evaluate the one source-CV-approved salient boundary recipe on test.

The fixed recipe averages the soft boundaries of five recording-disjoint
routers and shrinks them by 25% toward the existing TemporalMaxer ensemble.
Ranking and reference Soft-NMS selection remain unchanged. No test sweep is
implemented by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dev.analyze_boundary_oracle_cv import boundary_candidates
from dev.eval_boundary_quality_router_cv import (
    candidate_features,
    post_nms_training_indices,
)
from dev.eval_boundary_router_post_nms_cv import evaluate_post_nms
from dev.eval_salient_boundary_router_cv import (
    SalientBoundaryRouter,
    SalientRouterDataset,
    relative_candidate_positions,
    route,
)
from dev.eval_temporal_boundary_router_cv import (
    load_master_scores,
    select_boundary_candidates,
)
from dev.train_temporalmaxer_dense import (
    cache_paths,
    load_cache,
    map_to_master,
    stable_proposal_index,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = [
    "raw",
    "reference_blend050",
    "reference_delta",
    "reference_distribution",
    "reference_point",
    "tespec_blend050",
    "tespec_delta",
    "tespec_distribution",
    "tespec_point",
    "mean_blend050",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-master",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument(
        "--source-fold-dir", default="tmp/temporalmaxer_dense/hybrid_source"
    )
    parser.add_argument(
        "--source-score-root",
        default="tmp/temporalmaxer_dense/boundary_quality_router_cv",
    )
    parser.add_argument(
        "--router-root",
        default="tmp/temporalmaxer_dense/salient_boundary_router_pilot",
    )
    parser.add_argument(
        "--test-proposals",
        default="tmp/temporalmaxer_dense/hybrid_test_proposals.csv",
    )
    parser.add_argument(
        "--test-cache", default="tmp/temporalmaxer_dense/hybrid_test_cache"
    )
    parser.add_argument(
        "--test-reference-root", default="tmp/temporalmaxer_dense/hybrid_test_eval"
    )
    parser.add_argument(
        "--test-tespec-root",
        default="tmp/temporalmaxer_dense/hybrid_test_tespec_combined_fixed",
    )
    parser.add_argument(
        "--base-ensemble",
        default="tmp/temporalmaxer_dense/hybrid_test_eval/scored_ensemble.csv",
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/salient_boundary_router_test"
    )
    parser.add_argument("--candidate-names", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--candidate-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--training-min-score", type=float, default=0.05)

    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--score-column", default="quality_score")
    parser.add_argument("--nms-boundary", default="reference_blend050")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def aligned(frame: pd.DataFrame, target: pd.DataFrame, label: str) -> pd.DataFrame:
    if len(frame) == len(target) and stable_proposal_index(frame).equals(
        stable_proposal_index(target)
    ):
        return frame.reset_index(drop=True)
    positions = map_to_master(frame, target)
    output = frame.iloc[positions].reset_index(drop=True)
    if not stable_proposal_index(output).equals(stable_proposal_index(target)):
        raise ValueError(f"{label} could not be aligned")
    return output


def source_normalization(
    fold: int,
    source_master: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    held_out = pd.read_csv(
        resolve(args.source_fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
    )
    held_out_recordings = set(held_out["rec_name"].astype(str))
    source_indices = np.flatnonzero(
        ~source_master["rec_name"].astype(str).isin(held_out_recordings).to_numpy()
    )
    source = source_master.iloc[source_indices].reset_index(drop=True)
    reference_all = load_master_scores(
        resolve(args.source_score_root), fold, "reference", source_master
    )
    tespec_all = load_master_scores(
        resolve(args.source_score_root), fold, "tespec", source_master
    )
    reference = reference_all.iloc[source_indices].reset_index(drop=True)
    tespec = tespec_all.iloc[source_indices].reset_index(drop=True)
    names, starts, ends = boundary_candidates(source, reference, tespec)
    names, starts, ends = select_boundary_candidates(
        names, starts, ends, args.candidate_names
    )
    candidates = candidate_features(source, reference, tespec, starts, ends)
    candidates = candidates[:, :, :-len(names)]
    selected = post_nms_training_indices(source, reference, args)
    flat = candidates[selected].reshape(-1, candidates.shape[-1])
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        flat.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4
    )
    return mean, std, names


def test_router_boundary(
    fold: int,
    proposals: pd.DataFrame,
    feature_path: Path,
    feature_dim: int,
    source_master: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = out_dir / f"router_soft_fold_{fold:02d}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["start"], cached["end"]
    mean, std, source_names = source_normalization(fold, source_master, args)
    reference = aligned(
        pd.read_csv(
            resolve(args.test_reference_root) / f"temporal_scored_fold_{fold:02d}.csv"
        ),
        proposals,
        f"test reference fold {fold}",
    )
    tespec = aligned(
        pd.read_csv(
            resolve(args.test_tespec_root) / f"scored_fold_{fold:02d}.csv"
        ),
        proposals,
        f"test TESPEC fold {fold}",
    )
    names, starts, ends = boundary_candidates(proposals, reference, tespec)
    names, starts, ends = select_boundary_candidates(
        names, starts, ends, args.candidate_names
    )
    if names != source_names:
        raise ValueError(f"Source and test candidates differ in fold {fold}")
    candidates = candidate_features(proposals, reference, tespec, starts, ends)
    candidates = (candidates[:, :, :-len(names)] - mean) / std
    positions = relative_candidate_positions(proposals, starts, ends)
    indices = np.arange(len(proposals), dtype=np.int64)
    dataset = SalientRouterDataset(
        feature_path,
        indices,
        candidates,
        positions,
    )
    model = SalientBoundaryRouter(
        feature_dim,
        candidates.shape[-1],
        args.hidden_dim,
        args.candidate_hidden_dim,
        args.dropout,
        args.augment_factor,
    ).to(device)
    checkpoint = torch.load(
        resolve(args.router_root) / f"fold_{fold:02d}" / "router_last.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    _, probability = route(model, dataset, args, device)
    soft_start = np.sum(probability * starts, axis=1)
    soft_end = np.sum(probability * ends, axis=1)
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, start=soft_start, end=soft_end)
    temporary.replace(cache_path)
    return soft_start, soft_end


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals = pd.read_csv(resolve(args.test_proposals)).reset_index(drop=True)
    source_master = pd.read_csv(resolve(args.source_master)).reset_index(drop=True)
    _, _, metadata = load_cache(resolve(args.test_cache))
    feature_path = cache_paths(resolve(args.test_cache))["features"]
    soft_boundaries = [
        test_router_boundary(
            fold,
            proposals,
            feature_path,
            int(metadata["feature_dim"]),
            source_master,
            args,
            device,
            out_dir,
        )
        for fold in range(5)
    ]
    router_start = np.mean([values[0] for values in soft_boundaries], axis=0)
    router_end = np.mean([values[1] for values in soft_boundaries], axis=0)

    base = aligned(
        pd.read_csv(resolve(args.base_ensemble)), proposals, "test base ensemble"
    )
    if not {"quality_score", "hybrid_t_start", "hybrid_t_end"} <= set(base.columns):
        raise ValueError("Base ensemble misses the frozen score or hybrid boundary")
    scored = proposals.copy()
    scored["quality_score"] = base["quality_score"].to_numpy(dtype=np.float64)
    scored["reference_blend050_t_start"] = base["hybrid_t_start"].to_numpy(
        dtype=np.float64
    )
    scored["reference_blend050_t_end"] = base["hybrid_t_end"].to_numpy(
        dtype=np.float64
    )
    scored["router_soft_t_start"] = router_start
    scored["router_soft_t_end"] = router_end
    scored["router_shrink025_t_start"] = (
        0.75 * scored["reference_blend050_t_start"] + 0.25 * router_start
    )
    scored["router_shrink025_t_end"] = (
        0.75 * scored["reference_blend050_t_end"] + 0.25 * router_end
    )
    scored.to_csv(out_dir / "scored.csv", index=False)

    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {str(row["boundary_mode"]) for row in rows}
    for boundary in ("reference_blend050", "router_shrink025"):
        if boundary in completed:
            continue
        rows.append(
            evaluate_post_nms(
                scored,
                boundary,
                0,
                args,
                out_dir / "predictions",
            )
        )
        temporary = partial_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(partial_path)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "recipe.json").write_text(
        json.dumps(
            {
                "routers": 5,
                "candidate_names": args.candidate_names,
                "soft_boundary_ensemble": "mean",
                "reference_weight": 0.75,
                "router_weight": 0.25,
                "ranking": "unchanged GroupDRO ensemble",
                "nms_selection": "unchanged reference_blend050",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
