"""Train a strict recording-disjoint BREM-style boundary quality router."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.analyze_boundary_oracle_cv import (  # noqa: E402
    aligned,
    boundary_candidates,
    oracle_choice,
)
from dev.eval_boundary_router_post_nms_cv import temporal_soft_nms_indices  # noqa: E402
from dev.train_temporalmaxer_dense import (  # noqa: E402
    autocast_context,
    atomic_torch_save,
    cache_paths,
    evaluate_variant,
    load_annotation_index,
    load_cache,
    make_model,
    map_to_master,
    roi_key,
    score_model,
    set_seed,
    stable_proposal_index,
)


class CandidateQualityRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


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
        "--reference-checkpoint-root",
        default="tmp/temporalmaxer_dense/screened_cv_blend075_erm",
    )
    parser.add_argument(
        "--tespec-checkpoint-root",
        default="tmp/temporalmaxer_dense/screened_cv_tespec_combined",
    )
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/boundary_quality_router_cv"
    )
    parser.add_argument(
        "--score-cache-root",
        default=None,
        help="Optional validated master-score cache from a previous router run.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-temperature", type=float, default=0.10)
    parser.add_argument("--min-training-tiou", type=float, default=0.0)
    parser.add_argument(
        "--train-selection",
        choices=["all", "reference_post_nms"],
        default="reference_post_nms",
    )
    parser.add_argument("--training-min-score", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
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


def score_checkpoint(
    checkpoint_path: Path,
    proposals: pd.DataFrame,
    indices: np.ndarray,
    logits: np.ndarray,
    metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
    event_cache: str | None,
) -> pd.DataFrame:
    local_args = copy.deepcopy(args)
    local_args.event_feature_cache_dir = event_cache
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for name in (
        "hidden_dim",
        "pyramid_levels",
        "dropout",
        "trident_bins",
        "event_features_only",
    ):
        if name in checkpoint["args"]:
            setattr(local_args, name, checkpoint["args"][name])
    model = make_model(metadata, local_args).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    scored = score_model(
        model,
        proposals,
        indices,
        cache_paths(resolve(args.cache_dir))["features"],
        logits,
        local_args,
        device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scored


def candidate_features(
    proposals: pd.DataFrame,
    reference: pd.DataFrame,
    tespec: pd.DataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    raw_start = proposals["t_start"].to_numpy(dtype=np.float64)
    raw_end = proposals["t_end"].to_numpy(dtype=np.float64)
    raw_duration = np.maximum(raw_end - raw_start, 1.0)
    common_columns = [
        proposals["score"].to_numpy(dtype=np.float64),
        reference["cnn_score"].to_numpy(dtype=np.float64),
    ]
    for frame in (reference, tespec):
        for column in (
            "dense_quality",
            "dense_action",
            "dense_point",
            "dense_score",
            "brem_score",
        ):
            common_columns.append(frame[column].to_numpy(dtype=np.float64))
    common_columns.extend(
        [
            np.log1p(raw_duration / 1e6),
            (
                tespec["delta_t_start"].to_numpy(dtype=np.float64)
                - reference["delta_t_start"].to_numpy(dtype=np.float64)
            )
            / raw_duration,
            (
                tespec["delta_t_end"].to_numpy(dtype=np.float64)
                - reference["delta_t_end"].to_numpy(dtype=np.float64)
            )
            / raw_duration,
        ]
    )
    common = np.stack(common_columns, axis=1)
    common = np.repeat(common[:, None, :], starts.shape[1], axis=1)
    candidate_duration = np.maximum(ends - starts, 1.0)
    candidate_specific = np.stack(
        [
            (starts - raw_start[:, None]) / raw_duration[:, None],
            (ends - raw_end[:, None]) / raw_duration[:, None],
            np.log(candidate_duration / raw_duration[:, None]),
            (0.5 * (starts + ends) - 0.5 * (raw_start + raw_end)[:, None])
            / raw_duration[:, None],
            np.log1p(candidate_duration / 1e6),
        ],
        axis=2,
    )
    identity = np.broadcast_to(
        np.eye(starts.shape[1], dtype=np.float64)[None, :, :],
        (len(proposals), starts.shape[1], starts.shape[1]),
    )
    features = np.concatenate((common, candidate_specific, identity), axis=2)
    return np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def candidate_tiou(
    proposals: pd.DataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
    annotations: dict[tuple[str, str], np.ndarray],
    chunk_size: int = 2048,
) -> np.ndarray:
    quality = np.zeros(starts.shape, dtype=np.float32)
    keys = pd.DataFrame(
        {
            "rec_name": proposals["rec_name"].astype(str),
            "roi": proposals["roi_id"].map(roi_key),
        }
    )
    for (recording, roi), local in keys.groupby(["rec_name", "roi"]).groups.items():
        indices = np.asarray(list(local), dtype=np.int64)
        segments = annotations.get((recording, roi), np.empty((0, 2), dtype=np.float64))
        if segments.size == 0:
            continue
        for offset in range(0, len(indices), chunk_size):
            selected = indices[offset : offset + chunk_size]
            candidate_start = starts[selected, :, None] / 1e6
            candidate_end = ends[selected, :, None] / 1e6
            intersection = np.maximum(
                0.0,
                np.minimum(candidate_end, segments[None, None, :, 1])
                - np.maximum(candidate_start, segments[None, None, :, 0]),
            )
            union = (
                candidate_end
                - candidate_start
                + segments[None, None, :, 1]
                - segments[None, None, :, 0]
                - intersection
            )
            tiou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            quality[selected] = tiou.max(axis=2).astype(np.float32)
    return quality


def post_nms_training_indices(
    proposals: pd.DataFrame,
    reference: pd.DataFrame,
    args: argparse.Namespace,
) -> np.ndarray:
    raw_start = proposals["t_start"].to_numpy(dtype=np.float64)
    raw_end = proposals["t_end"].to_numpy(dtype=np.float64)
    start = 0.5 * (
        raw_start + reference["delta_t_start"].to_numpy(dtype=np.float64)
    )
    end = 0.5 * (
        raw_end + reference["delta_t_end"].to_numpy(dtype=np.float64)
    )
    duration_seconds = np.maximum(end - start, 0.0) / 1e6
    score = reference["brem_score"].to_numpy(dtype=np.float64) * np.exp(
        -np.maximum(0.0, duration_seconds - args.duration_dmax) / args.duration_sigma
    )
    selected = []
    grouping = proposals.groupby(["rec_name", "roi_id"]).groups
    for local in grouping.values():
        indices = np.asarray(list(local), dtype=np.int64)
        indices = indices[score[indices] >= args.training_min_score]
        if args.pre_nms_topk_per_roi > 0 and len(indices) > args.pre_nms_topk_per_roi:
            order = np.argsort(score[indices])[::-1][: args.pre_nms_topk_per_roi]
            indices = indices[order]
        if len(indices) == 0:
            continue
        candidates = np.stack((start[indices], end[indices], score[indices]), axis=1)
        keep, _ = temporal_soft_nms_indices(
            candidates,
            sigma=args.soft_nms_sigma,
            score_threshold=args.soft_nms_score_threshold,
        )
        selected.append(indices[keep])
    if not selected:
        raise ValueError("Reference post-NMS selection produced no router samples")
    return np.concatenate(selected)


def fit_router(
    features: np.ndarray,
    quality: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[CandidateQualityRouter, np.ndarray, np.ndarray]:
    relevant = quality.max(axis=1) >= args.min_training_tiou
    selected_features = features[relevant]
    selected_quality = quality[relevant]
    flat = selected_features.reshape(-1, selected_features.shape[-1])
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    selected_features = (selected_features - mean) / std
    dataset = TensorDataset(
        torch.from_numpy(selected_features),
        torch.from_numpy(selected_quality),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    model = CandidateQualityRouter(features.shape[-1], args.router_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 1
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for batch_features, batch_quality in loader:
            batch_features = batch_features.to(device)
            batch_quality = batch_quality.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(batch_features)
                quality_loss = F.binary_cross_entropy_with_logits(
                    logits, batch_quality
                )
                target_distribution = torch.softmax(
                    batch_quality / args.rank_temperature, dim=1
                )
                rank_loss = -(
                    target_distribution * F.log_softmax(logits, dim=1)
                ).sum(dim=1).mean()
                loss = quality_loss + args.rank_weight * rank_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        atomic_torch_save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "mean": mean,
                "std": std,
            },
            checkpoint_path,
        )
    return model, mean, std


@torch.no_grad()
def route_candidates(
    model: CandidateQualityRouter,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = (features - mean) / std
    loader = DataLoader(
        TensorDataset(torch.from_numpy(standardized)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    batches = []
    model.eval()
    for (batch,) in loader:
        batches.append(torch.sigmoid(model(batch.to(device))).float().cpu().numpy())
    quality = np.concatenate(batches, axis=0)
    return quality.argmax(axis=1), quality


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, logits, metadata = load_cache(resolve(args.cache_dir))
    all_indices = np.arange(len(master), dtype=np.int64)
    annotations = load_annotation_index(resolve(args.ann_path))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        fold_out = out_dir / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        metrics_path = fold_out / "metrics.csv"
        if metrics_path.exists():
            continue
        scored_frames = {}
        for name, root, event_cache in (
            ("reference", args.reference_checkpoint_root, None),
            ("tespec", args.tespec_checkpoint_root, args.event_feature_cache_dir),
        ):
            scored_path = fold_out / f"{name}_scored_master.csv"
            if args.score_cache_root:
                shared_path = (
                    resolve(args.score_cache_root)
                    / f"fold_{fold:02d}"
                    / f"{name}_scored_master.csv"
                )
                if shared_path.exists():
                    scored_path = shared_path
            if scored_path.exists():
                cached = pd.read_csv(scored_path)
                if len(cached) == len(master) and stable_proposal_index(cached).equals(
                    stable_proposal_index(master)
                ):
                    scored_frames[name] = cached
                else:
                    local_path = fold_out / f"{name}_scored_master.csv"
                    if scored_path == local_path:
                        scored_path.unlink()
                    else:
                        scored_path = local_path
            if name not in scored_frames:
                scored_frames[name] = score_checkpoint(
                    resolve(root) / f"fold_{fold:02d}" / "best.pt",
                    master,
                    all_indices,
                    logits,
                    metadata,
                    args,
                    device,
                    event_cache,
                )
                temporary = scored_path.with_suffix(scored_path.suffix + ".tmp")
                scored_frames[name].to_csv(temporary, index=False)
                temporary.replace(scored_path)

        target = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        target_indices = map_to_master(master, target)
        held_out = set(target["rec_name"].astype(str))
        source_indices = np.flatnonzero(
            ~master["rec_name"].astype(str).isin(held_out).to_numpy()
        )
        source = master.iloc[source_indices].reset_index(drop=True)
        source_reference = scored_frames["reference"].iloc[source_indices].reset_index(drop=True)
        source_tespec = scored_frames["tespec"].iloc[source_indices].reset_index(drop=True)
        names, source_starts, source_ends = boundary_candidates(
            source, source_reference, source_tespec
        )
        source_features = candidate_features(
            source, source_reference, source_tespec, source_starts, source_ends
        )
        source_quality = candidate_tiou(
            source, source_starts, source_ends, annotations
        )
        if args.train_selection == "reference_post_nms":
            training_indices = post_nms_training_indices(
                source, source_reference, args
            )
            source_features = source_features[training_indices]
            source_quality = source_quality[training_indices]
        model, mean, std = fit_router(
            source_features,
            source_quality,
            args,
            device,
            fold_out / "router_last.pt",
        )

        target_reference = scored_frames["reference"].iloc[target_indices].reset_index(drop=True)
        target_tespec = scored_frames["tespec"].iloc[target_indices].reset_index(drop=True)
        target_names, target_starts, target_ends = boundary_candidates(
            target, target_reference, target_tespec
        )
        if target_names != names:
            raise ValueError("Source and target candidate sets differ")
        target_features = candidate_features(
            target, target_reference, target_tespec, target_starts, target_ends
        )
        choice, predicted_quality = route_candidates(
            model, target_features, mean, std, device, args.batch_size
        )
        oracle, oracle_tiou = oracle_choice(
            target, target_starts, target_ends, annotations
        )
        groupdro = aligned(
            pd.read_csv(
                resolve(args.groupdro_root)
                / f"fold_{fold:02d}"
                / "cache"
                / "val_scores_qhead_qfl_only.csv"
            ),
            target,
        )
        scored = target.copy()
        scored["quality_score"] = groupdro["quality_score"].to_numpy(dtype=np.float64)
        for candidate, name in enumerate(names):
            scored[f"{name}_t_start"] = target_starts[:, candidate]
            scored[f"{name}_t_end"] = target_ends[:, candidate]
        rows_index = np.arange(len(scored))
        scored["router_t_start"] = target_starts[rows_index, choice]
        scored["router_t_end"] = target_ends[rows_index, choice]
        scored["router_quality"] = predicted_quality[rows_index, choice]
        scored["router_candidate"] = np.asarray(names, dtype=object)[choice]
        scored["oracle_t_start"] = target_starts[rows_index, oracle]
        scored["oracle_t_end"] = target_ends[rows_index, oracle]
        scored["oracle_tiou"] = oracle_tiou
        scored.to_csv(fold_out / "scored.csv", index=False)
        partial_path = fold_out / "metrics_partial.csv"
        rows = (
            pd.read_csv(partial_path).to_dict("records")
            if partial_path.exists()
            else []
        )
        completed_modes = {str(row["boundary_mode"]) for row in rows}
        for mode in ("reference_blend050", "tespec_blend050", "router", "oracle"):
            if mode in completed_modes:
                continue
            row = evaluate_variant(
                scored,
                "quality_score",
                mode,
                f"boundary_quality_router_fold_{fold:02d}",
                args,
                fold_out / "predictions",
            )
            row["fold"] = fold
            row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
            rows.append(row)
            temporary = partial_path.with_suffix(partial_path.suffix + ".tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(partial_path)
        if len(completed_modes | {str(row["boundary_mode"]) for row in rows}) == 4:
            temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(metrics_path)

    metric_paths = [out_dir / f"fold_{fold:02d}" / "metrics.csv" for fold in range(5)]
    if not all(path.exists() for path in metric_paths):
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = []
    for mode, group in metrics.groupby("boundary_mode"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row = {"boundary_mode": mode}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        summary.append(row)
    result = pd.DataFrame(summary).sort_values("mean_mAP", ascending=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
