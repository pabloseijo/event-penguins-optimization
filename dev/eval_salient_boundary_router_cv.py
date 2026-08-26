"""Route temporal boundary experts with candidate-aligned boundary pooling.

Unlike the previous temporal router, this model samples frozen ATSN features
immediately outside and inside each candidate start/end. The architecture is a
small source-only adaptation of the salient boundary feature idea (AFSD,
CVPR 2021). Candidate identity is deliberately excluded to prevent memorizing
which expert wins in the source recordings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dev.analyze_boundary_oracle_cv import (
    boundary_candidates,
    oracle_choice,
)
from dev.eval_boundary_quality_router_cv import (
    candidate_features,
    candidate_tiou,
    post_nms_training_indices,
)
from dev.eval_boundary_router_post_nms_cv import evaluate_post_nms
from dev.eval_temporal_boundary_router_cv import (
    load_master_scores,
    select_boundary_candidates,
)
from dev.train_temporalmaxer_dense import (
    DenseFeatureDataset,
    atomic_torch_save,
    autocast_context,
    cache_paths,
    load_annotation_index,
    load_cache,
    map_to_master,
    set_seed,
    stable_proposal_index,
)


ROOT = Path(__file__).resolve().parents[1]


def sample_at_relative_positions(
    temporal: torch.Tensor,
    positions: torch.Tensor,
    augment_factor: int = 5,
) -> torch.Tensor:
    """Linearly sample [B,T,C] features at [B,K,M] proposal-relative positions."""
    if temporal.ndim != 3 or positions.ndim != 3:
        raise ValueError("Expected temporal [B,T,C] and positions [B,K,M]")
    if temporal.shape[0] != positions.shape[0]:
        raise ValueError("Temporal features and positions use different batch sizes")
    batch_size, length, channels = temporal.shape
    lower_relative = -1.0 / augment_factor
    upper_relative = 1.0 + 1.0 / augment_factor
    continuous = (
        (positions - lower_relative)
        / (upper_relative - lower_relative)
        * (length - 1)
    ).clamp(0.0, float(length - 1))
    lower = continuous.floor().long()
    upper = (lower + 1).clamp_max(length - 1)
    weight = (continuous - lower.to(continuous.dtype)).unsqueeze(-1)
    flat_lower = lower.reshape(batch_size, -1)
    flat_upper = upper.reshape(batch_size, -1)
    lower_values = temporal.gather(
        1, flat_lower.unsqueeze(2).expand(-1, -1, channels)
    )
    upper_values = temporal.gather(
        1, flat_upper.unsqueeze(2).expand(-1, -1, channels)
    )
    shape = (*positions.shape, channels)
    return (
        lower_values.reshape(shape) * (1.0 - weight)
        + upper_values.reshape(shape) * weight
    )


class SalientBoundaryRouter(nn.Module):
    def __init__(
        self,
        temporal_dim: int,
        candidate_dim: int,
        hidden_dim: int = 64,
        candidate_hidden_dim: int = 32,
        dropout: float = 0.10,
        augment_factor: int = 5,
    ) -> None:
        super().__init__()
        self.augment_factor = int(augment_factor)
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.temporal_projection = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal_tower = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden_dim),
            nn.GELU(),
        )
        # global mean/max, four boundary regions and two signed contrasts
        boundary_dim = 8 * hidden_dim + candidate_hidden_dim
        self.output = nn.Sequential(
            nn.Linear(boundary_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(
        self,
        temporal_features: torch.Tensor,
        candidate_features_tensor: torch.Tensor,
        candidate_positions: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_positions.ndim != 3 or candidate_positions.shape[2] != 2:
            raise ValueError("Expected candidate start/end positions with shape [B,K,2]")
        projected = self.temporal_projection(
            self.temporal_norm(temporal_features)
        )
        temporal = projected + self.temporal_tower(projected.transpose(1, 2)).transpose(1, 2)
        step = (1.0 + 2.0 / self.augment_factor) / max(temporal.shape[1] - 1, 1)
        offsets = temporal.new_tensor([-1.5, -0.5, 0.5, 1.5]) * step
        start_positions = candidate_positions[:, :, 0, None] + offsets
        end_positions = candidate_positions[:, :, 1, None] + offsets
        start_samples = sample_at_relative_positions(
            temporal, start_positions, self.augment_factor
        )
        end_samples = sample_at_relative_positions(
            temporal, end_positions, self.augment_factor
        )
        start_outside = start_samples[:, :, :2].mean(dim=2)
        start_inside = start_samples[:, :, 2:].mean(dim=2)
        end_inside = end_samples[:, :, :2].mean(dim=2)
        end_outside = end_samples[:, :, 2:].mean(dim=2)
        start_contrast = start_inside - start_outside
        end_contrast = end_inside - end_outside
        global_features = torch.cat(
            (temporal.mean(dim=1), temporal.amax(dim=1)), dim=1
        )
        global_features = global_features[:, None, :].expand(
            -1, candidate_positions.shape[1], -1
        )
        candidate_embedding = self.candidate_projection(candidate_features_tensor)
        fused = torch.cat(
            (
                global_features,
                start_outside,
                start_inside,
                end_inside,
                end_outside,
                start_contrast,
                end_contrast,
                candidate_embedding,
            ),
            dim=2,
        )
        return self.output(fused).squeeze(2)


class SalientRouterDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        master_indices: np.ndarray,
        candidate_features_array: np.ndarray,
        candidate_positions: np.ndarray,
        quality: np.ndarray | None = None,
    ) -> None:
        self.temporal = DenseFeatureDataset(feature_path, master_indices)
        self.candidate_features = candidate_features_array
        self.candidate_positions = candidate_positions
        self.quality = quality

    def __len__(self) -> int:
        return len(self.temporal)

    def __getitem__(self, index: int):
        temporal, _ = self.temporal[index]
        values = (
            temporal,
            torch.from_numpy(self.candidate_features[index]),
            torch.from_numpy(self.candidate_positions[index]),
        )
        if self.quality is None:
            return values
        return (*values, torch.from_numpy(self.quality[index]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-proposals",
        default="tmp/temporalmaxer_dense/hybrid_source/master_proposals.csv",
    )
    parser.add_argument(
        "--fold-dir", default="tmp/temporalmaxer_dense/hybrid_source"
    )
    parser.add_argument(
        "--cache-dir", default="tmp/temporalmaxer_dense/hybrid_source_cache"
    )
    parser.add_argument(
        "--score-cache-root",
        default="tmp/temporalmaxer_dense/boundary_quality_router_cv",
    )
    parser.add_argument(
        "--groupdro-root", default="tmp/quality_head/family_groupdro_cv"
    )
    parser.add_argument(
        "--manifest", default="tmp/cv/recording_folds_r5/manifest.csv"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/salient_boundary_router_cv"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--candidate-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--quality-weight", type=float, default=0.25)
    parser.add_argument("--rank-temperature", type=float, default=0.10)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument(
        "--candidate-names",
        nargs="+",
        default=[
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
        ],
    )
    parser.add_argument("--training-min-score", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
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


def relative_candidate_positions(
    proposals: pd.DataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    raw_start = proposals["t_start"].to_numpy(dtype=np.float64)
    raw_end = proposals["t_end"].to_numpy(dtype=np.float64)
    duration = np.maximum(raw_end - raw_start, 1.0)
    return np.stack(
        (
            (starts - raw_start[:, None]) / duration[:, None],
            (ends - raw_start[:, None]) / duration[:, None],
        ),
        axis=2,
    ).astype(np.float32)


def fit_router(
    dataset: SalientRouterDataset,
    temporal_dim: int,
    candidate_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> SalientBoundaryRouter:
    model = SalientBoundaryRouter(
        temporal_dim,
        candidate_dim,
        args.hidden_dim,
        args.candidate_hidden_dim,
        args.dropout,
        args.augment_factor,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 1
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for temporal, candidates, positions, quality in loader:
            temporal = temporal.to(device)
            candidates = candidates.to(device)
            positions = positions.to(device)
            quality = quality.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(temporal, candidates, positions)
                quality_loss = F.binary_cross_entropy_with_logits(logits, quality)
                target_distribution = torch.softmax(
                    quality / args.rank_temperature, dim=1
                )
                rank_loss = -(
                    target_distribution * F.log_softmax(logits, dim=1)
                ).sum(dim=1).mean()
                loss = args.quality_weight * quality_loss + rank_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        atomic_torch_save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )
    return model


@torch.no_grad()
def route(
    model: SalientBoundaryRouter,
    dataset: SalientRouterDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    probabilities = []
    model.eval()
    for temporal, candidates, positions in loader:
        with autocast_context(device):
            logits = model(
                temporal.to(device), candidates.to(device), positions.to(device)
            )
        probabilities.append(torch.softmax(logits, dim=1).float().cpu().numpy())
    probability = np.concatenate(probabilities, axis=0)
    return probability.argmax(axis=1), probability


def evaluate_fold(
    scored: pd.DataFrame,
    fold: int,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    fold_out: Path,
) -> None:
    scored = add_router_shrinkage(scored)
    partial_path = fold_out / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {str(row["boundary_mode"]) for row in rows}
    boundaries = (
        "reference_blend050",
        "router",
        "router_soft",
        "router_shrink025",
        "router_shrink050",
        "router_shrink075",
        "oracle",
    )
    for boundary in boundaries:
        if boundary in completed:
            continue
        row = evaluate_post_nms(
            scored, boundary, fold, args, fold_out / "predictions"
        )
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)
    if len({str(row["boundary_mode"]) for row in rows}) == len(boundaries):
        pd.DataFrame(rows).to_csv(fold_out / "metrics.csv", index=False)


def add_router_shrinkage(scored: pd.DataFrame) -> pd.DataFrame:
    output = scored.copy()
    required = {
        "reference_blend050_t_start",
        "reference_blend050_t_end",
        "router_soft_t_start",
        "router_soft_t_end",
    }
    missing = sorted(required - set(output.columns))
    if missing:
        raise ValueError(f"Cannot shrink router boundaries; missing {missing}")
    for alpha in (0.25, 0.50, 0.75):
        name = f"router_shrink{int(alpha * 100):03d}"
        for suffix in ("start", "end"):
            output[f"{name}_t_{suffix}"] = (
                (1.0 - alpha) * output[f"reference_blend050_t_{suffix}"]
                + alpha * output[f"router_soft_t_{suffix}"]
            )
    return output


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, _, metadata = load_cache(resolve(args.cache_dir))
    annotations = load_annotation_index(resolve(args.ann_path))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_root = resolve(args.score_cache_root)
    feature_path = cache_paths(resolve(args.cache_dir))["features"]

    for fold in args.folds:
        fold_out = out_dir / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        if (fold_out / "metrics.csv").exists():
            continue
        target = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        scored_path = fold_out / "scored.csv"
        if scored_path.exists():
            cached = pd.read_csv(scored_path)
            if len(cached) == len(target) and stable_proposal_index(cached).equals(
                stable_proposal_index(target)
            ):
                evaluate_fold(cached, fold, args, manifest, fold_out)
                continue

        target_indices = map_to_master(master, target)
        held_out = set(target["rec_name"].astype(str))
        source_indices = np.flatnonzero(
            ~master["rec_name"].astype(str).isin(held_out).to_numpy()
        )
        source = master.iloc[source_indices].reset_index(drop=True)
        reference_all = load_master_scores(score_root, fold, "reference", master)
        tespec_all = load_master_scores(score_root, fold, "tespec", master)
        source_reference = reference_all.iloc[source_indices].reset_index(drop=True)
        source_tespec = tespec_all.iloc[source_indices].reset_index(drop=True)
        names, source_starts, source_ends = boundary_candidates(
            source, source_reference, source_tespec
        )
        names, source_starts, source_ends = select_boundary_candidates(
            names, source_starts, source_ends, args.candidate_names
        )
        source_candidates = candidate_features(
            source, source_reference, source_tespec, source_starts, source_ends
        )
        source_candidates = source_candidates[:, :, :-len(names)]
        source_quality = candidate_tiou(
            source, source_starts, source_ends, annotations
        )
        source_positions = relative_candidate_positions(
            source, source_starts, source_ends
        )
        training_local = post_nms_training_indices(source, source_reference, args)
        training_candidates = source_candidates[training_local]
        flat = training_candidates.reshape(-1, training_candidates.shape[-1])
        mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(
            flat.std(axis=0, dtype=np.float64).astype(np.float32), 1e-4
        )
        training_candidates = (training_candidates - mean) / std
        training_dataset = SalientRouterDataset(
            feature_path,
            source_indices[training_local],
            training_candidates,
            source_positions[training_local],
            source_quality[training_local],
        )
        model = fit_router(
            training_dataset,
            int(metadata["feature_dim"]),
            training_candidates.shape[-1],
            args,
            device,
            fold_out / "router_last.pt",
        )

        target_reference = reference_all.iloc[target_indices].reset_index(drop=True)
        target_tespec = tespec_all.iloc[target_indices].reset_index(drop=True)
        target_names, target_starts, target_ends = boundary_candidates(
            target, target_reference, target_tespec
        )
        target_names, target_starts, target_ends = select_boundary_candidates(
            target_names, target_starts, target_ends, args.candidate_names
        )
        if target_names != names:
            raise ValueError("Source and target candidate sets differ")
        target_candidates = candidate_features(
            target, target_reference, target_tespec, target_starts, target_ends
        )[:, :, :-len(names)]
        target_candidates = (target_candidates - mean) / std
        target_positions = relative_candidate_positions(
            target, target_starts, target_ends
        )
        target_dataset = SalientRouterDataset(
            feature_path,
            target_indices,
            target_candidates,
            target_positions,
        )
        choice, probability = route(model, target_dataset, args, device)
        oracle, oracle_tiou = oracle_choice(
            target, target_starts, target_ends, annotations
        )
        groupdro_all = pd.read_csv(
            resolve(args.groupdro_root)
            / f"fold_{fold:02d}"
            / "cache"
            / "val_scores_qhead_qfl_only.csv"
        )
        groupdro = groupdro_all.iloc[
            map_to_master(groupdro_all, target)
        ].reset_index(drop=True)
        scored = target.copy()
        scored["quality_score"] = groupdro["quality_score"].to_numpy(dtype=np.float64)
        for candidate, name in enumerate(names):
            scored[f"{name}_t_start"] = target_starts[:, candidate]
            scored[f"{name}_t_end"] = target_ends[:, candidate]
        row_indices = np.arange(len(scored))
        scored["router_t_start"] = target_starts[row_indices, choice]
        scored["router_t_end"] = target_ends[row_indices, choice]
        scored["router_soft_t_start"] = np.sum(
            probability * target_starts, axis=1
        )
        scored["router_soft_t_end"] = np.sum(
            probability * target_ends, axis=1
        )
        scored["router_quality"] = probability[row_indices, choice]
        scored["router_candidate"] = np.asarray(names, dtype=object)[choice]
        scored["oracle_t_start"] = target_starts[row_indices, oracle]
        scored["oracle_t_end"] = target_ends[row_indices, oracle]
        scored["oracle_tiou"] = oracle_tiou
        temporary = scored_path.with_suffix(".csv.tmp")
        scored.to_csv(temporary, index=False)
        temporary.replace(scored_path)
        evaluate_fold(scored, fold, args, manifest, fold_out)

    metric_paths = [out_dir / f"fold_{fold:02d}" / "metrics.csv" for fold in range(5)]
    if not all(path.exists() for path in metric_paths):
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for boundary, group in metrics.groupby("boundary_mode"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row: dict[str, float | str] = {"boundary_mode": boundary}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
