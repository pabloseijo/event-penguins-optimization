"""Route boundary experts from the full ATSN+TESPEC temporal sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.analyze_boundary_oracle_cv import (  # noqa: E402
    aligned,
    boundary_candidates,
    oracle_choice,
)
from dev.eval_boundary_quality_router_cv import (  # noqa: E402
    candidate_features,
    candidate_tiou,
    post_nms_training_indices,
)
from dev.eval_boundary_router_post_nms_cv import evaluate_post_nms  # noqa: E402
from dev.train_temporalmaxer_dense import (  # noqa: E402
    DenseFeatureDataset,
    atomic_torch_save,
    autocast_context,
    cache_paths,
    event_feature_configuration,
    load_annotation_index,
    load_cache,
    map_to_master,
    set_seed,
    stable_proposal_index,
)


class TemporalCandidateRouter(nn.Module):
    def __init__(
        self,
        temporal_dim: int,
        auxiliary_dim: int,
        candidate_dim: int,
        hidden_dim: int = 64,
        candidate_hidden_dim: int = 32,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.auxiliary_dim = int(auxiliary_dim)
        base_dim = temporal_dim - self.auxiliary_dim
        self.base_norm = nn.LayerNorm(base_dim)
        self.auxiliary_norm = (
            nn.LayerNorm(self.auxiliary_dim) if self.auxiliary_dim > 0 else None
        )
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
        self.output = nn.Sequential(
            nn.Linear(2 * hidden_dim + candidate_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        temporal_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        split = temporal_features.shape[-1] - self.auxiliary_dim
        normalized = self.base_norm(temporal_features[:, :, :split])
        if self.auxiliary_norm is not None:
            normalized = torch.cat(
                (
                    normalized,
                    self.auxiliary_norm(temporal_features[:, :, split:]),
                ),
                dim=2,
            )
        temporal = self.temporal_projection(normalized).transpose(1, 2)
        temporal = temporal + self.temporal_tower(temporal)
        pooled = torch.cat((temporal.mean(dim=2), temporal.amax(dim=2)), dim=1)
        candidates = self.candidate_projection(candidate_features)
        pooled = pooled[:, None, :].expand(-1, candidates.shape[1], -1)
        return self.output(torch.cat((pooled, candidates), dim=2)).squeeze(2)


class TemporalRouterDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        master_indices: np.ndarray,
        event_feature_path: Path,
        candidate_features_array: np.ndarray,
        quality: np.ndarray | None = None,
    ) -> None:
        self.temporal = DenseFeatureDataset(
            feature_path,
            master_indices,
            event_feature_path=event_feature_path,
        )
        self.candidates = candidate_features_array
        self.quality = quality

    def __len__(self) -> int:
        return len(self.temporal)

    def __getitem__(self, index: int):
        temporal, _ = self.temporal[index]
        candidates = torch.from_numpy(self.candidates[index])
        if self.quality is None:
            return temporal, candidates
        return temporal, candidates, torch.from_numpy(self.quality[index])


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
        "--score-cache-root",
        default="tmp/temporalmaxer_dense/boundary_quality_router_cv",
    )
    parser.add_argument("--groupdro-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/temporal_boundary_router_cv"
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
    parser.add_argument(
        "--candidate-names",
        nargs="+",
        default=[
            "raw",
            "reference_blend050",
            "tespec_blend050",
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


def load_master_scores(
    root: Path,
    fold: int,
    name: str,
    master: pd.DataFrame,
) -> pd.DataFrame:
    path = root / f"fold_{fold:02d}" / f"{name}_scored_master.csv"
    frame = pd.read_csv(path)
    if len(frame) != len(master) or not stable_proposal_index(frame).equals(
        stable_proposal_index(master)
    ):
        raise ValueError(f"Invalid shared score cache: {path}")
    return frame


def select_boundary_candidates(
    names: list[str],
    starts: np.ndarray,
    ends: np.ndarray,
    selected_names: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    missing = sorted(set(selected_names) - set(names))
    if missing:
        raise ValueError(f"Unknown boundary candidates: {missing}")
    indices = [names.index(name) for name in selected_names]
    return list(selected_names), starts[:, indices], ends[:, indices]


def fit_temporal_router(
    dataset: TemporalRouterDataset,
    temporal_dim: int,
    auxiliary_dim: int,
    candidate_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> TemporalCandidateRouter:
    model = TemporalCandidateRouter(
        temporal_dim,
        auxiliary_dim,
        candidate_dim,
        args.hidden_dim,
        args.candidate_hidden_dim,
        args.dropout,
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
        for temporal, candidates, quality in loader:
            temporal = temporal.to(device)
            candidates = candidates.to(device)
            quality = quality.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(temporal, candidates)
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
    model: TemporalCandidateRouter,
    dataset: TemporalRouterDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions = []
    model.eval()
    for temporal, candidates in loader:
        with autocast_context(device):
            logits = model(temporal.to(device), candidates.to(device))
        predictions.append(torch.softmax(logits, dim=1).float().cpu().numpy())
    probability = np.concatenate(predictions, axis=0)
    return probability.argmax(axis=1), probability


def evaluate_scored_fold(
    scored: pd.DataFrame,
    fold: int,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    fold_out: Path,
) -> None:
    metrics_path = fold_out / "metrics.csv"
    partial_path = fold_out / "metrics_partial.csv"
    metric_rows = (
        pd.read_csv(partial_path).to_dict("records")
        if partial_path.exists()
        else []
    )
    completed = {str(row["boundary_mode"]) for row in metric_rows}
    for boundary in ("reference_blend050", "router", "oracle"):
        if boundary in completed:
            continue
        row = evaluate_post_nms(
            scored, boundary, fold, args, fold_out / "predictions"
        )
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        metric_rows.append(row)
        temporary = partial_path.with_suffix(partial_path.suffix + ".tmp")
        pd.DataFrame(metric_rows).to_csv(temporary, index=False)
        temporary.replace(partial_path)
    if len({str(row["boundary_mode"]) for row in metric_rows}) == 3:
        temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        pd.DataFrame(metric_rows).to_csv(temporary, index=False)
        temporary.replace(metrics_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, _, metadata = load_cache(resolve(args.cache_dir))
    event_path, auxiliary_dim = event_feature_configuration(args, metadata)
    if event_path is None:
        raise ValueError("The temporal router requires TESPEC event features")
    temporal_dim = int(metadata["feature_dim"]) + auxiliary_dim
    annotations = load_annotation_index(resolve(args.ann_path))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_root = resolve(args.score_cache_root)

    for fold in args.folds:
        fold_out = out_dir / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        metrics_path = fold_out / "metrics.csv"
        if metrics_path.exists():
            continue
        target = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        scored_path = fold_out / "scored.csv"
        if scored_path.exists():
            cached_scored = pd.read_csv(scored_path)
            if len(cached_scored) == len(target) and stable_proposal_index(
                cached_scored
            ).equals(stable_proposal_index(target)):
                evaluate_scored_fold(
                    cached_scored, fold, args, manifest, fold_out
                )
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
        source_quality = candidate_tiou(
            source, source_starts, source_ends, annotations
        )
        training_local = post_nms_training_indices(source, source_reference, args)
        training_candidates = source_candidates[training_local]
        mean = training_candidates.reshape(-1, training_candidates.shape[-1]).mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        std = training_candidates.reshape(-1, training_candidates.shape[-1]).std(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        std = np.maximum(std, 1e-4)
        training_candidates = (training_candidates - mean) / std
        training_dataset = TemporalRouterDataset(
            cache_paths(resolve(args.cache_dir))["features"],
            source_indices[training_local],
            event_path,
            training_candidates,
            source_quality[training_local],
        )
        model = fit_temporal_router(
            training_dataset,
            temporal_dim,
            auxiliary_dim,
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
        )
        target_candidates = (target_candidates - mean) / std
        target_dataset = TemporalRouterDataset(
            cache_paths(resolve(args.cache_dir))["features"],
            target_indices,
            event_path,
            target_candidates,
        )
        choice, probability = route(model, target_dataset, args, device)
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
        rows = np.arange(len(scored))
        scored["router_t_start"] = target_starts[rows, choice]
        scored["router_t_end"] = target_ends[rows, choice]
        scored["router_quality"] = probability[rows, choice]
        scored["router_candidate"] = np.asarray(names, dtype=object)[choice]
        scored["oracle_t_start"] = target_starts[rows, oracle]
        scored["oracle_t_end"] = target_ends[rows, oracle]
        scored["oracle_tiou"] = oracle_tiou
        temporary = scored_path.with_suffix(scored_path.suffix + ".tmp")
        scored.to_csv(temporary, index=False)
        temporary.replace(scored_path)
        evaluate_scored_fold(scored, fold, args, manifest, fold_out)

    metric_paths = [out_dir / f"fold_{fold:02d}" / "metrics.csv" for fold in range(5)]
    if not all(path.exists() for path in metric_paths):
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = []
    for boundary, group in metrics.groupby("boundary_mode"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        row = {"boundary_mode": boundary}
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
