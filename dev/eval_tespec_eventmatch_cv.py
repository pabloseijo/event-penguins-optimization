"""Evaluate a lightweight EventMatch-style TESPEC adapter on source CV."""

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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.eval_temporalmaxer_groupdro_cv import add_ranking_fusions  # noqa: E402
from dev.train_temporalmaxer_dense import (  # noqa: E402
    DenseFeatureDataset,
    atomic_torch_save,
    autocast_context,
    build_targets,
    cache_paths,
    choose_training_indices,
    dense_loss,
    evaluate_variant,
    event_feature_configuration,
    load_cache,
    make_model,
    map_to_master,
    score_model,
    set_seed,
    softmax_ed,
    subset_targets,
)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


def reverse_gradient(value: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(value, float(strength))


def _limited_anchors(features: torch.Tensor, maximum: int) -> torch.Tensor:
    if maximum > 0 and len(features) > maximum:
        indices = torch.randperm(len(features), device=features.device)[:maximum]
        return features[indices]
    return features


def semantic_domain_loss(
    discriminator: nn.Module,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_action: torch.Tensor,
    target_action_probability: torch.Tensor,
    pseudo_foreground: float,
    pseudo_background: float,
    max_anchors: int,
) -> torch.Tensor:
    """SADA-style class-conditional alignment of action/background anchors."""
    losses = []
    masks = (
        (source_action >= 0.5, target_action_probability >= pseudo_foreground),
        (source_action <= 0.05, target_action_probability <= pseudo_background),
    )
    for semantic_class, (source_mask, target_mask) in enumerate(masks):
        source = _limited_anchors(source_features[source_mask], max_anchors)
        target = _limited_anchors(target_features[target_mask], max_anchors)
        if len(source) == 0 or len(target) == 0:
            continue
        features = reverse_gradient(torch.cat((source, target), dim=0))
        labels = torch.cat(
            (
                torch.zeros(len(source), device=features.device),
                torch.ones(len(target), device=features.device),
            )
        )
        logits = discriminator(features)[:, semantic_class]
        losses.append(F.binary_cross_entropy_with_logits(logits, labels))
    if not losses:
        return 0.0 * (source_features.sum() + target_features.sum())
    return torch.stack(losses).mean()


class ConditionalSemanticDiscriminator(nn.Module):
    """Single-class SADA discriminator with learned foreground/background tokens."""

    def __init__(self, feature_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.semantic_embeddings = nn.Parameter(torch.zeros(2, feature_dim))
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.network[-1].bias, 0.25)

    def forward(self, features: torch.Tensor, semantic_class: int) -> torch.Tensor:
        token = self.semantic_embeddings[semantic_class].expand(len(features), -1)
        return self.network(torch.cat((features, token), dim=1)).squeeze(1)


def sada_conditional_domain_loss(
    discriminator: ConditionalSemanticDiscriminator,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_action: torch.Tensor,
    target_action_probability: torch.Tensor,
    pseudo_threshold: float,
    max_anchors: int,
    beta: float = 0.5,
    target_weight: float = 0.2,
    background_weight: float = 0.3,
) -> torch.Tensor:
    source_losses = []
    target_losses = []
    masks = (
        (source_action >= 0.5, target_action_probability >= pseudo_threshold, 1.0),
        (source_action <= 0.05, target_action_probability < pseudo_threshold, background_weight),
    )
    for semantic_class, (source_mask, target_mask, class_weight) in enumerate(masks):
        source = _limited_anchors(source_features[source_mask], max_anchors)
        target = _limited_anchors(target_features[target_mask], max_anchors)
        if len(source) > 0:
            source_logits = discriminator(
                reverse_gradient(source, beta), semantic_class
            )
            source_losses.append(
                class_weight
                * F.binary_cross_entropy_with_logits(
                    source_logits, torch.zeros_like(source_logits)
                )
            )
        if len(target) > 0:
            target_logits = discriminator(
                reverse_gradient(target, beta), semantic_class
            )
            target_losses.append(
                class_weight
                * F.binary_cross_entropy_with_logits(
                    target_logits, torch.ones_like(target_logits)
                )
            )
    zero = 0.0 * (source_features.sum() + target_features.sum())
    source_loss = torch.stack(source_losses).mean() if source_losses else zero
    target_loss = torch.stack(target_losses).mean() if target_losses else zero
    return source_loss + target_weight * target_loss


class ResidualEventAdapter(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(features))))
        return features + residual


class TemporalResidualEventAdapter(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.temporal = nn.Conv1d(
            bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1
        )
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.down(self.norm(features))).transpose(1, 2)
        hidden = F.gelu(self.temporal(hidden)).transpose(1, 2)
        return features + self.up(hidden)


class AdaptedTemporalDetector(nn.Module):
    def __init__(
        self,
        detector: nn.Module,
        auxiliary_dim: int,
        bottleneck_dim: int,
        adapter_mode: str = "pointwise",
    ) -> None:
        super().__init__()
        self.detector = detector
        self.auxiliary_dim = auxiliary_dim
        adapter_class = (
            TemporalResidualEventAdapter
            if adapter_mode == "temporal"
            else ResidualEventAdapter
        )
        self.adapter = adapter_class(auxiliary_dim, bottleneck_dim)

    def forward_with_temporal_domain(
        self, features: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        split = features.shape[-1] - self.auxiliary_dim
        auxiliary = features[:, :, split:]
        adapted = self.adapter(auxiliary)
        detector_input = torch.cat((features[:, :, :split], adapted), dim=2)
        output = self.detector(detector_input)
        domain_feature = F.layer_norm(adapted, (adapted.shape[-1],))
        drift = torch.square(adapted - auxiliary).mean(dim=(1, 2))
        return output, domain_feature, drift

    def forward_with_domain(
        self, features: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        output, temporal_domain, drift = self.forward_with_temporal_domain(features)
        return output, temporal_domain.mean(dim=1), drift

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        output, _, _ = self.forward_with_domain(features)
        return output


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
        "--out-dir", default="tmp/temporalmaxer_dense/hybrid_cv_tespec_eventmatch"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--domain-weight", type=float, default=0.1)
    parser.add_argument(
        "--domain-mode", choices=["global", "semantic", "sada"], default="global"
    )
    parser.add_argument("--pseudo-foreground", type=float, default=0.70)
    parser.add_argument("--pseudo-background", type=float, default=0.30)
    parser.add_argument("--max-domain-anchors", type=int, default=2048)
    parser.add_argument("--discriminator-lr", type=float, default=7e-3)
    parser.add_argument("--domain-warmup-epochs", type=int, default=0)
    parser.add_argument("--adapter-l2", type=float, default=0.01)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument(
        "--adapter-mode", choices=["pointwise", "temporal"], default="pointwise"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--neg-tiou", type=float, default=0.1)
    parser.add_argument("--pos-tiou", type=float, default=0.5)
    parser.add_argument("--high-pos-tiou", type=float, default=0.7)
    parser.add_argument("--boundary-min-tiou", type=float, default=0.3)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--boundary-blend", type=float, default=0.75)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--trident-weight", type=float, default=0.0)
    parser.add_argument("--qfl-beta", type=float, default=2.0)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def load_base_detector(
    checkpoint_path: Path,
    metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for name in (
        "hidden_dim",
        "pyramid_levels",
        "dropout",
        "trident_bins",
        "event_features_only",
    ):
        if name in checkpoint["args"]:
            setattr(args, name, checkpoint["args"][name])
    detector = make_model(metadata, args).to(device)
    detector.load_state_dict(checkpoint["state_dict"])
    detector.eval()
    detector.requires_grad_(False)
    return detector, checkpoint


def train_adapter(
    fold: int,
    source: pd.DataFrame,
    target: pd.DataFrame,
    master: pd.DataFrame,
    features: np.ndarray,
    logits: np.ndarray,
    metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
    fold_out: Path,
) -> AdaptedTemporalDetector:
    source_indices = map_to_master(master, source)
    target_indices = map_to_master(master, target)
    targets_all = build_targets(source, features.shape[1], args)
    source_scores = softmax_ed(np.asarray(logits[source_indices]))
    sampled = choose_training_indices(
        source,
        targets_all,
        source_scores,
        args.max_train_samples,
        args.seed + fold,
    )
    source_indices = source_indices[sampled]
    source_targets = subset_targets(targets_all, sampled)
    event_path, event_dim = event_feature_configuration(args, metadata)
    source_dataset = DenseFeatureDataset(
        cache_paths(resolve(args.cache_dir))["features"],
        source_indices,
        source_targets,
        event_feature_path=event_path,
    )
    target_dataset = DenseFeatureDataset(
        cache_paths(resolve(args.cache_dir))["features"],
        target_indices,
        event_feature_path=event_path,
    )
    generator = torch.Generator().manual_seed(args.seed + fold)
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 100 + fold),
        num_workers=0,
    )
    detector, _ = load_base_detector(
        resolve(args.checkpoint_root) / f"fold_{fold:02d}" / "best.pt",
        metadata,
        args,
        device,
    )
    model = AdaptedTemporalDetector(
        detector, event_dim, args.bottleneck_dim, args.adapter_mode
    ).to(device)
    if args.domain_mode == "sada":
        discriminator = ConditionalSemanticDiscriminator(event_dim).to(device)
    else:
        discriminator = nn.Sequential(
            nn.Linear(event_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 if args.domain_mode == "semantic" else 1),
        ).to(device)
    parameter_groups = [
        {"params": model.adapter.parameters(), "lr": args.lr},
        {
            "params": discriminator.parameters(),
            "lr": args.discriminator_lr if args.domain_mode == "sada" else args.lr,
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    last_path = fold_out / "last.pt"
    start_epoch = 1
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.adapter.load_state_dict(state["adapter"])
        discriminator.load_state_dict(state["discriminator"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.adapter.train()
        discriminator.train()
        target_iterator = iter(target_loader)
        for source_batch in source_loader:
            try:
                target_features, _ = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_features, _ = next(target_iterator)
            (
                source_features,
                quality_target,
                action_target,
                point_distance_target,
                start_target,
                end_target,
                delta_target,
                boundary_weight,
                _,
            ) = source_batch
            source_features = source_features.to(device)
            target_features = target_features.to(device)
            quality_target = quality_target.to(device)
            action_target = action_target.to(device)
            point_distance_target = point_distance_target.to(device)
            start_target = start_target.to(device)
            end_target = end_target.to(device)
            delta_target = delta_target.to(device)
            boundary_weight = boundary_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                if args.domain_mode in ("semantic", "sada"):
                    (
                        source_output,
                        source_domain,
                        source_drift,
                    ) = model.forward_with_temporal_domain(source_features)
                    _, target_domain, target_drift = model.forward_with_temporal_domain(
                        target_features
                    )
                else:
                    source_output, source_domain, source_drift = model.forward_with_domain(
                        source_features
                    )
                    _, target_domain, target_drift = model.forward_with_domain(
                        target_features
                    )
                task_loss, _ = dense_loss(
                    source_output,
                    quality_target,
                    action_target,
                    point_distance_target,
                    start_target,
                    end_target,
                    delta_target,
                    boundary_weight,
                    args,
                )
                if args.domain_mode in ("semantic", "sada"):
                    with torch.no_grad():
                        target_base_output = model.detector(target_features)
                        target_action_probability = torch.sigmoid(
                            target_base_output["action_logits"]
                        )
                    if args.domain_mode == "sada":
                        domain_loss = sada_conditional_domain_loss(
                            discriminator,
                            source_domain,
                            target_domain,
                            action_target,
                            target_action_probability,
                            args.pseudo_foreground,
                            args.max_domain_anchors,
                        )
                    else:
                        domain_loss = semantic_domain_loss(
                            discriminator,
                            source_domain,
                            target_domain,
                            action_target,
                            target_action_probability,
                            args.pseudo_foreground,
                            args.pseudo_background,
                            args.max_domain_anchors,
                        )
                else:
                    domain_features = reverse_gradient(
                        torch.cat((source_domain, target_domain), dim=0)
                    )
                    domain_labels = torch.cat(
                        (
                            torch.zeros(len(source_domain), device=device),
                            torch.ones(len(target_domain), device=device),
                        )
                    )
                    domain_logits = discriminator(domain_features).squeeze(1)
                    domain_loss = F.binary_cross_entropy_with_logits(
                        domain_logits, domain_labels
                    )
                drift = 0.5 * (source_drift.mean() + target_drift.mean())
                domain_weight = (
                    0.0
                    if epoch <= args.domain_warmup_epochs
                    else args.domain_weight
                )
                loss = (
                    task_loss.mean()
                    + domain_weight * domain_loss
                    + args.adapter_l2 * drift
                )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [*model.adapter.parameters(), *discriminator.parameters()], 5.0
            )
            optimizer.step()
        atomic_torch_save(
            {
                "epoch": epoch,
                "adapter": model.adapter.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            last_path,
        )
    atomic_torch_save(
        {
            "adapter": model.adapter.state_dict(),
            "args": vars(args),
        },
        fold_out / "adapter.pt",
    )
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    features, logits, metadata = load_cache(resolve(args.cache_dir))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {int(row["fold"]) for row in rows}

    for fold in args.folds:
        if fold in completed:
            continue
        target = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        held_out = set(target["rec_name"].astype(str))
        source = master[
            ~master["rec_name"].astype(str).isin(held_out)
        ].reset_index(drop=True)
        fold_out = out_dir / f"fold_{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)
        scored_path = fold_out / "scored.csv"
        if scored_path.exists():
            scored = pd.read_csv(scored_path)
        else:
            model = train_adapter(
                fold,
                source,
                target,
                master,
                features,
                logits,
                metadata,
                copy.deepcopy(args),
                device,
                fold_out,
            )
            scored = score_model(
                model,
                target,
                map_to_master(master, target),
                cache_paths(resolve(args.cache_dir))["features"],
                logits,
                args,
                device,
            )
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
            "qhead_brem_score",
            "blend050",
            f"tespec_eventmatch_fold_{fold:02d}",
            args,
            fold_out / "predictions",
        )
        row["fold"] = fold
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_path, index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    weights = metrics["val_ed_instances"].to_numpy(dtype=np.float64)
    summary = {"score_column": "qhead_brem_score", "boundary_mode": "blend050"}
    for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
        values = metrics[column].to_numpy(dtype=np.float64)
        summary[f"mean_{column}"] = float(values.mean())
        summary[f"weighted_{column}"] = float(np.average(values, weights=weights))
        summary[f"worst_{column}"] = float(values.min())
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
