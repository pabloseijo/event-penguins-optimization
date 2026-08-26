"""Episodic, recording-level test-time adaptation for the ATSN classifier.

No target annotations are used for adaptation.  Each recording starts from the
same source checkpoint, adapts normalization on a label-free subset of its
proposals, and is then scored independently.  The script supports target
BatchNorm statistics (AdaBN) and entropy-minimizing affine updates (TENT/EATA-
style filtering and source regularization).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.augmented_tsn import AugmentedTsn

from dev.train_atsn_lpft import (
    build_loader,
    evaluate_scored,
    make_model,
    resolve_path,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Episodic ATSN test-time adaptation by recording.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["adabn", "tent", "affine"], default="adabn")
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument(
        "--reuse-base-score-col",
        default=None,
        help="Reuse a cached source-model probability column instead of rescoring the source model.",
    )
    parser.add_argument("--max-adapt-proposals", type=int, default=2048)
    parser.add_argument(
        "--adapt-selection",
        choices=["uniform", "top_score", "roi_balanced"],
        default="roi_balanced",
    )
    parser.add_argument("--adapt-epochs", type=int, default=1)
    parser.add_argument("--adabn-momentum", type=float, default=0.01)
    parser.add_argument("--adabn-reset-stats", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-margin", type=float, default=0.65)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    parser.add_argument("--l2sp-weight", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet-progress", action="store_true")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    # build_loader expects these training-only options even though TTA uses the
    # deterministic proposal representation.
    parser.set_defaults(
        temporal_scale_jitter=0.0,
        temporal_shift_jitter=0.0,
        sample_duration_jitter=0.0,
        event_drop_prob=0.0,
    )

    parser.add_argument("--min-ed-score", type=float, nargs="+", default=[0.02])
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def batch_norm_modules(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def configure_adabn(model: AugmentedTsn, reset_stats: bool, momentum: float) -> None:
    freeze_model(model)
    model.train()
    model.dropout.eval()
    model.fc_cls.eval()
    for module in batch_norm_modules(model):
        if reset_stats:
            module.reset_running_stats()
            module.momentum = None
        else:
            module.momentum = momentum
        module.train()


def configure_tent(
    model: AugmentedTsn,
    use_target_batch_stats: bool,
) -> list[tuple[str, nn.Parameter]]:
    freeze_model(model)
    model.train()
    model.dropout.eval()
    model.fc_cls.eval()
    trainable = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if use_target_batch_stats:
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        else:
            module.eval()
        if module.weight is not None:
            module.weight.requires_grad = True
            trainable.append((f"{module_name}.weight", module.weight))
        if module.bias is not None:
            module.bias.requires_grad = True
            trainable.append((f"{module_name}.bias", module.bias))
    if not trainable:
        raise RuntimeError("ATSN does not contain affine BatchNorm parameters.")
    return trainable


def set_tent_inference_mode(model: AugmentedTsn) -> None:
    model.eval()
    model.dropout.eval()
    for module in batch_norm_modules(model):
        module.train()
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None


def select_adaptation_proposals(
    proposals: pd.DataFrame,
    maximum: int,
    mode: str,
    seed: int,
) -> pd.DataFrame:
    frame = proposals.reset_index(drop=True)
    if maximum <= 0 or len(frame) <= maximum:
        return frame
    if mode == "uniform":
        return frame.sample(n=maximum, random_state=seed).reset_index(drop=True)
    if mode == "top_score":
        score_column = "score" if "score" in frame else "cnn_score"
        return frame.nlargest(maximum, score_column).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    roi_groups = list(frame.groupby("roi_id", sort=True))
    per_roi = max(1, int(math.ceil(maximum / max(len(roi_groups), 1))))
    selected_indices: list[int] = []
    for _, group in roi_groups:
        take = min(per_roi, len(group))
        chosen = rng.choice(group.index.to_numpy(dtype=np.int64), size=take, replace=False)
        selected_indices.extend(chosen.tolist())
    selected_indices = selected_indices[:maximum]
    if len(selected_indices) < maximum:
        remaining = frame.index.difference(pd.Index(selected_indices)).to_numpy(dtype=np.int64)
        take = min(maximum - len(selected_indices), len(remaining))
        selected_indices.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return frame.loc[selected_indices].reset_index(drop=True)


@torch.no_grad()
def adapt_batch_norm_statistics(
    model: AugmentedTsn,
    proposals: pd.DataFrame,
    args: argparse.Namespace,
    data_path: Path,
    device: torch.device,
) -> None:
    configure_adabn(model, args.adabn_reset_stats, args.adabn_momentum)
    loader = build_loader(proposals, args, data_path, require_label=False)
    for images, _ in tqdm(loader, desc="adabn", leave=False, disable=args.quiet_progress):
        model(images.to(device, non_blocking=True))
    model.eval()


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)


def adapt_tent(
    model: AugmentedTsn,
    teacher: AugmentedTsn,
    proposals: pd.DataFrame,
    args: argparse.Namespace,
    data_path: Path,
    device: torch.device,
    use_target_batch_stats: bool,
) -> dict[str, float]:
    trainable = configure_tent(model, use_target_batch_stats)
    teacher.eval()
    initial = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer = torch.optim.Adam([parameter for _, parameter in trainable], lr=args.lr)
    loader = build_loader(proposals, args, data_path, require_label=False)
    totals = {"loss": 0.0, "entropy": 0.0, "selected": 0.0, "items": 0.0}

    for _ in range(args.adapt_epochs):
        for images, _ in tqdm(loader, desc="tent", leave=False, disable=args.quiet_progress):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            entropy = prediction_entropy(logits)
            selected = entropy < args.entropy_margin if args.entropy_margin > 0 else torch.ones_like(
                entropy, dtype=torch.bool
            )
            if not selected.any():
                continue
            probabilities = torch.softmax(logits[selected], dim=1)
            marginal = probabilities.mean(dim=0)
            marginal_entropy = -(marginal * torch.log(marginal.clamp_min(1e-8))).sum()
            entropy_loss = entropy[selected].mean() - args.diversity_weight * marginal_entropy
            with torch.no_grad():
                teacher_logits = teacher(images[selected])
            distill = F.kl_div(
                F.log_softmax(logits[selected], dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction="batchmean",
            )
            regularizer = torch.stack(
                [(parameter - initial[name]).pow(2).sum() for name, parameter in trainable]
            ).sum()
            loss = entropy_loss + args.distill_weight * distill + args.l2sp_weight * regularizer
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], args.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach().item()) * int(selected.sum().item())
            totals["entropy"] += float(entropy[selected].sum().detach().item())
            totals["selected"] += float(selected.sum().item())
            totals["items"] += float(len(entropy))
    if use_target_batch_stats:
        set_tent_inference_mode(model)
    else:
        model.eval()
    selected_count = max(totals["selected"], 1.0)
    return {
        "adapt_loss": totals["loss"] / selected_count,
        "adapt_entropy": totals["entropy"] / selected_count,
        "adapt_selected_fraction": totals["selected"] / max(totals["items"], 1.0),
    }


@torch.no_grad()
def score_recording(
    model: AugmentedTsn,
    proposals: pd.DataFrame,
    args: argparse.Namespace,
    data_path: Path,
    device: torch.device,
    tent_mode: bool,
) -> np.ndarray:
    if tent_mode:
        set_tent_inference_mode(model)
    else:
        model.eval()
    loader = build_loader(proposals, args, data_path, require_label=False)
    scores = np.zeros(len(proposals), dtype=np.float64)
    for images, indices in tqdm(loader, desc="score", leave=False, disable=args.quiet_progress):
        logits = model(images.to(device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        scores[np.asarray(indices, dtype=np.int64)] = probabilities
    return scores


def score_source_model(
    proposals: pd.DataFrame,
    args: argparse.Namespace,
    data_path: Path,
    device: torch.device,
) -> np.ndarray:
    model = make_model(args, device)
    return score_recording(model, proposals, args, data_path, device, tent_mode=False)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_adapt_proposals = min(args.max_adapt_proposals, 16)
        args.num_workers = 0
        args.batch_size = min(args.batch_size, 4)
        args.quiet_progress = True

    set_seed(args.seed)
    data_path = resolve_path(args.data_path)
    ann_path = resolve_path(args.ann_path)
    out_dir = resolve_path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    proposals = pd.read_csv(resolve_path(args.proposals)).reset_index(drop=True)
    if args.smoke:
        proposals = proposals.head(32).reset_index(drop=True)
    print(f"[INFO] Device={device} method={args.method} proposals={len(proposals)}")

    if args.reuse_base_score_col is not None:
        if args.reuse_base_score_col not in proposals:
            raise ValueError(f"Missing cached base score column: {args.reuse_base_score_col}")
        base_scores = proposals[args.reuse_base_score_col].to_numpy(dtype=np.float64)
        if not np.isfinite(base_scores).all() or ((base_scores < 0.0) | (base_scores > 1.0)).any():
            raise ValueError("Cached base scores must be finite probabilities in [0, 1]")
        print(f"[INFO] Reusing base scores from column={args.reuse_base_score_col}")
    else:
        base_scores = score_source_model(proposals, args, data_path, device)
    adapted_scores = np.zeros(len(proposals), dtype=np.float64)
    adaptation_rows = []
    for recording_index, (recording, group) in enumerate(proposals.groupby("rec_name", sort=True)):
        recording_frame = group.reset_index().rename(columns={"index": "global_index"})
        adapt_frame = select_adaptation_proposals(
            recording_frame,
            args.max_adapt_proposals,
            args.adapt_selection,
            args.seed + recording_index,
        )
        model = make_model(args, device)
        diagnostics: dict[str, float] = {}
        if args.method == "adabn":
            adapt_batch_norm_statistics(model, adapt_frame, args, data_path, device)
        else:
            teacher = make_model(args, device)
            freeze_model(teacher)
            diagnostics = adapt_tent(
                model,
                teacher,
                adapt_frame,
                args,
                data_path,
                device,
                use_target_batch_stats=args.method == "tent",
            )
        scores = score_recording(
            model,
            recording_frame,
            args,
            data_path,
            device,
            tent_mode=args.method == "tent",
        )
        global_indices = recording_frame["global_index"].to_numpy(dtype=np.int64)
        adapted_scores[global_indices] = scores
        adaptation_rows.append(
            {
                "rec_name": recording,
                "proposals": int(len(recording_frame)),
                "adapt_proposals": int(len(adapt_frame)),
                **diagnostics,
            }
        )
        print(
            f"[ADAPT] rec={recording} n={len(recording_frame)} adapt={len(adapt_frame)}",
            flush=True,
        )

    scored = proposals.copy()
    scored["cnn_score_base"] = base_scores
    scored["cnn_score"] = adapted_scores
    scored.to_csv(out_dir / "scored_proposals.csv", index=False)
    pd.DataFrame(adaptation_rows).to_csv(out_dir / "adaptation.csv", index=False)

    base_frame = proposals.copy()
    base_frame["cnn_score"] = base_scores
    base_metrics = evaluate_scored(base_frame, ann_path, args, pred_dir, "base")
    adapted_metrics = evaluate_scored(scored, ann_path, args, pred_dir, args.method)
    summary = {
        "method": args.method,
        "base": base_metrics,
        "adapted": adapted_metrics,
        "delta_mAP": float(adapted_metrics["mAP"] - base_metrics["mAP"]),
        "recordings": int(proposals["rec_name"].nunique()),
        "proposals": int(len(proposals)),
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[RESULTADO] base={base_metrics['mAP']:.6f} adapted={adapted_metrics['mAP']:.6f} "
        f"delta={summary['delta_mAP']:+.6f}"
    )


if __name__ == "__main__":
    main()
