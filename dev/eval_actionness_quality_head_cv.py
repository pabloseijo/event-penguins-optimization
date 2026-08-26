"""Cross-fit a linear QFL proposal-quality head on OOF actionness features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.diagnose_atsn_linear_separability import load_annotations  # noqa: E402
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import (  # noqa: E402
    extract_actionness,
    frame_prediction,
)
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402


FEATURE_COLUMNS = (
    "raw_score",
    "raw_global_rank",
    "raw_recording_rank",
    "raw_roi_rank",
    "log_duration",
    "inside_mean",
    "inside_std",
    "inside_q10",
    "inside_q25",
    "inside_median",
    "inside_q75",
    "inside_q90",
    "left_context_mean",
    "right_context_mean",
    "completeness",
    "start_contrast",
    "end_contrast",
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def safe_mean(values: np.ndarray, fallback: float) -> float:
    return float(values.mean()) if len(values) else fallback


def candidate_features(
    proposal: dict,
    actionness: dict[tuple[str, int], np.ndarray],
    stride_s: float,
    annotations: dict[tuple[str, int], list[tuple[float, float]]],
    fold: int,
    context_ratio: float = 0.5,
) -> pd.DataFrame:
    frame = prediction_rows(proposal, "proposal").drop(columns=["rank_score", "model"])
    frame["fold"] = fold
    frame["raw_global_rank"] = frame["raw_score"].rank(method="average", pct=True)
    frame["raw_recording_rank"] = frame.groupby("rec_name")["raw_score"].rank(
        method="average", pct=True
    )
    frame["raw_roi_rank"] = frame.groupby(["rec_name", "roi_id"])["raw_score"].rank(
        method="average", pct=True
    )
    rows = []
    for row in frame.itertuples(index=False):
        sequence = actionness[(str(row.rec_name), int(row.roi_id))]
        start = max(0, int(np.floor(float(row.t_start) / stride_s)))
        end = min(len(sequence), int(np.ceil(float(row.t_end) / stride_s)))
        if end <= start:
            end = min(len(sequence), start + 1)
        inside = sequence[start:end]
        inside_mean = float(inside.mean())
        context_length = max(1, int(round(len(inside) * context_ratio)))
        left = sequence[max(0, start - context_length) : start]
        right = sequence[end : min(len(sequence), end + context_length)]
        left_mean = safe_mean(left, inside_mean)
        right_mean = safe_mean(right, inside_mean)
        edge_length = max(1, min(context_length, len(inside)))
        target = 0.0
        for gt_start, gt_end in annotations.get(
            (str(row.rec_name), int(row.roi_id)), []
        ):
            intersection = max(
                0.0,
                min(float(row.t_end), gt_end) - max(float(row.t_start), gt_start),
            )
            union = float(row.t_end) - float(row.t_start) + gt_end - gt_start - intersection
            target = max(target, intersection / union if union > 0 else 0.0)
        rows.append(
            {
                **row._asdict(),
                "log_duration": float(
                    np.log1p(max(float(row.t_end) - float(row.t_start), 0.0))
                ),
                "inside_mean": inside_mean,
                "inside_std": float(inside.std()),
                "inside_q10": float(np.quantile(inside, 0.10)),
                "inside_q25": float(np.quantile(inside, 0.25)),
                "inside_median": float(np.median(inside)),
                "inside_q75": float(np.quantile(inside, 0.75)),
                "inside_q90": float(np.quantile(inside, 0.90)),
                "left_context_mean": left_mean,
                "right_context_mean": right_mean,
                "completeness": inside_mean - 0.5 * (left_mean + right_mean),
                "start_contrast": float(inside[:edge_length].mean()) - left_mean,
                "end_contrast": float(inside[-edge_length:].mean()) - right_mean,
                "target_tiou": target,
            }
        )
    return pd.DataFrame(rows)


def recording_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("rec_name")["rec_name"].transform("size").to_numpy(np.float64)
    return 1.0 / counts


def weighted_mean_std(
    features: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / weights.sum()
    mean = np.sum(features * normalized[:, None], axis=0)
    variance = np.sum(np.square(features - mean) * normalized[:, None], axis=0)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)


def fit_linear_qfl(
    frame: pd.DataFrame,
    device: torch.device,
    steps: int,
    learning_rate: float,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    features = frame[list(feature_columns)].to_numpy(np.float32)
    targets = frame["target_tiou"].to_numpy(np.float32)
    weights = recording_weights(frame).astype(np.float32)
    mean, std = weighted_mean_std(features, weights)
    inputs = torch.from_numpy((features - mean) / std).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    weight_tensor = torch.from_numpy(weights).to(device)
    model = torch.nn.Linear(features.shape[1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).squeeze(1)
        probabilities = logits.sigmoid()
        qfl = F.binary_cross_entropy_with_logits(
            logits, target_tensor, reduction="none"
        ) * (target_tensor - probabilities).abs().square()
        loss = (qfl * weight_tensor).sum() / weight_tensor.sum()
        loss.backward()
        optimizer.step()
    return (
        mean,
        std,
        model.weight.detach().float().cpu().numpy().reshape(-1),
        float(model.bias.detach().float().cpu()),
    )


def score_quality_head(
    frame: pd.DataFrame,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    blend: float,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    mean, std, weight, bias = model
    features = frame[list(feature_columns)].to_numpy(np.float32)
    learned = ((features - mean) / std) @ weight + bias
    learned_rank = pd.Series(learned).rank(method="average", pct=True).to_numpy(np.float64)
    original_rank = frame["raw_global_rank"].to_numpy(np.float64)
    output = frame[["rec_name", "roi_id", "t_start", "t_end"]].copy()
    output["raw_score"] = (1.0 - blend) * learned_rank + blend * original_rank
    output["rank_score"] = output["raw_score"].rank(method="average", pct=True)
    output["model"] = "proposal"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features_v1")
    parser.add_argument("--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1")
    parser.add_argument("--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1")
    parser.add_argument(
        "--proposal-root",
        default="tmp/temporalmaxer_dense/multi_expert_boundary_voting_pilot_v2",
    )
    parser.add_argument("--proposal-variant", default="multi_median_blend050")
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/actionness_qfl_cv_v1"
    )
    parser.add_argument("--blends", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-action-duration", type=float, default=2.0)
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    args = parser.parse_args()

    feature_dir = resolve(args.feature_dir)
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    proposal_root = resolve(args.proposal_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    annotations = load_annotations(
        resolve(args.ann_path), min_duration_s=args.min_action_duration
    )
    input_dim = int(metadata["feature_dim"])
    stride_s = float(metadata["grid_stride_s"])
    device = torch.device(args.device)
    cache_path = out_dir / "candidate_features.csv"
    if cache_path.exists():
        candidates = pd.read_csv(cache_path)
    else:
        feature_parts = []
        for fold in range(5):
            recordings = str(manifest.loc[fold, "val_record_names"]).split()
            selected = sequences[sequences["rec_name"].isin(recordings)].copy()
            models = load_models(
                continuous_root,
                input_dim,
                device,
                [continuous_root / f"fold_{fold:02d}" / "best.pt"],
            )
            actionness = extract_actionness(
                models,
                make_loader(feature_dir, selected, args.batch_size, args.num_workers, device),
                device,
            )
            proposal = json.loads(
                (
                    proposal_root
                    / f"fold_{fold:02d}"
                    / "predictions"
                    / f"{args.proposal_variant}.json"
                ).read_text(encoding="utf-8")
            )
            feature_parts.append(
                candidate_features(
                    proposal, actionness, stride_s, annotations, fold
                )
            )
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
        candidates = pd.concat(feature_parts, ignore_index=True)
        candidates.to_csv(cache_path, index=False)

    rows = []
    for fold in range(5):
        train = candidates[candidates["fold"] != fold].copy()
        validation = candidates[candidates["fold"] == fold].copy()
        model = fit_linear_qfl(train, device, args.steps, args.learning_rate)
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        continuous_frame = prediction_rows(
            best_prediction(continuous_root, fold), "continuous"
        )
        event_frame = prediction_rows(best_prediction(event_root, fold), "event")
        for blend in args.blends:
            proposal_frame = score_quality_head(validation, model, blend)
            suffix = f"blend{int(round(blend * 100)):03d}"
            proposal_prediction = frame_prediction(proposal_frame, 0.5, 200)
            rows.append(
                {
                    "fold": fold,
                    "variant": f"proposal_qfl_{suffix}",
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        proposal_prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_proposal_{suffix}.json",
                        args.tiou,
                        args.min_action_duration,
                    ),
                }
            )
            fusion = build_prediction(
                [continuous_frame, event_frame, proposal_frame],
                {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
                min_action_duration=args.min_action_duration,
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": f"fusion_qfl_{suffix}",
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        fusion,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_fusion_{suffix}.json",
                        args.tiou,
                        args.min_action_duration,
                    ),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, values in metrics.groupby("variant", sort=False):
        instance_weights = values["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "mean_mAP": float(values["mAP"].mean()),
                "weighted_mAP": float(np.average(values["mAP"], weights=instance_weights)),
                "worst_mAP": float(values["mAP"].min()),
                "mean_AP@0.1": float(values["AP@0.1"].mean()),
                "mean_AP@0.3": float(values["AP@0.3"].mean()),
                "mean_AP@0.5": float(values["AP@0.5"].mean()),
                "mean_AP@0.7": float(values["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
