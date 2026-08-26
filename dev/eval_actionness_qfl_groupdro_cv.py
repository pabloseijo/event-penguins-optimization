"""Evaluate recording-robust GroupDRO on the final linear QFL scorer."""

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

from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    FEATURE_COLUMNS,
    recording_weights,
    score_quality_head,
    weighted_mean_std,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    best_prediction,
    build_prediction,
    evaluate,
    prediction_rows,
)
from dev.train_temporalmaxer_continuous import group_dro_reduce  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def fit_groupdro_qfl(
    frame: pd.DataFrame,
    device: torch.device,
    steps: int,
    learning_rate: float,
    eta: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, float],
    pd.DataFrame,
]:
    """Fit the same linear QFL head with recording-level GroupDRO."""
    if eta <= 0:
        raise ValueError("GroupDRO eta must be positive")
    features = frame[list(FEATURE_COLUMNS)].to_numpy(np.float32)
    targets = frame["target_tiou"].to_numpy(np.float32)
    equal_recording_weights = recording_weights(frame).astype(np.float32)
    mean, std = weighted_mean_std(features, equal_recording_weights)
    inputs = torch.from_numpy((features - mean) / std).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    group_codes, recording_names = pd.factorize(
        frame["rec_name"].astype(str), sort=True
    )
    group_tensor = torch.from_numpy(group_codes.astype(np.int64)).to(device)
    group_weights = torch.full(
        (len(recording_names),),
        1.0 / len(recording_names),
        dtype=torch.float32,
        device=device,
    )
    model = torch.nn.Linear(features.shape[1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).squeeze(1)
        probabilities = logits.sigmoid()
        qfl = F.binary_cross_entropy_with_logits(
            logits, target_tensor, reduction="none"
        ) * (target_tensor - probabilities).abs().square()
        loss = group_dro_reduce(
            qfl,
            group_tensor,
            group_weights,
            eta,
        )
        loss.backward()
        optimizer.step()
    fitted = (
        mean,
        std,
        model.weight.detach().float().cpu().numpy().reshape(-1),
        float(model.bias.detach().float().cpu()),
    )
    diagnostics = pd.DataFrame(
        {
            "rec_name": recording_names.astype(str),
            "group_weight": group_weights.detach().cpu().numpy(),
        }
    )
    return fitted, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features",
        default=(
            "tmp/temporalmaxer_continuous/actionness_qfl_cv_v1/"
            "candidate_features.csv"
        ),
    )
    parser.add_argument(
        "--base-root",
        default="tmp/temporalmaxer_continuous/cv_recipe_v1",
    )
    parser.add_argument(
        "--pal-root",
        default=(
            "tmp/temporalmaxer_continuous/cv_pal_consistency_pilot_v1"
        ),
    )
    parser.add_argument(
        "--event-root",
        default="tmp/temporalmaxer_continuous/cv_eventstats_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default=(
            "tmp/temporalmaxer_continuous/"
            "actionness_qfl_groupdro_cv_v1"
        ),
    )
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(resolve(args.candidate_features))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    weight_parts = []
    for fold in range(5):
        train = candidates[candidates["fold"] != fold].copy()
        validation = candidates[candidates["fold"] == fold].copy()
        model, group_weights = fit_groupdro_qfl(
            train,
            device,
            args.steps,
            args.learning_rate,
            args.eta,
        )
        group_weights.insert(0, "fold", fold)
        weight_parts.append(group_weights)
        frames = [
            prediction_rows(
                best_prediction(resolve(args.base_root), fold), "base"
            ),
            prediction_rows(
                best_prediction(resolve(args.pal_root), fold), "pal_consistency"
            ),
            prediction_rows(
                best_prediction(resolve(args.event_root), fold), "event"
            ),
            score_quality_head(validation, model, args.score_blend),
        ]
        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        prediction = build_prediction(
            frames,
            {
                "base": 0.1,
                "pal_consistency": 0.1,
                "event": 0.4,
                "proposal": 0.4,
            },
            sigma=0.5,
            per_model_topk=100,
            max_predictions=200,
        )
        rows.append(
            {
                "fold": fold,
                "eta": args.eta,
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **evaluate(
                    prediction,
                    recordings,
                    resolve(args.ann_path),
                    out_dir / "predictions" / f"fold{fold:02d}.json",
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    pd.concat(weight_parts, ignore_index=True).to_csv(
        out_dir / "group_weights.csv", index=False
    )
    instance_weights = metrics["val_ed_instances"].to_numpy(np.float64)
    summary = {
        "eta": args.eta,
        "mean_mAP": float(metrics["mAP"].mean()),
        "weighted_mAP": float(
            np.average(metrics["mAP"], weights=instance_weights)
        ),
        "worst_mAP": float(metrics["mAP"].min()),
        "mean_AP@0.1": float(metrics["AP@0.1"].mean()),
        "mean_AP@0.3": float(metrics["AP@0.3"].mean()),
        "mean_AP@0.5": float(metrics["AP@0.5"].mean()),
        "mean_AP@0.7": float(metrics["AP@0.7"].mean()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
