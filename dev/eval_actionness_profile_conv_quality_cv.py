"""Cross-fit a small Conv1D QFL head on ordered actionness profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.eval_actionness_profile_quality_head_cv import PROFILE_COLUMNS  # noqa: E402
from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    FEATURE_COLUMNS,
    fit_linear_qfl,
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
from dev.diagnose_final_prediction_oracles import source_prediction_path  # noqa: E402


class ProfileConvQFL(nn.Module):
    def __init__(self, summary_dim: int) -> None:
        super().__init__()
        self.profile_tower = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * len(PROFILE_COLUMNS), 16),
            nn.ReLU(inplace=True),
        )
        self.summary_tower = nn.Sequential(
            nn.Linear(summary_dim, 16),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        profile: torch.Tensor,
        summary: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            (
                self.profile_tower(profile[:, None, :]),
                self.summary_tower(summary),
            ),
            dim=1,
        )
        return self.output(combined).squeeze(1)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        default=(
            "tmp/temporalmaxer_continuous/actionness_profile_qfl_cv_v1/"
            "candidate_profiles.csv"
        ),
    )
    parser.add_argument(
        "--continuous-root",
        default="tmp/temporalmaxer_continuous/cv_recipe_v1",
    )
    parser.add_argument(
        "--event-root",
        default="tmp/temporalmaxer_continuous/cv_eventstats_v1",
    )
    parser.add_argument(
        "--control-root",
        default="tmp/temporalmaxer_continuous/actionness_qfl_fusion_weights_cv_v1",
    )
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_continuous/actionness_profile_conv_qfl_cv_v1",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-per-recording", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2027, 4099])
    parser.add_argument("--score-blend", type=float, default=0.5)
    parser.add_argument(
        "--summary-blends",
        type=float,
        nargs="+",
        default=[0.25, 0.5],
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def feature_normalization(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = recording_weights(frame).astype(np.float64)
    profile = frame[list(PROFILE_COLUMNS)].to_numpy(np.float64)
    summary = frame[list(FEATURE_COLUMNS)].to_numpy(np.float64)
    profile_mean, profile_std = weighted_mean_std(profile, weights)
    summary_mean, summary_std = weighted_mean_std(summary, weights)
    return profile_mean, profile_std, summary_mean, summary_std


def balanced_indices(
    frame: pd.DataFrame,
    batch_per_recording: int,
    rng: np.random.Generator,
) -> np.ndarray:
    parts = []
    for _, group in frame.groupby("rec_name", sort=False):
        indices = group.index.to_numpy(np.int64)
        parts.append(
            rng.choice(
                indices,
                size=batch_per_recording,
                replace=len(indices) < batch_per_recording,
            )
        )
    return np.concatenate(parts)


def fit_profile_conv_qfl(
    frame: pd.DataFrame,
    device: torch.device,
    steps: int,
    batch_per_recording: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[ProfileConvQFL, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    local = frame.reset_index(drop=True)
    normalization = feature_normalization(local)
    profile_mean, profile_std, summary_mean, summary_std = normalization
    profile = (
        local[list(PROFILE_COLUMNS)].to_numpy(np.float32) - profile_mean
    ) / profile_std
    summary = (
        local[list(FEATURE_COLUMNS)].to_numpy(np.float32) - summary_mean
    ) / summary_std
    targets = local["target_tiou"].to_numpy(np.float32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ProfileConvQFL(len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        indices = balanced_indices(local, batch_per_recording, rng)
        profile_batch = torch.from_numpy(profile[indices]).to(device)
        summary_batch = torch.from_numpy(summary[indices]).to(device)
        target_batch = torch.from_numpy(targets[indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(profile_batch, summary_batch)
        probabilities = logits.sigmoid()
        qfl = F.binary_cross_entropy_with_logits(
            logits,
            target_batch,
            reduction="none",
        ) * (target_batch - probabilities).abs().square()
        qfl.mean().backward()
        optimizer.step()
    return model.eval(), normalization


@torch.no_grad()
def predict_profile_conv(
    frame: pd.DataFrame,
    models,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    for model, normalization in models:
        profile_mean, profile_std, summary_mean, summary_std = normalization
        profile = (
            frame[list(PROFILE_COLUMNS)].to_numpy(np.float32) - profile_mean
        ) / profile_std
        summary = (
            frame[list(FEATURE_COLUMNS)].to_numpy(np.float32) - summary_mean
        ) / summary_std
        predictions.append(
            model(
                torch.from_numpy(profile).to(device),
                torch.from_numpy(summary).to(device),
            )
            .sigmoid()
            .float()
            .cpu()
            .numpy()
        )
    return np.mean(predictions, axis=0)


def quality_frame(
    frame: pd.DataFrame,
    quality: np.ndarray,
    score_blend: float,
) -> pd.DataFrame:
    learned_rank = pd.Series(quality).rank(
        method="average",
        pct=True,
    ).to_numpy(np.float64)
    original_rank = frame["raw_global_rank"].to_numpy(np.float64)
    output = frame[["rec_name", "roi_id", "t_start", "t_end"]].copy()
    output["raw_score"] = (
        (1.0 - score_blend) * learned_rank + score_blend * original_rank
    )
    output["rank_score"] = output["raw_score"].rank(method="average", pct=True)
    output["model"] = "proposal"
    return output


def main() -> None:
    args = parse_args()
    profiles = pd.read_csv(resolve(args.profiles))
    continuous_root = resolve(args.continuous_root)
    event_root = resolve(args.event_root)
    control_root = resolve(args.control_root)
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []

    for fold in args.folds:
        train = profiles[profiles["fold"] != fold].copy()
        validation = profiles[profiles["fold"] == fold].copy()
        models = [
            fit_profile_conv_qfl(
                train,
                device,
                args.steps,
                args.batch_per_recording,
                args.learning_rate,
                args.weight_decay,
                seed,
            )
            for seed in args.seeds
        ]
        quality = predict_profile_conv(validation, models, device)
        conv_frame = quality_frame(
            validation,
            quality,
            args.score_blend,
        )
        summary_model = fit_linear_qfl(
            train,
            device,
            args.steps,
            0.03,
            feature_columns=FEATURE_COLUMNS,
        )
        summary_frame = score_quality_head(
            validation,
            summary_model,
            args.score_blend,
            feature_columns=FEATURE_COLUMNS,
        )
        proposal_frames = {"profile_conv": conv_frame}
        for blend in args.summary_blends:
            blended = conv_frame.copy()
            blended["raw_score"] = (
                (1.0 - blend) * conv_frame["raw_score"].to_numpy(np.float64)
                + blend * summary_frame["raw_score"].to_numpy(np.float64)
            )
            blended["rank_score"] = blended["raw_score"].rank(
                method="average",
                pct=True,
            )
            proposal_frames[
                f"profile_conv_summary_w{int(round(100 * blend)):03d}"
            ] = blended

        recordings = str(manifest.loc[fold, "val_record_names"]).split()
        control = json.loads(
            source_prediction_path(control_root, fold).read_text(encoding="utf-8")
        )
        rows.append(
            {
                "fold": fold,
                "variant": "control",
                "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                **evaluate(
                    control,
                    recordings,
                    resolve(args.ann_path),
                    out_dir / "predictions" / f"fold_{fold:02d}_control.json",
                ),
            }
        )
        continuous_frame = prediction_rows(
            best_prediction(continuous_root, fold),
            "continuous",
        )
        event_frame = prediction_rows(
            best_prediction(event_root, fold),
            "event",
        )
        for variant, proposal_frame in proposal_frames.items():
            prediction = build_prediction(
                [continuous_frame, event_frame, proposal_frame],
                {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
                sigma=0.5,
                per_model_topk=100,
                max_predictions=200,
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    **evaluate(
                        prediction,
                        recordings,
                        resolve(args.ann_path),
                        out_dir / "predictions" / f"fold_{fold:02d}_{variant}.json",
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / "fold_metrics_partial.csv", index=False)
        print(f"Completed fold {fold}", flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_rows = []
    for variant, group in metrics.groupby("variant", sort=False):
        weights = group["val_ed_instances"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "folds": len(group),
                "mean_mAP": float(group["mAP"].mean()),
                "weighted_mAP": float(np.average(group["mAP"], weights=weights)),
                "worst_mAP": float(group["mAP"].min()),
                "mean_AP@0.1": float(group["AP@0.1"].mean()),
                "mean_AP@0.3": float(group["AP@0.3"].mean()),
                "mean_AP@0.5": float(group["AP@0.5"].mean()),
                "mean_AP@0.7": float(group["AP@0.7"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
