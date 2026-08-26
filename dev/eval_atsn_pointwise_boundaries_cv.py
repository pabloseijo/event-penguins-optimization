"""Evaluate training-free ATSN pointwise boundaries in source external CV.

The original ATSN classifier is linear over start/main/end feature roles. A
single temporal feature is repeated in all three roles to obtain an
instantaneous ED probability without fitting another model. Threshold
crossings are interpolated densely and combined with the fixed TemporalMaxer
boundary; all configuration selection is recording-disjoint and source-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dev.train_temporalmaxer_dense import (
    evaluate_variant,
    load_cache,
    map_to_master,
)
from src.augmented_tsn import AugmentedTsn


ROOT = Path(__file__).resolve().parents[1]


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
        "--reference-root", default="tmp/temporalmaxer_dense/hybrid_cv_eval"
    )
    parser.add_argument(
        "--groupdro-root", default="tmp/quality_head/family_groupdro_cv"
    )
    parser.add_argument(
        "--manifest", default="tmp/cv/recording_folds_r5/manifest.csv"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/atsn_pointwise_boundary_cv"
    )
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--dense-grid-size", type=int, default=201)
    parser.add_argument("--minimum-range", type=float, default=0.05)
    parser.add_argument("--min-duration", type=float, default=2.0e6)

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


def load_atsn(args: argparse.Namespace, device: torch.device) -> AugmentedTsn:
    model = AugmentedTsn(2, args.num_tsn_samples, args.augment_factor)
    try:
        state = torch.load(resolve(args.model_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(resolve(args.model_path), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def pointwise_probabilities(
    feature_path: Path,
    master_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    features = np.load(feature_path, mmap_mode="r")
    model = load_atsn(args, device)
    output = np.empty((len(master_indices), features.shape[1]), dtype=np.float32)
    for start in range(0, len(master_indices), args.batch_size):
        end = min(start + args.batch_size, len(master_indices))
        batch = torch.from_numpy(
            np.asarray(features[master_indices[start:end]], dtype=np.float32).copy()
        ).to(device)
        repeated_roles = torch.cat((batch, batch, batch), dim=2)
        logits = model.fc_cls(repeated_roles)
        output[start:end] = torch.softmax(logits, dim=2)[..., 1].cpu().numpy()
    return output


def _smooth_probabilities(probabilities: np.ndarray) -> np.ndarray:
    padded = np.pad(probabilities, ((0, 0), (1, 1)), mode="edge")
    return (
        0.25 * padded[:, :-2]
        + 0.50 * padded[:, 1:-1]
        + 0.25 * padded[:, 2:]
    )


def crossing_intervals(
    probabilities: np.ndarray,
    threshold: float,
    padding: float,
    minimum_range: float,
    dense_grid_size: int,
    smooth: bool,
    augment_factor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return relative threshold-crossing intervals and a validity mask."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 3:
        raise ValueError("Expected point probabilities with shape [N,T], T >= 3")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if padding < 0.0:
        raise ValueError("padding must be non-negative")
    if dense_grid_size < probabilities.shape[1]:
        raise ValueError("dense_grid_size cannot be shorter than the input sequence")
    values = _smooth_probabilities(probabilities) if smooth else probabilities
    relative = np.linspace(
        -1.0 / augment_factor,
        1.0 + 1.0 / augment_factor,
        values.shape[1],
        dtype=np.float64,
    )
    dense_relative = np.linspace(relative[0], relative[-1], dense_grid_size)
    starts = np.zeros(len(values), dtype=np.float64)
    ends = np.ones(len(values), dtype=np.float64)
    valid = np.ptp(values, axis=1) >= minimum_range
    for index in np.flatnonzero(valid):
        dense = np.interp(dense_relative, relative, values[index])
        low = float(dense.min())
        high = float(dense.max())
        normalized = (dense - low) / max(high - low, 1e-12)
        active = normalized >= threshold
        peak = int(np.argmax(normalized))
        if not active[peak]:
            valid[index] = False
            continue
        left = peak
        right = peak + 1
        while left > 0 and active[left - 1]:
            left -= 1
        while right < len(active) and active[right]:
            right += 1
        starts[index] = dense_relative[left] - padding
        ends[index] = dense_relative[min(right, len(active) - 1)] + padding
        starts[index] = np.clip(starts[index], relative[0], relative[-1])
        ends[index] = np.clip(ends[index], relative[0], relative[-1])
        if ends[index] <= starts[index]:
            valid[index] = False
    return starts, ends, valid


def stabilized_boundaries(
    raw_start: np.ndarray,
    raw_end: np.ndarray,
    relative_start: np.ndarray,
    relative_end: np.ndarray,
    valid: np.ndarray,
    minimum_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    duration = np.maximum(raw_end - raw_start, 1.0)
    start = raw_start.copy()
    end = raw_end.copy()
    start[valid] = raw_start[valid] + relative_start[valid] * duration[valid]
    end[valid] = raw_start[valid] + relative_end[valid] * duration[valid]
    start = np.maximum(start, 0.0)
    center = 0.5 * (start + end)
    short = end - start < minimum_duration
    start[short] = np.maximum(0.0, center[short] - 0.5 * minimum_duration)
    end[short] = start[short] + minimum_duration
    return start, end


def candidate_configurations() -> list[dict[str, float | bool]]:
    return [
        {
            "threshold": threshold,
            "padding": padding,
            "point_weight": point_weight,
            "smooth": True,
        }
        for threshold in (0.30, 0.50)
        for padding in (0.00, 0.05)
        for point_weight in (0.50, 0.75, 1.00)
    ]


def configuration_name(configuration: dict[str, float | bool]) -> str:
    return (
        f"point_thr{float(configuration['threshold']):.2f}"
        f"_pad{float(configuration['padding']):.2f}"
        f"_w{float(configuration['point_weight']):.2f}"
    ).replace(".", "p")


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, _, _ = load_cache(resolve(args.cache_dir))
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    partial_path = out_dir / "metrics_partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {
        (int(row["fold"]), str(row["boundary_mode"])) for row in rows
    }

    for fold in range(5):
        proposals = pd.read_csv(
            resolve(args.fold_dir) / f"fold_{fold:02d}" / "val_proposals.csv"
        ).reset_index(drop=True)
        master_indices = map_to_master(master, proposals)
        probability_path = out_dir / f"point_probabilities_fold_{fold:02d}.npy"
        if probability_path.exists():
            probabilities = np.load(probability_path)
        else:
            probabilities = pointwise_probabilities(
                resolve(args.cache_dir) / "frame_features.npy",
                master_indices,
                args,
                device,
            )
            np.save(probability_path, probabilities)

        reference_all = pd.read_csv(
            resolve(args.reference_root) / f"scored_fold_{fold:02d}.csv"
        )
        reference = reference_all.iloc[
            map_to_master(reference_all, proposals)
        ].reset_index(drop=True)
        groupdro_all = pd.read_csv(
            resolve(args.groupdro_root)
            / f"fold_{fold:02d}"
            / "cache"
            / "val_scores_qhead_qfl_only.csv"
        )
        groupdro = groupdro_all.iloc[
            map_to_master(groupdro_all, proposals)
        ].reset_index(drop=True)
        scored = proposals.copy()
        scored["quality_score"] = groupdro["quality_score"].to_numpy(dtype=np.float64)
        raw_start = scored["t_start"].to_numpy(dtype=np.float64)
        raw_end = scored["t_end"].to_numpy(dtype=np.float64)
        reference_start = 0.5 * (
            raw_start + reference["delta_t_start"].to_numpy(dtype=np.float64)
        )
        reference_end = 0.5 * (
            raw_end + reference["delta_t_end"].to_numpy(dtype=np.float64)
        )
        scored["reference_t_start"] = reference_start
        scored["reference_t_end"] = reference_end

        configurations = candidate_configurations()
        for configuration in configurations:
            name = configuration_name(configuration)
            relative_start, relative_end, valid = crossing_intervals(
                probabilities,
                threshold=float(configuration["threshold"]),
                padding=float(configuration["padding"]),
                minimum_range=args.minimum_range,
                dense_grid_size=args.dense_grid_size,
                smooth=bool(configuration["smooth"]),
                augment_factor=args.augment_factor,
            )
            point_start, point_end = stabilized_boundaries(
                raw_start,
                raw_end,
                relative_start,
                relative_end,
                valid,
                args.min_duration,
            )
            weight = float(configuration["point_weight"])
            scored[f"{name}_t_start"] = (
                (1.0 - weight) * reference_start + weight * point_start
            )
            scored[f"{name}_t_end"] = (
                (1.0 - weight) * reference_end + weight * point_end
            )
            if (fold, name) in completed:
                continue
            row = evaluate_variant(
                scored,
                "quality_score",
                name,
                f"atsn_pointwise_fold_{fold:02d}",
                args,
                out_dir / "predictions",
            )
            row.update(
                {
                    "fold": fold,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    "valid_fraction": float(valid.mean()),
                    **configuration,
                }
            )
            rows.append(row)
            completed.add((fold, name))
            pd.DataFrame(rows).to_csv(partial_path, index=False)

        if (fold, "reference") not in completed:
            baseline = evaluate_variant(
                scored,
                "quality_score",
                "reference",
                f"atsn_pointwise_fold_{fold:02d}",
                args,
                out_dir / "predictions",
            )
            baseline.update(
                {
                    "fold": fold,
                    "val_ed_instances": int(manifest.loc[fold, "val_ed_instances"]),
                    "valid_fraction": 0.0,
                    "threshold": np.nan,
                    "padding": np.nan,
                    "point_weight": 0.0,
                    "smooth": True,
                }
            )
            rows.append(baseline)
            completed.add((fold, "reference"))
            pd.DataFrame(rows).to_csv(partial_path, index=False)
        scored.to_csv(out_dir / f"scored_fold_{fold:02d}.csv", index=False)
        print(f"[FOLD {fold:02d}] completo")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary_rows = []
    for mode, group in metrics.groupby("boundary_mode"):
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        result: dict[str, float | str] = {"boundary_mode": mode}
        for column in ("mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"):
            values = group[column].to_numpy(dtype=np.float64)
            result[f"mean_{column}"] = float(values.mean())
            result[f"weighted_{column}"] = float(np.average(values, weights=weights))
            result[f"worst_{column}"] = float(values.min())
        result["mean_valid_fraction"] = float(group["valid_fraction"].mean())
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
