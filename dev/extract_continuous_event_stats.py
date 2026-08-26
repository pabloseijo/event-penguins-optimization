"""Extract compact event statistics aligned with the continuous ATSN grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = [
    "log_event_count",
    "polarity_balance",
    "half_window_contrast",
    "mean_x",
    "mean_y",
    "std_x",
    "std_y",
    "xy_correlation",
    "spectral_energy_ratio",
    "dominant_frequency",
]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--spectral-bins", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def binned_sum(indices: np.ndarray, values: np.ndarray, length: int) -> np.ndarray:
    return np.bincount(indices, weights=values, minlength=length)[:length]


def two_bin_window(values: np.ndarray) -> np.ndarray:
    previous = np.concatenate((np.zeros(1, dtype=values.dtype), values[:-1]))
    return previous + values


def sequence_features(
    events: np.ndarray,
    length: int,
    stride_s: float,
    width: int,
    height: int,
    spectral_bins: int,
) -> np.ndarray:
    if len(events) == 0:
        return np.zeros((length, len(FEATURE_NAMES)), dtype=np.float32)
    bin_width_us = stride_s * 1e6
    indices = np.floor(events[:, 2].astype(np.float64) / bin_width_us).astype(np.int64)
    valid = (indices >= 0) & (indices < length)
    indices = indices[valid]
    x = events[valid, 0].astype(np.float64)
    y = events[valid, 1].astype(np.float64)
    polarity = np.where(events[valid, 3] > 0, 1.0, -1.0)

    count_half = np.bincount(indices, minlength=length)[:length].astype(np.float64)
    polarity_half = binned_sum(indices, polarity, length)
    sum_x_half = binned_sum(indices, x, length)
    sum_y_half = binned_sum(indices, y, length)
    sum_x2_half = binned_sum(indices, x * x, length)
    sum_y2_half = binned_sum(indices, y * y, length)
    sum_xy_half = binned_sum(indices, x * y, length)

    count = two_bin_window(count_half)
    polarity_sum = two_bin_window(polarity_half)
    sum_x = two_bin_window(sum_x_half)
    sum_y = two_bin_window(sum_y_half)
    sum_x2 = two_bin_window(sum_x2_half)
    sum_y2 = two_bin_window(sum_y2_half)
    sum_xy = two_bin_window(sum_xy_half)
    safe_count = np.maximum(count, 1.0)
    mean_x_raw = sum_x / safe_count
    mean_y_raw = sum_y / safe_count
    variance_x = np.maximum(sum_x2 / safe_count - mean_x_raw**2, 0.0)
    variance_y = np.maximum(sum_y2 / safe_count - mean_y_raw**2, 0.0)
    covariance = sum_xy / safe_count - mean_x_raw * mean_y_raw
    std_x_raw = np.sqrt(variance_x)
    std_y_raw = np.sqrt(variance_y)
    correlation = covariance / np.maximum(std_x_raw * std_y_raw, 1e-6)
    correlation[count < 2] = 0.0

    previous_half = np.concatenate((np.zeros(1), count_half[:-1]))
    contrast = (count_half - previous_half) / np.maximum(count, 1.0)
    spectral_energy = np.zeros(length, dtype=np.float64)
    dominant_frequency = np.zeros(length, dtype=np.float64)
    for index in range(length):
        start = max(0, index - spectral_bins + 1)
        history = np.log1p(count_half[start : index + 1])
        if len(history) < spectral_bins:
            history = np.pad(history, (spectral_bins - len(history), 0))
        spectrum = np.abs(np.fft.rfft(history)) ** 2
        total_energy = float(spectrum.sum())
        non_dc = spectrum[1:]
        if total_energy > 0 and len(non_dc):
            spectral_energy[index] = float(non_dc.sum() / total_energy)
            dominant_frequency[index] = float((np.argmax(non_dc) + 1) / len(non_dc))

    output = np.column_stack(
        (
            np.log1p(count),
            polarity_sum / safe_count,
            contrast,
            mean_x_raw / max(width - 1, 1) - 0.5,
            mean_y_raw / max(height - 1, 1) - 0.5,
            std_x_raw / max(width - 1, 1),
            std_y_raw / max(height - 1, 1),
            np.clip(correlation, -1.0, 1.0),
            spectral_energy,
            dominant_frequency,
        )
    )
    output[count == 0, 3:8] = 0.0
    return output.astype(np.float32)


def main() -> None:
    args = parse_args()
    feature_dir = resolve(args.feature_dir)
    data_path = resolve(args.data_path)
    base_metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "event_stats.npy"
    metadata_path = out_dir / "metadata.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists; use --force to replace it")
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(base_metadata["num_points"]), len(FEATURE_NAMES)),
    )
    with h5py.File(data_path, "r") as handle:
        for row in tqdm(sequences.itertuples(index=False), total=len(sequences), desc="event-stats"):
            group = handle[row.rec_name][row.roi_key]
            values = sequence_features(
                np.asarray(group["events"]),
                int(row.length),
                float(base_metadata["grid_stride_s"]),
                int(group.attrs["width"]),
                int(group.attrs["height"]),
                args.spectral_bins,
            )
            matrix[int(row.offset) : int(row.offset + row.length)] = values
    matrix.flush()
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.all():
        raise ValueError(f"Event-stat cache has {int((~finite).sum())} invalid rows")
    mean = np.asarray(matrix, dtype=np.float64).mean(axis=0)
    std = np.asarray(matrix, dtype=np.float64).std(axis=0)
    std[std < 1e-6] = 1.0
    metadata = {
        "format_version": 1,
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "num_points": len(matrix),
        "grid_stride_s": float(base_metadata["grid_stride_s"]),
        "spectral_bins": args.spectral_bins,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "feature_metadata_sha256": sha256_file(feature_dir / "metadata.json"),
        "sequence_index_sha256": sha256_file(feature_dir / "sequences.csv"),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] shape={matrix.shape} mean={mean.tolist()} std={std.tolist()}")


if __name__ == "__main__":
    main()
