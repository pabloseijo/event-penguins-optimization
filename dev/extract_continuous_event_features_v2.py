"""Extract aligned ON/OFF profiles and signed spectra for continuous TAD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
PROFILE_BINS = 8
SPECTRAL_BANDS = ((0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0))
FEATURE_NAMES = (
    [f"log_on_rate_bin_{index}" for index in range(PROFILE_BINS)]
    + [f"log_off_rate_bin_{index}" for index in range(PROFILE_BINS)]
    + ["window_polarity_balance", "window_log_rate"]
    + [f"signed_power_{low:g}_{high:g}hz" for low, high in SPECTRAL_BANDS]
    + [f"unsigned_power_{low:g}_{high:g}hz" for low, high in SPECTRAL_BANDS]
    + [
        "signed_band_fraction",
        "unsigned_band_fraction",
        "signed_dominant_frequency",
        "unsigned_dominant_frequency",
        "signed_band_entropy",
        "unsigned_band_entropy",
    ]
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile-bins", type=int, default=PROFILE_BINS)
    parser.add_argument("--spectral-window-s", type=float, default=5.0)
    parser.add_argument("--spectral-bin-s", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def binned_counts(
    timestamps_s: np.ndarray,
    polarity_on: np.ndarray,
    start_s: float,
    bin_s: float,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.floor((timestamps_s - start_s) / bin_s).astype(np.int64)
    valid = (indices >= 0) & (indices < num_bins)
    indices = indices[valid]
    on = np.bincount(indices[polarity_on[valid]], minlength=num_bins)[:num_bins]
    off = np.bincount(indices[~polarity_on[valid]], minlength=num_bins)[:num_bins]
    return on.astype(np.float64), off.astype(np.float64)


def normalized_band_features(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    non_dc_total = power[:, frequencies > 0].sum(axis=1, keepdims=True)
    denominator = np.maximum(non_dc_total, 1e-12)
    values = []
    for low, high in SPECTRAL_BANDS:
        selected = (frequencies >= low) & (frequencies < high)
        if high == SPECTRAL_BANDS[-1][1]:
            selected = (frequencies >= low) & (frequencies <= high)
        values.append(power[:, selected].sum(axis=1) / denominator[:, 0])
    return np.column_stack(values)


def spectral_summary(signal: np.ndarray, bin_s: float) -> tuple[np.ndarray, ...]:
    tapered = (signal - signal.mean(axis=1, keepdims=True)) * np.hanning(signal.shape[1])
    power = np.abs(np.fft.rfft(tapered, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(signal.shape[1], d=bin_s)
    bands = normalized_band_features(power, frequencies)
    selected = (frequencies >= SPECTRAL_BANDS[0][0]) & (
        frequencies <= SPECTRAL_BANDS[-1][1]
    )
    band_power = power[:, selected]
    total = np.maximum(power[:, frequencies > 0].sum(axis=1), 1e-12)
    fraction = band_power.sum(axis=1) / total
    band_total = band_power.sum(axis=1, keepdims=True)
    probabilities = band_power / np.maximum(band_total, 1e-12)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    entropy /= np.log(max(band_power.shape[1], 2))
    band_frequencies = frequencies[selected]
    dominant = band_frequencies[np.argmax(band_power, axis=1)]
    silent = band_total[:, 0] <= 1e-12
    fraction[silent] = 0.0
    entropy[silent] = 0.0
    dominant[silent] = 0.0
    return bands, fraction, dominant / SPECTRAL_BANDS[-1][1], entropy


def sequence_features(
    events: np.ndarray,
    length: int,
    stride_s: float,
    window_s: float,
    width: int,
    height: int,
    profile_bins: int = PROFILE_BINS,
    spectral_window_s: float = 5.0,
    spectral_bin_s: float = 0.1,
) -> np.ndarray:
    if profile_bins != PROFILE_BINS:
        raise ValueError(f"This format requires exactly {PROFILE_BINS} profile bins")
    centers = (np.arange(length, dtype=np.float64) + 0.5) * stride_s
    timestamps_s = events[:, 2].astype(np.float64) / 1e6 if len(events) else np.empty(0)
    polarity_on = events[:, 3] > 0 if len(events) else np.empty(0, dtype=bool)
    area = float(max(width * height, 1))

    profile_bin_s = window_s / profile_bins
    profile_start = float(centers[0] - 0.5 * window_s)
    profile_num_bins = (length - 1) * round(stride_s / profile_bin_s) + profile_bins
    on_fine, off_fine = binned_counts(
        timestamps_s, polarity_on, profile_start, profile_bin_s, profile_num_bins
    )
    profile_step = round(stride_s / profile_bin_s)
    profile_indices = np.arange(profile_bins)[None, :] + profile_step * np.arange(length)[:, None]
    on_profile = on_fine[profile_indices]
    off_profile = off_fine[profile_indices]
    on_rate = on_profile / (area * profile_bin_s)
    off_rate = off_profile / (area * profile_bin_s)
    count = on_profile.sum(axis=1) + off_profile.sum(axis=1)
    balance = (on_profile.sum(axis=1) - off_profile.sum(axis=1)) / np.maximum(count, 1.0)
    log_rate = np.log1p(count / (area * window_s))

    spectral_bins = round(spectral_window_s / spectral_bin_s)
    spectral_start = float(centers[0] - 0.5 * spectral_window_s)
    spectral_step = round(stride_s / spectral_bin_s)
    spectral_num_bins = (length - 1) * spectral_step + spectral_bins
    on_spectral, off_spectral = binned_counts(
        timestamps_s,
        polarity_on,
        spectral_start,
        spectral_bin_s,
        spectral_num_bins,
    )
    spectral_indices = (
        np.arange(spectral_bins)[None, :] + spectral_step * np.arange(length)[:, None]
    )
    signed = on_spectral[spectral_indices] - off_spectral[spectral_indices]
    unsigned = on_spectral[spectral_indices] + off_spectral[spectral_indices]
    signed_bands, signed_fraction, signed_dominant, signed_entropy = spectral_summary(
        signed, spectral_bin_s
    )
    unsigned_bands, unsigned_fraction, unsigned_dominant, unsigned_entropy = spectral_summary(
        unsigned, spectral_bin_s
    )
    output = np.column_stack(
        (
            np.log1p(on_rate),
            np.log1p(off_rate),
            balance,
            log_rate,
            signed_bands,
            unsigned_bands,
            signed_fraction,
            unsigned_fraction,
            signed_dominant,
            unsigned_dominant,
            signed_entropy,
            unsigned_entropy,
        )
    )
    return output.astype(np.float32)


def main() -> None:
    args = parse_args()
    feature_dir = resolve(args.feature_dir)
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
    with h5py.File(resolve(args.data_path), "r") as handle:
        for row in tqdm(sequences.itertuples(index=False), total=len(sequences), desc="event-v2"):
            group = handle[row.rec_name][row.roi_key]
            values = sequence_features(
                np.asarray(group["events"]),
                int(row.length),
                float(base_metadata["grid_stride_s"]),
                float(base_metadata["window_duration_s"]),
                int(group.attrs["width"]),
                int(group.attrs["height"]),
                args.profile_bins,
                args.spectral_window_s,
                args.spectral_bin_s,
            )
            matrix[int(row.offset) : int(row.offset + row.length)] = values
    matrix.flush()
    if not np.isfinite(matrix).all():
        raise ValueError("Event-v2 cache contains invalid values")
    mean = np.asarray(matrix, dtype=np.float64).mean(axis=0)
    std = np.asarray(matrix, dtype=np.float64).std(axis=0)
    std[std < 1e-6] = 1.0
    metadata = {
        "format_version": 2,
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "num_points": len(matrix),
        "grid_stride_s": float(base_metadata["grid_stride_s"]),
        "profile_bins": args.profile_bins,
        "spectral_window_s": args.spectral_window_s,
        "spectral_bin_s": args.spectral_bin_s,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] shape={matrix.shape} finite=True")


if __name__ == "__main__":
    main()
