"""Extract local ON/OFF and spectral features at the dense ATSN sample times.

The output is aligned row-for-row with a proposal master and has shape
``[N, T, 8]``. Event rates are normalized by ROI area, while the remaining
features are scale-free. Extraction is resumable at ROI-group granularity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]
FEATURE_NAMES = [
    "log_on_rate_per_pixel_s",
    "log_off_rate_per_pixel_s",
    "polarity_balance",
    "unsigned_rate_cv",
    "signed_band_energy_fraction",
    "unsigned_band_energy_fraction",
    "signed_dominant_frequency",
    "signed_band_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--bin-s", type=float, default=0.1)
    parser.add_argument("--freq-min", type=float, default=0.5)
    parser.add_argument("--freq-max", type=float, default=3.0)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def proposal_fingerprint(proposals: pd.DataFrame) -> str:
    missing = sorted(set(KEY_COLUMNS) - set(proposals.columns))
    if missing:
        raise ValueError(f"Proposal file misses key columns: {missing}")
    hashed = pd.util.hash_pandas_object(proposals[KEY_COLUMNS], index=False)
    return hashlib.sha256(hashed.to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def expanded_tsn_samples(num_tsn_samples: int, augment_factor: int) -> int:
    return num_tsn_samples + 2 * int(math.ceil(num_tsn_samples / augment_factor))


def sample_centers_us(
    proposals: pd.DataFrame,
    num_segments: int,
    augment_factor: int,
) -> np.ndarray:
    starts = proposals["t_start"].to_numpy(dtype=np.float64)
    ends = proposals["t_end"].to_numpy(dtype=np.float64)
    durations = ends - starts
    relative = np.linspace(
        -1.0 / augment_factor,
        1.0 + 1.0 / augment_factor,
        num_segments,
        dtype=np.float64,
    )
    return starts[:, None] + durations[:, None] * relative[None, :]


def normalized_entropy(power: np.ndarray) -> np.ndarray:
    total = power.sum(axis=-1, keepdims=True)
    probability = np.divide(
        power,
        total,
        out=np.zeros_like(power),
        where=total > 1e-12,
    )
    terms = np.where(probability > 0, probability * np.log(probability), 0.0)
    denominator = math.log(power.shape[-1]) if power.shape[-1] > 1 else 1.0
    return -terms.sum(axis=-1) / denominator


def local_event_descriptors(
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    centers_us: np.ndarray,
    area: float,
    bin_s: float,
    window_s: float,
    freq_min: float,
    freq_max: float,
) -> np.ndarray:
    """Compute the eight descriptors from globally binned ROI event counts."""
    window_bins = max(4, int(round(window_s / bin_s)))
    left = window_bins // 2
    offsets = np.arange(-left, window_bins - left, dtype=np.int64)
    center_bins = np.floor(centers_us / (bin_s * 1e6)).astype(np.int64)
    indices = center_bins[..., None] + offsets
    valid = (indices >= 0) & (indices < len(on_counts))
    clipped = np.clip(indices, 0, max(len(on_counts) - 1, 0))
    on = np.where(valid, on_counts[clipped], 0.0).astype(np.float64)
    off = np.where(valid, off_counts[clipped], 0.0).astype(np.float64)
    unsigned = on + off
    signed = on - off

    density_scale = max(float(area) * bin_s, 1e-12)
    on_rate = on.mean(axis=-1) / density_scale
    off_rate = off.mean(axis=-1) / density_scale
    mean_unsigned = unsigned.mean(axis=-1)
    balance = signed.sum(axis=-1) / np.maximum(unsigned.sum(axis=-1), 1.0)
    unsigned_cv = unsigned.std(axis=-1) / np.maximum(mean_unsigned, 1.0)

    hann = np.hanning(window_bins)
    signed_centered = signed - signed.mean(axis=-1, keepdims=True)
    unsigned_centered = unsigned - unsigned.mean(axis=-1, keepdims=True)
    signed_power = np.abs(np.fft.rfft(signed_centered * hann, axis=-1)) ** 2
    unsigned_power = np.abs(np.fft.rfft(unsigned_centered * hann, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(window_bins, d=bin_s)
    positive = frequencies > 0
    band = (frequencies >= freq_min) & (frequencies <= freq_max)
    if not np.any(band):
        raise ValueError("The requested frequency band has no FFT bins")
    signed_total = signed_power[..., positive].sum(axis=-1)
    unsigned_total = unsigned_power[..., positive].sum(axis=-1)
    signed_band = signed_power[..., band]
    unsigned_band = unsigned_power[..., band]
    signed_band_fraction = signed_band.sum(axis=-1) / np.maximum(signed_total, 1e-12)
    unsigned_band_fraction = unsigned_band.sum(axis=-1) / np.maximum(unsigned_total, 1e-12)
    band_frequencies = frequencies[band]
    dominant_index = signed_band.argmax(axis=-1)
    dominant_frequency = band_frequencies[dominant_index] / max(freq_max, 1e-12)
    no_signed_power = signed_band.sum(axis=-1) <= 1e-12
    dominant_frequency[no_signed_power] = 0.0
    entropy = normalized_entropy(signed_band)

    result = np.stack(
        (
            np.log1p(on_rate),
            np.log1p(off_rate),
            balance,
            unsigned_cv,
            signed_band_fraction,
            unsigned_band_fraction,
            dominant_frequency,
            entropy,
        ),
        axis=-1,
    )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def roi_name(value: str) -> str:
    text = str(value)
    return text if text.startswith("N") else f"N{int(text):02d}"


def main() -> None:
    args = parse_args()
    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    fingerprint = proposal_fingerprint(proposals)
    num_segments = expanded_tsn_samples(args.num_tsn_samples, args.augment_factor)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "event_features.npy"
    metadata_path = out_dir / "metadata.json"
    state_path = out_dir / "extraction_state.json"
    expected = {
        "fingerprint": fingerprint,
        "rows": len(proposals),
        "num_segments": num_segments,
        "feature_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "window_s": args.window_s,
        "bin_s": args.bin_s,
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
    }
    if not args.restart and feature_path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            print(f"[INFO] Event-spectral cache reutilizada: {out_dir}")
            return

    completed: set[str] = set()
    can_resume = not args.restart and feature_path.exists() and state_path.exists()
    if can_resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if all(state.get(key) == value for key, value in expected.items() if key != "feature_names"):
            completed = set(state.get("completed_groups", []))
        else:
            can_resume = False
    mode = "r+" if can_resume else "w+"
    output = np.lib.format.open_memmap(
        feature_path,
        mode=mode,
        dtype=np.float32,
        shape=(len(proposals), num_segments, len(FEATURE_NAMES)),
    )
    if not can_resume:
        output[:] = 0.0

    groups = list(proposals.groupby(["rec_name", "roi_id"], sort=False).groups.items())
    with h5py.File(resolve(args.data_path), "r") as hf:
        for group_number, ((recording, roi_id), row_indices) in enumerate(groups, start=1):
            group_key = f"{recording}/{roi_id}"
            if group_key in completed:
                continue
            roi = roi_name(str(roi_id))
            roi_group = hf[str(recording)][roi]
            events = np.asarray(roi_group["events"])
            timestamps = events[:, 2].astype(np.float64)
            max_time = max(
                float(timestamps[-1]) if len(timestamps) else 0.0,
                float(proposals.loc[row_indices, "t_end"].max()),
            )
            bin_us = args.bin_s * 1e6
            num_bins = max(1, int(math.floor(max_time / bin_us)) + 2)
            bins = np.minimum((timestamps / bin_us).astype(np.int64), num_bins - 1)
            polarity_on = events[:, 3] > 0 if len(events) else np.zeros(0, dtype=bool)
            on_counts = np.bincount(bins[polarity_on], minlength=num_bins)
            off_counts = np.bincount(bins[~polarity_on], minlength=num_bins)
            group_proposals = proposals.loc[row_indices]
            centers = sample_centers_us(group_proposals, num_segments, args.augment_factor)
            area = float(roi_group.attrs["height"] * roi_group.attrs["width"])
            output[np.asarray(row_indices, dtype=np.int64)] = local_event_descriptors(
                on_counts,
                off_counts,
                centers,
                area,
                args.bin_s,
                args.window_s,
                args.freq_min,
                args.freq_max,
            )
            output.flush()
            completed.add(group_key)
            state_path.write_text(
                json.dumps({**expected, "completed_groups": sorted(completed)}, indent=2),
                encoding="utf-8",
            )
            print(
                f"[EVENT {group_number:04d}/{len(groups):04d}] {group_key} "
                f"rows={len(row_indices)}",
                flush=True,
            )

    output.flush()
    metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    state_path.unlink(missing_ok=True)
    print(f"[RESULTADO] Event-spectral cache: {output.shape} en {out_dir}")


if __name__ == "__main__":
    main()
