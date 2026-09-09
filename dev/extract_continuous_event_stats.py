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


#: Eventos por bloque ao acumular. Un vídeo pode ter 1.600 millóns de eventos (25,8 GB como
#: uint32), así que cargalos enteiros mata o proceso por OOM incluso con 62 GB de RAM. Todas as
#: magnitudes que se acumulan son sumas por bin, e polo tanto aditivas entre bloques.
EVENT_CHUNK = 20_000_000


def _empty_accumulators(length: int) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(length, dtype=np.float64)
        for name in ("count", "polarity", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy")
    }


def accumulate_chunk(
    events: np.ndarray, length: int, bin_width_us: float, acc: dict[str, np.ndarray]
) -> None:
    """Acumula as sumas por bin dun bloque de eventos. Aditivo: pódese chamar en secuencia."""
    if len(events) == 0:
        return
    indices = np.floor(events[:, 2].astype(np.float64) / bin_width_us).astype(np.int64)
    valid = (indices >= 0) & (indices < length)
    indices = indices[valid]
    if len(indices) == 0:
        return
    x = events[valid, 0].astype(np.float64)
    y = events[valid, 1].astype(np.float64)
    polarity = np.where(events[valid, 3] > 0, 1.0, -1.0)

    acc["count"] += np.bincount(indices, minlength=length)[:length].astype(np.float64)
    acc["polarity"] += binned_sum(indices, polarity, length)
    acc["sum_x"] += binned_sum(indices, x, length)
    acc["sum_y"] += binned_sum(indices, y, length)
    acc["sum_x2"] += binned_sum(indices, x * x, length)
    acc["sum_y2"] += binned_sum(indices, y * y, length)
    acc["sum_xy"] += binned_sum(indices, x * y, length)


def features_from_accumulators(
    acc: dict[str, np.ndarray],
    length: int,
    width: int,
    height: int,
    spectral_bins: int,
) -> np.ndarray:
    """Segunda metade do cálculo. Opera só sobre arrays de tamaño `length`, non sobre eventos."""
    count_half = acc["count"]
    polarity_half = acc["polarity"]
    sum_x_half = acc["sum_x"]
    sum_y_half = acc["sum_y"]
    sum_x2_half = acc["sum_x2"]
    sum_y2_half = acc["sum_y2"]
    sum_xy_half = acc["sum_xy"]

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


def sequence_features_chunked(
    dataset,
    length: int,
    stride_s: float,
    width: int,
    height: int,
    spectral_bins: int,
    chunk: int = EVENT_CHUNK,
) -> np.ndarray:
    """Igual que a versión antiga pero sen materializar os eventos enteiros en RAM.

    `dataset` é o dataset HDF5 sen ler (non un array), e lese en bloques de `chunk` eventos.
    """
    total = int(dataset.shape[0])
    if total == 0:
        return np.zeros((length, len(FEATURE_NAMES)), dtype=np.float32)
    acc = _empty_accumulators(length)
    bin_width_us = stride_s * 1e6
    for start in range(0, total, chunk):
        accumulate_chunk(dataset[start : start + chunk], length, bin_width_us, acc)
    return features_from_accumulators(acc, length, width, height, spectral_bins)


def sequence_features(
    events: np.ndarray,
    length: int,
    stride_s: float,
    width: int,
    height: int,
    spectral_bins: int,
) -> np.ndarray:
    """Envoltorio compatible para quen xa teña os eventos en memoria."""
    if len(events) == 0:
        return np.zeros((length, len(FEATURE_NAMES)), dtype=np.float32)
    acc = _empty_accumulators(length)
    accumulate_chunk(events, length, stride_s * 1e6, acc)
    return features_from_accumulators(acc, length, width, height, spectral_bins)


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
            # Pásase o dataset SEN ler: `np.asarray(group["events"])` cargaba os eventos enteiros,
            # e hai vídeos de 1.600 millóns de eventos (25,8 GB) que mataban o proceso por OOM.
            values = sequence_features_chunked(
                group["events"],
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
    # As estatísticas de normalización axústanse SÓ con train+val. Incluír o test faría que a
    # normalización da rama de eventos dependese do conxunto de avaliación: é transdución, aínda
    # que non use etiquetas, e un revisor pode sinalala con razón.
    fit_mask = np.zeros(len(matrix), dtype=bool)
    for row in sequences.itertuples(index=False):
        if str(row.split) != "test":
            fit_mask[int(row.offset) : int(row.offset + row.length)] = True
    if not fit_mask.any():
        raise ValueError("Non hai filas de train/val coas que axustar a normalización")
    fit_rows = np.asarray(matrix[fit_mask], dtype=np.float64)
    mean = fit_rows.mean(axis=0)
    std = fit_rows.std(axis=0)
    std[std < 1e-6] = 1.0
    metadata = {
        "format_version": 2,
        "normalisation_fit_splits": ["train", "val"],
        "normalisation_fit_rows": int(fit_mask.sum()),
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
