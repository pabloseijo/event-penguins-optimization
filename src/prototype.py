"""Spatial prototype of the Ecstatic Display and its per-bin similarity score.

The prototype is the mean normalised event-density grid over annotated ED
instances of the training split. Scoring a recording against it answers a
question the raw event rate cannot: not *how much* motion there is, but whether
it is laid out over the nest the way an ED is.
"""

from __future__ import annotations

import json

import h5py
import numpy as np


def _events_to_grid(
    events: np.ndarray,
    roi_height: int,
    roi_width: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    if len(events) == 0:
        return np.zeros((grid_h, grid_w), dtype=np.float64)

    gy = np.clip((events[:, 1] / roi_height * grid_h).astype(int), 0, grid_h - 1)
    gx = np.clip((events[:, 0] / roi_width * grid_w).astype(int), 0, grid_w - 1)

    grid = np.zeros((grid_h, grid_w), dtype=np.float64)
    np.add.at(grid, (gy, gx), 1)

    max_val = grid.max()
    return grid / max_val if max_val > 0 else grid


def build_ed_prototype(
    data_path: str,
    ann_path: str,
    split: str = "train",
    label: str = "ed",
    grid_h: int = 16,
    grid_w: int = 16,
    min_duration: float = 2.0,
    recordings: set[str] | None = None,
) -> np.ndarray:
    """Average the event-density grids of every annotated ED instance.

    Args:
        data_path: HDF5 file produced by ``scripts/preprocess.py``.
        ann_path: ActivityNet-style annotation file.
        split: split whose recordings feed the prototype when ``recordings`` is None.
        label: annotation label to average over.
        grid_h: prototype height in cells.
        grid_w: prototype width in cells.
        min_duration: instances shorter than this (seconds) are skipped.
        recordings: explicit recording set, overriding ``split``. Pass the training
            fold here when cross-validating, so no test recording leaks in.

    Returns:
        L2-normalised ``[grid_h, grid_w]`` prototype, or zeros when no instance matched.
    """
    with open(ann_path) as f:
        ann = json.load(f)

    with h5py.File(data_path, "r") as hf:
        split_recs = (
            set(recordings)
            if recordings is not None
            else {r for r in hf if hf[r].attrs.get("split") == split}
        )
        missing = split_recs - set(hf.keys())
        if missing:
            raise ValueError(f"Prototype recordings are missing from HDF5: {sorted(missing)}")

        grids: list[np.ndarray] = []
        n_skipped = 0

        for rec, v in ann["database"].items():
            if rec not in split_recs or rec not in hf:
                continue
            for roi_str, roi_anns in v["annotations"].items():
                if roi_str == "null":
                    continue
                matching_annotations = [
                    item
                    for item in roi_anns
                    if item["label"] == label
                    and (item["segment"][1] - item["segment"][0]) >= min_duration
                ]
                n_skipped += sum(
                    item["label"] == label
                    and (item["segment"][1] - item["segment"][0]) < min_duration
                    for item in roi_anns
                )
                if not matching_annotations:
                    continue
                roi_id = f"N{int(roi_str):02d}"
                if roi_id not in hf[rec]:
                    continue

                roi_height = int(hf[rec][roi_id].attrs["height"])
                roi_width = int(hf[rec][roi_id].attrs["width"])
                all_events = np.array(hf[rec][roi_id]["events"])

                for a in matching_annotations:
                    t_start, t_end = a["segment"]

                    mask = (all_events[:, 2] >= t_start * 1e6) & (all_events[:, 2] <= t_end * 1e6)
                    events = all_events[mask]

                    if len(events) < 10:
                        n_skipped += 1
                        continue

                    grids.append(_events_to_grid(events, roi_height, roi_width, grid_h, grid_w))

    print(f"[prototipo] Instancias usadas: {len(grids)}  |  Descartadas: {n_skipped}")

    if not grids:
        print("[prototipo] AVISO: non se atoparon instancias. Retornando prototipo cero.")
        return np.zeros((grid_h, grid_w), dtype=np.float64)

    prototype = np.mean(grids, axis=0)
    norm = np.linalg.norm(prototype)
    return prototype / norm if norm > 0 else prototype


def get_prototype_score(
    events: np.ndarray,
    bins: np.ndarray,
    prototype: np.ndarray,
    roi_height: int,
    roi_width: int,
    min_events_per_bin: int = 5,
) -> np.ndarray:
    """Per-bin cosine similarity between the event layout and the ED prototype.

    Bins holding fewer than ``min_events_per_bin`` events score 0: there is not
    enough evidence to judge their spatial layout. Since the prototype is
    L2-normalised, the cosine similarity is a plain dot product.

    Args:
        events: ``[n, 4]`` array of ``[x, y, t, p]``.
        bins: bin edges from :func:`src.proposals.get_event_rate`.
        prototype: grid from :func:`build_ed_prototype`.
        roi_height: ROI height in pixels.
        roi_width: ROI width in pixels.
        min_events_per_bin: evidence required to score a bin.

    Returns:
        Per-bin similarity in ``[0, 1]``.
    """
    grid_h, grid_w = prototype.shape
    bin_num = len(bins) - 1

    bin_idx = np.searchsorted(bins[1:], events[:, 2], side="right")
    bin_idx = np.clip(bin_idx, 0, bin_num - 1)

    gy = np.clip((events[:, 1] / roi_height * grid_h).astype(int), 0, grid_h - 1)
    gx = np.clip((events[:, 0] / roi_width * grid_w).astype(int), 0, grid_w - 1)

    flat_idx = gy * grid_w + gx
    n_cells = grid_h * grid_w
    proto_flat = prototype.ravel()

    counts_per_bin = np.bincount(bin_idx, minlength=bin_num)
    scores = np.zeros(bin_num, dtype=np.float64)
    active_bins = np.where(counts_per_bin >= min_events_per_bin)[0]

    for b in active_bins:
        mask = bin_idx == b
        grid_flat = np.bincount(flat_idx[mask], minlength=n_cells).astype(np.float64)
        norm = np.linalg.norm(grid_flat)
        if norm > 0:
            grid_flat /= norm
            scores[b] = np.clip(np.dot(grid_flat, proto_flat), 0.0, 1.0)

    return scores
