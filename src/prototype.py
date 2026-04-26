"""ED spatial prototype: build a reference 2D event-density map from training
annotations and compute per-bin cosine similarity scores at inference time.

The prototype captures the typical spatial distribution of events during an
Ecstatic Display (ED). Bins that resemble the prototype get a score close to 1;
bins with a very different layout (e.g. wing-flap, background) score near 0.

Usage pattern:
    prototype = build_ed_prototype(data_path, ann_path, split="train")
    # pass prototype to ProposalGenerator(prototype=prototype, ...)
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
    """Accumulate events into a normalised 2D density grid.

    Coordinates are scaled to [0, grid_h) × [0, grid_w) regardless of ROI
    size, so prototypes computed across different ROIs are spatially comparable.

    Args:
        events: Array of shape (N, 4) with columns [x, y, t, p].
        roi_height: ROI height in pixels.
        roi_width: ROI width in pixels.
        grid_h: Grid height in cells.
        grid_w: Grid width in cells.

    Returns:
        Normalised density grid of shape (grid_h, grid_w) with values in [0, 1].
        Returns zeros if the event array is empty.
    """
    if len(events) == 0:
        return np.zeros((grid_h, grid_w), dtype=np.float64)

    # Map pixel coordinates to grid cells — clip to avoid off-by-one at edges
    gy = np.clip((events[:, 1] / roi_height * grid_h).astype(int), 0, grid_h - 1)
    gx = np.clip((events[:, 0] / roi_width  * grid_w).astype(int), 0, grid_w - 1)

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
) -> np.ndarray:
    """Build a mean 2D spatial prototype from labelled ED instances.

    Loads every segment labelled *label* in *split*, accumulates its events
    into a normalised grid, and returns the per-cell mean across all instances.
    Each instance is individually normalised before averaging so that longer
    or higher-rate segments do not dominate the prototype.

    Only uses the *split* subset to avoid leaking test labels.

    Args:
        data_path: Path to the preprocessed HDF5 file.
        ann_path: Path to the annotations JSON.
        split: HDF5 split attribute to restrict to (default 'train').
        label: Annotation label to treat as the positive class.
        grid_h: Prototype grid height in cells.
        grid_w: Prototype grid width in cells.
        min_duration: Ignore instances shorter than this (seconds).

    Returns:
        Prototype array of shape (grid_h, grid_w), L2-normalised to unit norm.
        Returns zeros if no valid instances were found.
    """
    with open(ann_path) as f:
        ann = json.load(f)

    with h5py.File(data_path, "r") as hf:
        split_recs = {r for r in hf if hf[r].attrs.get("split") == split}

        grids: list[np.ndarray] = []
        n_skipped = 0

        for rec, v in ann["database"].items():
            if rec not in split_recs or rec not in hf:
                continue
            for roi_str, roi_anns in v["annotations"].items():
                if roi_str == "null":
                    continue
                roi_id = f"N{int(roi_str):02d}"
                if roi_id not in hf[rec]:
                    continue

                roi_height = int(hf[rec][roi_id].attrs["height"])
                roi_width  = int(hf[rec][roi_id].attrs["width"])
                all_events = np.array(hf[rec][roi_id]["events"])  # (N, 4) in µs

                for a in roi_anns:
                    if a["label"] != label:
                        continue
                    t_start, t_end = a["segment"]
                    if (t_end - t_start) < min_duration:
                        n_skipped += 1
                        continue

                    t_start_us = t_start * 1e6
                    t_end_us   = t_end   * 1e6
                    mask   = (all_events[:, 2] >= t_start_us) & (all_events[:, 2] <= t_end_us)
                    events = all_events[mask]

                    if len(events) < 10:
                        n_skipped += 1
                        continue

                    grids.append(_events_to_grid(events, roi_height, roi_width, grid_h, grid_w))

    print(f"[prototype] Instancias usadas: {len(grids)}  |  Descartadas: {n_skipped}")

    if not grids:
        print("[prototype] AVISO: non se atoparon instancias. Retornando prototipo cero.")
        return np.zeros((grid_h, grid_w), dtype=np.float64)

    prototype = np.mean(grids, axis=0)

    # L2-normalise so cosine similarity is a simple dot product at inference time
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
    """Compute per-bin cosine similarity to the ED prototype.

    For each temporal bin, accumulates events into a normalised grid and
    computes the cosine similarity against the pre-built prototype.
    Bins with fewer than *min_events_per_bin* events receive a score of 0
    (too noisy to be informative).

    Because the prototype is L2-normalised (see build_ed_prototype), the
    similarity reduces to: dot(grid_norm, prototype).

    Args:
        events: Array of shape (N, 4) with columns [x, y, t, p].
        bins: Bin edges in microseconds, shape (bin_num + 1,).
        prototype: L2-normalised prototype array of shape (grid_h, grid_w).
        roi_height: ROI height in pixels.
        roi_width: ROI width in pixels.
        min_events_per_bin: Minimum events required to compute a valid score.

    Returns:
        Similarity scores in [0, 1] per bin, shape (bin_num,).
    """
    grid_h, grid_w = prototype.shape
    bin_num = len(bins) - 1

    # Assign each event to a bin
    bin_idx = np.searchsorted(bins[1:], events[:, 2], side="right")
    bin_idx = np.clip(bin_idx, 0, bin_num - 1)

    # Map to grid coordinates
    gy = np.clip((events[:, 1] / roi_height * grid_h).astype(int), 0, grid_h - 1)
    gx = np.clip((events[:, 0] / roi_width  * grid_w).astype(int), 0, grid_w - 1)

    # Flatten grid coordinates for vectorised bincount
    flat_idx = gy * grid_w + gx  # shape (N,)
    n_cells  = grid_h * grid_w

    proto_flat = prototype.ravel()
    counts_per_bin = np.bincount(bin_idx, minlength=bin_num)

    scores = np.zeros(bin_num, dtype=np.float64)

    # Batch bins that have enough events; skip the rest (score stays 0)
    active_bins = np.where(counts_per_bin >= min_events_per_bin)[0]

    for b in active_bins:
        mask       = bin_idx == b
        cell_idx   = flat_idx[mask]
        grid_flat  = np.bincount(cell_idx, minlength=n_cells).astype(np.float64)
        grid_norm  = np.linalg.norm(grid_flat)
        if grid_norm > 0:
            grid_flat /= grid_norm
            scores[b]  = np.clip(np.dot(grid_flat, proto_flat), 0.0, 1.0)

    return scores
