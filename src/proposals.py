"""Proposal generation pipeline for event-camera action detection (reTAG).

Builds an actionness signal from binned event rates and extracts temporal
candidate intervals. Includes adaptive thresholding, spatial compactness
modulation, and noise penalisation (sustained and dispersed).
"""

import os
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
import h5py
from absl import logging

from src.utils import temporal_nms
from src.prototype import get_prototype_score


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log(path: str, text: str) -> None:
    with open(path, "a") as f:
        f.write(text + "\n")


def _detect_runs(
    mask: np.ndarray,
    bin_width_us: float,
    min_duration_s: float,
) -> np.ndarray:
    """Mark runs of True in *mask* that span at least *min_duration_s* seconds.

    Args:
        mask: Boolean array of shape (T,).
        bin_width_us: Bin width in microseconds.
        min_duration_s: Minimum run length in seconds.

    Returns:
        Float64 binary array in {0, 1} of shape (T,).
    """
    min_bins = max(1, int((min_duration_s * 1e6) / bin_width_us))
    result = np.zeros(len(mask), dtype=np.float64)
    diff   = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) >= min_bins:
            result[s:e] = 1.0
    return result


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_event_rate(events: np.ndarray, bin_width: float) -> tuple:
    """Bin events into fixed-width temporal intervals and count events per bin.

    Args:
        events: Array of shape (N, 4) with columns [x, y, t, p].
            Events must be time-sorted; t is in microseconds.
        bin_width: Bin width in microseconds.

    Returns:
        counts: Array of shape (bin_num,) with event counts per bin.
        bins: Bin edges in microseconds, shape (bin_num + 1,).
    """
    t_min, t_max = events[0, 2], events[-1, 2]
    bin_num = int((t_max - t_min) / bin_width)
    counts, bins = np.histogram(events[:, 2], bins=bin_num)
    return counts, bins


def apply_robust_min_max(rate: np.ndarray, percentile: float) -> np.ndarray:
    """Clip per-bin event rates to suppress outliers.

    Saturates values outside the symmetric percentile band
    [0.5*percentile, 100 - 0.5*percentile]. Modifies the array in-place.

    Args:
        rate: Per-bin event counts, shape (bin_num,).
        percentile: Clipping strength; higher values clip more.

    Returns:
        The same array, clipped in-place.
    """
    rmin = np.percentile(rate.flat, 0.5 * percentile)
    rmax = np.percentile(rate.flat, 100 - 0.5 * percentile)
    rate[rate < rmin] = rmin
    rate[rate > rmax] = rmax
    return rate


def get_index_proposals_from_1d_score(
    score1d: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Extract contiguous above-threshold segments from a 1D score signal.

    Args:
        score1d: Actionness signal of shape (T,).
        threshold: Activation threshold λ.

    Returns:
        Array of shape (n, 2) with [start_idx, end_idx] for each segment
        where score1d > threshold.
    """
    return np.where(np.diff(score1d > threshold, prepend=0, append=0))[0].reshape(-1, 2)


def check_merge_possible(
    proposal_1: np.ndarray,
    proposal_2: np.ndarray,
    accumulated_duration: float,
    threshold: float,
) -> bool:
    """Decide whether merging proposal_2 into proposal_1 is warranted.

    Merges if the ratio of active bins to total span would exceed *threshold*
    after incorporating proposal_2.

    Args:
        proposal_1: Current proposal [start, end] in bin indices.
        proposal_2: Next proposal [start, end] in bin indices.
        accumulated_duration: Active-bin count accumulated so far.
        threshold: Minimum activity ratio to trigger a merge.

    Returns:
        True if the proposals should be merged.
    """
    candidate_active = accumulated_duration + (proposal_2[1] - proposal_2[0])
    candidate_span   = proposal_2[1] - proposal_1[0]
    return candidate_active / candidate_span > threshold


def merge_proposals(
    unmerged: np.ndarray,
    score: np.ndarray,
    grouping_thres: float,
    times: np.ndarray,
) -> list:
    """Merge an initial proposal list into longer, coherent intervals.

    Iterates proposals in temporal order and merges consecutive ones when
    their activity ratio (active bins / total span) exceeds *grouping_thres*.
    The merged proposal score is the mean of *score* over its span.

    Args:
        unmerged: Proposals as bin-index pairs, shape (n, 2).
        score: Actionness signal, shape (T,).
        grouping_thres: Activity-ratio threshold for merging.
        times: Bin-edge timestamps in microseconds, shape (bin_num + 1,).

    Returns:
        List of [t_start, t_end, mean_score] for each merged proposal.
    """
    merged = []
    current = None
    accumulated_basin_durations = 0

    for next_proposal in unmerged:
        if current is None:
            current = next_proposal
            accumulated_basin_durations += next_proposal[1] - next_proposal[0]
        else:
            do_merge = check_merge_possible(
                current, next_proposal, accumulated_basin_durations, grouping_thres
            )
            if do_merge:
                current[1] = next_proposal[1]
                accumulated_basin_durations += next_proposal[1] - next_proposal[0]
            elif not do_merge or (next_proposal == unmerged[-1]).all():
                merged.append([
                    times[current[0]],
                    times[current[1]],
                    np.mean(score[current[0]:current[1]]),
                ])
                current = next_proposal
                accumulated_basin_durations = 0

    return merged


def get_adaptive_actioness_thresholds(
    actioness: np.ndarray,
    central_percentile: float = 80,
    delta: float = 0.10,
    step: float = 0.05,
) -> np.ndarray:
    """Compute a threshold grid centred on the actionness distribution.

    Instead of a fixed global grid, the grid adapts to each sequence by
    anchoring its centre at *central_percentile* of the actionness signal.

    Args:
        actioness: Normalised signal in [0, 1], shape (T,).
        central_percentile: Percentile used as the grid centre.
        delta: Half-width of the threshold range around the centre.
        step: Step between consecutive thresholds.

    Returns:
        Sorted unique thresholds, clipped to [0.05, 0.95].
    """
    lambda_center = np.percentile(actioness, central_percentile)
    # +1e-9 ensures np.arange reliably includes the upper bound despite float rounding
    thresholds = np.arange(lambda_center - delta, lambda_center + delta + 1e-9, step)
    thresholds = np.clip(thresholds, 0.05, 0.95)
    return np.unique(np.round(thresholds, 4))


def get_spatial_compactness(
    events: np.ndarray,
    bins: np.ndarray,
    roi_height: int,
    roi_width: int,
) -> np.ndarray:
    """Compute a spatial compactness signal per temporal bin.

    Measures how concentrated events are within the ROI for each bin.
    High values → events clustered in a small region → likely real action.
    Low values → events spread across the ROI → likely background or noise.

    Uses the RMS distance from the per-bin centroid, normalised by the ROI
    semi-diagonal. Fully vectorised via np.bincount.

    Args:
        events: Array of shape (N, 4) with columns [x, y, t, p].
            Coordinates are relative to the ROI bounding box.
        bins: Bin edges in microseconds, shape (bin_num + 1,).
        roi_height: ROI height in pixels.
        roi_width: ROI width in pixels.

    Returns:
        Compactness signal in [0, 1] per bin, shape (bin_num,).
        Bins with fewer than 2 events receive 0 (insufficient data).
    """
    bin_num = len(bins) - 1
    bin_idx = np.searchsorted(bins[1:], events[:, 2], side='right')
    bin_idx = np.clip(bin_idx, 0, bin_num - 1)

    x = events[:, 0].astype(np.float64)
    y = events[:, 1].astype(np.float64)

    count  = np.bincount(bin_idx, minlength=bin_num).astype(np.float64)
    sum_x  = np.bincount(bin_idx, weights=x,     minlength=bin_num)
    sum_y  = np.bincount(bin_idx, weights=y,     minlength=bin_num)
    sum_x2 = np.bincount(bin_idx, weights=x * x, minlength=bin_num)
    sum_y2 = np.bincount(bin_idx, weights=y * y, minlength=bin_num)

    safe_count = np.maximum(count, 1.0)
    mean_x = sum_x / safe_count
    mean_y = sum_y / safe_count

    # Clamp negatives: floating-point cancellation can produce tiny negative variances
    var_x = np.maximum(sum_x2 / safe_count - mean_x ** 2, 0.0)
    var_y = np.maximum(sum_y2 / safe_count - mean_y ** 2, 0.0)

    spread     = np.sqrt(var_x + var_y)
    max_spread = np.sqrt((roi_width / 2.0) ** 2 + (roi_height / 2.0) ** 2) + 1e-9

    # Bins with fewer than 2 events have no reliable centroid estimate
    return np.where(count >= 2, 1.0 - np.clip(spread / max_spread, 0.0, 1.0), 0.0)


def get_sustained_noise_indicator(
    rate: np.ndarray,
    bin_width_us: float,
    min_duration_s: float = 20.0,
    high_percentile: float = 98,
    variance_window: int = 10,
) -> np.ndarray:
    """Identify bins with high, sustained, and temporally flat activity.

    Targets persistent background noise: sensor drift, wind, camera vibration.
    These produce high event rates that remain approximately constant over time,
    unlike real actions which have internal temporal structure.

    Three criteria must hold simultaneously:
    1. Rate ≥ *high_percentile* of the global distribution.
    2. Low local temporal variance (flat signal, not a transient peak).
    3. Sustained for at least *min_duration_s* seconds.

    Args:
        rate: Per-bin event counts after robust scaling, shape (bin_num,).
        bin_width_us: Bin width in microseconds.
        min_duration_s: Minimum segment length in seconds.
        high_percentile: Percentile defining "high activity".
        variance_window: Rolling-variance window size in bins.

    Returns:
        Binary indicator in {0, 1}, shape (bin_num,).
    """
    # >= rather than > so that plateau values exactly at the percentile are included
    above = rate >= np.percentile(rate, high_percentile)

    # Rolling variance via Var = E[X²] − E[X]² (vectorised, no Python loops)
    kernel = np.ones(variance_window) / variance_window
    rolling_mean    = np.convolve(rate,      kernel, mode='same')
    rolling_sq_mean = np.convolve(rate ** 2, kernel, mode='same')
    # mode='same' uses partial windows at boundaries; clamp numerical negatives
    rolling_var = np.maximum(rolling_sq_mean - rolling_mean ** 2, 0.0)
    flat = rolling_var <= np.percentile(rolling_var, 33)

    return _detect_runs(above & flat, bin_width_us, min_duration_s)


def get_dispersed_noise_indicator(
    rate: np.ndarray,
    compactness: np.ndarray,
    bin_width_us: float,
    min_duration_s: float = 20.0,
    high_percentile: float = 90,
    dispersion_percentile: float = 20,
) -> np.ndarray:
    """Identify bins with high rate and high spatial dispersion sustained over time.

    Targets precipitation (snow, rain): events distributed randomly across the
    ROI. Complements get_sustained_noise_indicator — snow has HIGH temporal
    variance (random events), so the flat-variance criterion of the sustained
    indicator misses it; but its spatial compactness is near zero, which this
    indicator detects.

    Three criteria must hold simultaneously:
    1. Rate ≥ *high_percentile* of the global distribution.
    2. Compactness ≤ *dispersion_percentile* (highly dispersed bins).
    3. Sustained for at least *min_duration_s* seconds.

    Both percentile thresholds are relative to the ROI to adapt to varying
    conditions across sequences.

    Args:
        rate: Per-bin event counts after robust scaling, shape (bin_num,).
        compactness: Spatial compactness per bin (from get_spatial_compactness),
            shape (bin_num,).
        bin_width_us: Bin width in microseconds.
        min_duration_s: Minimum segment length in seconds.
        high_percentile: Percentile defining "high activity". Lower than the
            sustained indicator (90 vs 98) because precipitation does not
            always produce the most extreme rate spikes.
        dispersion_percentile: Compactness percentile below which a bin is
            considered highly dispersed.

    Returns:
        Binary indicator in {0, 1}, shape (bin_num,).
    """
    above     = rate >= np.percentile(rate, high_percentile)
    dispersed = compactness <= np.percentile(compactness, dispersion_percentile)
    return _detect_runs(above & dispersed, bin_width_us, min_duration_s)


def get_periodicity_indicator(
    rate: np.ndarray,
    bin_width_us: float,
    min_period_s: float = 0.3,
    max_period_s: float = 2.0,
    window_s: float = 3.0,
    global_threshold: float = 0.30,
    local_threshold: float = 0.30,
    min_duration_s: float = 3.0,
) -> np.ndarray:
    """Identify bins with locally periodic activity in the wing-flap frequency range.

    Wing flaps are rhythmic (0.5–3 Hz); Ecstatic Displays produce a single
    sustained peak with no periodicity. This indicator detects segments where
    the rate signal oscillates at a consistent lag within [min_period_s,
    max_period_s] and penalises them in the actionness.

    Two-stage detection:
    1. **Global screening** — FFT-based autocorrelation over the full ROI to
       check whether any dominant period exists in the target range. Returns
       zeros immediately if the global peak is below *global_threshold*.
    2. **Local confirmation** — sliding-window normalised cross-correlation at
       the dominant lag. Bins where the local cross-correlation exceeds
       *local_threshold* are marked as periodic. *_detect_runs* then requires
       the periodic pattern to be sustained for at least *min_duration_s*.

    Both steps are fully vectorised (FFT + convolution); no Python loops.

    Args:
        rate: Per-bin event counts after robust scaling, shape (bin_num,).
        bin_width_us: Bin width in microseconds.
        min_period_s: Minimum oscillation period in seconds (≈ max frequency).
        max_period_s: Maximum oscillation period in seconds (≈ min frequency).
        window_s: Sliding window length in seconds for local cross-correlation.
        global_threshold: Minimum normalised autocorrelation peak to proceed.
        local_threshold: Minimum local cross-correlation to mark a bin periodic.
        min_duration_s: Minimum sustained periodic segment length in seconds.

    Returns:
        Binary indicator in {0, 1}, shape (bin_num,).
    """
    n = len(rate)
    bin_width_s = bin_width_us / 1e6

    min_lag = max(1, int(min_period_s / bin_width_s))
    max_lag = min(n // 2, int(max_period_s / bin_width_s))
    if min_lag >= max_lag:
        return np.zeros(n, dtype=np.float64)

    # --- Stage 1: global autocorrelation via FFT ---
    centered = rate - rate.mean()
    fft_val  = np.fft.rfft(centered, n=2 * n)
    autocorr = np.fft.irfft(fft_val * np.conj(fft_val))[:n]
    norm     = autocorr[0]
    if norm < 1e-9:
        return np.zeros(n, dtype=np.float64)
    autocorr /= norm

    dominant_lag = int(np.argmax(autocorr[min_lag:max_lag + 1])) + min_lag
    if autocorr[dominant_lag] < global_threshold:
        return np.zeros(n, dtype=np.float64)

    # --- Stage 2: sliding local cross-correlation at dominant_lag ---
    T   = dominant_lag
    W   = max(2, int(window_s / bin_width_s))
    pad = np.zeros(T, dtype=np.float64)

    # Pairs: X = rate[:-T], Y = rate[T:], both length (n - T)
    X = rate[:-T]
    Y = rate[T:]
    m = len(X)

    kernel = np.ones(W) / W

    mean_X  = np.convolve(X,      kernel, mode='same')
    mean_Y  = np.convolve(Y,      kernel, mode='same')
    mean_XY = np.convolve(X * Y,  kernel, mode='same')
    mean_X2 = np.convolve(X ** 2, kernel, mode='same')
    mean_Y2 = np.convolve(Y ** 2, kernel, mode='same')

    std_X = np.sqrt(np.maximum(mean_X2 - mean_X ** 2, 0.0))
    std_Y = np.sqrt(np.maximum(mean_Y2 - mean_Y ** 2, 0.0))

    local_corr = (mean_XY - mean_X * mean_Y) / (std_X * std_Y + 1e-9)
    local_corr = np.clip(local_corr, 0.0, 1.0)

    # Pad back to full length (last T bins have no paired signal → 0)
    local_corr_full = np.concatenate([local_corr, pad])[:n]

    periodic_mask = local_corr_full >= local_threshold
    return _detect_runs(periodic_mask, bin_width_us, min_duration_s)


# ---------------------------------------------------------------------------
# Proposal generator
# ---------------------------------------------------------------------------

class ProposalGenerator:
    """Generate temporal action proposals from event-camera ROI data.

    Computes a per-bin actionness signal from the event rate, optionally
    modulated by spatial compactness and penalised for noise, then segments
    it into candidate action intervals via thresholding and temporal NMS.

    The actionness formula with all improvements active is:

        combined(t) = r(t) · (1 + w₂·compactness(t))
                            · (1 + w₅·prototype_score(t))
                            · (1 − w₃·noise(t))
                            · (1 − w₄·disp_noise(t))
                            · (1 − w₆·periodicity(t))
        a(t) = normalise(combined(t))

    Args:
        data_path: Path to the preprocessed HDF5 file.
        bin_width: Temporal bin width in seconds.
        percentile: Clipping percentile for robust rate scaling.
        nms_threshold: IoU threshold for temporal NMS.
        output_dir: Directory for the run log file.
        use_adaptive_lambda: Use per-sequence percentile-based thresholds
            instead of a fixed grid.
        lambda_percentile: Percentile of the actionness used as the grid centre.
        lambda_delta: Half-width of the threshold grid around the centre.
        lambda_step: Step between consecutive thresholds.
        use_spatial_compactness: Amplify actionness in spatially compact bins
            (w₂ modulation).
        spatial_weight: Compactness amplification weight w₂.
        use_noise_penalization: Dampen sustained flat-noise bins (wind,
            vibration) with weight w₃.
        noise_percentile: Rate percentile defining "high activity" for
            sustained noise detection.
        noise_min_duration: Minimum sustained-noise segment length (s).
        noise_weight: Sustained-noise damping weight w₃.
        noise_variance_window: Rolling-variance window in bins.
        use_dispersed_noise: Dampen dispersed-noise bins (snow, rain) with
            weight w₄.
        dispersed_noise_percentile: Rate percentile for dispersed-noise
            detection.
        dispersed_noise_dispersion_percentile: Compactness percentile below
            which a bin is considered spatially dispersed.
        dispersed_noise_min_duration: Minimum dispersed-noise segment length (s).
        dispersed_noise_weight: Dispersed-noise damping weight w₄.
        prototype: Pre-built ED spatial prototype array of shape (grid_h, grid_w),
            as returned by src.prototype.build_ed_prototype. When provided,
            bins whose spatial event distribution resembles an ED are amplified
            by w₅ (prototype_weight). This only amplifies — never suppresses —
            so AR cannot decrease.
        prototype_weight: Prototype amplification weight w₅.
    """

    def __init__(
        self,
        data_path: str,
        bin_width: float,
        percentile: float,
        nms_threshold: float,
        output_dir: str = "output",
        use_adaptive_lambda: bool = False,
        lambda_percentile: float = 75,
        lambda_delta: float = 0.20,
        lambda_step: float = 0.05,
        use_spatial_compactness: bool = False,
        spatial_weight: float = 0.2,
        use_noise_penalization: bool = False,
        noise_percentile: float = 98,
        noise_min_duration: float = 20.0,
        noise_weight: float = 0.5,
        noise_variance_window: int = 10,
        use_dispersed_noise: bool = False,
        dispersed_noise_percentile: float = 90,
        dispersed_noise_dispersion_percentile: float = 20,
        dispersed_noise_min_duration: float = 20.0,
        dispersed_noise_weight: float = 0.5,
        prototype: Optional[np.ndarray] = None,
        prototype_weight: float = 0.3,
        use_periodicity: bool = False,
        periodicity_min_period_s: float = 0.3,
        periodicity_max_period_s: float = 2.0,
        periodicity_window_s: float = 3.0,
        periodicity_global_threshold: float = 0.65,
        periodicity_local_threshold: float = 0.60,
        periodicity_min_duration_s: float = 3.0,
        periodicity_weight: float = 0.5,
    ) -> None:
        tags = []
        if use_adaptive_lambda:     tags.append("adaptive")
        if use_spatial_compactness: tags.append("spatial")
        if use_noise_penalization:  tags.append("noise")
        if use_dispersed_noise:     tags.append("dispersed")
        if prototype is not None:   tags.append("proto")
        if use_periodicity:         tags.append("period")
        self._run_tag = "_".join(tags) if tags else "baseline"

        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"retag_{self._run_tag}_full.txt")
        with open(self.log_file, "w") as f:
            f.write(f"=== RUN RETAG {self._run_tag.upper()} ===\n")

        self.data_path   = data_path
        self.bin_width   = bin_width * 1e6  # convert s → µs
        self.percentile  = percentile
        self.nms_threshold = nms_threshold

        self.use_adaptive_lambda = use_adaptive_lambda
        self.lambda_percentile   = lambda_percentile
        self.lambda_delta        = lambda_delta
        self.lambda_step         = lambda_step

        self.use_spatial_compactness = use_spatial_compactness
        self.spatial_weight          = spatial_weight

        self.use_noise_penalization = use_noise_penalization
        self.noise_percentile       = noise_percentile
        self.noise_min_duration     = noise_min_duration
        self.noise_weight           = noise_weight
        self.noise_variance_window  = noise_variance_window

        self.use_dispersed_noise                   = use_dispersed_noise
        self.dispersed_noise_percentile            = dispersed_noise_percentile
        self.dispersed_noise_dispersion_percentile = dispersed_noise_dispersion_percentile
        self.dispersed_noise_min_duration          = dispersed_noise_min_duration
        self.dispersed_noise_weight                = dispersed_noise_weight

        self.prototype        = prototype
        self.prototype_weight = prototype_weight

        self.use_periodicity                = use_periodicity
        self.periodicity_min_period_s       = periodicity_min_period_s
        self.periodicity_max_period_s       = periodicity_max_period_s
        self.periodicity_window_s           = periodicity_window_s
        self.periodicity_global_threshold   = periodicity_global_threshold
        self.periodicity_local_threshold    = periodicity_local_threshold
        self.periodicity_min_duration_s     = periodicity_min_duration_s
        self.periodicity_weight             = periodicity_weight

        self.actioness_thresholds = np.arange(0.05, 1, 0.05)
        self.grouping_thresholds  = np.arange(0.05, 1, 0.05)

    def process_recording(self, rec: str) -> dict:
        """Process all ROIs in one recording and return their proposals.

        Args:
            rec: Recording key in the HDF5 file.

        Returns:
            Dict mapping roi_id → proposal array of shape (n, 3) with columns
            [t_start_µs, t_end_µs, score].
        """
        rec_proposal_data = {}
        _log(self.log_file, f"\n[RECORDING] {rec}")

        with h5py.File(self.data_path, "r") as hf:
            data = hf[rec]
            for roi_id in data.keys():
                events     = np.array(data[roi_id]["events"])
                roi_height = int(data[roi_id].attrs["height"])
                roi_width  = int(data[roi_id].attrs["width"])

                _log(self.log_file, f"\n[ROI] {roi_id}")
                _log(self.log_file, f"n_events: {len(events)}")

                rate, bins = get_event_rate(events, self.bin_width)
                rate = apply_robust_min_max(rate, self.percentile)

                _log(self.log_file, f"rate min/max: {rate.min()} / {rate.max()}")
                _log(self.log_file, f"rate mean: {rate.mean():.4f}")

                r_min, r_max = rate.min(), rate.max()
                r_t = (
                    (rate - r_min) / (r_max - r_min)
                    if r_max > r_min
                    else np.zeros_like(rate, dtype=np.float64)
                )

                # Compute compactness once; reused by spatial and dispersed-noise steps
                needs_compactness = self.use_spatial_compactness or self.use_dispersed_noise
                if needs_compactness:
                    compactness = get_spatial_compactness(events, bins, roi_height, roi_width)
                    c_min, c_max = compactness.min(), compactness.max()
                    compactness_norm = (
                        (compactness - c_min) / (c_max - c_min)
                        if c_max > c_min
                        else np.zeros_like(compactness)
                    )

                # Build combined signal — multiplicative so noise steps only suppress
                # already-active bins and cannot create spurious peaks
                if self.use_spatial_compactness:
                    combined = r_t * (1.0 + self.spatial_weight * compactness_norm)
                    _log(self.log_file, f"compactness mean: {compactness_norm.mean():.4f}")
                    _log(self.log_file, f"spatial_weight: {self.spatial_weight}")
                else:
                    combined = r_t

                # Prototype amplification: bins that resemble the ED spatial
                # template get amplified. Pure amplification → AR cannot drop.
                if self.prototype is not None:
                    proto_score = get_prototype_score(
                        events, bins, self.prototype, roi_height, roi_width
                    )
                    combined = combined * (1.0 + self.prototype_weight * proto_score)
                    _log(self.log_file, f"proto_score mean: {proto_score.mean():.4f}")
                    _log(self.log_file, f"proto_score max:  {proto_score.max():.4f}")
                    _log(self.log_file, f"prototype_weight: {self.prototype_weight}")

                if self.use_periodicity:
                    periodicity = get_periodicity_indicator(
                        rate,
                        bin_width_us=self.bin_width,
                        min_period_s=self.periodicity_min_period_s,
                        max_period_s=self.periodicity_max_period_s,
                        window_s=self.periodicity_window_s,
                        global_threshold=self.periodicity_global_threshold,
                        local_threshold=self.periodicity_local_threshold,
                        min_duration_s=self.periodicity_min_duration_s,
                    )
                    combined = combined * (1.0 - self.periodicity_weight * periodicity)
                    _log(self.log_file, f"periodic bins: {int(periodicity.sum())} / {len(periodicity)}")
                    _log(self.log_file, f"periodicity_weight: {self.periodicity_weight}")

                if self.use_noise_penalization:
                    noise = get_sustained_noise_indicator(
                        rate,
                        bin_width_us=self.bin_width,
                        min_duration_s=self.noise_min_duration,
                        high_percentile=self.noise_percentile,
                        variance_window=self.noise_variance_window,
                    )
                    combined = combined * (1.0 - self.noise_weight * noise)
                    _log(self.log_file, f"noise bins: {int(noise.sum())} / {len(noise)}")
                    _log(self.log_file, f"noise_weight: {self.noise_weight}")

                if self.use_dispersed_noise:
                    disp_noise = get_dispersed_noise_indicator(
                        rate,
                        compactness,
                        bin_width_us=self.bin_width,
                        min_duration_s=self.dispersed_noise_min_duration,
                        high_percentile=self.dispersed_noise_percentile,
                        dispersion_percentile=self.dispersed_noise_dispersion_percentile,
                    )
                    combined = combined * (1.0 - self.dispersed_noise_weight * disp_noise)
                    _log(self.log_file, f"dispersed noise bins: {int(disp_noise.sum())} / {len(disp_noise)}")
                    _log(self.log_file, f"dispersed_noise_weight: {self.dispersed_noise_weight}")

                a_min, a_max = combined.min(), combined.max()
                actioness = (
                    (combined - a_min) / (a_max - a_min)
                    if a_max > a_min
                    else np.zeros_like(combined)
                )

                _log(self.log_file, f"actioness min/max: {actioness.min():.4f} / {actioness.max():.4f}")
                _log(self.log_file, f"actioness mean: {actioness.mean():.4f}")

                if self.use_adaptive_lambda:
                    thresholds = get_adaptive_actioness_thresholds(
                        actioness,
                        central_percentile=self.lambda_percentile,
                        delta=self.lambda_delta,
                        step=self.lambda_step,
                    )
                    _log(self.log_file, f"adaptive thresholds: {thresholds.tolist()}")
                else:
                    thresholds = self.actioness_thresholds
                    _log(self.log_file, f"fixed thresholds: {thresholds.tolist()}")

                proposals = [
                    proposal
                    for at in thresholds
                    for gt in self.grouping_thresholds
                    for proposal in merge_proposals(
                        get_index_proposals_from_1d_score(actioness, at),
                        actioness, gt, bins,
                    )
                ]

                _log(self.log_file, f"proposals raw: {len(proposals)}")
                proposals = np.array(proposals)

                if len(proposals) > 0:
                    proposals = proposals[proposals[:, 1] - proposals[:, 0] > 2e6]
                    proposals = temporal_nms(proposals, self.nms_threshold)
                    _log(self.log_file, f"proposals after NMS: {len(proposals)}")
                    if len(proposals) > 0:
                        durations = proposals[:, 1] - proposals[:, 0]
                        _log(self.log_file, f"mean duration: {durations.mean():.2f}")
                        _log(self.log_file, f"min duration:  {durations.min():.2f}")
                        _log(self.log_file, f"max duration:  {durations.max():.2f}")
                    else:
                        _log(self.log_file, "proposals after NMS: 0")
                else:
                    proposals = np.empty((0, 3))
                    _log(self.log_file, "proposals after NMS: 0")

                rec_proposal_data[roi_id] = proposals

        return rec_proposal_data

    def run(self, split: Optional[str] = "test") -> pd.DataFrame:
        """Process recordings and return proposals as a DataFrame.

        Args:
            split: If set, only process recordings whose HDF5 'split' attribute
                matches this value. Pass None to process all recordings.

        Returns:
            DataFrame with columns [rec_name, roi_id, t_start, t_end, score].
        """
        logging.info("Running Proposal Generator.")

        with h5py.File(self.data_path, "r") as hf:
            recordings = [
                rec for rec in hf.keys()
                if split is None or hf[rec].attrs.get("split") == split
            ]

        _log(self.log_file, f"\n[INFO] recordings: {recordings}")

        with Pool(processes=16) as pool:
            results = pool.map(self.process_recording, recordings)

        rows = [
            {
                "rec_name": os.path.splitext(rec)[0],
                "roi_id":   roi_id,
                "t_start":  proposal[0],
                "t_end":    proposal[1],
                "score":    proposal[2],
            }
            for rec, rec_proposals in zip(recordings, results)
            for roi_id, proposals in rec_proposals.items()
            for proposal in proposals
        ]
        return pd.DataFrame(rows)
