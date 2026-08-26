"""Stage 1: temporal proposal generation from per-ROI event streams.

The reTAG baseline (Hamann et al., CVPR 2024) reduces an event stream to a 1-D
event rate, normalises it robustly, thresholds it over a grid of actionness
values and groups the resulting basins into proposals. This module keeps that
skeleton intact and adds the descriptors studied in phase 1 of the project, each
one opt-in so the baseline stays reproducible unchanged:

* an adaptive threshold grid centred on the actionness distribution itself;
* spatial compactness of the event cloud;
* two noise indicators, for flat sustained noise and for precipitation;
* similarity to a spatial prototype of the Ecstatic Display;
* a periodicity penalty in the wing-flapping band.

Actionness is combined multiplicatively, so a descriptor that is silent leaves
the baseline signal untouched::

    a(t) = norm( r(t) · (1 + w_c·c(t)) · (1 + w_p·p(t)) · (1 − w_f·f(t)) · (1 − w_n·n(t)) )

where ``r`` is the normalised event rate, ``c`` compactness, ``p`` the prototype
similarity, ``f`` the periodicity indicator and ``n`` the noise indicators.

Times are handled in microseconds throughout, matching the raw event timestamps.
"""

from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import h5py
from absl import logging

from src.utils import temporal_nms
from src.prototype import get_prototype_score


def _log(path: str, text: str) -> None:
    with open(path, "a") as f:
        f.write(text + "\n")


def _detect_runs(
    mask: np.ndarray,
    bin_width_us: float,
    min_duration_s: float,
) -> np.ndarray:
    min_bins = max(1, int((min_duration_s * 1e6) / bin_width_us))
    result = np.zeros(len(mask), dtype=np.float64)
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) >= min_bins:
            result[s:e] = 1.0
    return result


def get_event_rate(events: np.ndarray, bin_width: float) -> tuple:
    """Histogram event timestamps into fixed-width temporal bins.

    Args:
        events: ``[n, 4]`` array of ``[x, y, t, p]``, sorted by timestamp.
        bin_width: bin width in microseconds.

    Returns:
        Tuple of per-bin event counts and the bin edges.
    """
    t_min, t_max = events[0, 2], events[-1, 2]
    bin_num = int((t_max - t_min) / bin_width)
    counts, bins = np.histogram(events[:, 2], bins=bin_num)
    return counts, bins


def apply_robust_min_max(rate: np.ndarray, percentile: float) -> np.ndarray:
    """Clip the event rate to a symmetric percentile range.

    ``percentile`` is the total share of samples to clip, split evenly between both
    tails, which is how reTAG keeps a single burst of events from compressing the
    whole normalised signal.

    Args:
        rate: per-bin event counts.
        percentile: total percentage clipped across both tails.

    Returns:
        The clipped rate (modified in place).
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
    """Return the ``[start, end)`` bin indices of every run above ``threshold``.

    Args:
        score1d: per-bin actionness.
        threshold: actionness value that separates active from inactive bins.

    Returns:
        ``[m, 2]`` array of bin index pairs, one row per basin.
    """
    return np.where(np.diff(score1d > threshold, prepend=0, append=0))[0].reshape(-1, 2)


def check_merge_possible(
    proposal_1: np.ndarray,
    proposal_2: np.ndarray,
    accumulated_duration: float,
    threshold: float,
) -> bool:
    """Decide whether two basins belong to the same proposal.

    Merging is accepted when the active time inside the candidate span stays above
    ``threshold``: a short gap between two long basins is bridged, a long gap is not.

    Args:
        proposal_1: bin indices of the group being accumulated.
        proposal_2: bin indices of the next basin.
        accumulated_duration: active bins accumulated so far, in bins.
        threshold: minimum active fraction of the merged span.

    Returns:
        True when the two basins should be merged.
    """
    candidate_active = accumulated_duration + (proposal_2[1] - proposal_2[0])
    candidate_span = proposal_2[1] - proposal_1[0]
    return candidate_active / candidate_span > threshold


def merge_proposals(
    unmerged: np.ndarray,
    score: np.ndarray,
    grouping_thres: float,
    times: np.ndarray,
) -> list:
    """Group consecutive basins into proposals under a grouping threshold.

    Args:
        unmerged: ``[m, 2]`` bin index pairs from ``get_index_proposals_from_1d_score``.
        score: per-bin actionness, averaged over each proposal to score it.
        grouping_thres: minimum active fraction required to bridge a gap.
        times: bin edges, used to convert bin indices back to timestamps.

    Returns:
        List of ``[t_start, t_end, mean_score]``, one per merged proposal.

    Note:
        The final group is flushed after the loop. The original implementation only
        appended when a merge was rejected, so it silently dropped the last proposal
        of every ROI.
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
            else:
                merged.append([
                    times[current[0]],
                    times[current[1]],
                    np.mean(score[current[0]:current[1]]),
                ])
                current = next_proposal
                accumulated_basin_durations = next_proposal[1] - next_proposal[0]

    # the original loop only flushed on a rejected merge, dropping the last group
    if current is not None:
        merged.append([
            times[current[0]],
            times[current[1]],
            np.mean(score[current[0]:current[1]]),
        ])

    return merged


def pad_short_proposals(
    proposals: list,
    target_duration_us: float,
    min_source_duration_us: float,
    max_source_duration_us: float,
    min_score: float,
    t_min: float,
    t_max: float,
    score_scale: float = 1.0,
) -> np.ndarray:
    """Pad short, salient bursts into windows long enough to survive.

    The generator drops any proposal shorter than the minimum duration. During a
    short display the signal can arrive as fragmented bursts that individually
    miss that bound even though the annotated action clears it. This helper keeps
    those hypotheses without looking at any annotation: it centres a minimum-length
    window on the short burst.

    Args:
        proposals: list of ``[t_start, t_end, score]`` in microseconds.
        target_duration_us: length of the padded window.
        min_source_duration_us: shortest burst eligible for padding.
        max_source_duration_us: longest burst eligible for padding.
        min_score: minimum score a burst needs to be padded.
        t_min: earliest timestamp of the recording, used to clamp the window.
        t_max: latest timestamp of the recording.
        score_scale: multiplier applied to the score of padded windows.

    Returns:
        The original proposals with the padded windows appended.
    """
    if len(proposals) == 0:
        return np.empty((0, 3), dtype=np.float64)

    arr = np.asarray(proposals, dtype=np.float64)
    durations = arr[:, 1] - arr[:, 0]
    mask = (
        (durations >= min_source_duration_us)
        & (durations < max_source_duration_us)
        & (arr[:, 2] >= min_score)
    )
    if not np.any(mask):
        return arr

    target_duration_us = min(target_duration_us, max(t_max - t_min, 0.0))
    if target_duration_us <= 0:
        return arr

    centers = 0.5 * (arr[mask, 0] + arr[mask, 1])
    starts = centers - 0.5 * target_duration_us
    starts = np.clip(starts, t_min, t_max - target_duration_us)
    ends = starts + target_duration_us
    padded = np.column_stack([starts, ends, arr[mask, 2] * score_scale])
    return np.vstack([arr, padded])


def get_adaptive_actioness_thresholds(
    actioness: np.ndarray,
    central_percentile: float = 80,
    delta: float = 0.10,
    step: float = 0.05,
) -> np.ndarray:
    """Build a threshold grid centred on the actionness distribution of this ROI.

    The baseline sweeps a fixed grid from 0.05 to 0.95, which spends most of its
    thresholds far from where the signal actually lives. Centring the grid on a
    percentile of the observed actionness adapts the sweep to each ROI without
    looking at any annotation.

    Args:
        actioness: per-bin actionness of one ROI.
        central_percentile: percentile used as the centre of the grid.
        delta: half-width of the grid around the centre.
        step: spacing between consecutive thresholds.

    Returns:
        Sorted unique thresholds, clipped to ``[0.05, 0.95]``.
    """
    lambda_center = np.percentile(actioness, central_percentile)
    # +1e-9 so np.arange keeps the upper end despite floating-point rounding
    thresholds = np.arange(lambda_center - delta, lambda_center + delta + 1e-9, step)
    thresholds = np.clip(thresholds, 0.05, 0.95)
    return np.unique(np.round(thresholds, 4))


def get_spatial_compactness(
    events: np.ndarray,
    bins: np.ndarray,
    roi_height: int,
    roi_width: int,
    sigmoid_k: float = 10.0,
    sigmoid_d0: float = 0.5,
) -> np.ndarray:
    """Per-bin compactness of the event cloud inside the ROI.

    An Ecstatic Display is one bird moving on its nest, so its events stay spatially
    concentrated; wind over the whole scene does not. Compactness is the spatial
    spread of the events of a bin, mapped through a sigmoid so that concentrated
    bins score near 1 and scattered bins near 0.

    Args:
        events: ``[n, 4]`` array of ``[x, y, t, p]``.
        bins: bin edges from ``get_event_rate``.
        roi_height: ROI height in pixels, used to normalise the spread.
        roi_width: ROI width in pixels.
        sigmoid_k: steepness of the sigmoid.
        sigmoid_d0: normalised spread at which the sigmoid crosses 0.5.

    Returns:
        Per-bin compactness in ``[0, 1]``; bins with fewer than two events score 0.
    """
    bin_num = len(bins) - 1
    bin_idx = np.searchsorted(bins[1:], events[:, 2], side='right')
    bin_idx = np.clip(bin_idx, 0, bin_num - 1)

    x = events[:, 0].astype(np.float64)
    y = events[:, 1].astype(np.float64)

    count = np.bincount(bin_idx, minlength=bin_num).astype(np.float64)
    sum_x = np.bincount(bin_idx, weights=x, minlength=bin_num)
    sum_y = np.bincount(bin_idx, weights=y, minlength=bin_num)
    sum_x2 = np.bincount(bin_idx, weights=x * x, minlength=bin_num)
    sum_y2 = np.bincount(bin_idx, weights=y * y, minlength=bin_num)

    safe_count = np.maximum(count, 1.0)
    mean_x = sum_x / safe_count
    mean_y = sum_y / safe_count

    # cancelación de coma flotante pode dar varianzas negativas pequenas
    var_x = np.maximum(sum_x2 / safe_count - mean_x ** 2, 0.0)
    var_y = np.maximum(sum_y2 / safe_count - mean_y ** 2, 0.0)

    spread = np.sqrt(var_x + var_y)
    max_spread = np.sqrt((roi_width / 2.0) ** 2 + (roi_height / 2.0) ** 2) + 1e-9

    spread_norm = np.clip(spread / max_spread, 0.0, 1.0)
    compactness = 1.0 / (1.0 + np.exp(sigmoid_k * (spread_norm - sigmoid_d0)))
    return np.where(count >= 2, compactness, 0.0)


def get_sustained_noise_indicator(
    rate: np.ndarray,
    bin_width_us: float,
    min_duration_s: float = 20.0,
    high_percentile: float = 98,
    variance_window: int = 10,
) -> np.ndarray:
    """Detect flat, sustained background noise such as wind or vibration.

    Three criteria must hold at once: a high event rate, low local variance, and a
    stretch lasting at least ``min_duration_s``. A display fails the second one,
    which is what separates it from a windy stretch of similar magnitude.

    Args:
        rate: per-bin event counts.
        bin_width_us: bin width in microseconds.
        min_duration_s: seconds a stretch must last to count as noise.
        high_percentile: rate percentile that counts as high activity.
        variance_window: bins in the local variance window.

    Returns:
        Per-bin indicator in ``{0, 1}``.
    """
    above = rate >= np.percentile(rate, high_percentile)

    kernel = np.ones(variance_window) / variance_window
    rolling_mean = np.convolve(rate, kernel, mode='same')
    rolling_sq_mean = np.convolve(rate ** 2, kernel, mode='same')
    # mode='same' usa ventás parciais nos bordos; recortar negativos numéricos
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
    """Detect precipitation: a high event rate spread over the whole ROI.

    This complements :func:`get_sustained_noise_indicator`. Snow and rain have high
    temporal variance, so they never pass the flat-signal test, but their
    compactness sits near zero because the events cover the frame.

    Args:
        rate: per-bin event counts.
        compactness: per-bin compactness from :func:`get_spatial_compactness`.
        bin_width_us: bin width in microseconds.
        min_duration_s: seconds a stretch must last to count as noise.
        high_percentile: rate percentile that counts as high activity.
        dispersion_percentile: compactness percentile that counts as dispersed.

    Returns:
        Per-bin indicator in ``{0, 1}``.
    """
    above = rate >= np.percentile(rate, high_percentile)
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
    """Detect rhythmic activity in the wing-flapping band (0.5-3 Hz).

    Wing flapping is periodic; an Ecstatic Display is not. Detection runs in two
    stages: an FFT autocorrelation screens the ROI for a dominant lag, and a local
    cross-correlation at that lag marks which bins are actually periodic.

    Args:
        rate: per-bin event counts.
        bin_width_us: bin width in microseconds.
        min_period_s: shortest period considered.
        max_period_s: longest period considered.
        window_s: window of the local cross-correlation.
        global_threshold: autocorrelation needed to accept a dominant lag.
        local_threshold: local correlation needed to mark a bin periodic.
        min_duration_s: seconds a periodic stretch must last.

    Returns:
        Per-bin indicator in ``{0, 1}``; all zeros when no dominant lag is found.
    """
    n = len(rate)
    bin_width_s = bin_width_us / 1e6

    min_lag = max(1, int(min_period_s / bin_width_s))
    max_lag = min(n // 2, int(max_period_s / bin_width_s))
    if min_lag >= max_lag:
        return np.zeros(n, dtype=np.float64)

    centered = rate - rate.mean()
    fft_val = np.fft.rfft(centered, n=2 * n)
    autocorr = np.fft.irfft(fft_val * np.conj(fft_val))[:n]
    norm = autocorr[0]
    if norm < 1e-9:
        return np.zeros(n, dtype=np.float64)
    autocorr /= norm

    dominant_lag = int(np.argmax(autocorr[min_lag:max_lag + 1])) + min_lag
    if autocorr[dominant_lag] < global_threshold:
        return np.zeros(n, dtype=np.float64)

    T = dominant_lag
    W = max(2, int(window_s / bin_width_s))
    pad = np.zeros(T, dtype=np.float64)

    X = rate[:-T]
    Y = rate[T:]

    kernel = np.ones(W) / W
    mean_X = np.convolve(X, kernel, mode='same')
    mean_Y = np.convolve(Y, kernel, mode='same')
    mean_XY = np.convolve(X * Y, kernel, mode='same')
    mean_X2 = np.convolve(X ** 2, kernel, mode='same')
    mean_Y2 = np.convolve(Y ** 2, kernel, mode='same')

    std_X = np.sqrt(np.maximum(mean_X2 - mean_X ** 2, 0.0))
    std_Y = np.sqrt(np.maximum(mean_Y2 - mean_Y ** 2, 0.0))
    local_corr = (mean_XY - mean_X * mean_Y) / (std_X * std_Y + 1e-9)
    local_corr = np.clip(local_corr, 0.0, 1.0)

    # the last T bins have no lagged counterpart to correlate against
    local_corr_full = np.concatenate([local_corr, pad])[:n]

    return _detect_runs(local_corr_full >= local_threshold, bin_width_us, min_duration_s)


class ProposalGenerator:
    """reTAG proposal generator with the optional phase-1 descriptors.

    With every switch left at its default the generator reproduces the CVPR 2024
    baseline: normalised event rate, a fixed grid of actionness and grouping
    thresholds, a minimum duration and temporal NMS. Each ``use_*`` flag turns on one
    descriptor and its weight, and the enabled set is recorded in the name of the run
    log so a result file always says which variant produced it.

    Args:
        data_path: HDF5 file produced by ``scripts/preprocess.py``.
        bin_width: temporal bin width in seconds (converted to microseconds inside).
        percentile: total share of samples clipped by the robust normalisation.
        nms_threshold: tIoU above which overlapping proposals are suppressed.
        output_dir: directory for the per-run text log.
        use_adaptive_lambda: centre the threshold grid on the actionness percentile.
        lambda_percentile: centre of the adaptive grid.
        lambda_delta: half-width of the adaptive grid.
        lambda_step: spacing of the adaptive grid.
        use_spatial_compactness: multiply actionness by the compactness descriptor.
        spatial_weight: weight of the compactness term.
        compact_sigmoid_k: steepness of the compactness sigmoid.
        compact_sigmoid_d0: normalised spread at the sigmoid midpoint.
        use_noise_penalization: damp flat sustained noise (wind, vibration).
        noise_percentile: rate percentile that counts as high activity.
        noise_min_duration: seconds a noisy stretch must last to be penalised.
        noise_weight: strength of the sustained-noise penalty.
        noise_variance_window: bins in the local variance window.
        use_dispersed_noise: damp precipitation (high rate, low compactness).
        dispersed_noise_percentile: rate percentile that counts as high activity.
        dispersed_noise_dispersion_percentile: compactness percentile that counts as dispersed.
        dispersed_noise_min_duration: seconds a dispersed stretch must last.
        dispersed_noise_weight: strength of the dispersed-noise penalty.
        prototype: ED prototype grid from ``build_ed_prototype``, or None to disable.
        prototype_weight: weight of the prototype similarity term.
        use_periodicity: damp rhythmic activity in the wing-flapping band.
        periodicity_min_period_s: shortest period considered periodic.
        periodicity_max_period_s: longest period considered periodic.
        periodicity_window_s: window of the local cross-correlation.
        periodicity_global_threshold: autocorrelation needed to accept a dominant lag.
        periodicity_local_threshold: local correlation needed to mark a bin periodic.
        periodicity_min_duration_s: seconds a periodic stretch must last.
        periodicity_weight: strength of the periodicity penalty.
        use_short_proposal_padding: keep short salient bursts by padding them.
        short_padding_target_duration_s: duration of the padded window.
        short_padding_min_source_duration_s: shortest burst eligible for padding.
        short_padding_max_source_duration_s: longest burst eligible for padding.
        short_padding_min_score: minimum score of a burst eligible for padding.
        short_padding_score_scale: score multiplier applied to padded windows.
        minimum_proposal_duration_s: proposals shorter than this are dropped.

    Raises:
        ValueError: if ``minimum_proposal_duration_s`` is negative.
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
        compact_sigmoid_k: float = 10.0,
        compact_sigmoid_d0: float = 0.5,
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
        use_short_proposal_padding: bool = False,
        short_padding_target_duration_s: float = 2.5,
        short_padding_min_source_duration_s: float = 0.5,
        short_padding_max_source_duration_s: float = 2.0,
        short_padding_min_score: float = 0.10,
        short_padding_score_scale: float = 1.0,
        minimum_proposal_duration_s: float = 2.0,
    ) -> None:
        if minimum_proposal_duration_s < 0:
            raise ValueError("minimum_proposal_duration_s must be non-negative")
        tags = []
        if use_adaptive_lambda:
            tags.append("adaptive")
        if use_spatial_compactness:
            tags.append("spatial")
        if use_noise_penalization:
            tags.append("noise")
        if use_dispersed_noise:
            tags.append("dispersed")
        if prototype is not None:
            tags.append("proto")
        if use_periodicity:
            tags.append("period")
        self._run_tag = "_".join(tags) if tags else "baseline"

        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"retag_{self._run_tag}_full.txt")
        with open(self.log_file, "w") as f:
            f.write(f"=== EXECUCIÓN RETAG {self._run_tag.upper()} ===\n")

        self.data_path = data_path
        self.bin_width = bin_width * 1e6  # s → µs
        self.percentile = percentile
        self.nms_threshold = nms_threshold

        self.use_adaptive_lambda = use_adaptive_lambda
        self.lambda_percentile = lambda_percentile
        self.lambda_delta = lambda_delta
        self.lambda_step = lambda_step

        self.use_spatial_compactness = use_spatial_compactness
        self.spatial_weight = spatial_weight
        self.compact_sigmoid_k = compact_sigmoid_k
        self.compact_sigmoid_d0 = compact_sigmoid_d0

        self.use_noise_penalization = use_noise_penalization
        self.noise_percentile = noise_percentile
        self.noise_min_duration = noise_min_duration
        self.noise_weight = noise_weight
        self.noise_variance_window = noise_variance_window

        self.use_dispersed_noise = use_dispersed_noise
        self.dispersed_noise_percentile = dispersed_noise_percentile
        self.dispersed_noise_dispersion_percentile = dispersed_noise_dispersion_percentile
        self.dispersed_noise_min_duration = dispersed_noise_min_duration
        self.dispersed_noise_weight = dispersed_noise_weight

        self.prototype = prototype
        self.prototype_weight = prototype_weight

        self.use_periodicity = use_periodicity
        self.periodicity_min_period_s = periodicity_min_period_s
        self.periodicity_max_period_s = periodicity_max_period_s
        self.periodicity_window_s = periodicity_window_s
        self.periodicity_global_threshold = periodicity_global_threshold
        self.periodicity_local_threshold = periodicity_local_threshold
        self.periodicity_min_duration_s = periodicity_min_duration_s
        self.periodicity_weight = periodicity_weight

        self.use_short_proposal_padding = use_short_proposal_padding
        self.short_padding_target_duration_s = short_padding_target_duration_s
        self.short_padding_min_source_duration_s = short_padding_min_source_duration_s
        self.short_padding_max_source_duration_s = short_padding_max_source_duration_s
        self.short_padding_min_score = short_padding_min_score
        self.short_padding_score_scale = short_padding_score_scale
        self.minimum_proposal_duration_us = minimum_proposal_duration_s * 1e6

        self.actioness_thresholds = np.arange(0.05, 1, 0.05)
        self.grouping_thresholds = np.arange(0.05, 1, 0.05)

    def process_recording(self, rec: str) -> dict:
        """Generate proposals for every ROI of one recording.

        Runs the whole stage-1 chain per ROI: event rate, robust normalisation, the
        enabled descriptors, the threshold and grouping sweep, the minimum duration and
        temporal NMS. Intermediate statistics go to the run log, which is what makes a
        variant auditable after the fact.

        Args:
            rec: recording name, i.e. a top-level group of the HDF5 file.

        Returns:
            Mapping from ROI id to a ``[n, 3]`` array of ``[t_start, t_end, score]``
            in microseconds.
        """
        rec_proposal_data = {}
        _log(self.log_file, f"\n[GRAVACIÓN] {rec}")

        with h5py.File(self.data_path, "r") as hf:
            data = hf[rec]
            for roi_id in data.keys():
                events = np.array(data[roi_id]["events"])
                roi_height = int(data[roi_id].attrs["height"])
                roi_width = int(data[roi_id].attrs["width"])

                _log(self.log_file, f"\n[ROI] {roi_id}")
                _log(self.log_file, f"n_eventos: {len(events)}")

                rate, bins = get_event_rate(events, self.bin_width)
                rate = apply_robust_min_max(rate, self.percentile)

                _log(self.log_file, f"taxa min/max: {rate.min()} / {rate.max()}")
                _log(self.log_file, f"taxa media: {rate.mean():.4f}")

                r_min, r_max = rate.min(), rate.max()
                r_t = (
                    (rate - r_min) / (r_max - r_min)
                    if r_max > r_min
                    else np.zeros_like(rate, dtype=np.float64)
                )

                # computed once: both compactness and dispersed noise consume it
                needs_compactness = self.use_spatial_compactness or self.use_dispersed_noise
                if needs_compactness:
                    compactness = get_spatial_compactness(
                        events, bins, roi_height, roi_width,
                        self.compact_sigmoid_k, self.compact_sigmoid_d0,
                    )
                    c_min, c_max = compactness.min(), compactness.max()
                    compactness_norm = (
                        (compactness - c_min) / (c_max - c_min)
                        if c_max > c_min
                        else np.zeros_like(compactness)
                    )

                if self.use_spatial_compactness:
                    combined = r_t * (1.0 + self.spatial_weight * compactness_norm)
                    _log(self.log_file, f"compacidade media: {compactness_norm.mean():.4f}")
                    _log(self.log_file, f"spatial_weight: {self.spatial_weight}")
                else:
                    combined = r_t

                if self.prototype is not None:
                    proto_score = get_prototype_score(
                        events, bins, self.prototype, roi_height, roi_width
                    )
                    combined = combined * (1.0 + self.prototype_weight * proto_score)
                    _log(self.log_file, f"proto_score media: {proto_score.mean():.4f}")
                    _log(self.log_file, f"proto_score max:   {proto_score.max():.4f}")
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
                    _log(self.log_file, f"bins periódicos: {int(periodicity.sum())} / {len(periodicity)}")
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
                    _log(self.log_file, f"bins de ruído: {int(noise.sum())} / {len(noise)}")
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
                    _log(self.log_file, f"bins ruído disperso: {int(disp_noise.sum())} / {len(disp_noise)}")
                    _log(self.log_file, f"dispersed_noise_weight: {self.dispersed_noise_weight}")

                a_min, a_max = combined.min(), combined.max()
                actioness = (
                    (combined - a_min) / (a_max - a_min)
                    if a_max > a_min
                    else np.zeros_like(combined)
                )

                _log(self.log_file, f"accionness min/max: {actioness.min():.4f} / {actioness.max():.4f}")
                _log(self.log_file, f"accionness media: {actioness.mean():.4f}")

                if self.use_adaptive_lambda:
                    thresholds = get_adaptive_actioness_thresholds(
                        actioness,
                        central_percentile=self.lambda_percentile,
                        delta=self.lambda_delta,
                        step=self.lambda_step,
                    )
                    _log(self.log_file, f"umbrais adaptativos: {thresholds.tolist()}")
                else:
                    thresholds = self.actioness_thresholds
                    _log(self.log_file, f"umbrais fixos: {thresholds.tolist()}")

                proposals = [
                    proposal
                    for at in thresholds
                    for gt in self.grouping_thresholds
                    for proposal in merge_proposals(
                        get_index_proposals_from_1d_score(actioness, at),
                        actioness, gt, bins,
                    )
                ]

                _log(self.log_file, f"propostas brutas: {len(proposals)}")

                if self.use_short_proposal_padding:
                    before_padding = len(proposals)
                    proposals = pad_short_proposals(
                        proposals,
                        target_duration_us=self.short_padding_target_duration_s * 1e6,
                        min_source_duration_us=self.short_padding_min_source_duration_s * 1e6,
                        max_source_duration_us=self.short_padding_max_source_duration_s * 1e6,
                        min_score=self.short_padding_min_score,
                        t_min=bins[0],
                        t_max=bins[-1],
                        score_scale=self.short_padding_score_scale,
                    )
                    _log(
                        self.log_file,
                        f"propostas curtas acolchadas: {len(proposals) - before_padding}",
                    )

                proposals = np.array(proposals)

                if len(proposals) > 0:
                    proposals = proposals[
                        proposals[:, 1] - proposals[:, 0]
                        > self.minimum_proposal_duration_us
                    ]
                    proposals = temporal_nms(proposals, self.nms_threshold)
                    _log(self.log_file, f"propostas tras NMS: {len(proposals)}")
                    if len(proposals) > 0:
                        durations = proposals[:, 1] - proposals[:, 0]
                        _log(self.log_file, f"duración media: {durations.mean():.2f}")
                        _log(self.log_file, f"duración mín:   {durations.min():.2f}")
                        _log(self.log_file, f"duración máx:   {durations.max():.2f}")
                    else:
                        _log(self.log_file, "propostas tras NMS: 0")
                else:
                    proposals = np.empty((0, 3))
                    _log(self.log_file, "propostas tras NMS: 0")

                rec_proposal_data[roi_id] = proposals

        return rec_proposal_data

    def run(
        self,
        split: Optional[str] = "test",
        recordings: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Generate proposals for a split, or for an explicit list of recordings.

        Recordings are processed in a process pool, one per worker.

        Args:
            split: split attribute to select recordings, or None for all of them.
            recordings: explicit recording names. When given, they are checked against
                ``split``, which is what keeps a cross-validation fold from silently
                pulling in a recording of another split.

        Returns:
            DataFrame with ``rec_name``, ``roi_id``, ``t_start``, ``t_end`` and ``score``.

        Raises:
            ValueError: if a requested recording is missing or belongs to another split.
        """
        logging.info("Executando o xerador de propostas.")

        with h5py.File(self.data_path, "r") as hf:
            if recordings is not None:
                selected = set(map(str, recordings))
                missing = selected - set(hf.keys())
                if missing:
                    raise ValueError(
                        f"Requested proposal recordings are missing: {sorted(missing)}"
                    )
                recording_names = sorted(selected)
                if split is not None:
                    wrong_split = [
                        rec
                        for rec in recording_names
                        if hf[rec].attrs.get("split") != split
                    ]
                    if wrong_split:
                        raise ValueError(
                            f"Explicit recordings do not belong to split {split!r}: "
                            f"{wrong_split[:10]}"
                        )
            else:
                recording_names = [
                    rec
                    for rec in hf.keys()
                    if split is None or hf[rec].attrs.get("split") == split
                ]

        _log(self.log_file, f"\n[INFO] gravacións: {recording_names}")

        with Pool(processes=16) as pool:
            results = pool.map(self.process_recording, recording_names)

        rows = [
            {
                "rec_name": os.path.splitext(rec)[0],
                "roi_id": roi_id,
                "t_start": proposal[0],
                "t_end": proposal[1],
                "score": proposal[2],
            }
            for rec, rec_proposals in zip(recording_names, results)
            for roi_id, proposals in rec_proposals.items()
            for proposal in proposals
        ]
        return pd.DataFrame(rows)

    def run_multiscale(
        self,
        bin_widths: list[float],
        split: Optional[str] = "test",
        merge_nms_threshold: float = 0.85,
    ) -> pd.DataFrame:
        """Generate proposals at several bin widths and merge them with NMS.

        A single bin width fixes the temporal resolution of the whole run, so short
        and long displays cannot both be favoured. Running the generator once per
        scale and merging keeps the best-scoring hypothesis of each.

        Args:
            bin_widths: bin widths in seconds, one run per value.
            split: split to process, or None for every recording.
            merge_nms_threshold: tIoU above which cross-scale duplicates are merged.

        Returns:
            DataFrame with the same columns as :meth:`run`.
        """
        import copy

        all_dfs = []
        for bw in bin_widths:
            gen = copy.copy(self)
            gen.bin_width = bw * 1e6
            all_dfs.append(gen.run(split=split))

        combined = pd.concat(all_dfs, ignore_index=True)

        merged_rows = []
        for (rec, roi), grp in combined.groupby(["rec_name", "roi_id"]):
            arr = grp[["t_start", "t_end", "score"]].values
            kept = temporal_nms(arr, merge_nms_threshold)
            for row in kept:
                merged_rows.append({
                    "rec_name": rec, "roi_id": roi,
                    "t_start": row[0], "t_end": row[1], "score": row[2],
                })

        return pd.DataFrame(merged_rows)
