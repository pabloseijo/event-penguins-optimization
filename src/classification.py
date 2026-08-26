"""Stage 2: proposal classification with the Augmented TSN.

Each proposal is turned into a fixed number of image-like representations of its
event window (exponentially decayed time surfaces), classified by the ATSN and
reduced to a single detection score. The stage also owns the post-processing
added in phase 2 of the project: temperature scaling, Platt calibration, a
duration prior, score fusion with the proposal score and Soft-NMS.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from absl import logging
from torch.utils.data import Dataset, DataLoader
import h5py

from .augmented_tsn import AugmentedTsn
from .utils import temporal_nms, temporal_soft_nms


def range_norm(
    matrix: np.ndarray,
    new_max: float = 255,
    lower: float = None,
    upper: float = None,
    dtype=None,
) -> np.ndarray:
    """Linearly rescale a matrix into ``[0, new_max]`` after clipping.

    Args:
        matrix: values to rescale.
        new_max: upper bound of the output range.
        lower: clipping floor; defaults to the matrix minimum.
        upper: clipping ceiling; defaults to the matrix maximum.
        dtype: optional output dtype, typically ``np.uint8`` for image channels.

    Returns:
        The rescaled matrix.
    """
    lower = matrix.min() if lower is None else lower
    upper = matrix.max() if upper is None else upper
    scaled = new_max * (np.clip(matrix, lower, upper) - lower) / (upper - lower)
    return scaled.astype(dtype) if dtype is not None else scaled


def create_time_map(
    events: np.ndarray,
    decay: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Build a signed, exponentially decayed time surface from an event window.

    Each pixel keeps the timestamp of its most recent event, decayed relative to the
    end of the window and signed by polarity, so recent ON events approach ``+1`` and
    recent OFF events ``-1``.

    Args:
        events: ``[n, 4]`` array of ``[x, y, t, p]``.
        decay: decay rate in inverse microseconds.
        height: ROI height in pixels.
        width: ROI width in pixels.

    Returns:
        ``[height, width]`` time surface.
    """
    time_map = np.zeros((height, width))
    time_map[events[:, 1], events[:, 0]] = events[:, 2]

    current_t = events[:, 2].max() if len(events) > 0 else 0
    time_map = np.exp(-decay * (current_t - time_map))

    polarity = events[:, 3].copy().astype(int)
    polarity[polarity == 0] = -1
    time_map[events[:, 1], events[:, 0]] *= polarity

    return time_map


def create_img_representation(
    events: np.ndarray,
    decay: float,
    height: int,
    width: int,
    transform=None,
) -> np.ndarray:
    """Render one event window as a 224x224 three-channel ATSN input.

    The signed time surface is repeated across the three channels, which is what the
    pretrained ATSN weights expect.

    Args:
        events: ``[n, 4]`` array of ``[x, y, t, p]``.
        decay: decay rate in inverse microseconds.
        height: ROI height in pixels.
        width: ROI width in pixels.
        transform: optional torchvision transform applied to the result.

    Returns:
        The rendered image, transformed when a transform was given.
    """
    img = create_time_map(events, decay, height, width)
    img = range_norm(img, lower=-1, upper=1, dtype=np.uint8)
    img = np.repeat(img[..., None], 3, axis=2)
    img = Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)
    return transform(img) if transform is not None else img


def create_polarity_img_representation(
    events: np.ndarray,
    decay: float,
    height: int,
    width: int,
    transform=None,
) -> np.ndarray:
    """Build signed, ON-recency, and OFF-recency channels for one event window."""
    signed = create_time_map(events, decay, height, width)
    signed_u8 = range_norm(signed, lower=-1, upper=1, dtype=np.uint8)
    on = np.zeros((height, width), dtype=np.float64)
    off = np.zeros((height, width), dtype=np.float64)
    if len(events) > 0:
        current_t = float(events[:, 2].max())
        for polarity_mask, output in ((events[:, 3] > 0, on), (events[:, 3] <= 0, off)):
            polarity_events = events[polarity_mask]
            if len(polarity_events) == 0:
                continue
            timestamps = np.zeros((height, width), dtype=np.float64)
            active = np.zeros((height, width), dtype=bool)
            y = polarity_events[:, 1].astype(np.int64)
            x = polarity_events[:, 0].astype(np.int64)
            timestamps[y, x] = polarity_events[:, 2]
            active[y, x] = True
            output[active] = np.exp(-decay * (current_t - timestamps[active]))
    on_u8 = range_norm(on, lower=0, upper=1, dtype=np.uint8)
    off_u8 = range_norm(off, lower=0, upper=1, dtype=np.uint8)
    img = np.stack((signed_u8, on_u8, off_u8), axis=2)
    img = Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)
    return transform(img) if transform is not None else img


def create_multiscale_decay_img_representation(
    events: np.ndarray,
    decays: tuple[float, float, float],
    height: int,
    width: int,
    transform=None,
) -> np.ndarray:
    """Build three signed time-surfaces with different temporal memories."""
    if len(decays) != 3:
        raise ValueError("Exactly three decay values are required")
    channels = [
        range_norm(
            create_time_map(events, decay, height, width),
            lower=-1,
            upper=1,
            dtype=np.uint8,
        )
        for decay in decays
    ]
    img = np.stack(channels, axis=2)
    img = Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)
    return transform(img) if transform is not None else img


class ProposalDataset(Dataset):
    """Torch dataset that renders the temporal samples of each proposal.

    A proposal is sampled at ``num_tsn_samples`` evenly spaced instants, extended on
    both sides by the augmentation fraction so the classifier also sees the context
    around the proposal. The HDF5 handle and the events of the current ROI are cached
    per worker, since consecutive proposals almost always share a ROI.
    """
    # class attribute, to avoid shadowing the torchvision transforms module
    _transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        proposals,
        augment_fraction: float,
        data_path: str,
        num_tsn_samples: int,
        sample_duration: float,
        decay: float,
        cache_full_events: bool = True,
        timestamp_cache_dir: str | None = None,
    ):
        self.proposals = proposals
        self.augment_fraction = augment_fraction
        self.data_path = data_path
        self.num_tsn_samples = num_tsn_samples
        self.sample_duration = sample_duration
        self.decay = decay
        self.cache_full_events = cache_full_events
        self.timestamp_cache_dir = (
            Path(timestamp_cache_dir) if timestamp_cache_dir is not None else None
        )
        self._hf = None
        self._cached_roi_key = None
        self._cached_roi_events = None
        self._cached_roi_timestamps = None
        self._cached_roi_height = None
        self._cached_roi_width = None

    def _get_h5(self):
        if self._hf is None:
            self._hf = h5py.File(self.data_path, "r")
        return self._hf

    def _get_roi_data(self, rec_name, roi_id):
        key = (rec_name, roi_id)
        if key != self._cached_roi_key:
            roi_group = self._get_h5()[rec_name][roi_id]
            events = roi_group["events"]
            if self.cache_full_events:
                self._cached_roi_events = np.asarray(events)
                self._cached_roi_timestamps = self._cached_roi_events[:, 2]
            else:
                self._cached_roi_events = events
                cache_path = None
                if self.timestamp_cache_dir is not None:
                    cache_path = self.timestamp_cache_dir / str(rec_name) / f"{roi_id}.npy"
                if cache_path is not None and cache_path.exists():
                    self._cached_roi_timestamps = np.load(cache_path, mmap_mode="r")
                else:
                    timestamps = np.asarray(events[:, 2])
                    if cache_path is not None:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = cache_path.with_suffix(f".{os.getpid()}.npy.tmp")
                        with temporary.open("wb") as stream:
                            np.save(stream, timestamps)
                        temporary.replace(cache_path)
                    self._cached_roi_timestamps = timestamps
            self._cached_roi_height = roi_group.attrs["height"]
            self._cached_roi_width = roi_group.attrs["width"]
            self._cached_roi_key = key
        return (
            self._cached_roi_events,
            self._cached_roi_timestamps,
            self._cached_roi_height,
            self._cached_roi_width,
        )

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, idx):
        t_start = self.proposals.loc[idx, "t_start"]
        t_end = self.proposals.loc[idx, "t_end"]
        rec_name = self.proposals.loc[idx, "rec_name"]
        roi_id = self.proposals.loc[idx, "roi_id"]

        roi_events, roi_timestamps, height, width = self._get_roi_data(rec_name, roi_id)

        t_delta = t_end - t_start
        t_aug_start = t_start - t_delta * self.augment_fraction
        t_aug_end = t_end + t_delta * self.augment_fraction

        img_times = torch.linspace(t_aug_start, t_aug_end, self.num_tsn_samples)
        sample_duration = (
            float(self.proposals.loc[idx, "sample_duration"])
            if "sample_duration" in self.proposals.columns
            else self.sample_duration
        )
        t_imgs_start = img_times - 0.5 * sample_duration
        t_imgs_end = img_times + 0.5 * sample_duration

        i_start = np.searchsorted(roi_timestamps, t_imgs_start)
        i_end = np.searchsorted(roi_timestamps, t_imgs_end)

        imgs = torch.stack([
            create_img_representation(
                np.asarray(roi_events[s:e]), self.decay, height, width, self._transform
            )
            for s, e in zip(i_start, i_end)
        ])
        proposal_score = float(self.proposals.loc[idx, "score"]) if "score" in self.proposals.columns else 0.0
        return imgs, rec_name, roi_id, t_start, t_end, proposal_score


class ProposalClassifier:
    """Score, calibrate and suppress proposals with the Augmented TSN.

    The ATSN weights are used exactly as released; everything that changed in phase 2
    lives around the model. Temperature and Platt parameters recalibrate the logits,
    ``score_fusion_weight`` mixes in the stage-1 proposal score, the duration prior
    penalises implausibly long detections and Soft-NMS replaces hard suppression
    when two displays are adjacent in time.

    Args:
        device: torch device the model runs on.
        model_path: pickled ATSN state dict.
        num_tsn_samples: temporal samples taken inside the proposal.
        augment_factor: inverse of the context fraction added on each side.
        data_path: HDF5 file produced by ``scripts/preprocess.py``.
        sample_duration: duration of each sampled event window, in seconds.
        decay: time-surface decay rate in inverse microseconds.
        nms_threshold: tIoU above which detections are suppressed.
        batch_size: inference batch size.
        use_soft_nms: decay overlapping scores instead of removing detections.
        soft_nms_sigma: width of the Soft-NMS gaussian.
        soft_nms_score_threshold: score below which a decayed detection is dropped.
        score_fusion_weight: weight of the stage-1 proposal score in the final score.
        max_duration_filter: drop detections longer than this many seconds.
        min_ed_score: score below which a detection is discarded.
        temperature: temperature scaling applied to the logits.
        duration_penalty_dmax: duration beyond which the prior starts penalising.
        duration_penalty_sigma: width of the duration penalty.
        platt_a: slope of the Platt calibration.
        platt_b: intercept of the Platt calibration.
        num_workers: dataloader workers.
    """

    def __init__(
        self,
        device,
        model_path: str,
        num_tsn_samples: int,
        augment_factor: int,
        data_path: str,
        sample_duration: float,
        decay: float,
        nms_threshold: float,
        batch_size: int,
        use_soft_nms: bool = False,
        soft_nms_sigma: float = 0.5,
        soft_nms_score_threshold: float = 0.001,
        score_fusion_weight: float = 0.0,
        max_duration_filter: float = None,
        min_ed_score: float = 0.5,
        temperature: float = 1.0,
        duration_penalty_dmax: float = None,
        duration_penalty_sigma: float = 20.0,
        platt_a: float = 1.0,
        platt_b: float = 0.0,
        num_workers: int = 16,
    ) -> None:
        self.device = device
        self.augment_fraction = 1 / augment_factor
        num_aug_samples = int(np.ceil(self.augment_fraction * num_tsn_samples))
        # segmento principal + aumentación esquerda + dereita
        self.num_tsn_samples = num_tsn_samples + 2 * num_aug_samples

        self.data_path = data_path
        self.sample_duration = 1e6 * sample_duration  # s → µs
        self.decay = float(decay)
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size
        self.use_soft_nms = use_soft_nms
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_score_threshold = soft_nms_score_threshold
        self.score_fusion_weight = score_fusion_weight
        self.max_duration_filter = max_duration_filter
        self.min_ed_score = min_ed_score
        self.temperature = temperature
        self.duration_penalty_dmax = duration_penalty_dmax
        self.duration_penalty_sigma = duration_penalty_sigma
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.num_workers = num_workers

        self.model = AugmentedTsn(2, num_tsn_samples, augment_factor)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()

    def run(self, proposals) -> dict:
        """Classify proposals and return calibrated, suppressed detections.

        Args:
            proposals: DataFrame from ``ProposalGenerator.run``.

        Returns:
            ActivityNet-style ``{"version": ..., "results": {recording: [detections]}}``
            dictionary, ready to be dumped as ``predictions.json``.
        """
        logging.info("Executando o clasificador de propostas.")

        dataset = ProposalDataset(
            proposals,
            self.augment_fraction,
            self.data_path,
            self.num_tsn_samples,
            self.sample_duration,
            self.decay,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        result = {
            rec: {roi: [] for roi in proposals[proposals["rec_name"] == rec]["roi_id"].unique()}
            for rec in proposals["rec_name"].unique()
        }

        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for imgs, rec_names, roi_ids, t_starts, t_ends, prop_scores in tqdm(loader):
                outputs = self.model(imgs.to(self.device))
                ed_scores = softmax(outputs / self.temperature)[:, 1]

                for i in range(len(ed_scores)):
                    score = float(ed_scores[i])
                    if score >= self.min_ed_score:
                        if self.score_fusion_weight > 0:
                            score *= (1.0 + self.score_fusion_weight * float(prop_scores[i]))
                        if self.duration_penalty_dmax is not None:
                            duration_s = (float(t_ends[i]) - float(t_starts[i])) / 1e6
                            excess = max(0.0, duration_s - self.duration_penalty_dmax)
                            score *= float(np.exp(-excess / self.duration_penalty_sigma))
                        result[rec_names[i]][roi_ids[i]].append([
                            float(t_starts[i]),
                            float(t_ends[i]),
                            score,
                        ])

        nmsed = {}
        for rec_name, rec_results in result.items():
            nmsed[rec_name] = {}
            for roi_id, roi_result in rec_results.items():
                if roi_result:
                    arr = np.array(roi_result)
                    processed = (
                        temporal_soft_nms(
                            arr,
                            sigma=self.soft_nms_sigma,
                            score_threshold=self.soft_nms_score_threshold,
                        )
                        if self.use_soft_nms
                        else temporal_nms(arr, self.nms_threshold)
                    )
                    if self.max_duration_filter is not None and len(processed) > 0:
                        max_dur_us = self.max_duration_filter * 1e6
                        processed = processed[processed[:, 1] - processed[:, 0] <= max_dur_us]
                else:
                    processed = []
                nmsed[rec_name][int(roi_id[1:])] = [
                    {
                        "label": "ed",
                        "segment": [a[0] / 1e6, a[1] / 1e6],
                        "score": a[2],
                    }
                    for a in processed
                ]

        return {"version": "VERSION 0.0", "results": nmsed}

    def collect_logits(self, proposals) -> tuple:
        """Return the raw ATSN logits and their proposal metadata.

        Calibration is fitted on these logits, which is why they are collected
        without any temperature, prior or suppression applied.

        Args:
            proposals: DataFrame from ``ProposalGenerator.run``.

        Returns:
            Tuple of the ``[n, num_classes]`` logit array and a list of
            ``(rec_name, roi_id, t_start, t_end)`` tuples.
        """
        logging.info("Recollendo logits para axuste de temperatura.")

        dataset = ProposalDataset(
            proposals,
            self.augment_fraction,
            self.data_path,
            self.num_tsn_samples,
            self.sample_duration,
            self.decay,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        all_logits = []
        all_meta = []

        with torch.no_grad():
            for imgs, rec_names, roi_ids, t_starts, t_ends, _ in tqdm(loader):
                outputs = self.model(imgs.to(self.device))
                all_logits.append(outputs.cpu().numpy())
                for i in range(len(t_starts)):
                    all_meta.append((rec_names[i], roi_ids[i], float(t_starts[i]), float(t_ends[i])))

        return np.concatenate(all_logits, axis=0), all_meta
