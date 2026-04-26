"""Proposal classification stage of the reTAG pipeline.

Converts each temporal proposal into a time-surface image representation,
runs it through AugmentedTSN, applies NMS, and returns JSON-formatted
detections.
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from absl import logging
from torch.utils.data import Dataset, DataLoader
import h5py

from .augmented_tsn import AugmentedTsn
from .utils import temporal_nms


def range_norm(
    matrix: np.ndarray,
    new_max: float = 255,
    lower: float = None,
    upper: float = None,
    dtype=None,
) -> np.ndarray:
    """Linearly rescale *matrix* to [0, new_max] after clipping to [lower, upper].

    Args:
        matrix: Input array.
        new_max: Target maximum value after scaling.
        lower: Clip minimum (defaults to matrix min).
        upper: Clip maximum (defaults to matrix max).
        dtype: Cast result to this dtype if provided.

    Returns:
        Scaled array of the same shape as *matrix*.
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
    """Build an exponential time-surface image from a set of events.

    Each pixel stores the timestamp of the last event that fell there,
    converted to an exponential decay score. Polarity is encoded as the sign.

    Args:
        events: Array of shape (N, 4) with columns [x, y, t, p].
        decay: Decay constant for the exponential (larger = faster decay).
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Time-surface array of shape (height, width).
    """
    time_map = np.zeros((height, width))
    time_map[events[:, 1], events[:, 0]] = events[:, 2]

    current_t = events[:, 2].max() if len(events) > 0 else 0
    time_map  = np.exp(-decay * (current_t - time_map))

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
    """Convert events to a 224×224 RGB time-surface image.

    Args:
        events: Array of shape (N, 4) with columns [x, y, t, p].
        decay: Exponential decay constant.
        height: ROI height in pixels.
        width: ROI width in pixels.
        transform: Optional torchvision transform applied after conversion.

    Returns:
        Image array (or tensor if *transform* is set).
    """
    img = create_time_map(events, decay, height, width)
    img = range_norm(img, lower=-1, upper=1, dtype=np.uint8)
    img = np.repeat(img[..., None], 3, axis=2)
    img = Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR)
    img = np.array(img)
    return transform(img) if transform is not None else img


class ProposalDataset(Dataset):
    """PyTorch Dataset that builds TSN image stacks for each proposal.

    Args:
        proposals: DataFrame with columns [rec_name, roi_id, t_start, t_end].
        augment_fraction: Fraction of the proposal duration added on each side.
        data_path: Path to the preprocessed HDF5 file.
        num_tsn_samples: Number of temporal samples per proposal.
        sample_duration: Duration (µs) of each image window.
        decay: Exponential decay constant for the time surface.
    """

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
    ):
        self.proposals       = proposals
        self.augment_fraction = augment_fraction
        self.data_path       = data_path
        self.num_tsn_samples = num_tsn_samples
        self.sample_duration = sample_duration
        self.decay           = decay

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, idx):
        t_start  = self.proposals.loc[idx, "t_start"]
        t_end    = self.proposals.loc[idx, "t_end"]
        rec_name = self.proposals.loc[idx, "rec_name"]
        roi_id   = self.proposals.loc[idx, "roi_id"]

        with h5py.File(self.data_path, "r") as hf:
            roi_events = np.array(hf[rec_name][roi_id]["events"])
            height     = hf[rec_name][roi_id].attrs["height"]
            width      = hf[rec_name][roi_id].attrs["width"]

        t_delta    = t_end - t_start
        t_aug_start = t_start - t_delta * self.augment_fraction
        t_aug_end   = t_end   + t_delta * self.augment_fraction

        img_times   = torch.linspace(t_aug_start, t_aug_end, self.num_tsn_samples)
        t_imgs_start = img_times - 0.5 * self.sample_duration
        t_imgs_end   = img_times + 0.5 * self.sample_duration

        i_start = np.searchsorted(roi_events[:, 2], t_imgs_start)
        i_end   = np.searchsorted(roi_events[:, 2], t_imgs_end)

        imgs = torch.stack([
            create_img_representation(
                roi_events[s:e], self.decay, height, width, self._transform
            )
            for s, e in zip(i_start, i_end)
        ])
        return imgs, rec_name, roi_id, t_start, t_end


class ProposalClassifier:
    """Run AugmentedTSN inference on a proposal set and apply NMS.

    Args:
        device: Torch device for inference.
        model_path: Path to the saved model state dict.
        num_tsn_samples: Number of TSN samples in the main segment.
        augment_factor: Controls the number of augmentation frames per side.
        data_path: Path to the preprocessed HDF5 file.
        sample_duration: Duration in seconds of each image window.
        decay: Exponential decay constant.
        nms_threshold: IoU threshold for temporal NMS.
        batch_size: Inference batch size.
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
    ) -> None:
        self.device           = device
        self.augment_fraction = 1 / augment_factor
        num_aug_samples       = int(np.ceil(self.augment_fraction * num_tsn_samples))
        # Total samples = main + left-augment + right-augment
        self.num_tsn_samples  = num_tsn_samples + 2 * num_aug_samples

        self.data_path     = data_path
        self.sample_duration = 1e6 * sample_duration  # s → µs
        self.decay         = float(decay)
        self.nms_threshold = nms_threshold
        self.batch_size    = batch_size

        self.model = AugmentedTsn(2, num_tsn_samples, augment_factor)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()

    def run(self, proposals) -> dict:
        """Classify proposals and return detections in reTAG JSON format.

        Args:
            proposals: DataFrame with columns [rec_name, roi_id, t_start, t_end].

        Returns:
            Dict with keys "version" and "results" (nested by rec_name → roi_id).
        """
        logging.info("Running Proposal Classifier.")

        dataset = ProposalDataset(
            proposals,
            self.augment_fraction,
            self.data_path,
            self.num_tsn_samples,
            self.sample_duration,
            self.decay,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        # Initialise result buckets for every (rec, roi) pair
        result = {
            rec: {roi: [] for roi in proposals[proposals["rec_name"] == rec]["roi_id"].unique()}
            for rec in proposals["rec_name"].unique()
        }

        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for imgs, rec_names, roi_ids, t_starts, t_ends in tqdm(loader):
                outputs   = self.model(imgs.to(self.device))
                preds     = outputs.argmax(dim=1)
                ed_scores = softmax(outputs)[:, 1]

                for i, pred in enumerate(preds):
                    if pred.item():
                        result[rec_names[i]][roi_ids[i]].append([
                            float(t_starts[i]),
                            float(t_ends[i]),
                            float(ed_scores[i]),
                        ])

        # Apply temporal NMS per (rec, roi)
        nmsed = {}
        for rec_name, rec_results in result.items():
            nmsed[rec_name] = {}
            for roi_id, roi_result in rec_results.items():
                processed = (
                    temporal_nms(np.array(roi_result), self.nms_threshold)
                    if roi_result else []
                )
                nmsed[rec_name][int(roi_id[1:])] = [
                    {
                        "label":   "ed",
                        "segment": [a[0] / 1e6, a[1] / 1e6],
                        "score":   a[2],
                    }
                    for a in processed
                ]

        return {"version": "VERSION 0.0", "results": nmsed}
