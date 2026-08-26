"""Tests for density-adaptive sample durations in the proposal dataset.

Checks that the window length tracks event density, stays inside its bounds,
rejects invalid arguments and lets a per-row duration override the default.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from dev.extract_continuous_features import adaptive_sample_durations
from src.classification import ProposalDataset


class AdaptiveSampleDurationTests(unittest.TestCase):
    def test_duration_tracks_density_and_respects_bounds(self) -> None:
        counts = np.asarray([0.0, 50.0, 100.0, 200.0, 1000.0])
        durations = adaptive_sample_durations(
            np.log1p(counts),
            target_count=100.0,
            min_duration=0.5,
            max_duration=2.0,
        )
        np.testing.assert_allclose(durations, [2.0, 2.0, 1.0, 0.5, 0.5])

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_sample_durations(np.asarray([1.0]), 0.0, 0.5, 2.0)
        with self.assertRaises(ValueError):
            adaptive_sample_durations(np.asarray([-1.0]), 100.0, 0.5, 2.0)
        with self.assertRaises(ValueError):
            adaptive_sample_durations(np.asarray([np.nan]), 100.0, 0.5, 2.0)


class ProposalDatasetAdaptiveWindowTests(unittest.TestCase):
    def test_row_duration_overrides_default_duration(self) -> None:
        proposals = pd.DataFrame(
            {
                "rec_name": ["recording"],
                "roi_id": ["N1"],
                "t_start": [500.0],
                "t_end": [500.0],
                "sample_duration": [600.0],
            }
        )
        dataset = ProposalDataset(
            proposals=proposals,
            augment_fraction=0.0,
            data_path="unused.h5",
            num_tsn_samples=1,
            sample_duration=200.0,
            decay=1.0,
        )
        events = np.column_stack(
            (
                np.zeros(11),
                np.zeros(11),
                np.arange(0.0, 1100.0, 100.0),
                np.ones(11),
            )
        )
        dataset._get_roi_data = lambda *_: (events, events[:, 2], 1, 1)

        def encode_count(selected, *_args, **_kwargs):
            return torch.tensor([float(len(selected))])

        with patch("src.classification.create_img_representation", side_effect=encode_count):
            images = dataset[0][0]

        self.assertEqual(images.shape, (1, 1))
        self.assertEqual(images.item(), 6.0)

        fixed_dataset = ProposalDataset(
            proposals=proposals.drop(columns="sample_duration"),
            augment_fraction=0.0,
            data_path="unused.h5",
            num_tsn_samples=1,
            sample_duration=200.0,
            decay=1.0,
        )
        fixed_dataset._get_roi_data = lambda *_: (events, events[:, 2], 1, 1)
        with patch("src.classification.create_img_representation", side_effect=encode_count):
            fixed_images = fixed_dataset[0][0]
        self.assertEqual(fixed_images.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
