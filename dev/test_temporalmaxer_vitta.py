"""Tests for continuous ViTTA primitives."""

from __future__ import annotations

import unittest

import torch

from dev.eval_temporalmaxer_vitta_cv import (
    localization_consistency,
    masked_channel_moments,
    moments_to_mean_variance,
    shift_temporal_view,
)


class TemporalMaxerViTTATest(unittest.TestCase):
    def test_masked_moments_ignore_padding(self) -> None:
        features = torch.tensor([[[1.0, 3.0, 99.0], [2.0, 4.0, 99.0]]])
        mask = torch.tensor([[True, True, False]])
        total, square, count = masked_channel_moments(features, mask)
        mean, variance = moments_to_mean_variance(total, square, count)
        self.assertTrue(torch.allclose(mean, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(variance, torch.tensor([1.0, 1.0])))

    def test_shifted_view_preserves_length_and_uses_previous_point(self) -> None:
        features = torch.arange(12).reshape(1, 4, 3)
        shifted = shift_temporal_view(features)
        self.assertTrue(torch.equal(shifted[:, 0], features[:, 0]))
        self.assertTrue(torch.equal(shifted[:, 1:], features[:, :-1]))

    def test_localization_consistency_aligns_shifted_offsets(self) -> None:
        first = {
            "offsets": [
                torch.tensor([[[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]]])
            ]
        }
        second = {
            "offsets": [
                torch.tensor([[[9.0, 9.0], [1.0, 2.0], [2.0, 3.0]]])
            ]
        }
        mask = torch.ones(1, 3, dtype=torch.bool)
        self.assertEqual(float(localization_consistency(first, second, mask)), 0.0)


if __name__ == "__main__":
    unittest.main()
