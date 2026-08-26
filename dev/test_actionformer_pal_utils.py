"""Tests for the supervised PAL helpers: background sampling and contrastive loss.

Checks that background starts keep the required margin from every annotated
segment and that the contrastive loss prefers aligned pairs.
"""

import unittest

import numpy as np
import torch

from actionformer_pal_utils import (
    pal_contrastive_loss,
    valid_background_starts,
)


class ActionFormerPalUtilsTest(unittest.TestCase):
    def test_background_starts_respect_segments_and_margin(self):
        starts = valid_background_starts(
            np.asarray([[3.0, 5.0]]),
            sequence_length=10,
            crop_length=2,
            margin=1,
        )
        np.testing.assert_array_equal(starts, [0, 6, 7, 8])

    def test_contrastive_loss_prefers_aligned_pairs(self):
        recipient = torch.eye(2)
        aligned = pal_contrastive_loss(recipient, torch.eye(2), 0.07)
        swapped = pal_contrastive_loss(
            recipient, torch.flip(torch.eye(2), dims=(0,)), 0.07
        )
        self.assertLess(float(aligned), float(swapped))

    def test_single_pair_uses_cosine_distance(self):
        loss = pal_contrastive_loss(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            0.07,
        )
        self.assertAlmostEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
