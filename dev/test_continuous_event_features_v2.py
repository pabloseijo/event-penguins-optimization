"""Focused tests for aligned continuous ON/OFF and spectral descriptors."""

from __future__ import annotations

import unittest

import numpy as np

from dev.extract_continuous_event_features_v2 import FEATURE_NAMES, sequence_features


class ContinuousEventFeaturesV2Test(unittest.TestCase):
    def test_synthetic_periodic_events_are_finite_and_aligned(self) -> None:
        timestamps = np.arange(0.0, 10.0, 0.25)
        events = np.column_stack(
            (
                np.ones(len(timestamps)),
                np.ones(len(timestamps)),
                timestamps * 1e6,
                np.where(np.arange(len(timestamps)) % 2 == 0, 1, 0),
            )
        )
        features = sequence_features(events, 20, 0.5, 1.0, 4, 4)
        self.assertEqual(features.shape, (20, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(np.abs(features).sum()), 0.0)

    def test_empty_sequence_is_zero_and_finite(self) -> None:
        features = sequence_features(np.empty((0, 4)), 12, 0.5, 1.0, 4, 4)
        self.assertEqual(features.shape, (12, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.allclose(features, 0.0))


if __name__ == "__main__":
    unittest.main()
