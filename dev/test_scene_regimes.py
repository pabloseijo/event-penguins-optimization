"""Tests for the label-free SAVS-style regime diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

from dev.analyze_scene_regimes import semantic_change_scores, select_boundaries


class SceneRegimeTest(unittest.TestCase):
    def test_abrupt_orthogonal_transition_is_recovered(self) -> None:
        features = np.zeros((100, 4), dtype=np.float64)
        features[:50, 0] = 1.0
        features[50:, 1] = 1.0

        scores = semantic_change_scores(features, window=10)
        boundaries, zscores = select_boundaries(scores, threshold_z=3.0, min_distance=20)

        self.assertEqual(boundaries.tolist(), [50])
        self.assertGreater(zscores[50], 3.0)

    def test_constant_sequence_has_no_boundary(self) -> None:
        features = np.ones((100, 4), dtype=np.float64)
        scores = semantic_change_scores(features, window=10)
        boundaries, _ = select_boundaries(scores, threshold_z=3.0, min_distance=20)
        self.assertEqual(boundaries.tolist(), [])


if __name__ == "__main__":
    unittest.main()
