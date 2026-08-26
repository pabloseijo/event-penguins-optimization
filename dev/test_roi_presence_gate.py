"""Focused tests for the cross-fit ROI presence gate."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dev.eval_roi_presence_gate_cv import (
    FEATURE_COLUMNS,
    fit_presence_model,
    presence_probabilities,
    roi_bag_features,
)


class RoiPresenceGateTest(unittest.TestCase):
    def test_bag_features_prioritize_high_scoring_evidence(self) -> None:
        weak = [
            {"score": 0.01, "segment": [0.0, 2.0]},
            {"score": 0.02, "segment": [3.0, 6.0]},
        ]
        strong = weak + [{"score": 0.8, "segment": [8.0, 14.0]}]
        weak_features = roi_bag_features(weak)
        strong_features = roi_bag_features(strong)

        self.assertGreater(
            strong_features["score_max"], weak_features["score_max"]
        )
        self.assertGreater(
            strong_features["top3_mean"], weak_features["top3_mean"]
        )
        self.assertEqual(set(strong_features), set(FEATURE_COLUMNS))

    def test_regularized_presence_model_separates_simple_bags(self) -> None:
        rows = []
        for target, offset in ((0.0, 0.0), (1.0, 1.0)):
            for index in range(8):
                row = {
                    name: offset + 0.01 * index
                    for name in FEATURE_COLUMNS
                }
                row["target_present"] = target
                rows.append(row)
        frame = pd.DataFrame(rows)
        model = fit_presence_model(frame, l2=0.01)
        probabilities = presence_probabilities(frame, model)

        self.assertGreater(
            float(probabilities[frame.target_present.eq(1)].mean()),
            float(probabilities[frame.target_present.eq(0)].mean()),
        )
        self.assertTrue(np.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
