"""Tests for frozen negative ROI gate test inference."""

from __future__ import annotations

import unittest

from dev.eval_roi_presence_gate_cv import FEATURE_COLUMNS
from dev.eval_roi_presence_negative_gate_test import unlabeled_prediction_bags


class RoiPresenceNegativeGateTestInferenceTest(unittest.TestCase):
    def test_bag_builder_does_not_require_or_emit_labels(self) -> None:
        prediction = {
            "results": {
                "rec": {
                    "2": [
                        {"score": 0.8, "segment": [1.0, 3.0]},
                    ]
                }
            }
        }
        frame = unlabeled_prediction_bags(prediction)
        self.assertEqual(len(frame), 1)
        self.assertNotIn("target_present", frame)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(frame.columns))


if __name__ == "__main__":
    unittest.main()
