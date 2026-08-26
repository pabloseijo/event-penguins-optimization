"""Tests for nested high-confidence negative ROI calibration."""

from __future__ import annotations

import unittest

import pandas as pd

from dev.eval_roi_presence_negative_gate_cv import select_nested_configuration


class RoiPresenceNegativeGateTest(unittest.TestCase):
    def test_nested_selection_rejects_active_roi_suppression(self) -> None:
        frame = pd.DataFrame(
            {
                "fold": [0, 1, 0, 1],
                "threshold": [0.1, 0.1, 0.2, 0.2],
                "suppression_factor": [0.0, 0.0, 0.0, 0.0],
                "suppressed_active_rois": [0, 0, 0, 1],
                "val_ed_instances": [10, 20, 10, 20],
                "mAP": [0.8, 0.9, 0.95, 0.95],
                "AP@0.7": [0.7, 0.8, 0.9, 0.9],
            }
        )
        control = pd.DataFrame(
            {
                "fold": [0, 1],
                "val_ed_instances": [10, 20],
                "mAP": [0.75, 0.85],
                "AP@0.7": [0.65, 0.75],
            }
        )
        selected = select_nested_configuration(frame, control)
        self.assertEqual(selected["threshold"], 0.1)
        self.assertEqual(selected["suppression_factor"], 0.0)


if __name__ == "__main__":
    unittest.main()
