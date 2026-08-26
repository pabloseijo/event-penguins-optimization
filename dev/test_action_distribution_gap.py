"""Tests for the action-distribution diagnostic."""

from __future__ import annotations

import unittest

import pandas as pd

from dev.analyze_action_distribution_gap import recording_statistics, union_duration


class ActionDistributionGapTest(unittest.TestCase):
    def test_union_duration_merges_overlaps(self) -> None:
        self.assertAlmostEqual(
            union_duration([(1.0, 4.0), (3.0, 7.0), (10.0, 12.0)]),
            8.0,
        )

    def test_recording_statistics_filters_and_clips_segments(self) -> None:
        sequences = pd.DataFrame(
            [
                {"roi_id": 1, "duration_s": 10.0},
                {"roi_id": 2, "duration_s": 10.0},
            ]
        )
        annotations = {
            "annotations": {
                "1": [
                    {"label": "ed", "segment": [1.0, 4.0]},
                    {"label": "ed", "segment": [3.0, 7.0]},
                    {"label": "other", "segment": [0.0, 10.0]},
                ],
                "2": [
                    {"label": "ed", "segment": [9.0, 12.0]},
                    {"label": "ed", "segment": [5.0, 5.5]},
                ],
            }
        }
        stats = recording_statistics("sample", sequences, annotations, min_duration=1.0)

        self.assertEqual(stats["gt_instances"], 3)
        self.assertAlmostEqual(stats["active_roi_fraction"], 1.0)
        self.assertAlmostEqual(stats["action_fraction"], 7.0 / 20.0)
        self.assertAlmostEqual(stats["median_duration_s"], 3.0)
        self.assertAlmostEqual(stats["median_gap_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
