"""Tests for the boundary-reliability router between two prediction sets.

Checks the routing directions, that merging swaps only the selected recordings,
and that the safeguards demand no degradation plus at least one gain.
"""

from __future__ import annotations

import unittest

import pandas as pd

from dev.eval_event_boundary_reliability_router_cv import (
    merge_recording_predictions,
    routed_recordings,
    safeguards_pass,
)


class EventBoundaryReliabilityRouterTest(unittest.TestCase):
    def test_routing_directions(self) -> None:
        features = pd.DataFrame(
            {
                "rec_name": ["a", "b", "c"],
                "uncertainty_mean": [0.1, 0.2, 0.3],
            }
        )
        self.assertEqual(
            routed_recordings(features, "uncertainty_mean", "low", 0.2),
            ["a", "b"],
        )
        self.assertEqual(
            routed_recordings(features, "uncertainty_mean", "high", 0.2),
            ["b", "c"],
        )

    def test_merge_swaps_only_selected_recordings(self) -> None:
        control = {
            "version": "control",
            "results": {"a": {"1": [{"score": 1}]}, "b": {"1": [{"score": 2}]}},
        }
        alternative = {
            "version": "alternative",
            "results": {"a": {"1": [{"score": 3}]}, "b": {"1": [{"score": 4}]}},
        }
        merged = merge_recording_predictions(control, alternative, ["b"], "router")
        self.assertEqual(merged["results"]["a"]["1"][0]["score"], 1)
        self.assertEqual(merged["results"]["b"]["1"][0]["score"], 4)

    def test_safeguards_require_non_degradation_and_one_gain(self) -> None:
        control = {
            "mean_mAP": 0.8,
            "weighted_mAP": 0.8,
            "worst_mAP": 0.7,
            "mean_AP@0.7": 0.6,
        }
        gain = {**control, "mean_mAP": 0.81}
        loss = {**gain, "mean_AP@0.7": 0.59}
        self.assertTrue(safeguards_pass(gain, control))
        self.assertFalse(safeguards_pass(control, control))
        self.assertFalse(safeguards_pass(loss, control))


if __name__ == "__main__":
    unittest.main()
