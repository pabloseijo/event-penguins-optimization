"""Tests for the bootstrap confidence intervals over out-of-fold transfer results.

Checks the weighted class AP against a hand-computed detection ordering and that
a video left out of a bootstrap sample carries zero weight.
"""

import unittest

import numpy as np

from bootstrap_actionformer_transfer_oof import (
    build_class_outcome,
    weighted_class_ap,
)


class BootstrapActionFormerTransferOofTest(unittest.TestCase):
    def test_weighted_ap_matches_known_detection_order(self):
        outcome = build_class_outcome(
            ground_truth={
                "v0": np.asarray([[0.0, 1.0]]),
                "v1": np.asarray([[0.0, 1.0]]),
            },
            prediction_video_ids=np.asarray(["v0", "v1", "v1"]),
            prediction_segments=np.asarray(
                [[0.0, 1.0], [2.0, 3.0], [0.0, 1.0]]
            ),
            prediction_scores=np.asarray([0.9, 0.8, 0.7]),
            video_to_index={"v0": 0, "v1": 1},
            thresholds=np.asarray([0.5]),
        )
        ap = weighted_class_ap(outcome, np.asarray([1, 1]))
        self.assertAlmostEqual(float(ap[0]), 5.0 / 6.0)

    def test_cluster_weight_removes_unsampled_video(self):
        outcome = build_class_outcome(
            ground_truth={
                "v0": np.asarray([[0.0, 1.0]]),
                "v1": np.asarray([[0.0, 1.0]]),
            },
            prediction_video_ids=np.asarray(["v0", "v1", "v1"]),
            prediction_segments=np.asarray(
                [[0.0, 1.0], [2.0, 3.0], [0.0, 1.0]]
            ),
            prediction_scores=np.asarray([0.9, 0.8, 0.7]),
            video_to_index={"v0": 0, "v1": 1},
            thresholds=np.asarray([0.5]),
        )
        ap = weighted_class_ap(outcome, np.asarray([0, 2]))
        self.assertAlmostEqual(float(ap[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
