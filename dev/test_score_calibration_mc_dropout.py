"""Tests for the MC-dropout score calibration diagnostics.

Checks that perfect calibration gives zero ECE, that overconfident errors raise
the NLL, that extreme scores are clipped, and that the best-scoring match claims
the ground-truth instance.
"""

import unittest

import numpy as np
import pandas as pd

from dev.eval_score_calibration_mc_dropout_cv import assign_tp_labels, calibration_metrics


class CalibrationMetricsTests(unittest.TestCase):
    def test_perfect_calibration_gives_zero_ece(self):
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        labels = np.array([1, 1, 0, 0])
        metrics = calibration_metrics(scores, labels, n_bins=10)
        self.assertAlmostEqual(metrics["ece"], 0.1, places=6)
        self.assertGreater(metrics["brier"], 0.0)
        self.assertEqual(metrics["n"], 4)

    def test_overconfident_wrong_predictions_increase_nll(self):
        confident_wrong = calibration_metrics(np.array([0.99]), np.array([0]), n_bins=10)
        uncertain_wrong = calibration_metrics(np.array([0.55]), np.array([0]), n_bins=10)
        self.assertGreater(confident_wrong["nll"], uncertain_wrong["nll"])
        self.assertGreater(confident_wrong["brier"], uncertain_wrong["brier"])

    def test_calibration_metrics_clip_extreme_scores(self):
        metrics = calibration_metrics(np.array([1.0, 0.0]), np.array([1, 0]), n_bins=10)
        self.assertTrue(np.isfinite(metrics["nll"]))
        self.assertTrue(np.isfinite(metrics["brier"]))


class AssignTpLabelsTests(unittest.TestCase):
    def test_best_scoring_match_wins_gt(self):
        rows = pd.DataFrame(
            [
                {"rec_name": "r1", "roi_id": 0, "t_start": 10.0, "t_end": 14.0, "raw_score": 0.9},
                {"rec_name": "r1", "roi_id": 0, "t_start": 10.2, "t_end": 14.1, "raw_score": 0.4},
            ]
        )
        annotations = {("r1", 0): np.array([[10.0, 14.0]], dtype=np.float64)}
        labels = assign_tp_labels(rows, annotations, iou_threshold=0.5)
        self.assertEqual(labels.iloc[0], 1)
        self.assertEqual(labels.iloc[1], 0)

    def test_no_overlap_is_false_positive(self):
        rows = pd.DataFrame(
            [{"rec_name": "r1", "roi_id": 0, "t_start": 50.0, "t_end": 54.0, "raw_score": 0.8}]
        )
        annotations = {("r1", 0): np.array([[10.0, 14.0]], dtype=np.float64)}
        labels = assign_tp_labels(rows, annotations, iou_threshold=0.5)
        self.assertEqual(labels.iloc[0], 0)

    def test_missing_annotations_default_to_false_positive(self):
        rows = pd.DataFrame(
            [{"rec_name": "r2", "roi_id": 3, "t_start": 5.0, "t_end": 8.0, "raw_score": 0.7}]
        )
        labels = assign_tp_labels(rows, {}, iou_threshold=0.5)
        self.assertEqual(labels.iloc[0], 0)

    def test_two_gt_instances_each_claim_one_detection(self):
        rows = pd.DataFrame(
            [
                {"rec_name": "r1", "roi_id": 0, "t_start": 10.0, "t_end": 14.0, "raw_score": 0.9},
                {"rec_name": "r1", "roi_id": 0, "t_start": 30.0, "t_end": 34.0, "raw_score": 0.8},
            ]
        )
        annotations = {
            ("r1", 0): np.array([[10.0, 14.0], [30.0, 34.0]], dtype=np.float64)
        }
        labels = assign_tp_labels(rows, annotations, iou_threshold=0.5)
        self.assertEqual(labels.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
