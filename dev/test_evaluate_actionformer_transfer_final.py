"""Tests for the calibration metrics of the final transfer evaluation.

Checks perfect calibration, a hand-computed single-bin ECE and Brier score, and
that misaligned arrays are rejected instead of silently broadcast.
"""

import unittest

import numpy as np

from evaluate_actionformer_transfer_final import calibration_metrics


class EvaluateActionFormerTransferFinalTest(unittest.TestCase):
    def test_perfect_calibration_metrics(self):
        metrics = calibration_metrics(
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            bins=2,
        )
        self.assertLess(metrics["ECE"], 1e-6)
        self.assertLess(metrics["Brier"], 1e-6)
        self.assertLess(metrics["NLL"], 1e-6)

    def test_known_single_bin_ece_and_brier(self):
        metrics = calibration_metrics(
            np.asarray([0.8, 0.6]),
            np.asarray([1.0, 0.0]),
            bins=1,
        )
        self.assertAlmostEqual(metrics["ECE"], 0.2)
        self.assertAlmostEqual(metrics["Brier"], 0.2)

    def test_rejects_misaligned_arrays(self):
        with self.assertRaises(ValueError):
            calibration_metrics(
                np.asarray([0.5]),
                np.asarray([0.0, 1.0]),
                bins=10,
            )


if __name__ == "__main__":
    unittest.main()
