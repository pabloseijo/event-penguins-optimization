"""Tests for the multiple-instance gate over ActionFormer detections.

Checks that bags are built for classes without detections, that the conservative
threshold keeps every training positive, that gating removes only the selected
video-class pair, and that the logistic gate separates simple bags.
"""

import unittest

import numpy as np
import torch

from eval_actionformer_mil_gate_cv import (
    apply_gate,
    build_bags,
    conservative_threshold,
    fit_logistic_gate,
    predict_gate,
)


class EvalActionFormerMilGateCvTest(unittest.TestCase):
    def test_bags_include_classes_without_detections(self):
        prediction = {
            "video_id": np.asarray(["v1"]),
            "t_start": np.asarray([0.0]),
            "t_end": np.asarray([2.0]),
            "label": np.asarray([0]),
            "score": np.asarray([0.8]),
        }
        targets = {("v1", 0): 1, ("v1", 1): 0}
        keys, features, labels = build_bags(prediction, targets, 2)
        self.assertEqual(keys, [("v1", 0), ("v1", 1)])
        self.assertEqual(features.shape, (2, 9))
        self.assertEqual(labels.tolist(), [1.0, 0.0])
        self.assertEqual(float(features[1, 0]), 0.0)

    def test_conservative_threshold_preserves_training_positives(self):
        probabilities = np.asarray([0.8, 0.06, 0.01])
        targets = np.asarray([1, 1, 0])
        threshold = conservative_threshold(probabilities, targets)
        self.assertEqual(threshold, 0.05)

    def test_gate_removes_only_selected_video_class(self):
        prediction = {
            "video_id": np.asarray(["v1", "v1", "v2"]),
            "t_start": np.asarray([0.0, 1.0, 2.0]),
            "t_end": np.asarray([1.0, 2.0, 3.0]),
            "label": np.asarray([0, 1, 0]),
            "score": np.asarray([0.8, 0.7, 0.6]),
        }
        probabilities = {
            ("v1", 0): 0.9,
            ("v1", 1): 0.01,
            ("v2", 0): 0.8,
        }
        output = apply_gate(prediction, probabilities, threshold=0.05)
        self.assertEqual(output["score"].tolist(), [0.8, 0.6])

    def test_logistic_gate_separates_simple_bags(self):
        features = np.asarray(
            [[0.0], [0.1], [0.9], [1.0]], dtype=np.float32
        )
        targets = np.asarray([0, 0, 1, 1], dtype=np.float32)
        model = fit_logistic_gate(
            features,
            targets,
            steps=200,
            learning_rate=0.05,
            device=torch.device("cpu"),
            seed=7,
        )
        probabilities = predict_gate(features, model)
        self.assertLess(probabilities[0], probabilities[-1])


if __name__ == "__main__":
    unittest.main()
