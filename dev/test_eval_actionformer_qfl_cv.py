"""Tests for the quality focal-loss head fitted in cross-validation.

Checks that the protocol audit marks meta-cross-validation as selection only,
that the design matrix holds the two ranks and the one-hot class, that training
selection keeps the quality and top-score rows, and that the fit is monotonic on
synthetic quality.
"""

import unittest

import numpy as np
import torch

from eval_actionformer_qfl_cv import (
    build_protocol_audit,
    build_design,
    collect_training_data,
    fit_qfl,
    predict_qfl,
    select_training_indices,
)


class EvalActionFormerQflCvTest(unittest.TestCase):
    def test_protocol_marks_meta_cv_as_selection_only(self):
        audit = build_protocol_audit(5)

        self.assertEqual(audit["use"], "model and recipe selection only")
        self.assertFalse(audit["meta_cv_fully_nested"])
        self.assertEqual(
            audit["required_for_unbiased_meta_cv"],
            "5 outer folds x 4 inner base-model fits",
        )

    def synthetic_video(self):
        return {
            "scores": np.asarray([0.9, 0.8, 0.1, 0.7], dtype=np.float32),
            "labels": np.asarray([0, 0, 0, 1], dtype=np.int64),
            "features": np.asarray(
                [[1, 0], [0.8, 0], [0, 1], [0.7, 0.2]],
                dtype=np.float32,
            ),
            "target_tiou": np.asarray(
                [1.0, 0.8, 0.0, 0.6], dtype=np.float32
            ),
        }

    def test_design_adds_two_ranks_and_one_hot_class(self):
        design = build_design([self.synthetic_video()], num_classes=2)[0]
        self.assertEqual(design.shape, (4, 6))
        np.testing.assert_allclose(design[:, -2:].sum(axis=1), 1.0)

    def test_training_selection_keeps_quality_and_top_scores(self):
        video = self.synthetic_video()
        indices = select_training_indices(video, topk_per_class=1)
        self.assertTrue({0, 1, 3}.issubset(set(indices.tolist())))

    def test_fit_qfl_learns_monotonic_synthetic_quality(self):
        video = self.synthetic_video()
        designs = build_design([video], num_classes=2)
        features, targets, weights = collect_training_data(
            [video], designs, topk_per_class=4
        )
        model = fit_qfl(
            features,
            targets,
            weights,
            steps=200,
            batch_size=4,
            learning_rate=0.05,
            device=torch.device("cpu"),
            seed=3,
        )
        predictions = predict_qfl(designs, model)[0]
        self.assertGreater(predictions[0], predictions[2])


if __name__ == "__main__":
    unittest.main()
