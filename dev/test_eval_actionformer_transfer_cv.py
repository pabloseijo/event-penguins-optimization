"""Tests for the class-wise ranking and calibration used in transfer evaluation.

Checks that percentile ranks average ties, that class-wise ranks never mix
classes, that the ECDF is fitted on training values only, and that the duration
penalty acts on the upper tail alone.
"""

import unittest

import numpy as np
import torch

from eval_actionformer_transfer_cv import (
    apply_classwise_ecdf,
    classwise_percentile_ranks,
    classwise_segment_voting,
    duration_penalty,
    fit_classwise_ecdf,
    percentile_rank,
)


class EvalActionFormerTransferCvTest(unittest.TestCase):
    def test_percentile_rank_averages_ties(self):
        actual = percentile_rank(np.asarray([1.0, 2.0, 2.0, 4.0]))
        np.testing.assert_allclose(actual, [0.25, 0.625, 0.625, 1.0])

    def test_classwise_ranks_do_not_mix_classes(self):
        videos = [
            {
                "labels": np.asarray([0, 1]),
            },
            {
                "labels": np.asarray([0, 1]),
            },
        ]
        values = [
            np.asarray([0.1, 100.0]),
            np.asarray([0.2, 200.0]),
        ]
        ranks = classwise_percentile_ranks(videos, values)
        np.testing.assert_allclose(ranks[0], [0.5, 0.5])
        np.testing.assert_allclose(ranks[1], [1.0, 1.0])

    def test_classwise_ecdf_is_fitted_only_on_training_values(self):
        training = [
            {
                "labels": np.asarray([0, 0, 1, 1]),
            }
        ]
        references = fit_classwise_ecdf(
            training,
            [np.asarray([0.1, 0.2, 100.0, 200.0])],
            num_classes=2,
        )
        validation = [{"labels": np.asarray([0, 1])}]
        ranks = apply_classwise_ecdf(
            validation,
            [np.asarray([0.15, 150.0])],
            references,
        )
        np.testing.assert_allclose(ranks[0], [0.5, 0.5])

    def test_duration_penalty_only_affects_upper_tail(self):
        video = {
            "segments": np.asarray([[0, 2], [0, 20]], dtype=np.float32),
            "labels": np.asarray([0, 0], dtype=np.int64),
        }
        penalty = duration_penalty(
            video,
            upper=np.asarray([np.log(5.0)], dtype=np.float32),
            scale=np.asarray([1.0], dtype=np.float32),
            gamma=1.0,
        )
        self.assertAlmostEqual(float(penalty[0]), 1.0)
        self.assertLess(float(penalty[1]), 1.0)

    def test_voting_is_applied_independently_by_class(self):
        selected_segments = torch.tensor([[0.0, 2.0], [8.0, 10.0]])
        selected_labels = torch.tensor([0, 1])
        all_segments = torch.tensor(
            [[0.0, 2.0], [0.2, 2.2], [8.0, 10.0], [7.8, 9.8]]
        )
        all_scores = torch.tensor([0.9, 0.8, 0.9, 0.8])
        all_labels = torch.tensor([0, 0, 1, 1])

        def fake_voting(selected, all_segs, scores, threshold):
            del scores, threshold
            return selected * 0 + all_segs.mean(dim=0)

        output = classwise_segment_voting(
            selected_segments,
            selected_labels,
            all_segments,
            all_scores,
            all_labels,
            0.7,
            fake_voting,
        )
        np.testing.assert_allclose(output[0], [0.1, 2.1])
        np.testing.assert_allclose(output[1], [7.9, 9.9])


if __name__ == "__main__":
    unittest.main()
