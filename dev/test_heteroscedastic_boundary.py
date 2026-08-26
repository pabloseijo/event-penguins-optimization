"""Tests for the heteroscedastic boundary model and its diagnostics.

Checks that the candidate descriptor separates start from end roles, that jitter
respects the sequence bounds and the minimum duration, that the model returns
bounded means with finite uncertainty, and that the loss backpropagates through
every output.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dev.diagnose_event_boundary_reliability import summarize_boundary_outputs
from dev.eval_heteroscedastic_boundary_cv import (
    DenseFeatureStore,
    HeteroscedasticBoundaryHead,
    apply_refinement,
    boundary_quality_loss,
    candidate_descriptor,
    jitter_candidates,
    rerank_with_local_quality,
)


class HeteroscedasticBoundaryTest(unittest.TestCase):
    def test_candidate_descriptor_separates_boundary_roles(self) -> None:
        sequence = np.repeat(
            np.arange(12, dtype=np.float32)[:, None],
            4,
            axis=1,
        )
        roles, scalars = candidate_descriptor(
            sequence,
            start_s=3.0,
            end_s=8.0,
            score=0.7,
            ranks=(0.8, 0.6, 0.4),
            stride_s=1.0,
            context_ratio=0.2,
            min_context_s=1.0,
            max_context_s=2.0,
        )
        self.assertEqual(roles.shape, (6, 4))
        np.testing.assert_allclose(roles[0], 2.0)
        np.testing.assert_allclose(roles[1], 3.0)
        np.testing.assert_allclose(roles[2], 7.0)
        np.testing.assert_allclose(roles[3], 8.0)
        self.assertEqual(scalars.shape, (10,))

    def test_jitter_preserves_sequence_bounds_and_minimum_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "frame_features.npy", np.zeros((20, 3), np.float32))
            (root / "metadata.json").write_text(
                '{"grid_stride_s": 1.0, "feature_dim": 3}',
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "rec_name": "r",
                        "roi_id": 1,
                        "offset": 0,
                        "length": 20,
                        "duration_s": 20.0,
                    }
                ]
            ).to_csv(root / "sequences.csv", index=False)
            store = DenseFeatureStore(root)
            frame = pd.DataFrame(
                [
                    {
                        "rec_name": "r",
                        "roi_id": 1,
                        "t_start": 1.0,
                        "t_end": 3.0,
                    }
                ]
            )
            jittered = jitter_candidates(frame, store, 8, 1.0, seed=7)
            self.assertTrue((jittered["t_start"] >= 0.0).all())
            self.assertTrue((jittered["t_end"] <= 20.0).all())
            self.assertTrue(
                ((jittered["t_end"] - jittered["t_start"]) >= 2.0 - 1e-6).all()
            )

    def test_model_returns_bounded_means_and_finite_uncertainty(self) -> None:
        model = HeteroscedasticBoundaryHead(8, 10, 4, 0.0, 0.5)
        mean, log_variance, quality = model(
            torch.randn(3, 6, 8),
            torch.randn(3, 10),
        )
        self.assertEqual(tuple(mean.shape), (3, 2))
        self.assertTrue((mean.abs() <= 0.5 + 1e-6).all())
        self.assertTrue(torch.isfinite(log_variance).all())
        self.assertEqual(tuple(quality.shape), (3,))

    def test_boundary_loss_backpropagates_through_all_outputs(self) -> None:
        mean = torch.zeros(3, 2, requires_grad=True)
        log_variance = torch.zeros(3, 2, requires_grad=True)
        quality = torch.zeros(3, requires_grad=True)
        offsets = torch.tensor([[0.2, -0.1], [0.0, 0.0], [0.1, 0.1]])
        target_quality = torch.tensor([0.8, 0.0, 0.4])
        weights = torch.ones(3)
        loss, quality_loss, boundary_loss = boundary_quality_loss(
            mean,
            log_variance,
            quality,
            offsets,
            target_quality,
            weights,
            positive_tiou=0.1,
            boundary_weight=1.0,
        )
        loss.backward()
        self.assertGreater(float(quality_loss), 0.0)
        self.assertTrue(torch.isfinite(boundary_loss))
        self.assertIsNotNone(mean.grad)
        self.assertIsNotNone(log_variance.grad)
        self.assertIsNotNone(quality.grad)

    def test_reliability_weighting_moves_uncertain_boundaries_less(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "rec_name": "r",
                    "roi_id": 1,
                    "t_start": 10.0,
                    "t_end": 20.0,
                    "score": 0.8,
                }
            ]
        )
        mean = np.asarray([[0.2, -0.2]], dtype=np.float32)
        low_variance = np.asarray([[0.01, 0.01]], dtype=np.float32)
        high_variance = np.asarray([[1.0, 1.0]], dtype=np.float32)
        quality = np.asarray([1.0], dtype=np.float32)
        low = apply_refinement(
            frame, mean, low_variance, quality, 1.0, True
        )
        high = apply_refinement(
            frame, mean, high_variance, quality, 1.0, True
        )
        low_shift = abs(float(low.loc[0, "t_start"]) - 10.0)
        high_shift = abs(float(high.loc[0, "t_start"]) - 10.0)
        self.assertGreater(low_shift, high_shift)

    def test_local_quality_reranking_respects_weight_extremes(self) -> None:
        frame = pd.DataFrame({"score": [0.9, 0.1]})
        local = np.asarray([0.0, 1.0], dtype=np.float32)
        original = rerank_with_local_quality(frame, local, 0.0)
        local_only = rerank_with_local_quality(frame, local, 1.0)
        self.assertGreater(float(original.loc[0, "score"]), float(original.loc[1, "score"]))
        self.assertLess(
            float(local_only.loc[0, "score"]),
            float(local_only.loc[1, "score"]),
        )

    def test_recording_summary_is_label_free_and_grouped(self) -> None:
        frame = pd.DataFrame(
            {
                "fold": [0, 0, 1],
                "rec_name": ["a", "a", "b"],
                "score": [0.9, 0.1, 0.5],
            }
        )
        mean = np.asarray([[0.1, -0.1], [0.2, 0.0], [0.0, 0.0]])
        variance = np.full((3, 2), 0.04)
        quality = np.asarray([0.8, 0.4, 0.6])
        summary = summarize_boundary_outputs(frame, mean, variance, quality)
        self.assertEqual(summary["rec_name"].tolist(), ["a", "b"])
        self.assertEqual(summary["detections"].tolist(), [2, 1])
        self.assertAlmostEqual(
            float(summary.loc[0, "correction_abs_mean"]),
            0.1,
        )
        self.assertGreater(float(summary.loc[0, "reliability_mean"]), 0.0)


if __name__ == "__main__":
    unittest.main()
