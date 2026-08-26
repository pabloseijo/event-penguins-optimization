"""Tests for the feature preparation of the ActionFormer transfer.

Checks temporal IoU and class matching, the fallback moments of empty intervals,
that completeness rewards the interior over the context, and that a prepared
video yields named, finite features.
"""

import unittest

import numpy as np

from prepare_actionformer_transfer_features import (
    FEATURE_NAMES,
    interval_mean_std,
    prepare_video,
    quality_targets,
    temporal_iou,
    temporal_statistics,
)


class PrepareActionFormerTransferFeaturesTest(unittest.TestCase):
    def test_temporal_iou_and_class_matching(self):
        segments = np.asarray([[0, 2], [1, 3], [5, 6]], dtype=np.float32)
        ground_truth = np.asarray([[0, 2], [5, 7]], dtype=np.float32)
        overlaps = temporal_iou(segments, ground_truth)
        self.assertAlmostEqual(float(overlaps[0, 0]), 1.0)
        self.assertAlmostEqual(float(overlaps[1, 0]), 1.0 / 3.0)
        targets = quality_targets(
            segments,
            np.asarray([0, 0, 1]),
            ground_truth,
            np.asarray([0, 1]),
        )
        np.testing.assert_allclose(targets, [1.0, 1.0 / 3.0, 0.5])

    def test_interval_moments_use_fallback_for_empty_intervals(self):
        times = np.asarray([0.0, 1.0, 2.0])
        values = np.asarray([0.0, 1.0, 0.0])
        means, standard_deviations, counts = interval_mean_std(
            times,
            values,
            np.asarray([0.0, 1.4]),
            np.asarray([1.0, 1.6]),
        )
        self.assertAlmostEqual(float(means[0]), 0.0)
        self.assertAlmostEqual(float(standard_deviations[0]), 0.0)
        self.assertEqual(counts.tolist(), [1, 0])
        self.assertAlmostEqual(float(means[1]), 0.5)

    def test_completeness_rewards_inside_over_context(self):
        times = np.arange(7, dtype=np.float32)
        probabilities = np.asarray(
            [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
            dtype=np.float32,
        )
        logits = np.log(probabilities / (1.0 - probabilities))[:, None]
        stats = temporal_statistics(
            dense_times=times,
            dense_logits=logits,
            segments=np.asarray([[2.0, 5.0]], dtype=np.float32),
            labels=np.asarray([0]),
            context_ratio=0.5,
        )
        self.assertGreater(float(stats["completeness"][0]), 0.5)
        self.assertGreater(float(stats["start_contrast"][0]), 0.5)
        self.assertGreater(float(stats["end_contrast"][0]), 0.5)

    def test_prepare_video_has_named_finite_features(self):
        probabilities = np.asarray(
            [[0.1], [0.8], [0.9], [0.1]], dtype=np.float32
        )
        raw = {
            "video_id": np.asarray("video"),
            "video_duration": np.asarray(4.0, dtype=np.float32),
            "segments": np.asarray([[1.0, 2.5]], dtype=np.float32),
            "scores": np.asarray([0.8], dtype=np.float32),
            "logits": np.asarray([1.3862944], dtype=np.float32),
            "labels": np.asarray([0], dtype=np.int64),
            "levels": np.asarray([0], dtype=np.int64),
            "point_strides": np.asarray([0.5], dtype=np.float32),
            "offsets": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "dense_times": np.arange(4, dtype=np.float32),
            "dense_logits": np.log(
                probabilities / (1.0 - probabilities)
            ),
        }
        output = prepare_video(
            raw,
            np.asarray([[1.0, 2.5]], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            context_ratio=0.5,
        )
        self.assertEqual(output["features"].shape, (1, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(output["features"]).all())
        self.assertAlmostEqual(float(output["target_tiou"][0]), 1.0)

    def test_inference_preparation_omits_quality_target(self):
        raw = {
            "video_id": np.asarray("video"),
            "video_duration": np.asarray(2.0, dtype=np.float32),
            "segments": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "scores": np.asarray([0.8], dtype=np.float32),
            "logits": np.asarray([1.3862944], dtype=np.float32),
            "labels": np.asarray([0], dtype=np.int64),
            "levels": np.asarray([0], dtype=np.int64),
            "point_strides": np.asarray([0.5], dtype=np.float32),
            "offsets": np.asarray([[1.0, 1.0]], dtype=np.float32),
            "dense_times": np.asarray([0.0, 1.0], dtype=np.float32),
            "dense_logits": np.asarray([[0.0], [0.0]], dtype=np.float32),
        }
        output = prepare_video(
            raw,
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            context_ratio=0.5,
            include_target=False,
        )
        self.assertNotIn("target_tiou", output)

    def test_quality_target_uses_post_nms_video_clipping(self):
        raw = {
            "video_id": np.asarray("video"),
            "video_duration": np.asarray(2.0, dtype=np.float32),
            "segments": np.asarray([[-1.0, 3.0]], dtype=np.float32),
            "scores": np.asarray([0.8], dtype=np.float32),
            "logits": np.asarray([1.3862944], dtype=np.float32),
            "labels": np.asarray([0], dtype=np.int64),
            "levels": np.asarray([0], dtype=np.int64),
            "point_strides": np.asarray([0.5], dtype=np.float32),
            "offsets": np.asarray([[1.0, 1.0]], dtype=np.float32),
            "dense_times": np.asarray([0.0, 1.0], dtype=np.float32),
            "dense_logits": np.asarray([[0.0], [0.0]], dtype=np.float32),
        }
        output = prepare_video(
            raw,
            np.asarray([[0.0, 2.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            context_ratio=0.5,
        )
        np.testing.assert_allclose(output["segments"], [[-1.0, 3.0]])
        np.testing.assert_allclose(output["clipped_segments"], [[0.0, 2.0]])
        self.assertAlmostEqual(float(output["target_tiou"][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
