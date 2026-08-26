"""Tests for gaussian instance fusion of overlapping candidates.

Checks the numerically stable softmax, that suppressed overlap uses the candidate
duration, that fusion moves a boundary towards the higher-scoring candidate, and
that the conservative variant preserves the detection count and scores.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dev.eval_gaussian_instance_fusion_cv import (
    gaussian_fuse_candidates,
    overlap_with_anchor,
    refine_control_prediction,
    stable_softmax,
)


class GaussianInstanceFusionTest(unittest.TestCase):
    def test_stable_softmax_prefers_high_confidence(self) -> None:
        weights = stable_softmax(np.asarray([0.2, 0.8]), temperature=0.1)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(float(weights[1]), 0.99)

    def test_suppressed_overlap_uses_candidate_duration(self) -> None:
        overlap = overlap_with_anchor(
            np.asarray([1.0]),
            np.asarray([3.0]),
            anchor_start=0.0,
            anchor_end=4.0,
            mode="suppressed",
        )
        self.assertAlmostEqual(float(overlap[0]), 1.0)

    def test_fusion_moves_boundary_toward_higher_score(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model": "a",
                    "t_start": 0.0,
                    "t_end": 10.0,
                    "fusion_score": 0.9,
                },
                {
                    "model": "b",
                    "t_start": 2.0,
                    "t_end": 12.0,
                    "fusion_score": 0.4,
                },
            ]
        )
        fused = gaussian_fuse_candidates(
            frame,
            overlap_threshold=0.5,
            temperature=0.2,
            overlap_mode="iou",
            per_model_topk=10,
        )
        self.assertEqual(fused.shape, (1, 3))
        self.assertLess(float(fused[0, 0]), 1.0)
        self.assertLess(float(fused[0, 1]), 11.0)

    def test_conservative_refinement_preserves_count_and_scores(self) -> None:
        control = {
            "results": {
                "r": {
                    "1": [
                        {
                            "label": "ed",
                            "segment": [0.0, 10.0],
                            "score": 0.8,
                        }
                    ]
                }
            }
        }
        frames = [
            pd.DataFrame(
                [
                    {
                        "rec_name": "r",
                        "roi_id": 1,
                        "model": model,
                        "rank_score": score,
                        "t_start": start,
                        "t_end": end,
                    }
                ]
            )
            for model, score, start, end in (
                ("continuous", 1.0, 0.0, 10.0),
                ("event", 0.8, 1.0, 11.0),
                ("proposal", 0.7, 2.0, 12.0),
            )
        ]
        refined = refine_control_prediction(
            control,
            frames,
            overlap_threshold=0.5,
            temperature=0.1,
            blend=0.5,
            per_model_topk=10,
            min_models=2,
        )
        detection = refined["results"]["r"]["1"][0]
        self.assertEqual(float(detection["score"]), 0.8)
        self.assertGreater(float(detection["segment"][0]), 0.0)
        self.assertEqual(len(refined["results"]["r"]["1"]), 1)


if __name__ == "__main__":
    unittest.main()
