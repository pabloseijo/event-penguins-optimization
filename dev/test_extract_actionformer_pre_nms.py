"""Tests for the extraction of ActionFormer candidates before NMS.

Checks the feature-grid to seconds conversion against the published formula, that
decoding preserves logit, label, level and offsets, that concatenation preserves
candidate order, and that candidates stay unclipped until after NMS.
"""

import unittest

import torch

from extract_actionformer_pre_nms import (
    concatenate_candidates,
    decode_level_candidates,
    feature_grid_to_seconds,
)


class ExtractActionFormerPreNmsTest(unittest.TestCase):
    def test_feature_grid_to_seconds_matches_actionformer_formula(self):
        values = torch.tensor([0.0, 1.0, 2.0])
        actual = feature_grid_to_seconds(
            values,
            feat_stride=4.0,
            feat_num_frames=16.0,
            fps=30.0,
        )
        expected = torch.tensor([8.0, 12.0, 16.0]) / 30.0
        self.assertTrue(torch.allclose(actual, expected))

    def test_decode_keeps_logit_label_level_and_offsets(self):
        logits = torch.tensor(
            [
                [-8.0, 2.0],
                [1.0, -8.0],
            ]
        )
        offsets = torch.tensor([[1.0, 2.0], [0.5, 1.0]])
        points = torch.tensor(
            [
                [5.0, 0.0, 100.0, 1.0],
                [10.0, 0.0, 100.0, 2.0],
            ]
        )
        output = decode_level_candidates(
            logits=logits,
            offsets=offsets,
            points=points,
            mask=torch.tensor([True, True]),
            level=3,
            num_classes=2,
            pre_nms_threshold=0.1,
            pre_nms_topk=10,
            duration_threshold=0.05,
            feat_stride=4.0,
            feat_num_frames=16.0,
            fps=4.0,
            video_duration=20.0,
        )
        self.assertEqual(output["labels"].tolist(), [1, 0])
        self.assertEqual(output["levels"].tolist(), [3, 3])
        self.assertTrue(torch.allclose(output["logits"], torch.tensor([2.0, 1.0])))
        self.assertTrue(
            torch.allclose(
                output["segments"],
                torch.tensor([[6.0, 9.0], [11.0, 14.0]]),
            )
        )

    def test_concatenate_preserves_candidate_order(self):
        first = {
            "scores": torch.tensor([0.9]),
            "segments": torch.tensor([[0.0, 1.0]]),
        }
        second = {
            "scores": torch.tensor([0.8]),
            "segments": torch.tensor([[1.0, 2.0]]),
        }
        output = concatenate_candidates([first, second])
        self.assertTrue(
            torch.allclose(output["scores"], torch.tensor([0.9, 0.8]))
        )
        self.assertEqual(output["segments"].shape, (2, 2))

    def test_candidates_remain_unclipped_until_after_nms(self):
        output = decode_level_candidates(
            logits=torch.tensor([[2.0]]),
            offsets=torch.tensor([[3.0, 20.0]]),
            points=torch.tensor([[0.0, 0.0, 100.0, 1.0]]),
            mask=torch.tensor([True]),
            level=0,
            num_classes=1,
            pre_nms_threshold=0.1,
            pre_nms_topk=10,
            duration_threshold=0.05,
            feat_stride=4.0,
            feat_num_frames=16.0,
            fps=4.0,
            video_duration=10.0,
        )
        self.assertLess(float(output["segments"][0, 0]), 0.0)
        self.assertGreater(float(output["segments"][0, 1]), 10.0)


if __name__ == "__main__":
    unittest.main()
