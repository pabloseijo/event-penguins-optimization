"""Tests for final PAL-consistency post-processing diagnostics."""

from __future__ import annotations

import unittest

from dev.eval_actionness_qfl_pal_postprocess_cv import detection_recall


class ActionnessQflPalPostprocessTest(unittest.TestCase):
    def test_recall_matches_targets_within_the_same_roi(self) -> None:
        prediction = {
            "results": {
                "rec": {
                    "1": [{"segment": [1.0, 3.0], "score": 0.8}],
                    "2": [],
                }
            }
        }
        annotations = {
            ("rec", 1): [(1.0, 3.0)],
            ("rec", 2): [(5.0, 7.0)],
        }
        self.assertEqual(
            detection_recall(prediction, annotations, ["rec"], 0.5), 0.5
        )


if __name__ == "__main__":
    unittest.main()
