"""Tests for the THUMOS14-E proposal evaluation and its protocol declaration.

Checks that the protocol declares one corpus and twenty class tasks, that the
manifest requires the canonical 212 test videos, and that microsecond proposals
reach the metrics with the right units.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dev.evaluate_thumos14e_proposals import (
    PROTOCOL_DECLARATION,
    canonical_test_recordings,
    flatten_result,
    load_proposals,
    method_result,
)


class EvaluateThumos14eProposalsTest(unittest.TestCase):
    def test_protocol_declares_one_corpus_and_twenty_class_tasks(self) -> None:
        scope = PROTOCOL_DECLARATION["scientific_scope"]
        self.assertEqual(scope["corpora"], 1)
        self.assertEqual(scope["class_tasks"], 20)
        self.assertFalse(scope["source_recordings_mixed_into_target"])
        self.assertIn(
            "twenty_one_vs_rest_binary_tasks",
            PROTOCOL_DECLARATION["predeclared_adaptations"],
        )

    def test_manifest_requires_canonical_212_test_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "video_id": f"video_test_{index:07d}",
                        "official_subset": "test",
                        "evaluation_included": index < 212,
                    }
                    for index in range(213)
                ]
            ).to_csv(path, index=False)
            self.assertEqual(len(canonical_test_recordings(path)), 212)

    def test_microsecond_proposals_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proposals.csv"
            pd.DataFrame(
                [
                    {
                        "rec_name": "positive",
                        "roi_id": "N01",
                        "t_start": 1_000_000,
                        "t_end": 3_000_000,
                        "score": 0.9,
                    },
                    {
                        "rec_name": "outside",
                        "roi_id": 1,
                        "t_start": 0,
                        "t_end": 1_000_000,
                        "score": 0.1,
                    },
                ]
            ).to_csv(path, index=False)
            proposals = load_proposals(path, ["positive"], "microseconds")
            self.assertEqual(proposals[["t_start", "t_end"]].iloc[0].tolist(), [1.0, 3.0])
            ground_truth = pd.DataFrame(
                [
                    {
                        "rec_name": "positive",
                        "roi_id": 1,
                        "t_start": 1.0,
                        "t_end": 3.0,
                        "source_label": "LongJump",
                    }
                ]
            )
            result = method_result(proposals, ground_truth)
            row = flatten_result("LongJump", "retag", 1, result)
            self.assertEqual(row["AR@20"], 1.0)
            self.assertEqual(row["oracle_mAP"], 1.0)


if __name__ == "__main__":
    unittest.main()
