"""Tests for the one-versus-rest aggregation on THUMOS14-E.

Checks that aggregation uses the canonical video list and the declared class,
which is what keeps a per-class number comparable with the published protocol.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dev.evaluate_thumos14e_ovr import canonical_test_protocol, prediction_rows


class EvaluateThumos14eOvrTest(unittest.TestCase):
    def test_aggregation_uses_canonical_videos_and_declared_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "canonical.json"
            annotations.write_text(
                json.dumps(
                    {
                        "database": {
                            "test_a": {
                                "subset": "Test",
                                "annotations": [
                                    {"label": "LongJump", "label_id": 0, "segment": [1.0, 2.0]}
                                ],
                            },
                            "validation_a": {
                                "subset": "Validation",
                                "annotations": [
                                    {"label": "LongJump", "label_id": 0, "segment": [1.0, 2.0]}
                                ],
                            },
                        }
                    }
                )
            )
            test_videos, label_to_id, _ = canonical_test_protocol(annotations)
            prediction_dir = root / "predictions" / "LongJump"
            prediction_dir.mkdir(parents=True)
            (prediction_dir / "predictions.json").write_text(
                json.dumps(
                    {
                        "target_class": "LongJump",
                        "results": {
                            "test_a": {
                                "1": [
                                    {"segment": [1.0, 2.0], "score": 0.9, "label": "ed"}
                                ]
                            },
                            "ambiguous_only": {
                                "1": [
                                    {"segment": [3.0, 4.0], "score": 0.8, "label": "ed"}
                                ]
                            },
                        },
                    }
                )
            )
            frame, audit = prediction_rows(
                root / "predictions",
                "predictions.json",
                test_videos,
                label_to_id,
            )
            self.assertEqual(test_videos, {"test_a"})
            self.assertEqual(frame["video-id"].tolist(), ["test_a"])
            self.assertEqual(frame["label"].tolist(), [0])
            self.assertEqual(audit["excluded_noncanonical_videos"], ["ambiguous_only"])


if __name__ == "__main__":
    unittest.main()
