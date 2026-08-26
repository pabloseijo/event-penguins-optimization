"""Tests for the THUMOS14 class analysis and its jackknife stability check.

Checks that class selection only reads the requested split, that filtering remaps
label ids while keeping every video, and that the jackknife reports a bounded
selection frequency.
"""

import copy
import unittest

from analyze_thumos14_classes import (
    analyze_classes,
    filter_annotations,
    jackknife_stability,
)


def annotation(label, label_id, start, end):
    return {
        "label": label,
        "label_id": label_id,
        "segment": [start, end],
    }


class AnalyzeThumos14ClassesTest(unittest.TestCase):
    def setUp(self):
        self.database = {
            "validation_1": {
                "subset": "validation",
                "annotations": [
                    annotation("Stable", 3, 0, 2),
                    annotation("Crowded", 8, 1, 4),
                ],
            },
            "validation_2": {
                "subset": "validation",
                "annotations": [
                    annotation("Stable", 3, 0, 2),
                    annotation("Stable", 3, 4, 6),
                    annotation("Sparse", 5, 8, 20),
                ],
            },
            "test_1": {
                "subset": "test",
                "annotations": [
                    annotation("TestOnly", 12, 0, 100),
                ],
            },
        }

    def test_selection_uses_only_requested_split(self):
        first = analyze_classes(self.database, "validation")
        changed = copy.deepcopy(self.database)
        changed["test_1"]["annotations"] = [
            annotation("Stable", 3, index, index + 0.1)
            for index in range(100)
        ]
        second = analyze_classes(changed, "validation")
        self.assertEqual(first, second)
        self.assertNotIn("TestOnly", {row["label"] for row in first})

    def test_filter_remaps_ids_and_keeps_all_videos(self):
        filtered, label_map = filter_annotations(
            {"database": self.database}, {"Stable", "Sparse"}
        )
        self.assertEqual(label_map, {"Stable": 0, "Sparse": 1})
        self.assertEqual(len(filtered["database"]), 3)
        labels = {
            item["label"]
            for video in filtered["database"].values()
            for item in video["annotations"]
        }
        label_ids = {
            item["label_id"]
            for video in filtered["database"].values()
            for item in video["annotations"]
        }
        self.assertEqual(labels, {"Stable", "Sparse"})
        self.assertEqual(label_ids, {0, 1})

    def test_jackknife_reports_bounded_selection_frequency(self):
        rows = analyze_classes(self.database, "validation")
        stability = jackknife_stability(
            self.database,
            "validation",
            [row["label"] for row in rows],
            number_selected=2,
        )
        self.assertEqual(len(stability), 3)
        self.assertTrue(
            all(0.0 <= row["selection_frequency"] <= 1.0 for row in stability)
        )


if __name__ == "__main__":
    unittest.main()
