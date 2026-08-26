"""Tests for the construction of the THUMOS14 transfer folds.

Checks that fold assignment is deterministic and split-safe, that every fold sees
every class plus negatives, and that a fold file differs from the source only in
which videos are marked for validation.
"""

import copy
import unittest

from build_thumos14_transfer_folds import (
    assign_folds,
    build_fold_annotations,
    collect_labels,
    summarize_folds,
    validate_partition,
)


def annotation(label, label_id, start=0.0, end=1.0):
    return {
        "label": label,
        "label_id": label_id,
        "segment": [start, end],
    }


class BuildThumos14TransferFoldsTest(unittest.TestCase):
    def setUp(self):
        self.database = {}
        for index in range(15):
            annotations = []
            if index % 5 != 0:
                if index % 3 != 0:
                    annotations.append(annotation("A", 0))
                if index % 3 != 1:
                    annotations.append(annotation("B", 1, 2.0, 4.0))
            self.database[f"validation_{index:02d}"] = {
                "subset": "validation",
                "duration": 30.0 + index,
                "annotations": annotations,
            }
        self.database["test_00"] = {
            "subset": "test",
            "duration": 20.0,
            "annotations": [annotation("A", 0)],
        }

    def test_assignment_is_deterministic_and_split_safe(self):
        labels = collect_labels(self.database, "validation")
        first = assign_folds(
            self.database,
            split="validation",
            labels=labels,
            folds=3,
            seed=7,
        )
        changed = copy.deepcopy(self.database)
        changed["test_00"]["annotations"] = [
            annotation("B", 1, index, index + 0.1)
            for index in range(100)
        ]
        second = assign_folds(
            changed,
            split="validation",
            labels=labels,
            folds=3,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertNotIn("test_00", first)

    def test_each_fold_contains_every_class_and_negatives(self):
        labels = collect_labels(self.database, "validation")
        assignments = assign_folds(
            self.database,
            split="validation",
            labels=labels,
            folds=3,
            seed=11,
        )
        validate_partition(
            self.database,
            assignments,
            split="validation",
            folds=3,
        )
        summaries = summarize_folds(
            self.database, assignments, labels, folds=3
        )
        for row in summaries:
            self.assertGreater(row["negative_videos"], 0)
            self.assertTrue(
                all(value > 0 for value in row["instances_by_class"].values())
            )

    def test_fold_json_changes_only_validation_subset_names(self):
        labels = collect_labels(self.database, "validation")
        assignments = assign_folds(
            self.database,
            split="validation",
            labels=labels,
            folds=3,
            seed=19,
        )
        source = {"version": "synthetic", "database": self.database}
        output = build_fold_annotations(
            source,
            assignments,
            fold_index=1,
            split="validation",
        )
        self.assertEqual(output["database"]["test_00"], self.database["test_00"])
        for video_id in assignments:
            expected_subset = (
                "transfer_val_1"
                if assignments[video_id] == 1
                else "transfer_train_1"
            )
            self.assertEqual(
                output["database"][video_id]["subset"], expected_subset
            )
            self.assertEqual(
                output["database"][video_id]["annotations"],
                self.database[video_id]["annotations"],
            )


if __name__ == "__main__":
    unittest.main()
