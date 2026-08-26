"""Tests for the cross-domain proposal protocol on THUMOS14-E.

Checks that the target prototype is built only from the explicit validation
recordings, that both domains declare their minimum proposal duration, that a
missing video is rejected instead of skipped, and that stage 1 matches the frozen
source recipe.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.proposals import ProposalGenerator
from src.prototype import build_ed_prototype

from dev.run_thumos14_event_generalization import (
    FULL_STAGE1_CLASSIFIER_RECIPE,
    audit_target_h5,
    detection_ap,
    load_generic_ground_truth,
    prefix_order,
    proposal_recall,
    proposals_in_seconds,
    write_empty_target_metadata,
)


class Thumos14EventGeneralizationTest(unittest.TestCase):
    def test_target_prototype_uses_only_explicit_validation_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "events.h5"
            with h5py.File(data_path, "w") as handle:
                for recording, x in (("validation_video", 0), ("test_sentinel", 3)):
                    group = handle.create_group(recording)
                    roi = group.create_group("N01")
                    roi.attrs["width"] = 4
                    roi.attrs["height"] = 3
                    roi.create_dataset(
                        "events",
                        data=np.asarray(
                            [[x, 0, 100_000 + index * 100_000, index % 2] for index in range(12)],
                            dtype=np.uint32,
                        ),
                    )
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "database": {
                            recording: {
                                "annotations": {
                                    "1": [{"label": "ed", "segment": [0.0, 2.0]}]
                                }
                            }
                            for recording in ("validation_video", "test_sentinel")
                        }
                    }
                ),
                encoding="utf-8",
            )
            validation_only = build_ed_prototype(
                str(data_path),
                str(annotations),
                min_duration=0.0,
                recordings={"validation_video"},
            )
        self.assertGreater(float(validation_only[:, :4].sum()), 0.0)
        self.assertEqual(float(validation_only[:, 12:].sum()), 0.0)

    def test_source_and_target_minimum_proposal_duration_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = ProposalGenerator(
                data_path="unused.h5",
                bin_width=0.033,
                percentile=1.0,
                nms_threshold=0.95,
                output_dir=tmp,
            )
            target = ProposalGenerator(
                data_path="unused.h5",
                bin_width=0.033,
                percentile=1.0,
                nms_threshold=0.95,
                minimum_proposal_duration_s=0.0,
                output_dir=tmp,
            )
        self.assertEqual(source.minimum_proposal_duration_us, 2e6)
        self.assertEqual(target.minimum_proposal_duration_us, 0.0)

    def test_explicit_proposal_recordings_reject_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "events.h5"
            with h5py.File(data_path, "w") as handle:
                handle.create_group("validation_video")
            generator = ProposalGenerator(
                data_path=str(data_path),
                bin_width=0.033,
                percentile=1.0,
                nms_threshold=0.95,
                output_dir=tmp,
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                generator.run(split=None, recordings=["test_sentinel"])

    def test_stage1_classifier_matches_frozen_r5_recipe(self) -> None:
        self.assertEqual(FULL_STAGE1_CLASSIFIER_RECIPE["augment_factor"], 5)
        self.assertTrue(FULL_STAGE1_CLASSIFIER_RECIPE["use_soft_nms"])
        self.assertEqual(FULL_STAGE1_CLASSIFIER_RECIPE["soft_nms_sigma"], 0.25)
        self.assertEqual(FULL_STAGE1_CLASSIFIER_RECIPE["min_ed_score"], 0.3)
        self.assertEqual(FULL_STAGE1_CLASSIFIER_RECIPE["temperature"], 2.0)
        self.assertEqual(FULL_STAGE1_CLASSIFIER_RECIPE["duration_penalty_dmax"], 60.0)
        self.assertEqual(FULL_STAGE1_CLASSIFIER_RECIPE["duration_penalty_sigma"], 20.0)

    def test_target_audit_accepts_only_selected_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.h5"
            with h5py.File(path, "w") as handle:
                for recording, split in (("positive", "test"), ("unused", "train")):
                    group = handle.create_group(recording)
                    group.attrs["split"] = split
                    roi = group.create_group("N01")
                    roi.attrs["width"] = 4
                    roi.attrs["height"] = 3
                    roi.attrs["duration_s"] = 1.0
                    roi.create_dataset(
                        "events",
                        data=np.asarray(
                            [[0, 0, 1, 0], [3, 2, 1_000_000, 1]], dtype=np.uint32
                        ),
                    )

            report = audit_target_h5(path, "test")
            self.assertEqual(report["recordings"], ["positive"])
            self.assertEqual(report["rois"][0]["events"], 2)

    def test_generic_metrics_match_known_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            annotation_path = Path(tmp) / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "database": {
                            "positive": {
                                "annotations": {
                                    "1": [
                                        {
                                            "label": "ed",
                                            "source_label": "LongJump",
                                            "segment": [1.0, 3.0],
                                        }
                                    ]
                                }
                            },
                            "negative": {"annotations": {"1": []}},
                        }
                    }
                ),
                encoding="utf-8",
            )
            ground_truth = load_generic_ground_truth(
                annotation_path, ["positive", "negative"]
            )
            proposals = proposals_in_seconds(
                pd.DataFrame(
                    [
                        {
                            "rec_name": "positive",
                            "roi_id": "N01",
                            "t_start": 1_000_000,
                            "t_end": 3_000_000,
                            "score": 0.8,
                        },
                        {
                            "rec_name": "negative",
                            "roi_id": "N01",
                            "t_start": 1_000_000,
                            "t_end": 2_000_000,
                            "score": 0.2,
                        },
                    ]
                )
            )
            recall = proposal_recall(proposals, ground_truth, thresholds=[0.5], budgets=[20])
            self.assertEqual(recall, {"20": {"AR@0.5": 1.0, "mean_AR": 1.0}})
            ap = detection_ap(proposals, ground_truth, thresholds=[0.5])
            self.assertEqual(ap, {"AP@0.5": 1.0, "mAP": 1.0})

    def test_target_audit_rejects_non_monotonic_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("video")
                group.attrs["split"] = "test"
                roi = group.create_group("N01")
                roi.attrs["width"] = 4
                roi.attrs["height"] = 3
                roi.create_dataset(
                    "events",
                    data=np.asarray([[0, 0, 20, 0], [1, 1, 10, 1]], dtype=np.uint32),
                )

            with self.assertRaisesRegex(ValueError, "not monotonic"):
                audit_target_h5(path, "test")

    def test_prefix_order_keeps_frozen_cnn_candidates_first(self) -> None:
        frame = pd.DataFrame(
            [
                {"rec_name": "r", "roi_id": "N01", "t_start": 3.0, "t_end": 4.0, "score": 0.3},
                {"rec_name": "r", "roi_id": "N01", "t_start": 1.0, "t_end": 2.0, "score": 0.1},
                {"rec_name": "r", "roi_id": "N01", "t_start": 5.0, "t_end": 6.0, "score": 0.5},
            ]
        )
        ordered = prefix_order(frame, frame.iloc[[1, 2]])
        self.assertEqual(ordered["t_start"].tolist(), [1.0, 5.0, 3.0])

    def test_empty_target_metadata_contains_no_annotations(self) -> None:
        manifest = {
            "target": {
                "split": "test",
                "recordings": ["video"],
                "rois": [{"recording": "video", "roi": "N01"}],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            annotation_path, info_path = write_empty_target_metadata(manifest, Path(tmp))
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["database"]["video"]["annotations"], {"1": []})
            info = pd.read_csv(info_path)
            self.assertEqual(info.to_dict("records"), [{"timestamp": "video", "split": "test"}])


if __name__ == "__main__":
    unittest.main()
