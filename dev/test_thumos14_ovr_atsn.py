"""Tests for the one-versus-rest ATSN training on THUMOS14-E.

Checks that the final fit uses all validation videos and never test, that fold
selection is video-disjoint, that labels use strict IoU and ignore ambiguous
overlap, and that negative downsampling is stable while keeping every positive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from dev.train_thumos14_ovr_atsn import (
    downsample_training_rows,
    label_proposals,
    split_recordings,
    stable_fraction_mask,
    train,
)


class Thumos14OvrAtsnTest(unittest.TestCase):
    def test_final_fit_uses_all_validation_and_never_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_model = root / "source.pt"
            source_model.write_bytes(b"frozen-source-checkpoint")
            source_hash = hashlib.sha256(source_model.read_bytes()).hexdigest()
            cache = root / "features"
            cache.mkdir()
            np.save(
                cache / "features.npy",
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            pd.DataFrame(
                [
                    {"rec_name": "val_a", "roi_id": "N01", "t_start": 0, "t_end": 10e6},
                    {"rec_name": "val_b", "roi_id": "N01", "t_start": 20e6, "t_end": 30e6},
                    {"rec_name": "test_a", "roi_id": "N01", "t_start": 0, "t_end": 10e6},
                ]
            ).to_csv(cache / "proposals.csv", index=False)
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "source_model_sha256": source_hash,
                        "feature_dim": 4,
                        "num_tsn_samples": 7,
                        "augment_factor": 3,
                    }
                )
            )
            config = root / "config"
            config.mkdir()
            annotations = config / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "target_class": "LongJump",
                        "database": {
                            "val_a": {
                                "annotations": {"1": [{"label": "ed", "segment": [0.0, 10.0]}]}
                            },
                            "val_b": {"annotations": {"1": []}},
                            "test_a": {
                                "annotations": {"1": [{"label": "ed", "segment": [0.0, 10.0]}]}
                            },
                        },
                    }
                )
            )
            with (config / "recording_info.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "split", "official_subset", "cv_fold"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"timestamp": "val_a", "split": "val", "official_subset": "validation", "cv_fold": "0"},
                        {"timestamp": "val_b", "split": "train", "official_subset": "validation", "cv_fold": "1"},
                        {"timestamp": "test_a", "split": "test", "official_subset": "test", "cv_fold": ""},
                    ]
                )
            out_dir = root / "out"
            train(
                Namespace(
                    epochs=10,
                    seed=7,
                    annotations=annotations,
                    source_model=source_model,
                    out_dir=out_dir,
                    train_features_dir=cache,
                    val_features_dir=None,
                    train_all_validation=True,
                    cv_fold=None,
                    positive_tiou=0.7,
                    negative_keep_fraction=1.0,
                    device="cpu",
                    head_init="reset",
                    learning_rate=1e-3,
                    momentum=0.9,
                    weight_decay=0.0,
                    batch_size=2,
                    num_workers=0,
                    command="train",
                )
            )
            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(summary["train_proposals_all"], 2)
            self.assertEqual(summary["train_positive"], 1)
            self.assertEqual(summary["train_negative_after_10x_downsampling"], 1)
            self.assertIsNone(summary["val_positive"])

    def test_fold_selection_is_video_disjoint_and_final_fit_excludes_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            annotation_path = Path(tmp) / "annotations.json"
            annotation_path.write_text("{}")
            with (Path(tmp) / "recording_info.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp",
                        "split",
                        "official_subset",
                        "cv_fold",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "timestamp": "validation_0",
                            "split": "val",
                            "official_subset": "validation",
                            "cv_fold": "0",
                        },
                        {
                            "timestamp": "validation_1",
                            "split": "train",
                            "official_subset": "validation",
                            "cv_fold": "1",
                        },
                        {
                            "timestamp": "test_0",
                            "split": "test",
                            "official_subset": "test",
                            "cv_fold": "",
                        },
                    ]
                )
            self.assertEqual(split_recordings(annotation_path, "val", cv_fold=0), {"validation_0"})
            self.assertEqual(split_recordings(annotation_path, "train", cv_fold=0), {"validation_1"})
            self.assertEqual(
                split_recordings(annotation_path, "train", all_validation=True),
                {"validation_0", "validation_1"},
            )

    def test_labels_use_strict_iou_and_ignore_ambiguous_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "target_class": "LongJump",
                        "database": {
                            "video_a": {
                                "annotations": {
                                    "1": [
                                        {"label": "ed", "segment": [0.0, 10.0]},
                                        {"label": "ambiguous", "segment": [20.0, 22.0]},
                                    ]
                                }
                            }
                        },
                    }
                )
            )
            with (root / "recording_info.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "split"])
                writer.writeheader()
                writer.writerow({"timestamp": "video_a", "split": "train"})
            proposals = pd.DataFrame(
                [
                    {"rec_name": "video_a", "roi_id": "N01", "t_start": 0, "t_end": 10e6},
                    {"rec_name": "video_a", "roi_id": "N01", "t_start": 0, "t_end": 7e6},
                    {"rec_name": "video_a", "roi_id": "N01", "t_start": 19e6, "t_end": 21e6},
                    {"rec_name": "video_a", "roi_id": "N01", "t_start": 30e6, "t_end": 32e6},
                ]
            )
            labeled = label_proposals(proposals, annotations, "train", 0.7)
            self.assertEqual(labeled["label"].tolist(), [1, 0, -1, 0])
            self.assertEqual(labeled["best_target_tiou"].tolist(), [1.0, 0.7, 0.0, 0.0])

    def test_negative_downsampling_is_stable_and_keeps_all_positives(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "feature_index": index,
                    "rec_name": f"video_{index // 10}",
                    "roi_id": "N01",
                    "t_start": float(index),
                    "t_end": float(index + 1),
                    "label": 1 if index < 3 else 0,
                }
                for index in range(103)
            ]
        )
        negatives = rows[rows["label"] == 0]
        first = stable_fraction_mask(negatives, 0.1, 17)
        second = stable_fraction_mask(negatives, 0.1, 17)
        self.assertEqual(first.tolist(), second.tolist())
        sampled = downsample_training_rows(rows, 0.1, 17)
        self.assertEqual(sampled[sampled["label"] == 1]["feature_index"].tolist(), [0, 1, 2])
        self.assertGreater(len(sampled[sampled["label"] == 0]), 0)


if __name__ == "__main__":
    unittest.main()
