"""Tests for the end-to-end THUMOS14-E pipeline runner.

Checks that canonical evaluation refuses a partial class set, that the shared
cache rejects a foreign recording or a missing shard, that a fold split cannot
absorb test videos, and that the quality screen uses the frozen source thresholds.
"""

from __future__ import annotations

import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dev.run_thumos14e_full_pipeline import (
    continuous_command,
    dense_train_command,
    local_prediction_from_scores,
    quality_train_command,
    run_evaluate_all,
    screen_lattice,
    split_proposals,
    validate_shared_feature_cache,
)


def proposal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rec_name": ["train_video", "val_video"],
            "roi_id": ["N01", "N01"],
            "t_start": [0.0, 0.0],
            "t_end": [1_000_000.0, 1_000_000.0],
            "score": [0.8, 0.9],
            "cnn_score": [0.11, 0.10],
            "quality_score": [0.2, 0.3],
        }
    )


class RunThumos14eFullPipelineTest(unittest.TestCase):
    def test_canonical_evaluation_refuses_a_partial_class_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            pd.DataFrame(
                {
                    "video_id": [f"test_{index:04d}" for index in range(213)],
                    "official_subset": ["test"] * 213,
                }
            ).to_csv(manifest, index=False)
            plan = {
                "inputs": {
                    "corpus_manifest": {"path": str(manifest)},
                    "canonical_annotations": {"path": str(root / "annotations.json")},
                },
                "paths": {"out_root": str(root / "output")},
            }
            args = Namespace(
                plan=str(root / "plan.json"),
                seed=1337,
                actionformer_root=str(root / "actionformer"),
                num_workers=1,
            )
            with patch(
                "dev.run_thumos14e_full_pipeline.load_plan", return_value=plan
            ), self.assertRaises(FileNotFoundError):
                run_evaluate_all(args)

    def test_shared_cache_rejects_foreign_recording_and_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            recordings = [f"video_{index:04d}" for index in range(413)]
            pd.DataFrame({"video_id": recordings}).to_csv(manifest_path, index=False)
            event_hdf5 = root / "events.h5"
            event_hdf5.write_bytes(b"events")
            source = root / "model.pt"
            source.write_bytes(b"model")
            shared = root / "output" / "shared_features"
            feature_dir = shared / "continuous"
            event_dir = shared / "event_stats"
            feature_dir.mkdir(parents=True)
            event_dir.mkdir(parents=True)
            altered = recordings[:-1] + ["foreign_video"]
            pd.DataFrame(
                {
                    "rec_name": altered,
                    "offset": np.arange(413),
                    "length": np.ones(413, dtype=int),
                }
            ).to_csv(feature_dir / "sequences.csv", index=False)
            (feature_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "recordings": altered,
                        "num_points": 413,
                        "feature_dim": 512,
                        "grid_stride_s": 0.5,
                        "window_duration_s": 1.0,
                        "data_path": str(event_hdf5),
                    }
                ),
                encoding="utf-8",
            )
            np.save(feature_dir / "frame_features.npy", np.zeros((413, 512), np.float16))
            np.save(event_dir / "event_stats.npy", np.zeros((413, 10), np.float32))
            (event_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "num_points": 413,
                        "grid_stride_s": 0.5,
                        "spectral_bins": 8,
                    }
                ),
                encoding="utf-8",
            )
            plan = {
                "inputs": {
                    "corpus_manifest": {"path": str(manifest_path)},
                    "event_hdf5": {"path": str(event_hdf5), "sha256": "data"},
                    "source_atsn": {"path": str(source), "sha256": "model"},
                },
                "paths": {"out_root": str(root / "output")},
            }
            with self.assertRaisesRegex(ValueError, "exactly one ROI"):
                validate_shared_feature_cache(plan, num_shards=2)
            sequences = pd.read_csv(feature_dir / "sequences.csv")
            sequences["rec_name"] = recordings
            sequences.to_csv(feature_dir / "sequences.csv", index=False)
            metadata = json.loads((feature_dir / "metadata.json").read_text())
            metadata["recordings"] = recordings
            (feature_dir / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(FileNotFoundError):
                validate_shared_feature_cache(plan, num_shards=2)

    def test_fold_split_cannot_absorb_test(self) -> None:
        proposals = proposal_frame()
        train, validation = split_proposals(
            proposals,
            {"train_video"},
            {"val_video"},
        )
        self.assertEqual(train["rec_name"].tolist(), ["train_video"])
        self.assertEqual(validation["rec_name"].tolist(), ["val_video"])
        leaked = pd.concat(
            (
                proposals,
                proposals.iloc[[0]].assign(rec_name="test_sentinel"),
            ),
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "escape"):
            split_proposals(leaked, {"train_video"}, {"val_video"})

    def test_quality_screen_uses_source_fixed_thresholds(self) -> None:
        prefix, hybrid = screen_lattice([proposal_frame()])
        self.assertEqual(prefix["rec_name"].tolist(), ["train_video", "val_video"])
        self.assertEqual(hybrid["rec_name"].tolist(), ["train_video", "val_video"])

    def test_local_prediction_keeps_one_second_target(self) -> None:
        quality = proposal_frame().iloc[[1]].reset_index(drop=True)
        hybrid = quality[["rec_name", "roi_id", "t_start", "t_end", "score"]].copy()
        dense = hybrid.copy()
        for prefix in ("delta", "distribution", "point"):
            dense[f"{prefix}_t_start"] = 0.0
            dense[f"{prefix}_t_end"] = 1_000_000.0
        scored, prediction = local_prediction_from_scores(
            hybrid,
            [quality],
            [dense],
            "LongJump",
        )
        self.assertEqual(len(scored), 1)
        self.assertEqual(prediction["target_class"], "LongJump")
        self.assertEqual(prediction["minimum_action_duration_s"], 0.0)
        self.assertEqual(len(prediction["results"]["val_video"][1]), 1)

    def test_training_commands_lock_article_recipe_and_target_duration(self) -> None:
        args = Namespace(
            seed=1337,
            device="cuda",
            num_workers=8,
            qhead_repr_batch_size=16,
            dense_repr_batch_size=32,
        )
        quality = quality_train_command(
            Path("events.h5"),
            Path("annotations.json"),
            Path("model.pt"),
            Path("train.csv"),
            Path("val.csv"),
            Path("quality"),
            args,
        )
        dense = dense_train_command(
            Path("events.h5"),
            Path("annotations.json"),
            Path("model.pt"),
            Path("master.csv"),
            Path("train.csv"),
            Path("val.csv"),
            Path("cache"),
            Path("dense"),
            args,
        )
        self.assertIn("--group-dro", quality)
        self.assertEqual(quality[quality.index("--min-gt-duration") + 1], "0.0")
        self.assertEqual(dense[dense.index("--min-action-duration") + 1], "0.0")
        self.assertNotIn("test", " ".join(quality + dense).lower())

    def test_continuous_command_has_no_test_adaptation(self) -> None:
        command = continuous_command(
            Path("features"),
            Path("events"),
            Path("annotations.json"),
            Path("folds.csv"),
            Path("out"),
            0,
            1337,
            "cuda",
            8,
            False,
        )
        self.assertIn("--min-action-duration", command)
        self.assertEqual(command[command.index("--min-action-duration") + 1], "0.0")
        self.assertNotIn("test", " ".join(command).lower())


if __name__ == "__main__":
    unittest.main()
