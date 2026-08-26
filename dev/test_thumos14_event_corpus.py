"""Tests for the assembly of the THUMOS14 event corpus.

Checks that a fork shares sources while separating conversion outputs, that
parsing preserves every label and the ambiguous regions, that multilabel folds are
deterministic and cover every class, and that shards are disjoint.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np

from dev.prepare_thumos14_event_corpus import (
    THUMOS_CLASSES,
    annotation_reachability,
    assemble,
    class_summary,
    conversion_recipe,
    corpus_paths,
    parse_official_annotations,
    fork,
    selected_rows,
    valid_annotations,
    validate,
    validation_fold_assignments,
    v2e_command,
    write_annotation_views,
)


class Thumos14EventCorpusTest(unittest.TestCase):
    def test_fork_shares_sources_but_separates_conversion_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            target_root = Path(tmp) / "variant"
            source = corpus_paths(source_root)
            for name in ("official", "videos", "source_metadata"):
                source[name].mkdir(parents=True)
            for name in ("manifest", "split_audit", "source_audit"):
                source[name].write_text(f"{name}\n")
            fork(
                Namespace(
                    source_work_dir=source_root,
                    work_dir=target_root,
                    purpose="original-rate sensitivity",
                )
            )
            target = corpus_paths(target_root)
            self.assertTrue(target["videos"].is_symlink())
            self.assertEqual(target["manifest"].read_text(), "manifest\n")
            self.assertFalse(target["v2e"].exists())
            self.assertEqual(
                json.loads(target["variant_parent"].read_text())["purpose"],
                "original-rate sensitivity",
            )

    def test_parse_preserves_all_labels_and_ambiguous_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "annotations.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index, label in enumerate(THUMOS_CLASSES):
                    archive.writestr(
                        f"annotation/{label}_val.txt",
                        f"video_validation_0000001 {index + 0.1} {index + 0.9}\n",
                    )
                archive.writestr(
                    "annotation/Ambiguous_val.txt",
                    "video_validation_0000002 1.0 2.0\n",
                )
            actions, ambiguous = parse_official_annotations(
                archive_path, "validation"
            )
            self.assertEqual(len(actions["video_validation_0000001"]), 20)
            self.assertEqual(
                {item["label"] for item in actions["video_validation_0000001"]},
                set(THUMOS_CLASSES),
            )
            self.assertEqual(ambiguous, {"video_validation_0000002": [[1.0, 2.0]]})

    def test_multilabel_folds_are_deterministic_and_cover_every_class(self) -> None:
        actions = {
            f"video_validation_{index:07d}": [
                {"label": label, "segment": [float(class_id), float(class_id + 1)]}
                for class_id, label in enumerate(THUMOS_CLASSES)
            ]
            for index in range(25)
        }
        first = validation_fold_assignments(actions, folds=5, seed=11)
        second = validation_fold_assignments(actions, folds=5, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(
            [first[f"video_validation_{index:07d}"] for index in range(10)],
            [3, 2, 4, 4, 4, 4, 1, 1, 0, 0],
        )
        self.assertEqual(set(first.values()), set(range(5)))
        for fold in range(5):
            labels = {
                item["label"]
                for video_id, annotations in actions.items()
                if first[video_id] == fold
                for item in annotations
            }
            self.assertEqual(labels, set(THUMOS_CLASSES))

    def test_shards_are_disjoint_and_cover_the_selection(self) -> None:
        rows = [{"video_id": f"v{index:02d}"} for index in range(17)]
        shards = [selected_rows(rows, None, index, 3) for index in range(3)]
        sets = [{row["video_id"] for row in shard} for shard in shards]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(set.union(*sets), {row["video_id"] for row in rows})

    @staticmethod
    def _convert_args(**changes) -> Namespace:
        values = {
            "v2e_python": "python",
            "v2e_entry": Path("/tmp/v2e.py"),
            "output_width": 346,
            "output_height": 260,
            "pos_thres": 0.2,
            "neg_thres": 0.2,
            "sigma_thres": 0.03,
            "cutoff_hz": 300.0,
            "leak_rate_hz": 0.01,
            "shot_noise_rate_hz": 0.001,
            "refractory_period": 0.0005,
            "seed": 7,
            "batch_size": 8,
            "disable_slomo": False,
            "fixed_timestamp_resolution": True,
            "timestamp_resolution": 0.003,
            "dvs_profile": "clean",
            "start_time": None,
            "stop_time": None,
        }
        values.update(changes)
        return Namespace(**values)

    def test_primary_v2e_command_uses_fixed_three_ms_interpolation(self) -> None:
        row = {"source_path": "/tmp/video.mp4"}
        primary = v2e_command(self._convert_args(), row, Path("/tmp/out"))
        original_rate = v2e_command(
            self._convert_args(disable_slomo=True), row, Path("/tmp/out")
        )
        adaptive = v2e_command(
            self._convert_args(
                fixed_timestamp_resolution=False,
                timestamp_resolution=None,
            ),
            row,
            Path("/tmp/out"),
        )
        bounded_auto = v2e_command(
            self._convert_args(
                fixed_timestamp_resolution=False,
                timestamp_resolution=0.001,
            ),
            row,
            Path("/tmp/out"),
        )
        self.assertIn("--auto_timestamp_resolution=False", primary)
        self.assertIn("--timestamp_resolution=0.003", primary)
        self.assertIn("--dvs_params=clean", primary)
        self.assertNotIn("--cutoff_hz", primary)
        self.assertNotIn("--disable_slomo", primary)
        self.assertIn("--disable_slomo", original_rate)
        self.assertFalse(
            any(item.startswith("--timestamp_resolution=") for item in original_rate)
        )
        self.assertIn("--auto_timestamp_resolution=True", adaptive)
        self.assertFalse(
            any(item.startswith("--timestamp_resolution=") for item in adaptive)
        )
        self.assertIn("--auto_timestamp_resolution=True", bounded_auto)
        self.assertIn("--timestamp_resolution=0.001", bounded_auto)

    def test_recipe_records_effective_preset_parameters_and_profile_times(self) -> None:
        recipe = conversion_recipe(
            self._convert_args(start_time=12.0, stop_time=22.0)
        )
        self.assertEqual(recipe["start_time_s"], 12.0)
        self.assertEqual(recipe["stop_time_s"], 22.0)
        self.assertEqual(
            recipe["effective_dvs_parameters"],
            {
                "pos_thres": 0.2,
                "neg_thres": 0.2,
                "sigma_thres": 0.02,
                "cutoff_hz": 0.0,
                "leak_rate_hz": 0.0,
                "shot_noise_rate_hz": 0.0,
                "refractory_period_s": 0.0,
                "leak_jitter_fraction": 0.0,
                "noise_rate_cov_decades": 0.0,
                "photoreceptor_noise": False,
            },
        )

    def test_original_rate_recipe_does_not_claim_an_ignored_timestamp_resolution(self) -> None:
        recipe = conversion_recipe(self._convert_args(disable_slomo=True))
        self.assertEqual(recipe["timing"], "original_rate_ablation")
        self.assertIsNone(recipe["auto_timestamp_resolution"])
        self.assertIsNone(recipe["timestamp_resolution_s"])

    def test_profile_command_passes_absolute_start_and_stop_times(self) -> None:
        command = v2e_command(
            self._convert_args(start_time=12.0, stop_time=22.0),
            {"source_path": "/tmp/video.mp4"},
            Path("/tmp/out"),
        )
        self.assertEqual(command[-4:], ["--start_time", "12.0", "--stop_time", "22.0"])

    def test_partial_annotation_view_drops_annotations_after_stop_time(self) -> None:
        row = {
            "video_id": "video_validation_0000001",
            "annotations_json": json.dumps(
                [
                    {"label": "LongJump", "segment": [1.0, 4.0]},
                    {"label": "LongJump", "segment": [8.0, 12.0]},
                    {"label": "LongJump", "segment": [15.0, 20.0]},
                ]
            ),
        }
        self.assertEqual(
            valid_annotations(row, 10.0, trim_to_duration=True),
            [
                {"label": "LongJump", "segment": [1.0, 4.0]},
                {"label": "LongJump", "segment": [8.0, 10.0]},
            ],
        )

    def test_canonical_unreachable_gt_is_preserved_but_not_trainable(self) -> None:
        row = {
            "video_id": "video_test_0000001",
            "official_subset": "test",
            "split": "test",
            "cv_fold": "",
            "evaluation_included": "1",
            "labels_json": json.dumps(["HammerThrow"]),
            "annotations_json": json.dumps(
                [
                    {"label": "HammerThrow", "segment": [1.0, 2.0]},
                    {"label": "HammerThrow", "segment": [12.0, 14.0]},
                ]
            ),
            "ambiguous_json": "[]",
        }
        with self.assertRaises(ValueError):
            valid_annotations(row, 10.0)
        self.assertEqual(
            len(valid_annotations(row, 10.0, allow_unreachable=True)), 2
        )
        issues = annotation_reachability(row, 10.0)
        self.assertEqual(len(issues), 1)
        self.assertTrue(issues[0]["starts_after_video"])

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            write_annotation_views(
                config,
                [row],
                {row["video_id"]: 10.0},
                {row["video_id"]: 123},
            )
            official = json.loads(
                (config / "by_class/HammerThrow/annotations.json").read_text()
            )
            trainable = json.loads(
                (
                    config
                    / "by_class/HammerThrow/annotations_trainable.json"
                ).read_text()
            )
            official_items = official["database"][row["video_id"]]["annotations"]["1"]
            trainable_items = trainable["database"][row["video_id"]]["annotations"]["1"]
            self.assertEqual(len(official_items), 2)
            self.assertEqual(len(trainable_items), 1)

    def test_annotation_views_keep_binary_and_multiclass_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            row = {
                "video_id": "video_validation_0000001",
                "official_subset": "validation",
                "split": "train",
                "cv_fold": "1",
                "evaluation_included": "1",
                "labels_json": json.dumps(["Diving", "LongJump"]),
                "annotations_json": json.dumps(
                    [
                        {"label": "Diving", "segment": [1.0, 2.0]},
                        {"label": "LongJump", "segment": [4.0, 6.0]},
                    ]
                ),
                "ambiguous_json": json.dumps([[6.5, 7.0]]),
            }
            write_annotation_views(
                config,
                [row],
                {row["video_id"]: 8.0},
                {row["video_id"]: 123},
            )
            binary = json.loads((config / "annotations.json").read_text())
            multiclass = json.loads(
                (config / "annotations_multiclass.json").read_text()
            )
            binary_items = binary["database"][row["video_id"]]["annotations"]["1"]
            multiclass_items = multiclass["database"][row["video_id"]]["annotations"]["1"]
            self.assertEqual(
                [item["label"] for item in binary_items],
                ["ed", "ed", "ambiguous"],
            )
            self.assertEqual(
                [item["label"] for item in multiclass_items],
                ["Diving", "LongJump", "ambiguous"],
            )
            long_jump = json.loads(
                (config / "by_class/LongJump/annotations.json").read_text()
            )
            ovr_items = long_jump["database"][row["video_id"]]["annotations"]["1"]
            self.assertEqual(
                [item["label"] for item in ovr_items],
                ["other_action", "ed", "ambiguous"],
            )
            with (config / "recording_info.csv").open(newline="") as handle:
                info = next(csv.DictReader(handle))
            self.assertEqual(info["event_count"], "123")
            with (config / "fold_manifest.csv").open(newline="") as handle:
                fold = next(csv.DictReader(handle))
            self.assertEqual(fold["fold"], "1")
            self.assertEqual(fold["val_record_names"], row["video_id"])
            self.assertEqual(fold["train_videos"], "0")
            summary = class_summary([row])
            self.assertEqual(
                summary["instances_by_split_and_class"]["train"]["LongJump"], 1
            )

    def test_partial_assembly_keeps_its_artifacts_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = corpus_paths(root)
            video_id = "video_validation_0000001"
            source = paths["videos"] / f"{video_id}.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
            with paths["manifest"].open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "video_id",
                        "official_subset",
                        "split",
                        "cv_fold",
                        "evaluation_included",
                        "source_url",
                        "source_path",
                        "labels_json",
                        "annotations_json",
                        "ambiguous_json",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video_id": video_id,
                        "official_subset": "validation",
                        "split": "val",
                        "cv_fold": "0",
                        "evaluation_included": "1",
                        "source_url": "https://example.invalid/video.mp4",
                        "source_path": source,
                        "labels_json": json.dumps(["LongJump"]),
                        "annotations_json": json.dumps(
                            [
                                {"label": "LongJump", "segment": [0.1, 0.8]},
                                {"label": "LongJump", "segment": [1.1, 1.5]},
                            ]
                        ),
                        "ambiguous_json": "[]",
                    }
                )
            metadata = paths["source_metadata"] / f"{video_id}.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                json.dumps({"duration_s": 1.0, "source_sha256": "source-sha"})
            )
            conversion_dir = paths["v2e"] / video_id
            conversion_dir.mkdir(parents=True)
            conversion = {
                "protocol_id": "test-protocol",
                "recipe": {
                    "output_width": 4,
                    "output_height": 3,
                    "stop_time_s": 1.0,
                },
                "implementation": {"test": True},
            }
            (conversion_dir / "conversion.json").write_text(json.dumps(conversion))
            with h5py.File(conversion_dir / "events.h5", "w") as handle:
                handle.create_dataset(
                    "events",
                    data=np.asarray([[100_000, 1, 2, 1], [800_000, 2, 1, 0]]),
                )
            output = root / "smoke" / "preprocessed.h5"
            args = Namespace(
                work_dir=root,
                video_ids=[video_id],
                shard_index=0,
                num_shards=1,
                output=output,
                chunk_size=100,
                force=False,
            )
            assemble(args)
            self.assertTrue(output.exists())
            self.assertTrue((output.parent / "config/annotations/annotations.json").exists())
            corpus_manifest = json.loads(
                (output.parent / "corpus_manifest.json").read_text()
            )
            self.assertEqual(
                corpus_manifest["conversion_protocols"]["test-protocol"]["implementation"],
                {"test": True},
            )
            self.assertFalse(paths["config"].exists())
            validate(
                Namespace(
                    work_dir=root,
                    video_ids=[video_id],
                    shard_index=0,
                    num_shards=1,
                    data_path=output,
                    chunk_size=100,
                )
            )
            report = json.loads((output.parent / "validation_report.json").read_text())
            self.assertEqual(
                report["class_summary"]["instances_by_split_and_class"]["val"]["LongJump"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
