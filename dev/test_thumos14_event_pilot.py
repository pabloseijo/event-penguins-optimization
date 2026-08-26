"""Tests for the pilot conversion of THUMOS14 to events with v2e.

Checks that the v2e command selects exactly one timing protocol, that official
class annotations parse correctly, that copying events changes only the column
order, and that the assembled index handles variable durations.
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

from dev.extract_continuous_features import build_index, paths as feature_paths
from dev.prepare_thumos14_event_pilot import (
    annotation_member,
    assemble,
    copy_v2e_events,
    parse_class_annotations,
    validate,
    v2e_command,
    work_paths,
    write_manifest,
)


class Thumos14EventPilotTest(unittest.TestCase):
    @staticmethod
    def _v2e_args(disable_slomo: bool) -> Namespace:
        return Namespace(
            v2e_python="python",
            v2e_entry=Path("/tmp/v2e.py"),
            timestamp_resolution=0.003,
            disable_slomo=disable_slomo,
            output_width=346,
            output_height=260,
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.03,
            cutoff_hz=15.0,
            leak_rate_hz=0.01,
            shot_noise_rate_hz=0.001,
            seed=1337,
            batch_size=8,
            stop_time=10.0,
        )

    def test_v2e_command_selects_exactly_one_timing_protocol(self) -> None:
        row = {"source_path": "/tmp/source.mp4"}
        slomo = v2e_command(self._v2e_args(False), row, Path("/tmp/slomo"))
        original_rate = v2e_command(
            self._v2e_args(True), row, Path("/tmp/original-rate")
        )

        self.assertIn("--timestamp_resolution=0.003", slomo)
        self.assertIn("--auto_timestamp_resolution=False", slomo)
        self.assertNotIn("--disable_slomo", slomo)
        self.assertIn("--disable_slomo", original_rate)
        self.assertFalse(any(item.startswith("--timestamp_resolution") for item in original_rate))
        self.assertNotIn("--auto_timestamp_resolution=False", original_rate)

    def test_parse_official_class_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "annotations.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "annotation/LongJump_val.txt",
                    "video_validation_1 1.0 3.0\nvideo_validation_1 5.0 8.5\n",
                )
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    annotation_member(archive, "LongJump", "validation"),
                    "annotation/LongJump_val.txt",
                )
            self.assertEqual(
                parse_class_annotations(archive_path, "LongJump", "validation"),
                {"video_validation_1": [[1.0, 3.0], [5.0, 8.5]]},
            )

    def test_copy_v2e_events_changes_only_column_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.h5"
            with h5py.File(path, "w") as handle:
                source = handle.create_dataset(
                    "source",
                    data=np.asarray([[100, 2, 3, 1], [200, 4, 5, 0]], dtype=np.uint32),
                )
                target_group = handle.create_group("target")
                copy_v2e_events(source, target_group, chunk_size=1)
                actual = np.asarray(target_group["events"])
            np.testing.assert_array_equal(
                actual,
                np.asarray([[2, 3, 100, 1], [4, 5, 200, 0]], dtype=np.uint32),
            )

    def test_assemble_validate_and_variable_duration_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "pilot"
            paths = work_paths(work_dir)
            video_id = "video_validation_0000001"
            source_video = paths["videos"] / "validation" / f"{video_id}.mp4"
            source_video.parent.mkdir(parents=True)
            source_video.write_bytes(b"fake-mp4")
            rows = [
                {
                    "video_id": video_id,
                    "class_name": "LongJump",
                    "official_subset": "validation",
                    "split": "train",
                    "source_url": "https://example.invalid/video.mp4",
                    "source_path": str(source_video),
                    "source_sha256": "test",
                    "source_bytes": str(source_video.stat().st_size),
                    "duration_s": "3.2",
                    "fps": "30",
                    "width": "640",
                    "height": "360",
                    "segments_json": json.dumps([[0.5, 2.5]]),
                }
            ]
            write_manifest(paths["manifest"], rows)
            v2e_dir = paths["v2e"] / video_id
            v2e_dir.mkdir(parents=True)
            events = np.asarray(
                [
                    [100_000, 10, 20, 1],
                    [600_000, 11, 21, 0],
                    [1_000_000, 12, 22, 1],
                    [2_000_000, 13, 23, 0],
                    [3_100_000, 14, 24, 1],
                ],
                dtype=np.uint32,
            )
            with h5py.File(v2e_dir / "events.h5", "w") as handle:
                handle.create_dataset("events", data=events)
            (v2e_dir / "conversion.json").write_text(
                json.dumps({"stop_time_s": None}), encoding="utf-8"
            )

            assemble(
                Namespace(
                    work_dir=work_dir,
                    video_ids=None,
                    output=None,
                    chunk_size=2,
                    force=False,
                )
            )
            validate(
                Namespace(
                    work_dir=work_dir,
                    data_path=None,
                    ann_path=None,
                    bin_width=0.5,
                    plot=False,
                )
            )
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["recordings"][0]["events"], len(events))

            feature_dir = work_dir / "features"
            build_index(
                Namespace(
                    data_path=paths["data"],
                    out_dir=feature_dir,
                    recordings=None,
                    splits=["train"],
                    grid_stride=0.5,
                    window_duration=1.0,
                    sequence_duration=None,
                    feature_dim=512,
                    adaptive_target_count=None,
                    adaptive_min_duration=0.5,
                    adaptive_max_duration=2.0,
                    force=False,
                )
            )
            with (feature_paths(feature_dir)["sequences"]).open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(int(row["length"]), 7)
            self.assertAlmostEqual(float(row["duration_s"]), 3.2)


if __name__ == "__main__":
    unittest.main()
