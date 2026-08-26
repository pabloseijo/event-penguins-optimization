"""Tests for the supervised THUMOS14-E runner and its protocol locks.

Checks that the protocol is one corpus with canonical counts, that annotation
views come from the assembler directory, that the corpus and the conversion role
stay locked, and that a fold manifest can never include test videos.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dev.prepare_thumos14_event_corpus import THUMOS_CLASSES
from dev.run_thumos14e_supervised import (
    DECISION_TRACEABILITY,
    NATIVE_MODEL_TOPOLOGY,
    REQUIRED_FULL_COMPONENTS,
    annotation_config_dir,
    class_selection,
    fold_recordings,
    full_stack_status,
    target_prototype_path,
    target_recordings,
    validate_assembled_corpus,
    validate_corpus_protocol,
)


class RunThumos14eSupervisedTest(unittest.TestCase):
    def make_manifest(self) -> pd.DataFrame:
        rows = [
            {
                "video_id": f"video_validation_{index:07d}",
                "official_subset": "validation",
                "evaluation_included": True,
            }
            for index in range(200)
        ]
        rows.extend(
            {
                "video_id": f"video_test_{index:07d}",
                "official_subset": "test",
                "evaluation_included": index < 212,
            }
            for index in range(213)
        )
        return pd.DataFrame(rows)

    def make_folds(self) -> pd.DataFrame:
        recordings = [f"video_validation_{index:07d}" for index in range(200)]
        rows = []
        for fold in range(5):
            validation = recordings[fold * 40 : (fold + 1) * 40]
            training = [name for name in recordings if name not in set(validation)]
            rows.append(
                {
                    "fold": fold,
                    "train_record_names": " ".join(training),
                    "val_record_names": " ".join(validation),
                }
            )
        return pd.DataFrame(rows)

    def test_protocol_is_one_corpus_with_canonical_counts(self) -> None:
        manifest = self.make_manifest()
        audit = validate_corpus_protocol(manifest, self.make_folds())
        self.assertEqual(audit["validation_videos"], 200)
        self.assertEqual(audit["test_files"], 213)
        self.assertEqual(audit["canonical_test_videos"], 212)
        self.assertEqual(len(target_recordings(manifest, "validation")), 200)

    def test_annotation_views_use_the_assembler_canonical_directory(self) -> None:
        self.assertEqual(
            annotation_config_dir(Path("/corpus")),
            Path("/corpus/config/annotations"),
        )

    def test_complete_corpus_and_conversion_role_are_locked(self) -> None:
        manifest = self.make_manifest()
        records = [
            {"video_id": video_id}
            for video_id in manifest["video_id"].tolist()
        ]
        counts = {label: 1 for label in THUMOS_CLASSES}
        assembly = {
            "partial": False,
            "index_sha256": "event-hash",
            "recordings": records,
            "conversion_protocols": {
                "protocol-hash": {
                    "recipe": {
                        "timing": "fixed_interpolated",
                        "dvs_profile": "clean",
                        "timestamp_resolution_s": 0.003,
                        "disable_slomo": False,
                    },
                    "implementation": {},
                }
            },
        }
        validation = {
            "status": "ok",
            "index_sha256": "event-hash",
            "recordings": records,
            "class_summary": {
                "instances_by_split_and_class": {
                    split: counts for split in ("train", "val", "test")
                }
            },
        }
        source_audit = {"videos": 413, "hashes_verified": True}
        audit = validate_assembled_corpus(
            manifest,
            assembly,
            validation,
            source_audit,
            data_sha256="event-hash",
            conversion_role="primary_fixed_interpolated",
        )
        self.assertEqual(audit["validated_recordings"], 413)
        with self.assertRaisesRegex(ValueError, "requires timing"):
            validate_assembled_corpus(
                manifest,
                assembly,
                validation,
                source_audit,
                data_sha256="event-hash",
                conversion_role="sensitivity_original_rate",
            )

    def test_fold_manifest_cannot_include_test(self) -> None:
        folds = self.make_folds()
        folds.loc[0, "train_record_names"] += " video_test_0000000"
        with self.assertRaisesRegex(ValueError, "exactly the validation pool"):
            validate_corpus_protocol(self.make_manifest(), folds)

    def test_fold_prototypes_use_train_pool_and_final_has_distinct_path(self) -> None:
        folds = self.make_folds()
        self.assertEqual(len(fold_recordings(folds, 2, "train")), 160)
        self.assertEqual(len(fold_recordings(folds, 2, "val")), 40)
        root = Path("/tmp/protocol")
        self.assertNotEqual(
            target_prototype_path(root, "LongJump", 2),
            target_prototype_path(root, "LongJump", None),
        )

    def test_full_name_requires_every_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / f"{name}.json" for name in REQUIRED_FULL_COMPONENTS}
            paths["stage1_actionness_reshape"].touch()
            partial = full_stack_status(paths)
            self.assertFalse(partial["reportable_as_eventpenguins_full"])
            for path in paths.values():
                path.touch()
            complete = full_stack_status(paths)
            self.assertTrue(complete["reportable_as_eventpenguins_full"])

    def test_every_protocol_decision_declares_evidence_and_status(self) -> None:
        self.assertGreaterEqual(len(DECISION_TRACEABILITY), 6)
        for record in DECISION_TRACEABILITY.values():
            self.assertIn(
                record["status"],
                {
                    "published_rule",
                    "published_precedent",
                    "predeclared_adaptation",
                    "method_under_test",
                },
            )
            self.assertTrue(record["evidence"])
            self.assertTrue(record["decision"])

    def test_native_topology_does_not_replace_published_retag_with_ensemble(self) -> None:
        self.assertIn("one target ATSN head", NATIVE_MODEL_TOPOLOGY["retag"]["primary"])
        self.assertIn("secondary_only", NATIVE_MODEL_TOPOLOGY["capacity_control"])

    def test_class_selection_rejects_unknown_and_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            class_selection(["NotAClass"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            class_selection(["LongJump", "LongJump"])


if __name__ == "__main__":
    unittest.main()
