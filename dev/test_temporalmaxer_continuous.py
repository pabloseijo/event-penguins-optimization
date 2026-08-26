"""Focused invariants for the full-sequence TemporalMaxer path."""

from __future__ import annotations

import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

import dev.eval_continuous_multi_rep_fusion_test as multi_rep_test
from dev.eval_temporalmaxer_continuous_test import (
    align_reversed_offsets,
    reverse_valid_features,
    reverse_valid_level,
)
from dev.eval_continuous_multi_rep_fusion_cv import build_prediction
from dev.pretrain_temporalmaxer_continuous_bsp import (
    clamp_window_start,
    overlaps_any,
)
from dev.train_temporalmaxer_continuous import (
    ActionRegion,
    align_temporal_feature_statistics,
    build_action_regions,
    feature_background_mix,
    feature_mixstyle,
    group_dro_reduce,
    load_annotations,
    manifest_validation_recordings,
    normalize_temporal_features,
    pal_region_consistency_loss,
    reverse_temporal_sample,
    split_fold_sequences,
    transplant_action_region,
    valid_transplant_starts,
)
from src.temporalmaxer_continuous import (
    TemporalMaxerContinuous,
    temporal_aware_normalization_perturbation,
    trident_offsets,
)


class TemporalMaxerContinuousTest(unittest.TestCase):
    def test_target_test_ensemble_propagates_duration_and_class(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            auxiliary_dir = root / "event_stats"
            feature_dir.mkdir()
            auxiliary_dir.mkdir()
            (feature_dir / "metadata.json").write_text(
                '{"feature_dim": 4, "num_points": 1, "grid_stride_s": 0.5}',
                encoding="utf-8",
            )
            pd.DataFrame(
                {"rec_name": ["video"], "roi_id": [1], "duration_s": [1.0]}
            ).to_csv(feature_dir / "sequences.csv", index=False)
            np.save(auxiliary_dir / "event_stats.npy", np.zeros((1, 2), np.float32))
            args = Namespace(
                feature_dir=str(feature_dir),
                auxiliary_feature_dir=str(auxiliary_dir),
                continuous_root=str(root / "continuous"),
                event_root=str(root / "event"),
                feature_normalization="none",
                proposal_prediction=str(root / "unused.json"),
                ann_path=str(root / "unused_annotations.json"),
                out_dir=str(root / "out"),
                batch_size=1,
                num_workers=0,
                device="cpu",
                target_class="LongJump",
                recording_manifest=None,
                recording_subset=None,
                min_action_duration=0.0,
                tiou=[0.3, 0.4, 0.5, 0.6, 0.7],
                ensemble_only=True,
                weights=(0.2, 0.4, 0.4),
            )
            prediction = {"results": {"video": {"1": []}}}
            with patch.object(
                multi_rep_test, "parse_args", return_value=args
            ), patch.object(
                multi_rep_test,
                "auxiliary_normalization",
                return_value=(
                    np.zeros(2, np.float32),
                    np.ones(2, np.float32),
                    2,
                ),
            ), patch.object(
                multi_rep_test, "make_loader", return_value=object()
            ):
                with patch.object(
                    multi_rep_test, "load_models", return_value=[object()]
                ) as load, patch.object(
                    multi_rep_test,
                    "cached_ensemble",
                    return_value=prediction,
                ) as cached:
                    multi_rep_test.main()

            self.assertEqual(len(load.call_args_list), 2)
            self.assertEqual(len(load.call_args_list[0].args), 4)
            self.assertEqual(len(cached.call_args_list), 2)
            for call in cached.call_args_list:
                self.assertEqual(call.args[-2:], (0.0, "LongJump"))

    def test_inference_manifest_selects_only_official_test(self) -> None:
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            pd.DataFrame(
                {
                    "video_id": ["train_video", "test_video"],
                    "official_subset": ["validation", "test"],
                }
            ).to_csv(manifest, index=False)
            sequences = pd.DataFrame(
                {
                    "rec_name": ["train_video", "test_video"],
                    "roi_id": [1, 1],
                }
            )
            selected = multi_rep_test.select_inference_sequences(
                sequences, str(manifest), "test"
            )
            self.assertEqual(selected["rec_name"].tolist(), ["test_video"])

    def test_cached_ensemble_rejects_foreign_recording_universe(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cached.json"
            path.write_text(
                '{"target_class":"LongJump","results":{"validation_video":{}}}',
                encoding="utf-8",
            )
            sequences = pd.DataFrame({"rec_name": ["test_video"]})
            with self.assertRaisesRegex(ValueError, "foreign recording"):
                multi_rep_test.cached_ensemble(
                    path,
                    [],
                    object(),
                    sequences,
                    0.5,
                    torch.device("cpu"),
                    0.0,
                    "LongJump",
                )

    def test_three_expert_fusion_keeps_short_thumos_candidates_at_zero(self) -> None:
        frame = pd.DataFrame(
            {
                "rec_name": ["video"],
                "roi_id": [1],
                "t_start": [0.0],
                "t_end": [1.0],
                "raw_score": [0.9],
                "rank_score": [1.0],
                "model": ["continuous"],
            }
        )
        target = build_prediction(
            [frame],
            {"continuous": 1.0},
            sigma=0.5,
            per_model_topk=100,
            max_predictions=200,
            min_action_duration=0.0,
        )
        source = build_prediction(
            [frame],
            {"continuous": 1.0},
            sigma=0.5,
            per_model_topk=100,
            max_predictions=200,
            min_action_duration=2.0,
        )
        self.assertEqual(len(target["results"]["video"]["1"]), 1)
        self.assertEqual(len(source["results"]["video"]["1"]), 0)

    def test_manifest_validation_pool_does_not_include_test_sentinel(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "fold": 0,
                    "train_record_names": "train_a train_b",
                    "val_record_names": "val_a",
                },
                {
                    "fold": 1,
                    "train_record_names": "train_a val_a",
                    "val_record_names": "train_b",
                },
            ]
        )
        recordings = manifest_validation_recordings(manifest)
        self.assertEqual(recordings, {"train_a", "train_b", "val_a"})
        self.assertNotIn("test_sentinel", recordings)

    def test_target_annotation_mode_keeps_sub_two_second_actions(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            path.write_text(
                '{"database":{"video":{"annotations":{"1":['
                '{"label":"ed","segment":[0.0,1.0]},'
                '{"label":"ed","segment":[2.0,5.0]}]}}}}',
                encoding="utf-8",
            )
            source = load_annotations(path)
            target = load_annotations(path, min_duration_s=0.0)
        np.testing.assert_allclose(source[("video", 1)], [[2.0, 5.0]])
        np.testing.assert_allclose(target[("video", 1)], [[0.0, 1.0], [2.0, 5.0]])

    def test_fold_split_excludes_test_sentinel(self) -> None:
        sequences = pd.DataFrame(
            {
                "rec_name": ["train_video", "val_video", "test_sentinel"],
                "roi_id": [1, 1, 1],
            }
        )
        manifest = pd.DataFrame(
            [
                {
                    "fold": 0,
                    "train_record_names": "train_video",
                    "val_record_names": "val_video",
                }
            ]
        )
        train, validation = split_fold_sequences(sequences, manifest, 0)
        self.assertEqual(train["rec_name"].tolist(), ["train_video"])
        self.assertEqual(validation["rec_name"].tolist(), ["val_video"])
        self.assertNotIn(
            "test_sentinel",
            set(pd.concat([train, validation])["rec_name"].astype(str)),
        )

    def make_model(
        self, quality: bool = True, reg_max: int = 0, boundaries: bool = False
    ) -> TemporalMaxerContinuous:
        return TemporalMaxerContinuous(
            input_dim=16,
            hidden_dim=8,
            pyramid_levels=3,
            head_layers=2,
            dropout=0.0,
            regression_ranges=((0.0, 8.0), (4.0, 16.0), (8.0, float("inf"))),
            use_quality=quality,
            reg_max=reg_max,
            use_boundary_heads=boundaries,
            boundary_refine_radius_seconds=2.0 if boundaries else 0.0,
        )

    def test_forward_preserves_dense_multiscale_geometry(self) -> None:
        model = self.make_model()
        features = torch.randn(2, 32, 16)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[1, 27:] = False
        output = model(features, mask)
        self.assertEqual(len(output["pyramid_features"]), 3)
        self.assertEqual(output["pyramid_features"][0].shape, (2, 8, 32))
        self.assertEqual([tuple(value.shape) for value in output["classification_logits"]], [(2, 32), (2, 16), (2, 8)])
        self.assertEqual([tuple(value.shape) for value in output["offsets"]], [(2, 32, 2), (2, 16, 2), (2, 8, 2)])
        self.assertTrue(all((value >= 0).all() for value in output["offsets"]))
        self.assertFalse(output["masks"][0][1, 27:].any())

    def test_cross_layer_task_decoupling_routes_auxiliary_to_localization(self) -> None:
        model = TemporalMaxerContinuous(
            input_dim=20,
            classification_input_dim=16,
            hidden_dim=8,
            pyramid_levels=2,
            head_layers=2,
            dropout=0.0,
            regression_ranges=((0.0, 8.0), (4.0, float("inf"))),
        ).eval()
        features = torch.randn(1, 16, 20)
        changed_auxiliary = features.clone()
        changed_auxiliary[..., 16:] += 10.0
        first = model(features)
        second = model(changed_auxiliary)
        for left, right in zip(
            first["classification_logits"],
            second["classification_logits"],
        ):
            torch.testing.assert_close(left, right)
        self.assertFalse(
            torch.allclose(first["offsets"][0], second["offsets"][0])
        )

    def test_temporal_reversal_keeps_padding_and_restores_level_order(self) -> None:
        features = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        reversed_features = reverse_valid_features(features, mask)
        torch.testing.assert_close(
            reversed_features[0], features[0].flip(0)
        )
        torch.testing.assert_close(
            reversed_features[1],
            torch.stack((features[1, 1], features[1, 0], features[1, 2])),
        )
        restored = reverse_valid_level(
            torch.tensor([[3.0, 2.0, 1.0], [5.0, 4.0, 0.0]]),
            mask,
        )
        torch.testing.assert_close(
            restored,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]),
        )

    def test_temporal_reversal_swaps_left_and_right_offsets(self) -> None:
        mask = torch.tensor([[True, True, True]])
        reversed_offsets = torch.tensor(
            [[[30.0, 3.0], [20.0, 2.0], [10.0, 1.0]]]
        )
        aligned = align_reversed_offsets(reversed_offsets, mask)
        torch.testing.assert_close(
            aligned,
            torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]),
        )

    def test_training_reversal_maps_features_and_segments_equivariantly(self) -> None:
        features = np.arange(12, dtype=np.float32).reshape(4, 3)
        segments = np.asarray([[1.0, 3.0], [6.0, 9.0]], dtype=np.float32)
        reversed_features, reversed_segments = reverse_temporal_sample(
            features,
            segments,
            duration_s=10.0,
        )
        np.testing.assert_array_equal(reversed_features, features[::-1])
        np.testing.assert_allclose(
            reversed_segments,
            np.asarray([[1.0, 4.0], [7.0, 9.0]], dtype=np.float32),
        )

    def test_trident_combines_center_and_neighbour_boundary_logits(self) -> None:
        center = torch.zeros(1, 5, 2, 4)
        start = torch.full((1, 5), -8.0)
        end = torch.full((1, 5), -8.0)
        start[0, 0] = 8.0
        end[0, 4] = 8.0
        offsets = trident_offsets(center, start, end, num_bins=3)
        self.assertAlmostEqual(float(offsets[0, 3, 0]), 3.0, places=4)
        self.assertAlmostEqual(float(offsets[0, 1, 1]), 3.0, places=4)

    def test_trident_head_backpropagates_through_relative_boundaries(self) -> None:
        model = TemporalMaxerContinuous(
            input_dim=16,
            hidden_dim=8,
            pyramid_levels=2,
            head_layers=2,
            dropout=0.0,
            regression_ranges=((0.0, 8.0), (4.0, float("inf"))),
            trident_bins=4,
        )
        output = model(torch.randn(1, 24, 16))
        self.assertEqual(output["offsets"][0].shape, (1, 24, 2))
        self.assertTrue((output["offsets"][0] <= 4.0).all())
        losses = model.losses(
            output,
            [torch.tensor([[4.0, 9.0]])],
            grid_stride_seconds=0.5,
        )
        losses["loss"].backward()
        self.assertIsNotNone(model.trident_start_head.weight.grad)
        self.assertIsNotNone(model.trident_end_head.weight.grad)

    def test_continuous_bsp_window_helpers_are_boundary_safe(self) -> None:
        self.assertEqual(clamp_window_start(-4, 20, 8), 0)
        self.assertEqual(clamp_window_start(18, 20, 8), 12)
        segments = np.asarray([[5.0, 8.0]], dtype=np.float32)
        self.assertTrue(overlaps_any(4.0, 6.0, segments))
        self.assertFalse(overlaps_any(8.0, 12.0, segments))

    def test_assignment_uses_full_sequence_points_and_ranges(self) -> None:
        model = self.make_model()
        classes, regression = model.targets_for_sequence(
            [32, 16, 8], torch.tensor([[8.0, 16.0]]), torch.device("cpu")
        )
        self.assertGreater(int(classes[0].sum()), 0)
        positive = classes[0].bool()
        self.assertTrue((regression[0][positive] >= 0).all())
        self.assertTrue((classes[0][:7] == 0).all())
        self.assertTrue((classes[0][16:] == 0).all())

    def test_empty_ground_truth_loss_is_finite_and_differentiable(self) -> None:
        model = self.make_model()
        output = model(torch.randn(2, 32, 16))
        losses = model.losses(
            output,
            [torch.empty(0, 2), torch.empty(0, 2)],
            grid_stride_seconds=0.5,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(model.classification_head.weight.grad)

    def test_positive_loss_optimizes_one_step(self) -> None:
        torch.manual_seed(4)
        model = self.make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        features = torch.randn(2, 32, 16)
        gt = [torch.tensor([[4.0, 9.0]]), torch.tensor([[8.0, 13.0]])]
        output = model(features)
        before = model.losses(output, gt, 0.5)["loss"]
        optimizer.zero_grad()
        before.backward()
        optimizer.step()
        after = model.losses(model(features), gt, 0.5)["loss"]
        self.assertTrue(torch.isfinite(after))
        self.assertGreater(float(before), 0.0)

    def test_rank_sort_classification_is_finite_and_trainable(self) -> None:
        torch.manual_seed(14)
        model = self.make_model(quality=False)
        output = model(torch.randn(2, 32, 16))
        losses = model.losses(
            output,
            [torch.tensor([[4.0, 9.0]]), torch.tensor([[8.0, 13.0]])],
            0.5,
            rank_sort=True,
        )
        self.assertTrue(torch.isfinite(losses["classification_loss"]))
        self.assertGreater(float(losses["classification_loss"]), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(model.classification_head.weight.grad)
        self.assertTrue(torch.isfinite(model.classification_head.weight.grad).all())

    def test_decode_maps_grid_coordinates_to_seconds(self) -> None:
        model = self.make_model(quality=False)
        logits = [torch.full((1, 8), -20.0), torch.full((1, 4), -20.0), torch.full((1, 2), -20.0)]
        logits[0][0, 4] = 20.0
        output = {
            "classification_logits": logits,
            "quality_logits": [torch.zeros_like(value) for value in logits],
            "offsets": [torch.zeros(1, 8, 2), torch.zeros(1, 4, 2), torch.zeros(1, 2, 2)],
            "masks": [torch.ones_like(value, dtype=torch.bool) for value in logits],
        }
        output["offsets"][0][0, 4] = torch.tensor([2.5, 3.5])
        decoded = model.decode(
            output,
            grid_stride_seconds=0.5,
            durations_seconds=torch.tensor([10.0]),
            score_threshold=0.5,
        )[0]
        self.assertEqual(len(decoded), 1)
        self.assertTrue(torch.allclose(decoded[0, :2], torch.tensor([1.0, 4.0])))

    def test_decode_visits_every_pyramid_level_for_a_batch(self) -> None:
        model = self.make_model(quality=False)
        output = model(torch.randn(2, 32, 16))
        for logits in output["classification_logits"]:
            logits.fill_(10.0)
        decoded = model.decode(
            output,
            grid_stride_seconds=0.5,
            durations_seconds=torch.tensor([16.0, 16.0]),
            score_threshold=0.5,
            min_duration_seconds=0.0,
        )
        self.assertEqual(len(decoded), 2)
        self.assertTrue(all(len(value) > 0 for value in decoded))

    def test_distributional_boundaries_are_finite_and_trainable(self) -> None:
        model = self.make_model(reg_max=16)
        features = torch.randn(2, 32, 16)
        output = model(features)
        self.assertEqual(output["offset_distributions"][0].shape, (2, 32, 2, 17))
        self.assertTrue(torch.isfinite(output["offsets"][0]).all())
        losses = model.losses(
            output,
            [torch.tensor([[4.0, 9.0]]), torch.tensor([[8.0, 13.0]])],
            grid_stride_seconds=0.5,
        )
        self.assertGreater(float(losses["distribution_loss"]), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(model.regression_head.weight.grad)

    def test_center_sampling_restricts_positive_assignments(self) -> None:
        unrestricted = self.make_model()
        centered = self.make_model()
        centered.center_sampling_radius = 1.5
        segment = torch.tensor([[4.0, 20.0]])
        unrestricted_targets, _ = unrestricted.targets_for_sequence(
            [32, 16, 8], segment, torch.device("cpu")
        )
        centered_targets, _ = centered.targets_for_sequence(
            [32, 16, 8], segment, torch.device("cpu")
        )
        self.assertLess(
            sum(int(value.sum()) for value in centered_targets),
            sum(int(value.sum()) for value in unrestricted_targets),
        )
        self.assertGreater(sum(int(value.sum()) for value in centered_targets), 0)

    def test_empty_sequence_weight_increases_background_penalty(self) -> None:
        torch.manual_seed(7)
        model = self.make_model()
        output = model(torch.randn(2, 32, 16))
        targets = [torch.tensor([[4.0, 9.0]]), torch.empty(0, 2)]
        control = model.losses(output, targets, 0.5, empty_sequence_weight=1.0)
        exposed = model.losses(output, targets, 0.5, empty_sequence_weight=2.0)
        self.assertGreater(
            float(exposed["classification_loss"]),
            float(control["classification_loss"]),
        )

    def test_boundary_heads_have_multiscale_targets_and_loss(self) -> None:
        model = self.make_model(boundaries=True)
        output = model(torch.randn(2, 32, 16))
        self.assertEqual(
            [tuple(value.shape) for value in output["start_boundary_logits"]],
            [(2, 32), (2, 16), (2, 8)],
        )
        starts, ends = model.boundary_targets_for_sequence(
            [32, 16, 8], torch.tensor([[8.0, 16.0]]), torch.device("cpu"), 1.0
        )
        self.assertGreater(float(starts[0].max()), 0.5)
        self.assertGreater(float(ends[0].max()), 0.5)
        losses = model.losses(
            output,
            [torch.tensor([[4.0, 9.0]]), torch.tensor([[8.0, 13.0]])],
            0.5,
        )
        self.assertGreater(float(losses["boundary_loss"]), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(model.start_boundary_head.weight.grad)

    def test_temporal_feature_normalization_is_label_free_and_stable(self) -> None:
        features = np.asarray(
            [[1.0, 4.0, 7.0], [2.0, 4.0, 9.0], [3.0, 4.0, 11.0]],
            dtype=np.float32,
        )
        self.assertIs(normalize_temporal_features(features, "none"), features)
        centered = normalize_temporal_features(features, "temporal-center")
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-6)
        normalized = normalize_temporal_features(features, "temporal-zscore")
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized[:, [0, 2]].std(axis=0), 1.0, atol=1e-6)
        np.testing.assert_allclose(normalized[:, 1], 0.0, atol=1e-6)

    def test_temporal_feature_alignment_matches_source_statistics(self) -> None:
        features = np.asarray(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32
        )
        source_mean = np.asarray([10.0, -3.0], dtype=np.float32)
        source_std = np.asarray([2.0, 4.0], dtype=np.float32)
        aligned = align_temporal_feature_statistics(
            features, source_mean, source_std, blend=1.0
        )
        np.testing.assert_allclose(aligned.mean(axis=0), source_mean, atol=1e-5)
        np.testing.assert_allclose(aligned.std(axis=0), source_std, atol=1e-5)
        identity = align_temporal_feature_statistics(
            features, source_mean, source_std, blend=0.0
        )
        np.testing.assert_allclose(identity, features)
        selected_mean = np.asarray([2.0, 5.0], dtype=np.float32)
        selected_std = np.asarray([0.5, 0.5], dtype=np.float32)
        selected = align_temporal_feature_statistics(
            features,
            source_mean,
            source_std,
            blend=1.0,
            target_mean=selected_mean,
            target_std=selected_std,
        )
        expected = (features - selected_mean) / selected_std * source_std + source_mean
        np.testing.assert_allclose(selected, expected)

    def test_dataset_can_standardize_a_non_atsn_feature_matrix(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "event_stats.npy"
            np.save(path, np.asarray([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32))
            sequences = __import__("pandas").DataFrame(
                [
                    {
                        "offset": 0,
                        "length": 2,
                        "rec_name": "recording",
                        "roi_id": 1,
                        "duration_s": 1.0,
                    }
                ]
            )
            from dev.train_temporalmaxer_continuous import ContinuousSequenceDataset

            dataset = ContinuousSequenceDataset(
                path,
                sequences,
                annotations={},
                feature_channel_mean=np.asarray([2.0, 6.0], dtype=np.float32),
                feature_channel_std=np.asarray([1.0, 2.0], dtype=np.float32),
            )
            np.testing.assert_allclose(
                dataset[0]["features"].numpy(), [[-1.0, -1.0], [1.0, 1.0]]
            )

    def test_group_dro_upweights_the_higher_risk_recording(self) -> None:
        losses = torch.tensor([1.0, 1.0, 3.0, 3.0], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])
        weights = torch.tensor([0.5, 0.5])
        robust = group_dro_reduce(losses, groups, weights, eta=0.1)
        self.assertGreater(float(weights[1]), float(weights[0]))
        robust.backward()
        self.assertGreater(float(losses.grad[2]), float(losses.grad[0]))

    def test_tanp_preserves_temporal_differences_up_to_channel_scale(self) -> None:
        features = torch.tensor(
            [[[1.0, 2.0, 4.0, 3.0], [2.0, 1.0, 3.0, 5.0]]],
            requires_grad=True,
        )
        mask = torch.ones(1, 4, dtype=torch.bool)
        alpha = torch.full((1, 2, 1), 2.0)
        beta = torch.full((1, 2, 1), 3.0)
        perturbed = temporal_aware_normalization_perturbation(
            features, mask, std=0.75, alpha=alpha, beta=beta
        )
        self.assertTrue(
            torch.allclose(
                perturbed[:, :, 1:] - perturbed[:, :, :-1],
                2.0 * (features[:, :, 1:] - features[:, :, :-1]),
                atol=1e-6,
            )
        )
        perturbed.sum().backward()
        self.assertIsNotNone(features.grad)

    def test_tanp_identity_noise_is_identity_and_masks_padding(self) -> None:
        features = torch.randn(2, 3, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[1, 4:] = False
        identity = torch.ones(2, 3, 1)
        perturbed = temporal_aware_normalization_perturbation(
            features, mask, alpha=identity, beta=identity
        )
        self.assertTrue(torch.allclose(perturbed[0], features[0], atol=1e-6))
        self.assertTrue(torch.allclose(perturbed[1, :, :4], features[1, :, :4], atol=1e-6))
        self.assertFalse(perturbed[1, :, 4:].bool().any())

    def test_feature_background_mix_matches_glad_equation(self) -> None:
        features = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        background = np.asarray([9.0, 11.0], dtype=np.float32)
        mixed = feature_background_mix(features, background, mix_ratio=0.25)
        np.testing.assert_allclose(
            mixed, 0.75 * features + 0.25 * background[None, :]
        )

    def test_feature_mixstyle_adopts_donor_statistics(self) -> None:
        features = np.asarray(
            [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]], dtype=np.float32
        )
        donor_mean = np.asarray([10.0, 20.0], dtype=np.float32)
        donor_std = np.asarray([2.0, 4.0], dtype=np.float32)
        mixed = feature_mixstyle(
            features, donor_mean, donor_std, recipient_weight=0.0
        )
        np.testing.assert_allclose(mixed.mean(axis=0), donor_mean, atol=1e-5)
        np.testing.assert_allclose(mixed.std(axis=0), donor_std, atol=1e-5)

    def test_pal_action_regions_retain_fractional_boundaries(self) -> None:
        sequences = pd.DataFrame(
            [
                {
                    "rec_name": "donor",
                    "roi_id": 1,
                    "length": 20,
                }
            ]
        )
        regions = build_action_regions(
            sequences,
            {("donor", 1): np.asarray([[2.25, 5.75]], dtype=np.float32)},
            grid_stride_s=0.5,
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual((regions[0].crop_start, regions[0].crop_end), (4, 12))
        self.assertAlmostEqual(regions[0].segment_start_offset_s, 0.25)
        self.assertAlmostEqual(regions[0].segment_end_offset_s, 3.75)

    def test_pal_transplant_uses_only_unoccupied_background(self) -> None:
        starts = valid_transplant_starts(
            sequence_length=12,
            region_length=3,
            segments=np.asarray([[2.0, 4.0]], dtype=np.float32),
            grid_stride_s=1.0,
            margin_bins=1,
        )
        self.assertEqual(starts.tolist(), [5, 6, 7, 8, 9])

        recipient = np.zeros((12, 2), dtype=np.float32)
        donor = np.arange(20, dtype=np.float32).reshape(10, 2)
        region = ActionRegion(
            sequence_index=0,
            rec_name="donor",
            crop_start=2,
            crop_end=5,
            segment_start_offset_s=0.25,
            segment_end_offset_s=2.75,
        )
        features, segments = transplant_action_region(
            recipient,
            np.asarray([[2.0, 4.0]], dtype=np.float32),
            donor,
            region,
            destination_start=6,
            grid_stride_s=1.0,
        )
        np.testing.assert_array_equal(features[6:9], donor[2:5])
        np.testing.assert_allclose(segments, [[2.0, 4.0], [6.25, 8.75]])

    def test_pal_transplant_softly_blends_action_and_background(self) -> None:
        recipient = np.full((6, 2), 2.0, dtype=np.float32)
        donor = np.full((4, 2), 10.0, dtype=np.float32)
        region = ActionRegion(
            sequence_index=0,
            rec_name="donor",
            crop_start=1,
            crop_end=3,
            segment_start_offset_s=0.0,
            segment_end_offset_s=2.0,
        )
        features, segments = transplant_action_region(
            recipient,
            np.empty((0, 2), dtype=np.float32),
            donor,
            region,
            destination_start=2,
            grid_stride_s=1.0,
            blend_ratio=0.75,
        )
        np.testing.assert_allclose(features[2:4], 8.0)
        np.testing.assert_allclose(features[:2], 2.0)
        np.testing.assert_allclose(features[4:], 2.0)
        np.testing.assert_allclose(segments, [[2.0, 4.0]])

    def test_pal_region_consistency_prefers_matching_pairs(self) -> None:
        recipient = torch.zeros(2, 3, 6)
        donor = torch.zeros(2, 3, 6)
        recipient[0, 0, 1:3] = 1.0
        recipient[1, 1, 2:4] = 1.0
        donor[0, 0, 3:5] = 1.0
        donor[1, 1, 1:3] = 1.0
        indices = torch.tensor([0, 1])
        recipient_spans = torch.tensor([[1, 3], [2, 4]])
        donor_spans = torch.tensor([[3, 5], [1, 3]])

        matching = pal_region_consistency_loss(
            recipient,
            donor,
            indices,
            recipient_spans,
            donor_spans,
            temperature=0.07,
        )
        mismatched = pal_region_consistency_loss(
            recipient,
            donor.flip(0),
            indices,
            recipient_spans,
            donor_spans.flip(0),
            temperature=0.07,
        )
        self.assertLess(float(matching), float(mismatched))

    def test_temporal_order_task_is_trainable(self) -> None:
        model = TemporalMaxerContinuous(
            input_dim=16,
            hidden_dim=8,
            pyramid_levels=2,
            head_layers=2,
            dropout=0.0,
            regression_ranges=((0.0, 8.0), (4.0, float("inf"))),
            use_temporal_order=True,
            temporal_order_chunks=3,
        )
        output = model(torch.randn(2, 30, 16))
        loss = model.temporal_order_loss(
            output["pyramid_features"][0], output["masks"][0]
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(model.temporal_order_head[-1].weight.grad)
        self.assertIsNotNone(model.input_projection[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
