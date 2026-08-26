"""Tests for the dense TemporalMaxer heads, their training loop and decoding.

Covers exposed frame features, duration handling, TANP behaviour at zero and
non-zero sigma, and the shape and gradient contracts of the detector.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from dev.extract_event_spectral_features import local_event_descriptors
from dev.extract_tespec_features import corrupt_stacked_sequence, stacked_histogram
from dev.extract_tism_features import tism_maps
from dev.analyze_boundary_oracle_cv import oracle_choice
from dev.eval_boundary_quality_router_cv import (
    CandidateQualityRouter,
    candidate_tiou,
    post_nms_training_indices,
)
from dev.eval_boundary_router_post_nms_cv import temporal_soft_nms_indices
from dev.eval_temporal_boundary_router_cv import (
    TemporalCandidateRouter,
    select_boundary_candidates,
)
from dev.eval_salient_boundary_router_cv import (
    SalientBoundaryRouter,
    add_router_shrinkage,
    sample_at_relative_positions,
)
from dev.eval_temporalmaxer_groupdro_cv import add_ranking_fusions
from dev.eval_temporalmaxer_cross_representation_cv import geometric_trimodal_score
from dev.eval_temporalmaxer_score_fusion_test import add_boundary_modes
from dev.eval_tespec_boundary_gate_cv import quality_aware_boundaries
from dev.eval_atsn_dense_soup_cv import soup_boundaries
from dev.eval_atsn_dense_lpft_test import ensemble_test_frames
from dev.eval_atsn_pointwise_boundaries_cv import (
    crossing_intervals,
    stabilized_boundaries,
)
from dev.eval_boundary_score_voting_cv import (
    consensus_rescore_detections,
    ensure_boundary_columns,
    vote_detection_boundaries,
)
from dev.eval_multi_expert_boundary_voting_cv import (
    expert_boundary_tensor,
    multi_expert_vote_boundaries,
    weighted_median,
)
from dev.eval_multi_expert_boundary_voting_test import add_test_expert_boundaries
from dev.eval_proposal_context_cv import (
    ProposalContextTCN,
    add_context_score_variants,
    build_proposal_chunks,
    chunks_cover_all,
    local_pairwise_loss,
    proposal_numeric_features,
    quantile_match_scores,
)
from dev.train_atsn_temporalmaxer_lpft import (
    EndToEndDenseDetector,
    configure_backbone_trainable,
    outputs_to_scored,
)
from dev.eval_tespec_coral_cv import feature_moments, recording_coral_affines
from dev.eval_tespec_eventmatch_cv import (
    ResidualEventAdapter,
    TemporalResidualEventAdapter,
    ConditionalSemanticDiscriminator,
    reverse_gradient,
    semantic_domain_loss,
    sada_conditional_domain_loss,
)
from dev.eval_temporalmaxer_bsp_cv import build_bsp_specification
from dev.train_temporalmaxer_dense import (
    DenseFeatureDataset,
    build_prediction as build_dense_prediction,
    build_targets,
    dense_loss,
    drop_one_event_frame,
    map_to_master,
    score_model,
    temporal_robust_consistency_loss,
)
from dev.train_quality_head import (
    apply_oof_hardness,
    predictions_to_df,
    tanp_standardized_roles,
)
from src.augmented_tsn import AugmentedTsn
from src.bsp import (
    DIFFERENT_CLASS,
    DIFFERENT_SPEED,
    SAME_CLASS,
    SAME_SPEED,
    BoundaryTypeHead,
    boundary_type_loss,
    synthesize_bsp_sequences,
)
from src.temporalmaxer_lite import (
    TemporalMaxerLiteHead,
    temporal_aware_normalization_perturbation,
)
from src.utils import temporal_soft_nms


class AugmentedTsnFeatureTest(unittest.TestCase):
    def test_exposed_frame_features_preserve_logits(self) -> None:
        torch.manual_seed(7)
        model = AugmentedTsn(2, num_tsn_samples=7, augment_factor=5).eval()
        images = torch.randn(2, 11, 3, 64, 64)
        with torch.no_grad():
            expected = model(images)
            actual, frame_features = model.forward_with_frame_features(images)
        self.assertEqual(tuple(frame_features.shape), (2, 11, 512))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class TemporalMaxerLiteHeadTest(unittest.TestCase):
    def test_target_zero_duration_keeps_short_predictions(self) -> None:
        scored = pd.DataFrame(
            {
                "rec_name": ["video_validation_0000001"],
                "roi_id": ["N01"],
                "t_start": [0.0],
                "t_end": [1.0e6],
                "score": [0.9],
            }
        )
        args = SimpleNamespace(
            min_score=0.0,
            duration_dmax=60.0,
            duration_sigma=20.0,
            pre_nms_topk_per_roi=100,
            soft_nms_sigma=0.5,
            soft_nms_score_threshold=0.0,
            min_action_duration=0.0,
        )
        target = build_dense_prediction(scored, "score", "raw", args)
        self.assertEqual(len(target["results"]["video_validation_0000001"][1]), 1)
        source = predictions_to_df(target, min_duration=2.0)
        self.assertTrue(source.empty)
        target_frame = predictions_to_df(target, min_duration=0.0)
        self.assertEqual(len(target_frame), 1)

    def test_tanp_identity_at_zero_sigma(self) -> None:
        features = torch.randn(3, 8, 11)
        actual = temporal_aware_normalization_perturbation(features, sigma=0.0)
        torch.testing.assert_close(actual, features)

    def test_tanp_is_finite_differentiable_and_shape_preserving(self) -> None:
        torch.manual_seed(7)
        features = torch.randn(3, 8, 11, requires_grad=True)
        actual = temporal_aware_normalization_perturbation(features, sigma=0.75)
        self.assertEqual(actual.shape, features.shape)
        self.assertTrue(torch.isfinite(actual).all())
        actual.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_tanp_is_disabled_during_evaluation(self) -> None:
        model = TemporalMaxerLiteHead(
            input_dim=32,
            hidden_dim=16,
            pyramid_levels=2,
            dropout=0.0,
            tanp_sigma=0.75,
        ).eval()
        features = torch.randn(2, 11, 32)
        with torch.no_grad():
            first = model(features)
            second = model(features)
        for key in first:
            if first[key] is not None:
                torch.testing.assert_close(first[key], second[key])

    def test_tanp_roles_preserve_numeric_tail_and_zero_sigma_identity(self) -> None:
        values = torch.randn(4, 15)
        mean = torch.randn(12)
        std = torch.rand(12) + 0.5
        identity = tanp_standardized_roles(values, 12, mean, std, sigma=0.0)
        torch.testing.assert_close(identity, values)

        torch.manual_seed(9)
        perturbed = tanp_standardized_roles(values, 12, mean, std, sigma=0.75)
        torch.testing.assert_close(perturbed[:, 12:], values[:, 12:])
        self.assertFalse(torch.equal(perturbed[:, :12], values[:, :12]))

    def test_oof_hardness_promotes_only_cross_fitted_high_score_negatives(self) -> None:
        frame = pd.DataFrame(
            {
                "rec_name": ["a", "a", "a"],
                "roi_id": ["N01", "N01", "N01"],
                "t_start": [0.0, 1.0, 2.0],
                "t_end": [1.0, 2.0, 3.0],
                "quality_target": [0.0, 0.0, 0.8],
                "sample_kind": ["easy_negative", "easy_negative", "high_positive"],
                "hardness_score": [0.01, 0.02, 0.03],
            }
        )
        hardness = frame[["rec_name", "roi_id", "t_start", "t_end"]].copy()
        hardness["oof_quality_score"] = [0.2, 0.05, 0.9]
        actual, promoted = apply_oof_hardness(frame, hardness, 0.1, 0.1)
        self.assertEqual(promoted, 1)
        self.assertEqual(actual["sample_kind"].tolist(), [
            "hard_negative", "easy_negative", "high_positive"
        ])
        self.assertAlmostEqual(float(actual.loc[0, "hardness_score"]), 0.2)

    def test_boundary_voting_preserves_scores_and_uses_overlapping_consensus(self) -> None:
        detections = np.asarray([[0.0, 10.0, 0.9], [30.0, 40.0, 0.4]])
        voters = np.asarray(
            [[0.0, 10.0], [2.0, 12.0], [30.0, 40.0], [100.0, 110.0]]
        )
        scores = np.asarray([0.2, 0.8, 0.5, 1.0])
        voted = vote_detection_boundaries(
            detections,
            voters,
            scores,
            tiou_threshold=0.5,
            blend=1.0,
            topk=20,
            minimum_duration=1.0,
        )
        np.testing.assert_allclose(voted[:, 2], detections[:, 2])
        self.assertGreater(voted[0, 0], detections[0, 0])
        self.assertGreater(voted[0, 1], detections[0, 1])
        np.testing.assert_allclose(voted[1, :2], detections[1, :2])

    def test_boundary_voting_blend_is_conservative(self) -> None:
        detections = np.asarray([[0.0, 10.0, 0.9]])
        voters = np.asarray([[2.0, 12.0]])
        voted = vote_detection_boundaries(
            detections,
            voters,
            np.asarray([1.0]),
            tiou_threshold=0.5,
            blend=0.25,
            minimum_duration=1.0,
        )
        np.testing.assert_allclose(voted[0, :2], [0.5, 10.5])

    def test_salient_shrinkage_boundary_is_derived_from_fixed_recipe(self) -> None:
        scored = pd.DataFrame(
            {
                "reference_blend050_t_start": [0.0],
                "reference_blend050_t_end": [10.0],
                "router_soft_t_start": [4.0],
                "router_soft_t_end": [14.0],
            }
        )
        output = ensure_boundary_columns(scored, "router_shrink025")
        self.assertEqual(float(output.loc[0, "router_shrink025_t_start"]), 1.0)
        self.assertEqual(float(output.loc[0, "router_shrink025_t_end"]), 11.0)

    def test_consensus_rescoring_preserves_zero_blend_and_soft_nms_decay(self) -> None:
        detections = np.asarray([[0.0, 10.0, 0.6], [30.0, 40.0, 0.2]])
        voters = np.asarray([[0.0, 10.0], [1.0, 11.0], [30.0, 40.0]])
        scores = np.asarray([0.6, 0.9, 0.4])
        control = consensus_rescore_detections(
            detections, voters, scores, 0.5, blend=0.0
        )
        np.testing.assert_allclose(control, detections)
        rescored = consensus_rescore_detections(
            detections, voters, scores, 0.5, blend=0.5
        )
        self.assertGreater(rescored[0, 2], detections[0, 2])
        self.assertLessEqual(rescored[1, 2], detections[1, 2] + 1e-12)
        self.assertLess(rescored[1, 2], scores[2])

    def test_multi_expert_voting_supports_robust_median(self) -> None:
        detections = np.asarray([[0.0, 10.0, 0.8]])
        voters = np.asarray(
            [[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [50.0, 60.0]]]
        )
        mean = multi_expert_vote_boundaries(
            detections, voters, np.asarray([1.0]), 0.5, 1.0, "mean", topk=10,
            minimum_duration=1.0,
        )
        median = multi_expert_vote_boundaries(
            detections, voters, np.asarray([1.0]), 0.5, 1.0, "median", topk=10,
            minimum_duration=1.0,
        )
        self.assertGreater(mean[0, 0], 0.0)
        self.assertIn(float(median[0, 0]), (0.0, 1.0, 2.0))
        self.assertEqual(float(median[0, 2]), 0.8)
        self.assertEqual(weighted_median(np.asarray([0.0, 5.0]), np.asarray([2.0, 1.0])), 0.0)

    def test_multi_expert_tensor_uses_raw_and_named_boundaries(self) -> None:
        scored = pd.DataFrame(
            {
                "t_start": [0.0],
                "t_end": [10.0],
                "reference_blend050_t_start": [1.0],
                "reference_blend050_t_end": [11.0],
            }
        )
        tensor = expert_boundary_tensor(
            scored, modes=("raw", "reference_blend050")
        )
        self.assertEqual(tensor.shape, (1, 2, 2))
        np.testing.assert_allclose(tensor[0, 1], [1.0, 11.0])

    def test_test_expert_boundaries_average_aligned_folds(self) -> None:
        base = pd.DataFrame(
            {
                "rec_name": ["recording"],
                "roi_id": ["N01"],
                "t_start": [0.0],
                "t_end": [10.0],
            }
        )
        frames = []
        for offset in (0.0, 2.0):
            frame = base.copy()
            for prefix in ("delta", "distribution", "point"):
                frame[f"{prefix}_t_start"] = offset
                frame[f"{prefix}_t_end"] = 10.0 + offset
            frames.append(frame)
        output = add_test_expert_boundaries(base, frames)
        self.assertEqual(float(output.loc[0, "reference_delta_t_start"]), 1.0)
        self.assertEqual(float(output.loc[0, "reference_point_t_end"]), 11.0)

    def test_proposal_chunks_cover_long_roi_in_temporal_order(self) -> None:
        count = 603
        starts = np.arange(count, dtype=np.float64)[::-1]
        proposals = pd.DataFrame(
            {
                "rec_name": ["recording"] * count,
                "roi_id": ["N01"] * count,
                "t_start": starts,
                "t_end": starts + 0.5,
            }
        )
        chunks = build_proposal_chunks(proposals, chunk_length=256, stride=192)
        self.assertTrue(chunks_cover_all(chunks, count))
        self.assertEqual(len(chunks[-1].positions), 256)
        for chunk in chunks:
            centers = 0.5 * (
                proposals.loc[chunk.positions, "t_start"].to_numpy()
                + proposals.loc[chunk.positions, "t_end"].to_numpy()
            )
            self.assertTrue(np.all(np.diff(centers) >= 0.0))

    def test_proposal_numeric_features_are_aligned_and_finite(self) -> None:
        proposals = pd.DataFrame(
            {
                "rec_name": ["recording"] * 3,
                "roi_id": ["N01"] * 3,
                "t_start": [20.0, 0.0, 10.0],
                "t_end": [30.0, 8.0, 18.0],
                "score": [0.3, 0.9, 0.6],
            }
        )
        reference = proposals.copy()
        for column, values in {
            "cnn_score": [0.2, 0.8, 0.5],
            "dense_quality": [0.1, 0.7, 0.4],
            "dense_action": [0.3, 0.9, 0.6],
            "dense_point": [0.4, 0.6, 0.5],
            "dense_score": [0.2, 0.7, 0.4],
            "brem_score": [0.1, 0.8, 0.5],
        }.items():
            reference[column] = values
        features = proposal_numeric_features(proposals, reference)
        self.assertEqual(features.shape, (3, 15))
        self.assertTrue(np.isfinite(features).all())
        self.assertLess(features[1, 8], features[2, 8])
        self.assertLess(features[2, 8], features[0, 8])

    def test_proposal_context_tcn_preserves_sequence_shape_and_gradients(self) -> None:
        torch.manual_seed(37)
        model = ProposalContextTCN(
            feature_dim=12,
            numeric_dim=15,
            hidden_dim=16,
            numeric_hidden_dim=4,
            dropout=0.0,
        )
        frame_features = torch.randn(2, 7, 11, 12)
        numeric = torch.randn(2, 7, 15)
        mask = torch.tensor(
            [[True] * 7, [True, True, True, True, False, False, False]]
        )
        logits = model(frame_features, numeric, mask)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.all(logits[1, 4:] == 0.0))
        logits[mask].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad != 0)
                for parameter in model.parameters()
            )
        )

    def test_proposal_context_pairwise_loss_ranks_positive_above_negative(self) -> None:
        mask = torch.tensor([[True, True, True, True]])
        targets = torch.tensor([[0.8, 0.7, 0.0, 0.0]])
        good = local_pairwise_loss(
            torch.tensor([[3.0, 2.0, -2.0, -3.0]]), targets, mask, max_pairs=2
        )
        bad = local_pairwise_loss(
            torch.tensor([[-3.0, -2.0, 2.0, 3.0]]), targets, mask, max_pairs=2
        )
        self.assertLess(float(good), float(bad))

    def test_context_calibration_preserves_reference_marginal(self) -> None:
        context = np.asarray([0.8, 0.1, 0.6, 0.2])
        qhead = np.asarray([0.7, 0.4, 0.9, 0.3])
        calibrated = quantile_match_scores(context, qhead)
        np.testing.assert_allclose(np.sort(calibrated), np.sort(qhead))
        np.testing.assert_array_equal(np.argsort(calibrated), np.argsort(context))
        scored = add_context_score_variants(
            pd.DataFrame({"quality_score": qhead, "context_score": context})
        )
        self.assertTrue(
            np.isfinite(scored.filter(like="context_").to_numpy()).all()
        )

    def test_salient_router_shrinkage_interpolates_from_reference(self) -> None:
        scored = pd.DataFrame(
            {
                "reference_blend050_t_start": [0.0],
                "reference_blend050_t_end": [10.0],
                "router_soft_t_start": [4.0],
                "router_soft_t_end": [14.0],
            }
        )
        output = add_router_shrinkage(scored)
        self.assertEqual(output.loc[0, "router_shrink025_t_start"], 1.0)
        self.assertEqual(output.loc[0, "router_shrink050_t_end"], 12.0)
        self.assertEqual(output.loc[0, "router_shrink075_t_start"], 3.0)

    def test_salient_boundary_sampler_interpolates_relative_positions(self) -> None:
        temporal = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
        positions = torch.tensor([[[-0.2, 0.5, 1.2]]], requires_grad=True)
        sampled = sample_at_relative_positions(temporal, positions)
        self.assertEqual(tuple(sampled.shape), (1, 1, 3, 1))
        torch.testing.assert_close(
            sampled.flatten(), torch.tensor([0.0, 2.0, 4.0]), atol=1e-6, rtol=0.0
        )
        sampled.sum().backward()
        self.assertIsNotNone(positions.grad)

    def test_salient_boundary_router_scores_candidate_aligned_features(self) -> None:
        router = SalientBoundaryRouter(
            temporal_dim=12,
            candidate_dim=7,
            hidden_dim=8,
            candidate_hidden_dim=4,
            dropout=0.0,
        )
        temporal = torch.randn(3, 11, 12)
        candidates = torch.randn(3, 5, 7)
        positions = torch.rand(3, 5, 2).sort(dim=2).values
        logits = router(temporal, candidates, positions)
        self.assertEqual(tuple(logits.shape), (3, 5))
        logits.sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad != 0)
                for parameter in router.parameters()
            )
        )

    def test_pointwise_crossing_localizes_peak_and_rejects_flat_signal(self) -> None:
        probabilities = np.asarray(
            [
                [0.0, 0.0, 0.1, 0.6, 1.0, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.4] * 11,
            ],
            dtype=np.float64,
        )
        start, end, valid = crossing_intervals(
            probabilities,
            threshold=0.5,
            padding=0.0,
            minimum_range=0.05,
            dense_grid_size=201,
            smooth=False,
        )
        np.testing.assert_array_equal(valid, [True, False])
        self.assertLess(start[0], 0.4)
        self.assertGreater(end[0], 0.4)
        self.assertLess(end[0] - start[0], 0.7)

        raw_start = np.asarray([10.0, 20.0])
        raw_end = np.asarray([110.0, 120.0])
        refined_start, refined_end = stabilized_boundaries(
            raw_start,
            raw_end,
            start,
            end,
            valid,
            minimum_duration=1.0,
        )
        self.assertEqual(refined_start[1], raw_start[1])
        self.assertEqual(refined_end[1], raw_end[1])
        self.assertGreater(refined_start[0], raw_start[0])
        self.assertLess(refined_end[0], raw_end[0])

    def test_shared_encoder_preserves_temporal_geometry_and_gradients(self) -> None:
        torch.manual_seed(31)
        model = TemporalMaxerLiteHead(
            input_dim=12,
            hidden_dim=8,
            pyramid_levels=3,
            dropout=0.0,
        )
        features = torch.randn(4, 11, 12)
        shared = model.encode_shared(features)
        self.assertEqual(tuple(shared.shape), (4, 8, 11))
        shared.square().mean().backward()
        self.assertIsNotNone(model.input_projection[0].weight.grad)

    def test_bsp_synthesis_implements_all_four_boundary_types(self) -> None:
        temporal = torch.arange(7, dtype=torch.float32).view(1, 7, 1)
        primary = temporal.repeat(4, 1, 2)
        secondary = primary + 100.0
        types = torch.tensor(
            [DIFFERENT_CLASS, SAME_CLASS, DIFFERENT_SPEED, SAME_SPEED]
        )
        synthetic = synthesize_bsp_sequences(
            primary,
            secondary,
            types,
            torch.tensor([3, 3, 3, 3]),
            torch.tensor([1.0, 1.0, 0.5, 1.0]),
        )
        torch.testing.assert_close(synthetic[0, :3], primary[0, :3])
        torch.testing.assert_close(synthetic[0, 3:], secondary[0, 3:])
        torch.testing.assert_close(synthetic[1, 3:], secondary[1, 3:])
        self.assertFalse(torch.equal(synthetic[2], primary[2]))
        torch.testing.assert_close(synthetic[3], primary[3])

    def test_bsp_classifier_backpropagates_into_shared_features(self) -> None:
        shared = torch.randn(8, 16, 11, requires_grad=True)
        head = BoundaryTypeHead(hidden_dim=16, dropout=0.0)
        targets = torch.arange(8) % 4
        loss = boundary_type_loss(head(shared), targets)
        loss.backward()
        self.assertIsNotNone(shared.grad)
        self.assertTrue(torch.any(shared.grad != 0))

    def test_bsp_pairs_are_balanced_and_cross_recordings(self) -> None:
        proposals = pd.DataFrame(
            {
                "rec_name": ["a", "a", "b", "b", "c", "c"],
                "roi_id": ["N01"] * 6,
                "t_start": np.arange(6, dtype=np.float64),
                "t_end": np.arange(6, dtype=np.float64) + 1.0,
            }
        )
        quality = np.asarray([0.8, 0.0, 0.9, 0.0, 0.7, 0.0])
        specification = build_bsp_specification(
            proposals, quality, count=40, num_segments=11, seed=37
        )
        counts = np.bincount(specification.boundary_type, minlength=4)
        np.testing.assert_array_equal(counts, [10, 10, 10, 10])
        recordings = proposals["rec_name"].to_numpy()
        spliced = np.isin(
            specification.boundary_type, [DIFFERENT_CLASS, SAME_CLASS]
        )
        self.assertTrue(
            np.all(
                recordings[specification.primary[spliced]]
                != recordings[specification.secondary[spliced]]
            )
        )

    def test_boundary_oracle_selects_candidate_with_highest_tiou(self) -> None:
        proposals = pd.DataFrame(
            {"rec_name": ["rec"], "roi_id": ["N01"], "t_start": [0.0], "t_end": [4e6]}
        )
        starts = np.asarray([[0.0, 1e6]])
        ends = np.asarray([[4e6, 3e6]])
        choice, quality = oracle_choice(
            proposals,
            starts,
            ends,
            {("rec", "1"): np.asarray([[1.0, 3.0]])},
        )
        np.testing.assert_array_equal(choice, [1])
        np.testing.assert_allclose(quality, [1.0])

    def test_boundary_router_quality_is_vectorized_per_candidate(self) -> None:
        proposals = pd.DataFrame(
            {"rec_name": ["rec"], "roi_id": ["N01"], "t_start": [0.0], "t_end": [4e6]}
        )
        quality = candidate_tiou(
            proposals,
            np.asarray([[0.0, 1e6]]),
            np.asarray([[4e6, 3e6]]),
            {("rec", "1"): np.asarray([[1.0, 3.0]])},
        )
        np.testing.assert_allclose(quality, [[0.5, 1.0]])
        router = CandidateQualityRouter(input_dim=7, hidden_dim=4)
        self.assertEqual(tuple(router(torch.zeros(2, 3, 7)).shape), (2, 3))

    def test_index_soft_nms_exactly_preserves_reference_implementation(self) -> None:
        detections = np.asarray(
            [[0.0, 4.0, 0.9], [1.0, 5.0, 0.8], [8.0, 10.0, 0.7]],
            dtype=np.float64,
        )
        expected = temporal_soft_nms(detections, sigma=0.25, score_threshold=0.01)
        keep, scores = temporal_soft_nms_indices(
            detections, sigma=0.25, score_threshold=0.01
        )
        actual = detections[keep].copy()
        actual[:, 2] = scores
        np.testing.assert_allclose(actual, expected)

    def test_router_training_selection_uses_reference_post_nms(self) -> None:
        proposals = pd.DataFrame(
            {
                "rec_name": ["rec", "rec"],
                "roi_id": ["N01", "N01"],
                "t_start": [0.0, 0.5e6],
                "t_end": [4e6, 4.5e6],
            }
        )
        reference = pd.DataFrame(
            {
                "delta_t_start": [0.0, 0.5e6],
                "delta_t_end": [4e6, 4.5e6],
                "brem_score": [0.9, 0.8],
            }
        )
        args = SimpleNamespace(
            duration_dmax=60.0,
            duration_sigma=20.0,
            training_min_score=0.05,
            pre_nms_topk_per_roi=1000,
            soft_nms_sigma=0.25,
            soft_nms_score_threshold=0.1,
        )
        selected = post_nms_training_indices(proposals, reference, args)
        self.assertGreaterEqual(len(selected), 1)
        self.assertIn(0, selected)

    def test_temporal_candidate_router_scores_each_boundary_expert(self) -> None:
        router = TemporalCandidateRouter(
            temporal_dim=20,
            auxiliary_dim=8,
            candidate_dim=7,
            hidden_dim=12,
            candidate_hidden_dim=4,
            dropout=0.0,
        )
        temporal = torch.randn(3, 11, 20)
        candidates = torch.randn(3, 5, 7)
        logits = router(temporal, candidates)
        self.assertEqual(tuple(logits.shape), (3, 5))
        logits.sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.any(parameter.grad != 0)
                for parameter in router.parameters()
            )
        )

    def test_temporal_router_can_restrict_unstable_boundary_experts(self) -> None:
        names, starts, ends = select_boundary_candidates(
            ["raw", "point", "blend"],
            np.asarray([[0.0, 1.0, 2.0]]),
            np.asarray([[3.0, 4.0, 5.0]]),
            ["raw", "blend"],
        )
        self.assertEqual(names, ["raw", "blend"])
        np.testing.assert_array_equal(starts, [[0.0, 2.0]])
        np.testing.assert_array_equal(ends, [[3.0, 5.0]])

    def test_eventmatch_adapter_starts_as_exact_identity(self) -> None:
        torch.manual_seed(19)
        adapter = ResidualEventAdapter(feature_dim=12, bottleneck_dim=4)
        features = torch.randn(3, 5, 12)
        torch.testing.assert_close(adapter(features), features, rtol=0.0, atol=0.0)

    def test_temporal_event_adapter_starts_as_exact_identity(self) -> None:
        torch.manual_seed(23)
        adapter = TemporalResidualEventAdapter(feature_dim=12, bottleneck_dim=4)
        features = torch.randn(3, 5, 12)
        torch.testing.assert_close(adapter(features), features, rtol=0.0, atol=0.0)

    def test_eventmatch_gradient_reversal_flips_and_scales_gradient(self) -> None:
        features = torch.ones(2, 3, requires_grad=True)
        reverse_gradient(features, strength=0.25).sum().backward()
        torch.testing.assert_close(
            features.grad,
            torch.full_like(features, -0.25),
            rtol=0.0,
            atol=0.0,
        )

    def test_semantic_domain_alignment_separates_action_and_background(self) -> None:
        source = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3, 4, requires_grad=True)
        source_action = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        target_action = torch.tensor([[0.9, 0.1, 0.2], [0.8, 0.1, 0.2]])
        discriminator = torch.nn.Linear(4, 2)
        loss = semantic_domain_loss(
            discriminator,
            source,
            target,
            source_action,
            target_action,
            pseudo_foreground=0.7,
            pseudo_background=0.3,
            max_anchors=16,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)

    def test_sada_conditional_loss_uses_learned_semantic_tokens(self) -> None:
        source = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3, 4, requires_grad=True)
        source_action = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        target_action = torch.tensor([[0.9, 0.1, 0.2], [0.8, 0.1, 0.2]])
        discriminator = ConditionalSemanticDiscriminator(4, hidden_dim=8)
        loss = sada_conditional_domain_loss(
            discriminator,
            source,
            target,
            source_action,
            target_action,
            pseudo_threshold=0.6,
            max_anchors=16,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(discriminator.semantic_embeddings.grad)

    def test_recording_coral_matches_source_first_two_moments(self) -> None:
        source = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
        features = np.concatenate(
            (source, 2.0 * source + 10.0, 0.5 * source - 3.0), axis=0
        )
        target = pd.DataFrame({"rec_name": ["a", "a", "b", "b"]})
        target_indices = np.asarray([2, 3, 4, 5])
        scale, bias = recording_coral_affines(
            features,
            np.asarray([0, 1]),
            target,
            target_indices,
        )
        transformed = (
            features[target_indices] * scale[:, None, :] + bias[:, None, :]
        )
        expected_mean, expected_std = feature_moments(features, np.asarray([0, 1]))
        for local in (np.asarray([0, 1]), np.asarray([2, 3])):
            actual_mean, actual_std = feature_moments(transformed, local)
            np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-5)
            np.testing.assert_allclose(actual_std, expected_std, atol=1e-5)

    def test_tespec_event_corruption_is_reproducible_and_nontrivial(self) -> None:
        sequence = np.zeros((2, 20, 8, 8), dtype=np.uint8)
        sequence[:, :, 2:6, 2:6] = 12
        first = corrupt_stacked_sequence(sequence, np.random.default_rng(17))
        second = corrupt_stacked_sequence(sequence, np.random.default_rng(17))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, sequence.shape)
        self.assertFalse(np.array_equal(first, sequence))

    def test_quality_gate_interpolates_reference_and_tespec_boundaries(self) -> None:
        start, end = quality_aware_boundaries(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([10.0, 10.0, 10.0]),
            np.asarray([2.0, 2.0, 2.0]),
            np.asarray([8.0, 8.0, 8.0]),
            np.asarray([4.0, 4.0, 4.0]),
            np.asarray([6.0, 6.0, 6.0]),
            np.asarray([0.0, 0.5, 1.0]),
        )
        np.testing.assert_allclose(start, [1.0, 1.5, 2.0])
        np.testing.assert_allclose(end, [9.0, 8.5, 8.0])

    def test_trimodal_score_is_an_equal_geometric_mean(self) -> None:
        score = geometric_trimodal_score(
            np.asarray([1.0, 0.125]),
            np.asarray([0.125, 1.0]),
            np.asarray([1.0, 1.0]),
        )
        np.testing.assert_allclose(score, [0.5, 0.5])

    def test_test_boundary_ensemble_matches_source_blend_definition(self) -> None:
        ensemble = pd.DataFrame(
            {
                "t_start": [0.0, 10.0],
                "t_end": [10.0, 20.0],
                "hybrid_t_start": [1.0, 11.0],
                "hybrid_t_end": [9.0, 19.0],
            }
        )
        frames = [
            pd.DataFrame(
                {
                    "delta_t_start": [2.0, 12.0],
                    "delta_t_end": [8.0, 18.0],
                }
            ),
            pd.DataFrame(
                {
                    "delta_t_start": [4.0, 14.0],
                    "delta_t_end": [6.0, 16.0],
                }
            ),
        ]
        fused = add_boundary_modes(ensemble, frames, boundary_blend=0.75)
        np.testing.assert_allclose(fused["delta_t_start"], [3.0, 13.0])
        np.testing.assert_allclose(fused["blend050_t_start"], [1.5, 11.5])
        np.testing.assert_allclose(fused["blend050_t_end"], [8.5, 18.5])
        np.testing.assert_allclose(fused["reference_t_start"], [1.0, 11.0])

    def test_ranking_fusions_are_fixed_geometric_means(self) -> None:
        scored = pd.DataFrame(
            {
                "quality_score": [0.25, 1.0],
                "dense_score": [1.0, 0.25],
                "brem_score": [0.36, 0.49],
                "dense_point": [0.64, 0.81],
            }
        )
        fused = add_ranking_fusions(scored)
        np.testing.assert_allclose(fused["qhead_dense_score"], [0.5, 0.5])
        np.testing.assert_allclose(fused["qhead_brem_score"], [0.3, 0.7])
        np.testing.assert_allclose(
            fused["qhead_brem_w020_score"],
            np.power([0.25, 1.0], 0.8) * np.power([0.36, 0.49], 0.2),
        )
        np.testing.assert_allclose(fused["qhead_point_score"], [0.4, 0.9])

    def test_tespec_histogram_preserves_time_and_polarity_bins(self) -> None:
        events = np.asarray(
            [
                [0, 0, 0, 0],
                [1, 1, 5, 1],
                [1, 0, 10, 1],
            ],
            dtype=np.float64,
        )
        histogram = stacked_histogram(events, height=2, width=2, bins=2, image_size=2)
        self.assertEqual(tuple(histogram.shape), (4, 2, 2))
        self.assertEqual(int(histogram.sum()), 3)
        self.assertEqual(int(histogram[0, 0, 0]), 1)
        self.assertEqual(int(histogram[3, 1, 1]), 1)
        self.assertEqual(int(histogram[3, 0, 1]), 1)

    def test_event_spectral_features_recover_two_hz_signal(self) -> None:
        bin_s = 0.1
        times = np.arange(400) * bin_s
        signed = 20.0 + 15.0 * np.sin(2.0 * np.pi * 2.0 * times)
        on = np.maximum(np.rint(signed), 0).astype(np.int64)
        off = np.zeros_like(on)
        features = local_event_descriptors(
            on,
            off,
            np.asarray([[20.0e6]]),
            area=1.0,
            bin_s=bin_s,
            window_s=5.0,
            freq_min=0.5,
            freq_max=3.0,
        )
        self.assertEqual(tuple(features.shape), (1, 1, 8))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features[0, 0, 6] * 3.0), 2.0, places=5)

    def test_event_features_are_concatenated_on_the_last_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "frame_features.npy"
            event_path = Path(directory) / "event_features.npy"
            np.save(feature_path, np.ones((2, 11, 4), dtype=np.float16))
            np.save(event_path, np.full((2, 11, 3), 2.0, dtype=np.float32))
            dataset = DenseFeatureDataset(
                feature_path,
                np.asarray([1, 0]),
                event_feature_path=event_path,
            )
            feature, index = dataset[0]
        self.assertEqual(index, 0)
        self.assertEqual(tuple(feature.shape), (11, 7))
        torch.testing.assert_close(feature[:, :4], torch.ones(11, 4))
        torch.testing.assert_close(feature[:, 4:], torch.full((11, 3), 2.0))

    def test_event_features_can_replace_atsn_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "frame_features.npy"
            event_path = Path(directory) / "event_features.npy"
            np.save(feature_path, np.ones((2, 11, 4), dtype=np.float16))
            np.save(event_path, np.full((2, 11, 3), 2.0, dtype=np.float32))
            dataset = DenseFeatureDataset(
                feature_path,
                np.asarray([0]),
                event_feature_path=event_path,
                event_features_only=True,
            )
            feature, _ = dataset[0]
        self.assertEqual(tuple(feature.shape), (11, 3))
        torch.testing.assert_close(feature, torch.full((11, 3), 2.0))

    def test_static_event_features_broadcast_over_dense_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "frame_features.npy"
            event_path = Path(directory) / "event_features.npy"
            np.save(feature_path, np.ones((1, 11, 4), dtype=np.float16))
            np.save(event_path, np.full((1, 1, 3), 2.0, dtype=np.float16))
            dataset = DenseFeatureDataset(
                feature_path,
                np.asarray([0]),
                event_feature_path=event_path,
            )
            feature, _ = dataset[0]
        self.assertEqual(tuple(feature.shape), (11, 7))
        torch.testing.assert_close(feature[:, 4:], torch.full((11, 3), 2.0))

    def test_tism_views_are_invariant_to_the_orthogonal_axis(self) -> None:
        events = np.asarray(
            [[1, 2, 0, 1], [3, 4, 5, 0], [5, 6, 10, 1]], dtype=np.float64
        )
        shifted_x = events.copy()
        shifted_x[:, 0] += 2
        shifted_y = events.copy()
        shifted_y[:, 1] += 2
        original = tism_maps(events, 0, 11, height=12, width=12, image_size=12)
        x_shifted = tism_maps(shifted_x, 0, 11, height=12, width=12, image_size=12)
        y_shifted = tism_maps(shifted_y, 0, 11, height=12, width=12, image_size=12)
        np.testing.assert_array_equal(original[0], x_shifted[0])
        np.testing.assert_array_equal(original[1], y_shifted[1])

    def test_trident_offsets_are_dense_nonnegative_and_differentiable(self) -> None:
        torch.manual_seed(3)
        model = TemporalMaxerLiteHead(
            input_dim=16,
            hidden_dim=8,
            pyramid_levels=2,
            trident_bins=10,
        )
        output = model(torch.randn(2, 11, 16))
        offsets = output["trident_offsets_bins"]
        self.assertEqual(tuple(offsets.shape), (2, 11, 2))
        self.assertTrue(torch.all(offsets >= 0))
        offsets.mean().backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("trident_")
        ]
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in gradients))

    def test_event_frame_drop_replaces_exactly_one_temporal_sample(self) -> None:
        torch.manual_seed(5)
        features = torch.randn(4, 11, 8)
        blank = torch.full((8,), -999.0)
        corrupted = drop_one_event_frame(features, blank)
        replaced = (corrupted == -999.0).all(dim=2)
        torch.testing.assert_close(replaced.sum(dim=1), torch.ones(4, dtype=torch.long))

    def test_trc_is_zero_for_identical_predictions_and_positive_after_change(self) -> None:
        distances = torch.full((3, 11, 2), 0.4)
        clean = {"boundary_distances": distances}
        same = {"boundary_distances": distances.clone()}
        changed = {"boundary_distances": distances.clone()}
        changed["boundary_distances"][:, 4:7] = 0.05
        deltas = torch.zeros(3, 2)
        weights = torch.ones(3)
        equal_loss = temporal_robust_consistency_loss(clean, same, deltas, weights, 5, 3)
        changed_loss = temporal_robust_consistency_loss(clean, changed, deltas, weights, 5, 3)
        torch.testing.assert_close(equal_loss, torch.zeros_like(equal_loss))
        self.assertTrue(torch.all(changed_loss > 0))

    def test_auxiliary_features_use_a_separate_normalization(self) -> None:
        torch.manual_seed(9)
        model = TemporalMaxerLiteHead(
            input_dim=12,
            auxiliary_dim=4,
            hidden_dim=8,
            pyramid_levels=2,
        )
        base = 100.0 * torch.randn(2, 11, 8)
        auxiliary = 0.01 * torch.randn(2, 11, 4)
        features = torch.cat((base, auxiliary), dim=2).requires_grad_(True)
        output = model(features)
        loss = output["quality_logit"].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(output["quality_logit"]).all())
        self.assertTrue(torch.any(features.grad[:, :, :8] != 0))
        self.assertTrue(torch.any(features.grad[:, :, 8:] != 0))

    def test_dense_outputs_and_gradients(self) -> None:
        torch.manual_seed(11)
        model = TemporalMaxerLiteHead(input_dim=32, hidden_dim=16, pyramid_levels=3)
        features = torch.randn(4, 11, 32)
        output = model(features)
        self.assertEqual(tuple(output["action_logits"].shape), (4, 11))
        self.assertEqual(tuple(output["point_quality_logits"].shape), (4, 11))
        self.assertEqual(tuple(output["start_logits"].shape), (4, 11))
        self.assertEqual(tuple(output["end_logits"].shape), (4, 11))
        self.assertEqual(tuple(output["quality_logit"].shape), (4,))
        self.assertEqual(tuple(output["boundary_deltas"].shape), (4, 2))
        self.assertEqual(tuple(output["boundary_distances"].shape), (4, 11, 2))

        args = SimpleNamespace(
            qfl_beta=2.0,
            quality_weight=1.0,
            action_weight=0.5,
            distribution_weight=0.25,
            boundary_weight=0.5,
        )
        quality = torch.tensor([0.0, 0.3, 0.7, 1.0])
        action = torch.zeros(4, 11)
        action[1:, 3:8] = 1.0
        start = torch.zeros(4, 11)
        end = torch.zeros(4, 11)
        start[1:, 3] = 1.0
        end[1:, 7] = 1.0
        deltas = torch.zeros(4, 2)
        point_distances = torch.zeros(4, 11, 2)
        boundary_weight = torch.tensor([0.0, 0.3, 0.7, 1.0])
        sample_loss, _ = dense_loss(
            output,
            quality,
            action,
            point_distances,
            start,
            end,
            deltas,
            boundary_weight,
            args,
        )
        self.assertTrue(torch.isfinite(sample_loss).all())
        sample_loss.mean().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None and torch.any(gradient != 0) for gradient in gradients))

    def test_point_boundary_scoring_is_finite(self) -> None:
        torch.manual_seed(13)
        model = TemporalMaxerLiteHead(input_dim=32, hidden_dim=16, pyramid_levels=2)
        proposals = pd.DataFrame(
            {
                "rec_name": ["recording", "recording"],
                "roi_id": ["N01", "N01"],
                "t_start": [10.0e6, 30.0e6],
                "t_end": [20.0e6, 35.0e6],
                "score": [0.8, 0.2],
            }
        )
        args = SimpleNamespace(
            batch_size=2,
            augment_factor=5,
            max_boundary_delta=0.75,
            boundary_blend=0.75,
        )
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.npy"
            np.save(feature_path, np.random.default_rng(3).normal(size=(2, 11, 32)).astype(np.float16))
            scored = score_model(
                model,
                proposals,
                np.asarray([0, 1]),
                feature_path,
                np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                args,
                torch.device("cpu"),
            )
        for column in ("dense_score", "point_t_start", "point_t_end"):
            self.assertTrue(np.isfinite(scored[column]).all())
        self.assertTrue(((scored["point_t_end"] - scored["point_t_start"]) >= 2.0e6).all())


class AtsnDenseFineTuneTest(unittest.TestCase):
    def test_test_ensemble_averages_folds_before_fixed_fusion(self) -> None:
        proposals = pd.DataFrame(
            {
                "rec_name": ["recording"],
                "roi_id": ["N01"],
                "t_start": [10.0],
                "t_end": [20.0],
            }
        )
        frames = [
            pd.DataFrame(
                {
                    "dense_score": [0.4 + 0.2 * index],
                    "brem_score": [0.3 + 0.4 * index],
                    "dense_point": [0.5],
                    "delta_t_start": [12.0 + 2.0 * index],
                    "delta_t_end": [18.0 + 2.0 * index],
                }
            )
            for index in range(2)
        ]
        ensemble = ensemble_test_frames(proposals, frames, np.asarray([0.25]))
        self.assertAlmostEqual(float(ensemble.loc[0, "brem_score"]), 0.5)
        self.assertAlmostEqual(
            float(ensemble.loc[0, "qhead_brem_score"]), np.sqrt(0.125)
        )
        self.assertAlmostEqual(float(ensemble.loc[0, "blend050_t_start"]), 11.5)
        self.assertAlmostEqual(float(ensemble.loc[0, "blend050_t_end"]), 19.5)

    def test_fixed_boundary_soup_is_an_equal_interpolation(self) -> None:
        scored = pd.DataFrame(
            {
                "reference_t_start": [0.0],
                "reference_t_end": [10.0],
                "blend050_t_start": [2.0],
                "blend050_t_end": [14.0],
            }
        )
        mixed = soup_boundaries(scored)
        self.assertEqual(float(mixed.loc[0, "soup050_t_start"]), 1.0)
        self.assertEqual(float(mixed.loc[0, "soup050_t_end"]), 12.0)

    def test_first_block_configuration_is_surgical(self) -> None:
        model = AugmentedTsn(2, num_tsn_samples=7, augment_factor=5)
        for parameter in model.parameters():
            parameter.requires_grad = False
        names = configure_backbone_trainable(model, "first")
        self.assertTrue(names)
        self.assertTrue(
            all(name.startswith(("conv1.", "bn1.", "layer1.")) for name in names)
        )
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.fc_cls.parameters())
        )

    def test_end_to_end_wrapper_keeps_temporal_features_differentiable(self) -> None:
        class TinyAtsn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(3, 4)

            def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
                return self.projection(images.mean(dim=(-1, -2)))

        class TinyDetector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output = nn.Linear(4, 1)

            def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"score": self.output(features).squeeze(-1)}

        model = EndToEndDenseDetector(TinyAtsn(), TinyDetector())
        output = model(torch.randn(2, 11, 3, 4, 4))["score"]
        output.mean().backward()
        self.assertIsNotNone(model.atsn.projection.weight.grad)
        self.assertIsNotNone(model.detector.output.weight.grad)

    def test_raw_outputs_reconstruct_valid_boundary_columns(self) -> None:
        proposals = pd.DataFrame(
            {
                "rec_name": ["recording", "recording"],
                "roi_id": ["N01", "N01"],
                "t_start": [10.0e6, 30.0e6],
                "t_end": [20.0e6, 35.0e6],
            }
        )
        outputs = {
            "quality": np.asarray([0.8, 0.2], dtype=np.float32),
            "action": np.asarray([0.7, 0.3], dtype=np.float32),
            "point_score": np.asarray([0.6, 0.4], dtype=np.float32),
            "point_position": np.asarray([0.5, 0.5], dtype=np.float32),
            "point_distances": np.asarray([[0.5, 0.5], [0.2, 0.2]], dtype=np.float32),
            "deltas": np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            "start_position": np.asarray([0.0, 0.2], dtype=np.float32),
            "end_position": np.asarray([1.0, 0.8], dtype=np.float32),
        }
        scored = outputs_to_scored(
            proposals,
            outputs,
            np.asarray([0.9, 0.1]),
            SimpleNamespace(max_boundary_delta=0.75),
        )
        self.assertTrue(np.isfinite(scored["brem_score"]).all())
        for prefix in ("delta", "point", "distribution"):
            duration = scored[f"{prefix}_t_end"] - scored[f"{prefix}_t_start"]
            self.assertTrue((duration >= 2.0e6).all())


class DenseTargetTest(unittest.TestCase):
    def test_targets_and_master_mapping(self) -> None:
        proposals = pd.DataFrame(
            [
                {
                    "rec_name": "recording",
                    "roi_id": "N01",
                    "t_start": 10.0e6,
                    "t_end": 20.0e6,
                    "score": 0.8,
                },
                {
                    "rec_name": "recording",
                    "roi_id": "N01",
                    "t_start": 30.0e6,
                    "t_end": 35.0e6,
                    "score": 0.2,
                },
            ]
        )
        annotation = {
            "database": {
                "recording": {
                    "annotations": {
                        "1": [{"label": "ed", "segment": [11.0, 19.0]}]
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            args = SimpleNamespace(
                ann_path=str(path),
                augment_factor=5,
                neg_tiou=0.1,
                pos_tiou=0.5,
                high_pos_tiou=0.7,
                boundary_min_tiou=0.3,
                max_boundary_delta=0.75,
                quiet_progress=True,
            )
            targets = build_targets(proposals, num_segments=11, args=args)
        self.assertAlmostEqual(float(targets.quality[0]), 0.8, places=6)
        self.assertEqual(float(targets.quality[1]), 0.0)
        self.assertGreater(float(targets.action[0].sum()), 0.0)
        self.assertGreater(float(targets.point_distances[0].sum()), 0.0)
        self.assertEqual(float(targets.action[1].sum()), 0.0)
        self.assertAlmostEqual(float(targets.start_distribution[0].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(targets.end_distribution[0].sum()), 1.0, places=6)
        self.assertGreater(float(targets.boundary_weight[0]), 0.0)

        reversed_proposals = proposals.iloc[::-1].reset_index(drop=True)
        reversed_proposals.loc[0, "t_start"] += 3.0e-8
        mapped = map_to_master(proposals, reversed_proposals)
        np.testing.assert_array_equal(mapped, np.asarray([1, 0]))


if __name__ == "__main__":
    unittest.main()
