"""Tests for recording-level expert reliability diagnostics."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dev.diagnose_recording_expert_reliability import (
    evaluate_recording,
    pair_agreement,
    prediction_rows,
    weight_prediction_path,
)
from dev.eval_recording_reliability_router_cv import routed_prediction
from dev.eval_actionness_profile_quality_head_cv import (
    PROFILE_COLUMNS,
    SHAPE_COLUMNS,
    add_shape_profiles,
    sample_actionness_profile,
)
from dev.diagnose_final_prediction_oracles import oracle_prediction
from dev.eval_final_boundary_gradient_cv import (
    local_transition_indices,
    refine_boundaries,
)
from dev.eval_final_boundary_ridge_cv import (
    apply_boundary_regression,
    fit_weighted_ridge,
    predict_ridge,
)
from dev.eval_bem_logits_boundary_refine_cv import (
    closest_boundary_peak,
    snap_bem_boundaries,
)
from dev.diagnose_dfl_transfer_reliability import boundary_transfer_features
from dev.eval_dfl_boundary_hypotheses_cv import add_boundary_hypotheses
from dev.eval_actionness_profile_conv_quality_cv import (
    ProfileConvQFL,
    quality_frame,
)
from dev.eval_recording_background_prototype_cv import (
    cosine_novelty,
    novelty_rescore,
    unit_centroid,
)
from dev.eval_feature_changepoint_boundary_cv import (
    feature_changepoint_saliency,
    local_saliency_peak,
)


def prediction(segment: tuple[float, float]) -> dict:
    return {
        "version": "test",
        "results": {
            "recording": {
                "1": [
                    {
                        "label": "ed",
                        "segment": list(segment),
                        "score": 0.9,
                    }
                ]
            }
        },
    }


class RecordingExpertReliabilityTest(unittest.TestCase):
    def test_evaluate_recording_perfect_prediction(self) -> None:
        ground_truth = pd.DataFrame(
            [{"video-id": "recording_1", "t-start": 1.0, "t-end": 5.0}]
        )
        metrics = evaluate_recording(
            prediction((1.0, 5.0)),
            "recording",
            ground_truth,
        )
        self.assertEqual(metrics["gt_instances"], 1)
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["AP@0.7"], 1.0)

    def test_pair_agreement_is_symmetric_for_identical_segments(self) -> None:
        first = prediction_rows(prediction((1.0, 5.0)), "continuous")
        second = prediction_rows(prediction((1.0, 5.0)), "event")
        frame = pd.concat([first, second], ignore_index=True)
        agreement = pair_agreement(frame, "continuous", "event", topk=10)
        self.assertAlmostEqual(agreement["agreement_continuous_event_mean"], 1.0)
        self.assertAlmostEqual(agreement["agreement_continuous_event_frac05"], 1.0)
        self.assertAlmostEqual(agreement["agreement_continuous_event_frac07"], 1.0)

    def test_pair_agreement_rejects_disjoint_segments(self) -> None:
        first = prediction_rows(prediction((1.0, 5.0)), "continuous")
        second = prediction_rows(prediction((10.0, 15.0)), "event")
        frame = pd.concat([first, second], ignore_index=True)
        agreement = pair_agreement(frame, "continuous", "event", topk=10)
        self.assertAlmostEqual(agreement["agreement_continuous_event_mean"], 0.0)
        self.assertAlmostEqual(agreement["agreement_continuous_event_frac05"], 0.0)

    def test_weight_prediction_path_matches_existing_convention(self) -> None:
        path = weight_prediction_path(Path("/tmp/root"), 2, (0.2, 0.4, 0.4))
        self.assertEqual(
            path.name,
            "fold02_cw0.2_ew0.4_pw0.4.json",
        )

    def test_router_scales_alternative_scores_to_canonical_ceiling(self) -> None:
        canonical = prediction((1.0, 5.0))
        alternative = prediction((2.0, 6.0))
        routed, names = routed_prediction(
            canonical,
            alternative,
            {"recording": 0.3},
            threshold=0.5,
            alternative_weights=(0.1, 0.1, 0.8),
        )
        self.assertEqual(names, ["recording"])
        detection = routed["results"]["recording"]["1"][0]
        self.assertEqual(detection["segment"], [2.0, 6.0])
        self.assertAlmostEqual(detection["score"], 0.45)

    def test_actionness_profile_preserves_left_inside_right_order(self) -> None:
        sequence = pd.Series(range(20), dtype=float).to_numpy()
        profile = sample_actionness_profile(
            sequence,
            t_start=4.0,
            t_end=8.0,
            stride_s=1.0,
            context_ratio=0.5,
        )
        self.assertEqual(profile.shape, (32,))
        self.assertLess(profile[:8].mean(), profile[8:24].mean())
        self.assertLess(profile[8:24].mean(), profile[24:].mean())

    def test_shape_profile_is_invariant_to_affine_amplitude(self) -> None:
        base = pd.DataFrame(
            [dict(zip(PROFILE_COLUMNS, range(32)))],
        )
        shifted = pd.DataFrame(
            [dict(zip(PROFILE_COLUMNS, 3.0 + 2.0 * pd.Series(range(32))))],
        )
        base_shape = add_shape_profiles(base)[list(SHAPE_COLUMNS)]
        shifted_shape = add_shape_profiles(shifted)[list(SHAPE_COLUMNS)]
        pd.testing.assert_frame_equal(base_shape, shifted_shape)

    def test_boundary_oracle_does_not_turn_background_into_gt(self) -> None:
        background = prediction((10.0, 15.0))
        transformed = oracle_prediction(
            background,
            {("recording", 1): [(1.0, 5.0)]},
            mode="boundary",
            localization_min_tiou=0.1,
        )
        detection = transformed["results"]["recording"]["1"][0]
        self.assertEqual(detection["segment"], [10.0, 15.0])

    def test_transition_indices_recover_candidate_boundaries(self) -> None:
        profile = np.concatenate(
            (
                np.zeros(8),
                np.ones(16),
                np.zeros(8),
            )
        )[None, :]
        start, end, start_strength, end_strength = local_transition_indices(
            profile,
            search_radius=2,
        )
        self.assertEqual(int(start[0]), 7)
        self.assertEqual(int(end[0]), 23)
        self.assertGreater(float(start_strength[0]), 0.0)
        self.assertLess(float(end_strength[0]), 0.0)

    def test_gradient_refinement_preserves_aligned_boundaries(self) -> None:
        profile = np.concatenate(
            (
                np.zeros(8),
                np.ones(16),
                np.zeros(8),
            )
        )
        frame = pd.DataFrame(
            [
                {
                    "fold": 0,
                    "rec_name": "recording",
                    "roi_id": 1,
                    "t_start": 10.0,
                    "t_end": 26.0,
                    "score": 0.8,
                    **dict(zip(PROFILE_COLUMNS, profile)),
                }
            ]
        )
        refined = refine_boundaries(frame, blend=1.0, search_radius=2)
        self.assertAlmostEqual(float(refined.iloc[0]["t_start"]), 10.0)
        self.assertAlmostEqual(float(refined.iloc[0]["t_end"]), 26.0)

    def test_ridge_recovers_constant_relative_boundary_offset(self) -> None:
        rows = []
        for index in range(4):
            row = {
                "rec_name": f"recording-{index // 2}",
                "target_tiou": 0.8,
                "target_start_delta": 0.1,
                "target_end_delta": -0.2,
                "log_duration": 1.0 + index,
                "score_global_rank": 0.5,
                "score_recording_rank": 0.5,
                "score_roi_rank": 0.5,
            }
            row.update(dict(zip(SHAPE_COLUMNS, np.linspace(-1.0, 1.0, 32))))
            rows.append(row)
        frame = pd.DataFrame(rows)
        model = fit_weighted_ridge(
            frame,
            alpha=1.0,
            positive_tiou=0.3,
            max_relative_offset=0.5,
        )
        prediction_offsets = predict_ridge(frame, model)
        np.testing.assert_allclose(
            prediction_offsets,
            np.asarray([[0.1, -0.2]] * 4),
            atol=1e-6,
        )

    def test_boundary_regression_uses_quality_as_a_gate(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "t_start": 10.0,
                    "t_end": 20.0,
                },
                {
                    "t_start": 10.0,
                    "t_end": 20.0,
                },
            ]
        )
        refined = apply_boundary_regression(
            frame,
            offsets=np.asarray([[0.2, -0.2], [0.2, -0.2]]),
            quality=np.asarray([1.0, 0.0]),
            blend=0.5,
            max_relative_offset=0.5,
        )
        self.assertAlmostEqual(float(refined.iloc[0]["t_start"]), 11.0)
        self.assertAlmostEqual(float(refined.iloc[0]["t_end"]), 19.0)
        self.assertAlmostEqual(float(refined.iloc[1]["t_start"]), 10.0)
        self.assertAlmostEqual(float(refined.iloc[1]["t_end"]), 20.0)

    def test_closest_boundary_peak_prefers_nearest_tied_peak(self) -> None:
        probabilities = np.asarray([0.1, 0.8, 0.2, 0.8, 0.1])
        boundary, confidence = closest_boundary_peak(
            probabilities,
            boundary_seconds=5.0,
            stride_seconds=2.0,
            radius_seconds=4.0,
        )
        self.assertAlmostEqual(boundary, 3.0)
        self.assertAlmostEqual(confidence, 0.8)

    def test_bem_confidence_threshold_blocks_weak_snap(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "rec_name": "recording",
                    "roi_id": 1,
                    "t_start": 4.0,
                    "t_end": 10.0,
                    "score": 0.9,
                }
            ]
        )
        maps = {
            ("recording", 1): {
                "start_level0": np.asarray([0.1, 0.2, 0.1, 0.1, 0.1, 0.1]),
                "end_level0": np.asarray([0.1, 0.1, 0.1, 0.1, 0.2, 0.1]),
            }
        }
        refined = snap_bem_boundaries(
            frame,
            maps,
            stride_seconds=2.0,
            map_mode="level0",
            radius_seconds=2.0,
            blend=1.0,
            confidence_threshold=0.3,
        )
        self.assertAlmostEqual(float(refined.iloc[0]["t_start"]), 4.0)
        self.assertAlmostEqual(float(refined.iloc[0]["t_end"]), 10.0)

    def test_boundary_transfer_features_measure_relative_shift(self) -> None:
        control = prediction((10.0, 20.0))
        transferred = prediction((11.0, 18.0))
        features = boundary_transfer_features(
            control,
            transferred,
            "recording",
            topk=25,
        )
        self.assertAlmostEqual(float(features["start_shift_mean"]), 0.1)
        self.assertAlmostEqual(float(features["end_shift_mean"]), -0.2)
        self.assertAlmostEqual(float(features["duration_ratio_mean"]), 0.7)
        self.assertAlmostEqual(float(features["changed_fraction"]), 1.0)

    def test_boundary_hypothesis_preserves_original_detection(self) -> None:
        control = prediction((10.0, 20.0))
        refined = prediction((11.0, 18.0))
        output = add_boundary_hypotheses(
            control,
            refined,
            topk_per_roi=1,
            score_scale=0.5,
        )
        detections = output["results"]["recording"]["1"]
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["segment"], [10.0, 20.0])
        self.assertEqual(detections[1]["segment"], [11.0, 18.0])
        self.assertAlmostEqual(detections[1]["score"], 0.45)

    def test_profile_conv_quality_head_shapes(self) -> None:
        model = ProfileConvQFL(summary_dim=17)
        logits = model(
            torch.zeros(3, 32),
            torch.zeros(3, 17),
        )
        self.assertEqual(tuple(logits.shape), (3,))

    def test_quality_frame_preserves_candidate_boundaries(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "rec_name": "recording",
                    "roi_id": 1,
                    "t_start": 10.0,
                    "t_end": 20.0,
                    "raw_global_rank": 0.5,
                }
            ]
        )
        output = quality_frame(frame, np.asarray([0.8]), score_blend=0.5)
        self.assertAlmostEqual(float(output.iloc[0]["t_start"]), 10.0)
        self.assertAlmostEqual(float(output.iloc[0]["t_end"]), 20.0)
        self.assertEqual(output.iloc[0]["model"], "proposal")

    def test_background_novelty_is_zero_for_prototype_direction(self) -> None:
        prototype = unit_centroid(np.asarray([[1.0, 0.0], [2.0, 0.0]]))
        self.assertAlmostEqual(
            cosine_novelty(np.asarray([3.0, 0.0]), prototype),
            0.0,
        )

    def test_novelty_rescore_keeps_row_count(self) -> None:
        frame = pd.DataFrame(
            {
                "score": [0.9, 0.1],
                "roi_background_novelty": [0.0, 1.0],
            }
        )
        rescored = novelty_rescore(
            frame,
            "roi_background_novelty",
            weight=0.5,
        )
        self.assertEqual(len(rescored), 2)
        self.assertTrue(np.isfinite(rescored["score"]).all())

    def test_feature_changepoint_saliency_peaks_at_direction_change(self) -> None:
        features = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        saliency = feature_changepoint_saliency(features, window_points=1)
        self.assertEqual(int(np.argmax(saliency)), 2)
        self.assertAlmostEqual(float(saliency[2]), 1.0)

    def test_local_saliency_peak_returns_grid_boundary(self) -> None:
        boundary, confidence = local_saliency_peak(
            np.asarray([0.0, 0.1, 0.9, 0.2]),
            boundary_seconds=3.0,
            stride_seconds=2.0,
            radius_seconds=2.0,
        )
        self.assertAlmostEqual(boundary, 4.0)
        self.assertAlmostEqual(confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
