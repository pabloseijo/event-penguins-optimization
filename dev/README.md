# `dev/` — experiment scripts

Every file here is a complete, self-contained experiment: it reads features or predictions,
applies one hypothesis and writes or prints its result. This directory is not a library. Stable
code lives in `src/`; what stays here is the record of everything that was tried, including
what did not work, because a discarded hypothesis is a result too.

Scripts marked ✦ produce numbers that appear in the paper. All of them run from the repository
root with the `eventpenguins` environment active.

```bash
conda activate eventpenguins
PYTHONPATH=.:dev python dev/<script>.py --help
```

Total: **227 scripts**.

## Contents

- [Data, subsets and splits](#data-subsets-and-splits) — 12 scripts
- [Feature extraction](#feature-extraction) — 10 scripts
- [Training](#training) — 15 scripts
- [Evaluation, cross-validation and fusion](#evaluation-cross-validation-and-fusion) — 107 scripts
- [Cross-domain transfer (THUMOS14-E)](#cross-domain-transfer-thumos14-e) — 27 scripts
- [Diagnostics and analysis](#diagnostics-and-analysis) — 16 scripts
- [Tests](#tests) — 40 scripts

## Data, subsets and splits

Everything downstream reads what these build: the ED prototype, recording-level folds,
condition subsets and the proposal lattice. A leaky split contaminates every number computed
after it, which is why these scripts carry the most assertions in the repository.

| Script | What it does |
| --- | --- |
| `build_condition_subset.py` | Build aligned proposal/representation subsets from acquisition metadata. |
| `build_oof_quality_hardness.py` | Assemble recording-disjoint quality scores for offline hard-example mining. |
| `build_proposal_lattice.py` | Build a boundary-rescue proposal lattice from an existing proposal CSV. |
| `build_prototype.py` | Build and evaluate the ED spatial prototype. |
| `build_quality_subset.py` | Materialize a stratified proposal/representation subset for quality-head training. |
| `build_recording_folds.py` | Build recording-level train/validation folds for proposal experiments. |
| `build_selftrain_feature_pool.py` | Merge source features with a few pseudo-labeled target recordings for self-training. |
| `build_selftraining_pseudolabels.py` | Consensus pseudo-labels for self-training on the hard day-15 test recordings. |
| `build_temporalmaxer_hybrid_sets.py` | Build source-CV and test sets selected by frozen CNN or GroupDRO scores. |
| `build_temporalmaxer_lattice_cv.py` | Build compact recording-disjoint TemporalMaxer folds from qhead lattice caches. |
| `build_temporalmaxer_screened_cv.py` | Build compact lattice folds using an operational frozen-CNN score screen. |
| `build_temporalmaxer_screened_test.py` | Build an operational test proposal set using only a frozen-CNN score screen. |

## Feature extraction

Turn events into the `[T, D]` per-ROI sequences the detectors consume: frozen-classifier
features, continuous event statistics, temporal and spectral descriptors, and the alternative
representations (TESPEC, TISM).

| Script | What it does |
| --- | --- |
| `extract_context_logits.py` | Extract and cache frozen-ATSN logits for transformed temporal windows. |
| `extract_continuous_event_features_v2.py` | Extract aligned ON/OFF profiles and signed spectra for continuous TAD. |
| `extract_continuous_event_stats.py` | Extract compact event statistics aligned with the continuous ATSN grid. |
| `extract_continuous_features.py` | Extract frozen ATSN features on a fixed grid over complete ROI timelines. |
| `extract_continuous_tespec_features.py` | Extract recurrent TESPEC features on the complete continuous ROI grid. |
| `extract_event_spectral_features.py` | Extract local ON/OFF and spectral features at the dense ATSN sample times. |
| `extract_sample.py` | Extract a small HDF5 subset for fast local debugging. |
| `extract_temporal_descriptors.py` | Append fixed temporal-order descriptors to cached ATSN representations. |
| `extract_tespec_features.py` | Extract aligned recurrent TESPEC features for a proposal master. |
| `extract_tism_features.py` | Extract proposal-level TISM features for external-CV ranking experiments. |

## Training

Train the added heads on top of frozen features. The reTAG backbone is never retrained; what is
trained here is the continuous detector, the quality head and the calibrators.

| Script | What it does |
| --- | --- |
| `fit_temperature.py` | Fit temperature scaling parameter T on the val split. |
| `pretrain_atsn_contrastive.py` | Temporal contrastive adaptation of the event-based ATSN representation. |
| `pretrain_temporalmaxer_continuous_bsp.py` | Boundary-Sensitive Pretraining on fixed windows from continuous source sequences. |
| `train_atsn_lpft.py` | Head-only LP-FT experiment for the Augmented TSN classifier. |
| `train_atsn_temporalmaxer_lpft.py` | Fine-tune ATSN layer4 with the dense TemporalMaxer localization objective. |
| `train_cascade_refiner.py` | Train a second-stage temporal refiner on recomputed ATSN features. |
| `train_fft_quality.py` | Train a lightweight quality predictor using FFT/event features (no GPU needed). |
| `train_oof_fusion_quality.py` | Cross-fitted second-stage quality calibration for detector fusion. |
| `train_precision_calibrator.py` | Precision recalibrator trained on a broad OOF hard-negative bank (all source recordings). |
| `train_quality_head.py` | Train a proposal quality/boundary head on top of frozen ATSN embeddings. |
| `train_score_calibrator.py` | Train an external proposal score calibrator/ranker. |
| ✦ `train_temporalmaxer_continuous.py` | Train and evaluate full-ROI TemporalMaxer with recording-disjoint folds. |
| `train_temporalmaxer_continuous_full.py` | Train the source-approved continuous detector on every source recording. |
| `train_temporalmaxer_dense.py` | Train a dense TemporalMaxer-lite head on ordered ATSN frame features. |
| `train_temporalmaxer_sada_lite_cv.py` | Pilot class-balanced semantic domain adaptation on recording-disjoint CV. |

## Evaluation, cross-validation and fusion

The bulk of the experimental work. By convention a `_cv` suffix means recording-disjoint cross-
validation, where hypotheses are selected, and a `_test` suffix means the frozen test split,
where results are reported. No `_cv` script reads test.

| Script | What it does |
| --- | --- |
| `aggregate_temporalmaxer_adaptation_cv.py` | Aggregate fixed-epoch adaptation metrics across recording folds. |
| `ar_dous_dominios.py` | Average Recall nos dous dominios, para a táboa comparativa do artigo. |
| `assemble_oof_scores.py` | Assemble recording-disjoint fold predictions in a reference proposal order. |
| `compute_f1.py` | Computa F1 score para o sistema final do TFG (R5) e compara co paper de Fourier. |
| `compute_segment_f1_current.py` | F1 a nivel de segmento do sistema actual, comparable co paper de Fourier. |
| `ensemble_logits.py` | Build a deterministic ensemble from aligned two-class logit caches. |
| `eval_actionness_lattice_rescue_cv.py` | Rescue high-recall lattice candidates with OOF actionness completeness. |
| `eval_actionness_profile_conv_quality_cv.py` | Cross-fit a small Conv1D QFL head on ordered actionness profiles. |
| `eval_actionness_profile_quality_head_cv.py` | Cross-fit a boundary-sensitive ordered-actionness quality head. |
| `eval_actionness_profile_quality_head_test.py` | Frozen test evaluation of the source-selected ordered-actionness QFL blend. |
| `eval_actionness_qfl_event_expert_blend_cv.py` | Test source-OOF complementarity between shared and decoupled event experts. |
| `eval_actionness_qfl_fusion_weights_cv.py` | Reselect three-expert fusion weights after replacing the proposal scorer. |
| `eval_actionness_qfl_groupdro_cv.py` | Evaluate recording-robust GroupDRO on the final linear QFL scorer. |
| `eval_actionness_qfl_pal_consistency_test.py` | Frozen test evaluation of the source-approved PAL-consistency QFL blend. |
| `eval_actionness_qfl_pal_postprocess_cv.py` | Evaluate capacity limits in the final PAL-consistency fusion funnel. |
| `eval_actionness_qfl_postprocess_cv.py` | Source-CV post-processing ablation for the actionness QFL fusion. |
| ✦ `eval_actionness_quality_head_cv.py` | Cross-fit a linear QFL proposal-quality head on OOF actionness features. |
| ✦ `eval_actionness_quality_head_test.py` | Single frozen test evaluation of the source-selected linear QFL head. |
| `eval_atsn_dense_lpft_test.py` | Evaluate the source-CV-approved surgical ATSN adaptation once on test. |
| `eval_atsn_dense_soup_cv.py` | Evaluate a fixed 50/50 base/adapted boundary soup in source CV. |
| `eval_atsn_pointwise_boundaries_cv.py` | Evaluate training-free ATSN pointwise boundaries in source external CV. |
| `eval_atsn_tta.py` | Episodic, recording-level test-time adaptation for the ATSN classifier. |
| `eval_aux_logits_fusion.py` | Evaluate an aligned auxiliary ATSN temporal view and fixed score fusions. |
| `eval_bem_logits_boundary_refine_cv.py` | Refine canonical detections with explicit BEM start/end probabilities. |
| `eval_boundary_consensus_score_cv.py` | Source-CV scoring from confidence consensus of overlapping proposals. |
| `eval_boundary_consensus_score_test.py` | Test the source-selected proposal-consensus confidence fusion. |
| `eval_boundary_quality_router_cv.py` | Train a strict recording-disjoint BREM-style boundary quality router. |
| `eval_boundary_refinement.py` | Evaluate event-rate boundary refinement on existing detections. |
| `eval_boundary_router_post_nms_cv.py` | Evaluate boundary experts strictly after reference Soft-NMS selection. |
| `eval_boundary_score_voting_cv.py` | Source-CV evaluation of score-weighted temporal boundary voting. |
| `eval_boundary_score_voting_test.py` | Evaluate the source-selected temporal boundary-voting recipe on test. |
| `eval_cascade_gates.py` | Evaluate label-free gates for trained temporal cascade checkpoints. |
| `eval_context_fusion.py` | Evaluate label-free surrounding-context penalties on proposal quality scores. |
| `eval_continuous_erm_groupdro_fusion_cv.py` | OOF fusion of complementary ERM and GroupDRO continuous detectors. |
| `eval_continuous_erm_groupdro_fusion_test.py` | Single test of the source-selected ERM/GroupDRO four-expert fusion. |
| `eval_continuous_feature_alignment_cv.py` | Evaluate ViTTA-style target-to-source feature-statistic alignment. |
| `eval_continuous_four_rep_fusion_cv.py` | OOF fusion of ATSN, two event representations, and proposal-local TAD. |
| `eval_continuous_four_rep_fusion_test.py` | Single test evaluation of the source-approved four-expert fusion. |
| ✦ `eval_continuous_multi_rep_fusion_cv.py` | Recording-disjoint CV fusion of continuous, event-stat, and proposal experts. |
| `eval_continuous_multi_rep_fusion_test.py` | Single fixed test evaluation of the source-selected three-expert fusion. |
| `eval_continuous_proposal_fusion_cv.py` | Late-fusion CV for complementary continuous and proposal-local detectors. |
| `eval_continuous_semantic_alignment_cv.py` | Align pseudo-background target statistics with labeled source background. |
| `eval_continuous_tag_cv.py` | Evaluate TAG proposals from source-OOF continuous actionness maps. |
| `eval_current_qfl_dfl_boundary_test.py` | Frozen test of source-selected DFL boundaries on the canonical QFL seed. |
| `eval_cv_score_ensemble.py` | Evaluate an equal-weight ensemble of aligned cross-validation predictions. |
| `eval_dfl_boundary_hypotheses_cv.py` | Evaluate conservative DFL boundary hypotheses on canonical predictions. |
| `eval_distributional_boundary_transfer_cv.py` | Transfer distributional-head boundaries to a fixed, source-validated ranking. |
| `eval_distributional_boundary_transfer_test.py` | Fixed test transfer of source-approved event-DFL boundaries. |
| `eval_embedding_prototypes.py` | Evaluate instance prototypes over frozen ATSN proposal embeddings. |
| `eval_event_boundary_reliability_router_cv.py` | Nested recording-disjoint CV for a conservative boundary-correction router. |
| `eval_event_quality_rescore_cv.py` | OOF evaluation of label-free event quality on fused temporal detections. |
| `eval_feature_changepoint_boundary_cv.py` | Refine candidate boundaries with local ATSN feature change points. |
| `eval_final_boundary_gradient_cv.py` | Refine final detection boundaries from local actionness transitions. |
| `eval_final_boundary_ridge_cv.py` | Cross-fit a regularized final-boundary regressor on source recordings. |
| `eval_gaussian_instance_fusion_cv.py` | Evaluate Gaussian weighted fusion on the current source-OOF QFL experts. |
| `eval_heteroscedastic_boundary_cv.py` | Cross-fit a candidate-conditioned uncertainty-aware boundary head. |
| `eval_multi_expert_boundary_voting_cv.py` | Source-CV robust voting across TemporalMaxer boundary experts. |
| `eval_multi_expert_boundary_voting_test.py` | Test the source-selected robust fusion of TemporalMaxer boundary experts. |
| `eval_multi_rep_consensus_cv.py` | Cross-model agreement scoring for the three source-validated TAD experts. |
| `eval_per_recording.py` | Break down a scored proposal file by recording and acquisition condition. |
| `eval_pre_nms_consensus_cv.py` | Compare source-selected consensus quality before versus after Soft-NMS. |
| `eval_precision_calibrated_fusion_test.py` | Apply the broad-pool precision calibrator to continuous+event test predictions, then fuse. |
| `eval_proposal_actionness_rescore_cv.py` | Rescore high-recall proposal segments with source-OOF actionness. |
| `eval_proposal_actionness_rescore_test.py` | Single frozen test evaluation of source-selected completeness rescoring. |
| `eval_proposal_context_cv.py` | Recording-disjoint proposal-context scoring inspired by P-GCN/ContextLoc. |
| `eval_proposal_csv_cnn.py` | Evaluate an arbitrary proposal CSV with the frozen ATSN classifier. |
| ✦ `eval_proposals.py` | Comparative evaluation of proposal variants on the test split. |
| `eval_recording_background_prototype_cv.py` | Rescore final detections by label-free recording background novelty. |
| `eval_recording_reliability_router_cv.py` | Nested-CV evaluation of a one-feature recording reliability router. |
| `eval_recording_reliability_router_test.py` | Frozen historical-test evaluation of the source-selected reliability router. |
| `eval_roi_presence_gate_cv.py` | Cross-fit MIL-style ROI presence calibration for a frozen TAD prediction. |
| `eval_roi_presence_negative_gate_cv.py` | Evaluate a nested high-confidence negative ROI gate on source OOF predictions. |
| `eval_roi_presence_negative_gate_test.py` | Frozen test evaluation of the source-approved negative ROI gate. |
| `eval_salient_boundary_router_cv.py` | Route temporal boundary experts with candidate-aligned boundary pooling. |
| `eval_salient_boundary_router_test.py` | Evaluate the one source-CV-approved salient boundary recipe on test. |
| `eval_score_calibration_mc_dropout_cv.py` | Calibration pilot: does MC Dropout give a better-calibrated confidence score? |
| `eval_score_ensemble.py` | Evaluate a fixed weighted ensemble of aligned proposal-score files. |
| `eval_score_normalization.py` | Evaluate label-free group normalization of temporal proposal scores. |
| `eval_scoring_variants.py` | Evaluate post-hoc scoring variants from cached proposal logits. |
| `eval_temporal_boundary_router_cv.py` | Route boundary experts from the full ATSN+TESPEC temporal sequence. |
| `eval_temporal_reversal_tta_cv.py` | Evaluate temporally equivariant reversal TTA on source recording folds. |
| `eval_temporalmaxer_bsp_cv.py` | Source-only BSP adaptation of a frozen-ATSN TemporalMaxer detector. |
| `eval_temporalmaxer_continuous_sweep.py` | Source-validation calibration of continuous TemporalMaxer scores and boundaries. |
| `eval_temporalmaxer_continuous_test.py` | Single fixed test evaluation of the source-approved continuous/proposal fusion. |
| `eval_temporalmaxer_cross_representation_cv.py` | Evaluate one fixed TESPEC/TISM/GroupDRO fusion on external source CV. |
| `eval_temporalmaxer_cross_representation_test.py` | Evaluate the fixed source-approved TESPEC/TISM fusion once on test. |
| `eval_temporalmaxer_ensemble.py` | Evaluate a fixed ensemble of TemporalMaxer boundary checkpoints. |
| `eval_temporalmaxer_groupdro_cv.py` | Evaluate GroupDRO scores with original or TemporalMaxer boundaries in CV. |
| `eval_temporalmaxer_groupdro_test.py` | Evaluate fixed GroupDRO scores with TemporalMaxer ensemble boundaries on test. |
| `eval_temporalmaxer_quality_boundary_cv.py` | Pair quality-focused TESPEC scores with fixed full-model TESPEC boundaries. |
| `eval_temporalmaxer_rank_fusion_cv.py` | Select a conservative GroupDRO/TemporalMaxer score fusion with source CV. |
| `eval_temporalmaxer_score_fusion_test.py` | Evaluate one source-CV-approved representation/score pair on test. |
| `eval_temporalmaxer_vitta_cv.py` | Recording-disjoint CV for one-step ViTTA on the continuous detector. |
| `eval_tespec_boundary_gate_cv.py` | Evaluate one BREM-style TESPEC quality-aware boundary gate on source CV. |
| `eval_tespec_coral_cv.py` | Evaluate label-free per-recording diagonal CORAL on TESPEC source CV. |
| `eval_tespec_coral_test.py` | Evaluate the source-CV-approved label-free TESPEC CORAL candidate on test. |
| `eval_tespec_eventmatch_cv.py` | Evaluate a lightweight EventMatch-style TESPEC adapter on source CV. |
| `eval_tta_guard.py` | Apply a label-free recording-level safety guard to cached TTA scores. |
| `eval_two_continuous_expert_blend_cv.py` | Evaluate source-OOF complementarity between two continuous experts. |
| `eval_vitta_multi_rep_fusion_cv.py` | Evaluate the fixed three-expert fusion with ViTTA continuous predictions. |
| `run_map_eval.py` | Full pipeline mAP evaluation across all proposal variants. |
| `run_nested_cascade_cv.py` | Run leakage-free nested CV for the recomputed-feature temporal cascade. |
| `run_quality_head_cv.py` | Run recording-disjoint quality-head folds across available GPUs. |
| `run_retag_debug.py` | Debug runner for the reTAG proposal pipeline. |
| `run_temporalmaxer_continuous_cv.py` | Run and aggregate full-ROI TemporalMaxer recording-disjoint CV folds. |
| `run_temporalmaxer_cv.py` | Run recording-disjoint TemporalMaxer-lite folds across available GPUs. |
| ✦ `tune_proposals.py` | Grid search over proposal hyperparameters on the validation split. |

## Cross-domain transfer (THUMOS14-E)

Conversion of THUMOS14 to events with v2e, the resulting THUMOS14-E corpus, and the
ActionFormer comparison arm. This is the evidence that the proposal generator does not depend
on the penguin domain.

| Script | What it does |
| --- | --- |
| `actionformer_pal_dataset.py` | Opt-in supervised Pseudo Action Localization dataset for ActionFormer. |
| `actionformer_pal_meta_arch.py` | ActionFormer meta-architecture with PAL region consistency. |
| `actionformer_pal_utils.py` | Pure helpers for the supervised PAL adaptation used with ActionFormer. |
| `analyze_thumos14_classes.py` | Select a reproducible THUMOS14 class subset from annotation structure. |
| `bootstrap_actionformer_transfer_oof.py` | Paired video-cluster bootstrap for OOF temporal detection predictions. |
| `build_actionformer_pal_config.py` | Create a fixed PAL-consistency config from an OOF ActionFormer config. |
| `build_thumos14_transfer_folds.py` | Build deterministic video-disjoint THUMOS14 folds for transfer studies. |
| `eval_actionformer_mil_gate_cv.py` | Cross-fit a conservative per-video/class negative MIL gate on OOF outputs. |
| `eval_actionformer_qfl_cv.py` | Select a linear QFL scorer with cross-fit THUMOS14 candidates. |
| `eval_actionformer_thumos14_chunked.py` | Run ActionFormer inference in restartable chunks and evaluate once. |
| `eval_actionformer_transfer_cv.py` | Evaluate domain-agnostic EventPenguins transfers on THUMOS14 OOF folds. |
| ✦ `evaluate_actionformer_transfer_final.py` | Evaluate the frozen THUMOS14 transfer recipe exactly once on test. |
| `evaluate_thumos14e_oof_proposals.py` | Avalía reTAG e o xerador CoTAD con propostas OOF en THUMOS14-E. |
| `evaluate_thumos14e_ovr.py` | Evaluate 20 one-vs-rest THUMOS14-E outputs with ActionFormer's evaluator. |
| ✦ `evaluate_thumos14e_proposals.py` | Evaluate frozen reTAG/EventPenguins proposals for all THUMOS14 classes. |
| `extract_actionformer_pre_nms.py` | Export restartable ActionFormer candidates before NMS and global truncation. |
| `prepare_actionformer_transfer_features.py` | Prepare domain-agnostic ActionFormer candidate features and tIoU targets. |
| ✦ `prepare_thumos14_event_corpus.py` | Prepare the complete, auditable THUMOS14-E corpus for reTAG comparisons. |
| `prepare_thumos14_event_pilot.py` | Prepare and validate a one-class THUMOS14 synthetic-event pilot. |
| `profile_thumos14_event_conversion.py` | Profile the THUMOS14-E conversion recipe on one action window per class. |
| `run_actionformer_transfer_final.py` | Fit the frozen OOF QFL recipe and emit target-free final predictions. |
| `run_thumos14_event_generalization.py` | Frozen EventPenguins -> THUMOS14-E generalization pilot. |
| ✦ `run_thumos14e_full_pipeline.py` | Run the complete supervised EventPenguins arm on THUMOS14-E. |
| `run_thumos14e_pilot.py` | Run the predeclared single-class THUMOS14-E integration pilot. |
| `run_thumos14e_supervised.py` | Orchestrate the literature-audited supervised THUMOS14-E comparison. |
| `train_actionformer_pal.py` | Register PAL extensions and delegate to the official ActionFormer trainer. |
| `train_thumos14_ovr_atsn.py` | Frozen-ATSN one-vs-rest transfer protocol for THUMOS14-E. |

## Diagnostics and analysis

These produce no publishable number; they explain why a number is what it is. Proposal
ceilings, boundary oracles, scene regimes, per-recording reliability and the error breakdown.

| Script | What it does |
| --- | --- |
| `analyze_action_distribution_gap.py` | Measure action-density and duration gaps without fitting to the test split. |
| `analyze_boundary_oracle_cv.py` | Measure the source-CV ceiling of routing between learned boundary experts. |
| `analyze_continuous_feature_shift.py` | Diagnose recording-level shifts in compact continuous event features. |
| `analyze_detection_errors.py` | Classify ranked temporal-detection errors after proposal post-processing. |
| `analyze_fft_phase.py` | Analyze FFT phase features for ED vs flap windows. |
| `analyze_scene_regimes.py` | Diagnose whether SAVS-like semantic regimes explain detection errors. |
| `check_env.py` | [EN] |
| `diagnose_atsn_linear_separability.py` | Measure whether frozen ATSN features linearly separate ED from background. |
| `diagnose_continuous_point_scores.py` | Audit pointwise action/background ranking inside continuous TemporalMaxer. |
| `diagnose_dfl_transfer_reliability.py` | Diagnose label-free reliability signals for DFL boundary transfer. |
| `diagnose_event_boundary_reliability.py` | Relate label-free boundary-head statistics to per-recording CV gains. |
| `diagnose_final_prediction_oracles.py` | Measure ranking and boundary ceilings of the final selected detections. |
| `diagnose_prediction_by_recording.py` | Diagnose localization ceiling and ranking errors of a prediction JSON. |
| `diagnose_proposal_ceiling.py` | Diagnose the proposal ceiling for a target mAP. |
| `diagnose_recording_expert_reliability.py` | Diagnose label-free recording signals for adaptive expert fusion. |
| `inspect_h5.py` | [EN] |

## Tests

The `unittest` suite. It covers protocol invariants above all — that a split cannot absorb
test, that a calibration is fitted on training data only — because that class of bug does not
surface as a failure but as a number that looks too good.

| Script | What it does |
| --- | --- |
| `test_action_distribution_gap.py` | Tests for the action-distribution diagnostic. |
| `test_actionformer_pal_utils.py` | Tests for the supervised PAL helpers: background sampling and contrastive loss. |
| `test_actionness_qfl_groupdro.py` | Tests for GroupDRO training of the final QFL scorer. |
| `test_actionness_qfl_pal_postprocess.py` | Tests for final PAL-consistency post-processing diagnostics. |
| `test_adaptive_event_windows.py` | Tests for density-adaptive sample durations in the proposal dataset. |
| `test_analyze_thumos14_classes.py` | Tests for the THUMOS14 class analysis and its jackknife stability check. |
| `test_attention_neck.py` | Probas do neck de atención local engadido para a comparación arquitectónica do artigo. |
| `test_bootstrap_actionformer_transfer_oof.py` | Tests for the bootstrap confidence intervals over out-of-fold transfer results. |
| `test_build_thumos14_transfer_folds.py` | Tests for the construction of the THUMOS14 transfer folds. |
| `test_continuous_event_features_v2.py` | Focused tests for aligned continuous ON/OFF and spectral descriptors. |
| `test_eval_actionformer_mil_gate_cv.py` | Tests for the multiple-instance gate over ActionFormer detections. |
| `test_eval_actionformer_qfl_cv.py` | Tests for the quality focal-loss head fitted in cross-validation. |
| `test_eval_actionformer_transfer_cv.py` | Tests for the class-wise ranking and calibration used in transfer evaluation. |
| `test_evaluate_actionformer_transfer_final.py` | Tests for the calibration metrics of the final transfer evaluation. |
| `test_evaluate_thumos14e_ovr.py` | Tests for the one-versus-rest aggregation on THUMOS14-E. |
| `test_evaluate_thumos14e_proposals.py` | Tests for the THUMOS14-E proposal evaluation and its protocol declaration. |
| `test_event_boundary_reliability_router.py` | Tests for the boundary-reliability router between two prediction sets. |
| `test_extract_actionformer_pre_nms.py` | Tests for the extraction of ActionFormer candidates before NMS. |
| `test_gaussian_instance_fusion.py` | Tests for gaussian instance fusion of overlapping candidates. |
| `test_heteroscedastic_boundary.py` | Tests for the heteroscedastic boundary model and its diagnostics. |
| `test_prepare_actionformer_transfer_features.py` | Tests for the feature preparation of the ActionFormer transfer. |
| `test_profile_thumos14_event_conversion.py` | Tests for the profiling helpers of the THUMOS14 event conversion. |
| `test_rank_sort_loss.py` | Tests for the Rank & Sort loss. |
| `test_recording_expert_reliability.py` | Tests for recording-level expert reliability diagnostics. |
| `test_roi_presence_gate.py` | Focused tests for the cross-fit ROI presence gate. |
| `test_roi_presence_negative_gate.py` | Tests for nested high-confidence negative ROI calibration. |
| `test_roi_presence_negative_gate_test.py` | Tests for frozen negative ROI gate test inference. |
| `test_run_actionformer_transfer_final.py` | Tests for the output-directory guard of the final transfer run. |
| `test_run_thumos14e_full_pipeline.py` | Tests for the end-to-end THUMOS14-E pipeline runner. |
| `test_run_thumos14e_supervised.py` | Tests for the supervised THUMOS14-E runner and its protocol locks. |
| `test_scene_regimes.py` | Tests for the label-free SAVS-style regime diagnostic. |
| `test_score_calibration_mc_dropout.py` | Tests for the MC-dropout score calibration diagnostics. |
| `test_temporalmaxer_continuous.py` | Focused invariants for the full-sequence TemporalMaxer path. |
| `test_temporalmaxer_dense.py` | Tests for the dense TemporalMaxer heads, their training loop and decoding. |
| `test_temporalmaxer_sada_lite.py` | Unit tests for class-balanced semantic adaptation helpers. |
| `test_temporalmaxer_vitta.py` | Tests for continuous ViTTA primitives. |
| `test_thumos14_event_corpus.py` | Tests for the assembly of the THUMOS14 event corpus. |
| `test_thumos14_event_generalization.py` | Tests for the cross-domain proposal protocol on THUMOS14-E. |
| `test_thumos14_event_pilot.py` | Tests for the pilot conversion of THUMOS14 to events with v2e. |
| `test_thumos14_ovr_atsn.py` | Tests for the one-versus-rest ATSN training on THUMOS14-E. |
