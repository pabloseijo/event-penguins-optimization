"""Frozen EventPenguins -> THUMOS14-E generalization pilot.

The command deliberately separates inference from evaluation:

1. ``freeze`` hashes every source artifact and the target HDF5.
2. ``stage1`` runs reTAG and the source-approved proposal stage.
3. ``local`` completes the lattice, GroupDRO and local TemporalMaxer branch.
4. ``full`` fuses the three frozen experts with completeness and QFL.
5. ``evaluate`` is the only command that reads real target annotations.

Run from the repository root. LongJump-only runs are exploratory diagnostics;
reportable AP requires the predeclared complete target test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification import ProposalClassifier  # noqa: E402
from src.proposals import ProposalGenerator  # noqa: E402
from dev.eval_actionness_quality_head_cv import (  # noqa: E402
    FEATURE_COLUMNS,
    candidate_features,
    fit_linear_qfl,
    score_quality_head,
)
from dev.eval_continuous_multi_rep_fusion_cv import (  # noqa: E402
    build_prediction,
    prediction_rows,
)
from dev.eval_continuous_tag_cv import make_loader  # noqa: E402
from dev.eval_proposal_actionness_rescore_cv import (  # noqa: E402
    extract_actionness,
    frame_prediction,
)
from dev.eval_temporalmaxer_continuous_test import load_models  # noqa: E402
from dev.eval_multi_expert_boundary_voting_cv import (  # noqa: E402
    build_multi_expert_prediction,
)
from dev.eval_multi_expert_boundary_voting_test import (  # noqa: E402
    add_test_expert_boundaries,
)
from dev.train_temporalmaxer_dense import (  # noqa: E402
    load_cache as load_dense_cache,
    make_model as make_dense_model,
    score_model as score_dense_model,
    stable_proposal_index,
)


PROTOCOL_VERSION = "eventpenguins-to-thumos14e-frozen-v2"
TIOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.7)
THUMOS_TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
PROPOSAL_BUDGETS = (20, 30, 50)

RETAG_PROPOSAL_RECIPE = {
    "bin_width": 0.033,
    "percentile": 1.0,
    "nms_threshold": 0.95,
}

FULL_STAGE1_PROPOSAL_RECIPE = {
    **RETAG_PROPOSAL_RECIPE,
    "use_adaptive_lambda": True,
    "lambda_percentile": 75,
    "lambda_delta": 0.20,
    "use_spatial_compactness": True,
    "spatial_weight": 0.2,
    "use_noise_penalization": True,
    "use_dispersed_noise": True,
    "use_periodicity": True,
    "periodicity_global_threshold": 0.65,
    "periodicity_local_threshold": 0.60,
}

RETAG_CLASSIFIER_RECIPE = {
    "num_tsn_samples": 7,
    "augment_factor": 3,
    "sample_duration": 1.0,
    "decay": 5e-6,
    "nms_threshold": 0.5,
    "use_soft_nms": False,
    "min_ed_score": 0.5,
}

FULL_STAGE1_CLASSIFIER_RECIPE = {
    **RETAG_CLASSIFIER_RECIPE,
    "augment_factor": 5,
    "use_soft_nms": True,
    "soft_nms_sigma": 0.25,
    "soft_nms_score_threshold": 0.001,
    "min_ed_score": 0.3,
    "temperature": 2.0,
    "duration_penalty_dmax": 60.0,
    "duration_penalty_sigma": 20.0,
}

FROZEN_FUSION_RECIPE = {
    "expert_weights": [0.20, 0.40, 0.40],
    "soft_nms_sigma": 0.50,
    "per_expert_roi_topk": 100,
    "max_predictions_per_roi": 200,
    "minimum_duration_s": 2.0,
    "completeness_rank_mix": [0.75, 0.25],
    "qfl_original_learned_mix": [0.50, 0.50],
    "score_normalization": "global percentile rank (transductive, label-free)",
}

LOCAL_EXPERT_RECIPE = {
    "lattice": {
        "min_duration_s": 2.0,
        "max_duration_s": 60.0,
        "top_k_per_roi": 300,
        "score_quantile": 0.35,
        "max_per_roi": 3500,
        "nms_threshold": 0.995,
    },
    "hybrid_screen": {"cnn_threshold": 0.09, "quality_threshold": 0.10},
    "prefix_screen": {"cnn_threshold": 0.10},
    "quality_ensemble": "mean of five source GroupDRO heads",
    "temporal_ensemble": "mean of five source TemporalMaxer-lite heads",
    "nms_boundary": "50% raw + 50% mean TemporalMaxer delta",
    "boundary_experts": ["raw", "blend050", "delta", "distribution", "point"],
    "boundary_estimator": "weighted median",
    "boundary_vote_tiou": 0.5,
    "boundary_vote_topk": 100,
    "consensus_score_blend": 0.25,
    "minimum_quality_score": 0.10,
    "soft_nms_sigma": 0.25,
}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_paths(root: Path) -> list[Path]:
    return [root / f"fold_{fold:02d}" / "best.pt" for fold in range(5)]


def quality_checkpoint_paths(root: Path) -> list[Path]:
    return [root / f"fold_{fold:02d}" / "qhead_qfl_only.pt" for fold in range(5)]


def source_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    artifacts = {
        "atsn": resolve(args.model_path),
        "eventpenguins_prototype": resolve(args.prototype_path),
        "qfl_source_candidates": resolve(args.qfl_source_features),
    }
    for family, root_value in (
        ("continuous", args.continuous_root),
        ("eventstats", args.event_root),
        ("dense", args.dense_root),
    ):
        for fold, path in enumerate(checkpoint_paths(resolve(root_value))):
            artifacts[f"{family}_fold_{fold:02d}"] = path
    for fold, path in enumerate(quality_checkpoint_paths(resolve(args.quality_root))):
        artifacts[f"quality_fold_{fold:02d}"] = path
    return artifacts


def selected_target_recordings(data_path: Path, split: str) -> list[str]:
    with h5py.File(data_path, "r") as handle:
        recordings = [
            str(recording)
            for recording in sorted(handle.keys())
            if str(handle[recording].attrs.get("split", "")) == split
        ]
    if not recordings:
        raise ValueError(f"No target recordings use split={split!r}")
    return recordings


def audit_target_h5(data_path: Path, split: str) -> dict:
    recordings = selected_target_recordings(data_path, split)
    rows = []
    with h5py.File(data_path, "r") as handle:
        for recording in recordings:
            group = handle[recording]
            for roi in sorted(group.keys()):
                roi_group = group[roi]
                events = roi_group["events"]
                if events.ndim != 2 or events.shape[1] != 4:
                    raise ValueError(f"{recording}/{roi} must have event shape [N,4]")
                if len(events) < 2:
                    raise ValueError(f"{recording}/{roi} has fewer than two events")
                timestamps = np.asarray(events[:, 2])
                if np.any(timestamps[1:] < timestamps[:-1]):
                    raise ValueError(f"{recording}/{roi} timestamps are not monotonic")
                width = int(roi_group.attrs["width"])
                height = int(roi_group.attrs["height"])
                x = np.asarray(events[:, 0])
                y = np.asarray(events[:, 1])
                polarity = np.asarray(events[:, 3])
                if x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height:
                    raise ValueError(f"{recording}/{roi} has out-of-bounds coordinates")
                if not set(np.unique(polarity)).issubset({0, 1}):
                    raise ValueError(f"{recording}/{roi} polarity must be binary")
                duration_s = float(
                    roi_group.attrs.get(
                        "duration_s",
                        group.attrs.get("duration_s", float(timestamps[-1]) / 1e6),
                    )
                )
                rows.append(
                    {
                        "recording": recording,
                        "roi": str(roi),
                        "events": int(len(events)),
                        "duration_s": duration_s,
                        "width": width,
                        "height": height,
                    }
                )
    return {
        "path": str(data_path),
        "sha256": sha256_file(data_path),
        "split": split,
        "recordings": recordings,
        "rois": rows,
    }


def artifact_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def create_manifest(args: argparse.Namespace) -> dict:
    data_path = resolve(args.data_path)
    target = audit_target_h5(data_path, args.split)
    artifacts = {
        name: artifact_record(path) for name, path in source_artifact_paths(args).items()
    }
    return {
        "protocol": PROTOCOL_VERSION,
        "created_before_target_evaluation": not args.exploratory_post_smoke,
        "exploratory_completion_after_prior_smoke": bool(args.exploratory_post_smoke),
        "target_annotations_accessed": False,
        "git_revision": git_revision(),
        "target": target,
        "source_artifacts": artifacts,
        "recipes": {
            "retag_proposals": RETAG_PROPOSAL_RECIPE,
            "full_stage1_proposals": FULL_STAGE1_PROPOSAL_RECIPE,
            "retag_classifier": RETAG_CLASSIFIER_RECIPE,
            "full_stage1_classifier": FULL_STAGE1_CLASSIFIER_RECIPE,
            "local_expert": LOCAL_EXPERT_RECIPE,
            "full_fusion": FROZEN_FUSION_RECIPE,
        },
        "forbidden": [
            "target training",
            "target validation selection",
            "target prototype construction",
            "target hyperparameter tuning",
            "target-label normalization",
        ],
    }


def verify_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported manifest protocol in {manifest_path}")
    target = manifest["target"]
    if sha256_file(Path(target["path"])) != target["sha256"]:
        raise ValueError("Target HDF5 changed after the source recipe was frozen")
    for name, record in manifest["source_artifacts"].items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen source artifact changed: {name}")
    return manifest


def save_predictions(path: Path, predictions: dict, version: str) -> None:
    output = dict(predictions)
    output["version"] = version
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def write_empty_target_metadata(manifest: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write an annotation-shaped file that contains no target labels."""
    annotation_dir = out_dir / "empty_target_metadata"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    rois_by_recording: dict[str, list[str]] = {}
    for row in manifest["target"]["rois"]:
        rois_by_recording.setdefault(str(row["recording"]), []).append(str(row["roi"]))
    database = {}
    for recording in manifest["target"]["recordings"]:
        roi_annotations = {
            str(int(roi[1:])): [] for roi in sorted(rois_by_recording[str(recording)])
        }
        database[str(recording)] = {
            "subset": str(manifest["target"]["split"]),
            "annotations": roi_annotations,
        }
    annotation_path = annotation_dir / "annotations.json"
    annotation_path.write_text(json.dumps({"database": database}, indent=2), encoding="utf-8")
    info_path = annotation_dir / "recording_info.csv"
    pd.DataFrame(
        {
            "timestamp": [str(value) for value in manifest["target"]["recordings"]],
            "split": str(manifest["target"]["split"]),
        }
    ).to_csv(info_path, index=False)
    return annotation_path, info_path


def unique_proposals(frame: pd.DataFrame) -> pd.DataFrame:
    index = stable_proposal_index(frame)
    return frame.loc[~index.duplicated()].reset_index(drop=True)


def prefix_order(frame: pd.DataFrame, prefix: pd.DataFrame) -> pd.DataFrame:
    frame = unique_proposals(frame)
    index = stable_proposal_index(frame)
    positions = index.get_indexer(stable_proposal_index(unique_proposals(prefix)))
    if np.any(positions < 0):
        raise ValueError("The frozen-CNN prefix is not contained in the hybrid proposal set")
    remaining = np.ones(len(frame), dtype=bool)
    remaining[positions] = False
    return pd.concat((frame.iloc[positions], frame.loc[remaining]), ignore_index=True)


def empty_prediction(manifest: dict) -> dict:
    results: dict[str, dict[str, list]] = {
        str(recording): {} for recording in manifest["target"]["recordings"]
    }
    for row in manifest["target"]["rois"]:
        results[str(row["recording"])][str(int(str(row["roi"])[1:]))] = []
    return {"version": f"{PROTOCOL_VERSION}:empty-local-expert", "results": results}


def run_stage1(args: argparse.Namespace) -> None:
    manifest_path = resolve(args.manifest)
    manifest = verify_manifest(manifest_path)
    data_path = Path(manifest["target"]["path"])
    split = str(manifest["target"]["split"])
    source = manifest["source_artifacts"]
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    prototype = np.load(source["eventpenguins_prototype"]["path"])

    variants = (
        (
            "retag",
            RETAG_PROPOSAL_RECIPE,
            RETAG_CLASSIFIER_RECIPE,
            None,
        ),
        (
            "eventpenguins_stage1",
            FULL_STAGE1_PROPOSAL_RECIPE,
            FULL_STAGE1_CLASSIFIER_RECIPE,
            prototype,
        ),
    )
    summary = []
    for name, proposal_recipe, classifier_recipe, source_prototype in variants:
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        generator_kwargs = {
            **proposal_recipe,
            "data_path": str(data_path),
            "output_dir": str(variant_dir / "proposal_logs"),
        }
        if source_prototype is not None:
            generator_kwargs["prototype"] = source_prototype
        proposals = ProposalGenerator(**generator_kwargs).run(split=split)
        proposals.to_csv(variant_dir / "proposals.csv", index=False)

        classifier = ProposalClassifier(
            device=device,
            model_path=source["atsn"]["path"],
            data_path=str(data_path),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **classifier_recipe,
        )
        predictions = classifier.run(proposals)
        save_predictions(
            variant_dir / "predictions.json",
            predictions,
            f"{PROTOCOL_VERSION}:{name}",
        )
        detections = sum(
            len(items)
            for recording in predictions["results"].values()
            for items in recording.values()
        )
        summary.append(
            {"variant": name, "proposals": int(len(proposals)), "detections": detections}
        )
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def run_local_expert(args: argparse.Namespace) -> None:
    """Run the complete source-frozen proposal expert without real target labels."""
    manifest = verify_manifest(resolve(args.manifest))
    source = manifest["source_artifacts"]
    data_path = Path(manifest["target"]["path"])
    target_recordings = sorted(str(value) for value in manifest["target"]["recordings"])
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = resolve(args.proposal_csv)
    base_proposals = pd.read_csv(proposal_path)
    proposal_recordings = sorted(base_proposals["rec_name"].astype(str).unique().tolist())
    if proposal_recordings != target_recordings:
        raise ValueError(
            f"Local proposal recordings {proposal_recordings} != frozen {target_recordings}"
        )

    lattice_path = out_dir / "lattice_proposals.csv"
    lattice_recipe = LOCAL_EXPERT_RECIPE["lattice"]
    run_checked(
        [
            sys.executable,
            "dev/build_proposal_lattice.py",
            "--proposals",
            str(proposal_path),
            "--out-proposals",
            str(lattice_path),
            "--data-path",
            str(data_path),
            "--min-duration-s",
            str(lattice_recipe["min_duration_s"]),
            "--max-duration-s",
            str(lattice_recipe["max_duration_s"]),
            "--top-k-per-roi",
            str(lattice_recipe["top_k_per_roi"]),
            "--score-quantile",
            str(lattice_recipe["score_quantile"]),
            "--max-per-roi",
            str(lattice_recipe["max_per_roi"]),
            "--nms-threshold",
            str(lattice_recipe["nms_threshold"]),
        ]
    )
    annotation_path, info_path = write_empty_target_metadata(manifest, out_dir)

    quality_root = out_dir / "quality_heads"
    representation_path = quality_root / "target_repr.npz"
    quality_frames = []
    for fold in range(5):
        fold_dir = quality_root / f"fold_{fold:02d}"
        label = f"target_fold_{fold:02d}"
        run_checked(
            [
                sys.executable,
                "dev/train_quality_head.py",
                "--data-path",
                str(data_path),
                "--ann-path",
                str(annotation_path),
                "--model-path",
                source["atsn"]["path"],
                "--val-proposals",
                str(lattice_path),
                "--val-repr",
                str(representation_path),
                "--out-dir",
                str(fold_dir),
                "--eval-checkpoint",
                source[f"quality_fold_{fold:02d}"]["path"],
                "--eval-label",
                label,
                "--skip-evaluation",
                "--repr-batch-size",
                str(args.qhead_repr_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--quiet-progress",
            ]
        )
        score_path = fold_dir / "cache" / f"{label}_scores_qhead_qfl_only.csv"
        quality_frames.append(pd.read_csv(score_path).reset_index(drop=True))

    expected = stable_proposal_index(quality_frames[0])
    for fold, frame in enumerate(quality_frames[1:], start=1):
        if len(frame) != len(quality_frames[0]) or not stable_proposal_index(frame).equals(
            expected
        ):
            raise ValueError(f"Target quality-head fold {fold} is misaligned")
    mean_quality = np.mean(
        [frame["quality_score"].to_numpy(dtype=np.float64) for frame in quality_frames],
        axis=0,
    )
    base = quality_frames[0]
    output_columns = ["rec_name", "roi_id", "t_start", "t_end", "score"]
    prefix = unique_proposals(
        base.loc[
            base["cnn_score"] >= LOCAL_EXPERT_RECIPE["prefix_screen"]["cnn_threshold"],
            output_columns,
        ]
    )
    keep = (
        base["cnn_score"].to_numpy(dtype=np.float64)
        >= LOCAL_EXPERT_RECIPE["hybrid_screen"]["cnn_threshold"]
    ) | (
        mean_quality >= LOCAL_EXPERT_RECIPE["hybrid_screen"]["quality_threshold"]
    )
    hybrid = prefix_order(base.loc[keep, output_columns], prefix)
    prefix.to_csv(out_dir / "screened_prefix.csv", index=False)
    hybrid_path = out_dir / "hybrid_proposals.csv"
    hybrid.to_csv(hybrid_path, index=False)

    prediction_path = out_dir / "predictions.json"
    if hybrid.empty:
        save_predictions(
            prediction_path,
            empty_prediction(manifest),
            f"{PROTOCOL_VERSION}:complete-local-expert-empty-after-source-screen",
        )
        report = {
            "base_stage1_proposals": int(len(base_proposals)),
            "lattice_proposals": int(len(base)),
            "hybrid_proposals": 0,
            "detections": 0,
            "real_target_annotations_accessed": False,
            "empty_annotation_sha256": sha256_file(annotation_path),
            "recording_info_sha256": sha256_file(info_path),
            "prediction_sha256": sha256_file(prediction_path),
        }
        (out_dir / "inference_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return

    base_index = stable_proposal_index(base)
    hybrid_positions = base_index.get_indexer(stable_proposal_index(hybrid))
    if np.any(hybrid_positions < 0):
        raise ValueError("Hybrid proposals cannot be aligned with quality-head outputs")
    aligned_quality = [
        frame.iloc[hybrid_positions].reset_index(drop=True) for frame in quality_frames
    ]

    dense_cache = out_dir / "dense_cache"
    run_checked(
        [
            sys.executable,
            "dev/train_temporalmaxer_dense.py",
            "--data-path",
            str(data_path),
            "--ann-path",
            str(annotation_path),
            "--model-path",
            source["atsn"]["path"],
            "--master-proposals",
            str(hybrid_path),
            "--cache-dir",
            str(dense_cache),
            "--timestamp-cache-dir",
            str(out_dir / "dense_roi_timestamps"),
            "--out-dir",
            str(out_dir / "dense_extract"),
            "--extract-only",
            "--repr-batch-size",
            str(args.dense_repr_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--quiet-progress",
        ]
    )
    _, logits, metadata = load_dense_cache(dense_cache)
    device = torch.device(args.device)
    temporal_frames = []
    indices = np.arange(len(hybrid), dtype=np.int64)
    for fold in range(5):
        checkpoint_path = Path(source[f"dense_fold_{fold:02d}"]["path"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_args = checkpoint.get("args", {})
        dense_args = SimpleNamespace(
            hidden_dim=int(checkpoint_args.get("hidden_dim", 128)),
            pyramid_levels=int(checkpoint_args.get("pyramid_levels", 3)),
            dropout=float(checkpoint_args.get("dropout", 0.15)),
            trident_bins=int(checkpoint_args.get("trident_bins") or 0),
            tanp_sigma=float(checkpoint_args.get("tanp_sigma", 0.0)),
            event_feature_cache_dir=None,
            event_features_only=False,
            batch_size=args.dense_batch_size,
            augment_factor=5,
            max_boundary_delta=float(checkpoint_args.get("max_boundary_delta", 0.75)),
            boundary_blend=float(checkpoint_args.get("boundary_blend", 0.75)),
        )
        model = make_dense_model(metadata, dense_args).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        scored = score_dense_model(
            model,
            hybrid,
            indices,
            dense_cache / "frame_features.npy",
            logits,
            dense_args,
            device,
        )
        scored.to_csv(out_dir / f"temporal_scored_fold_{fold:02d}.csv", index=False)
        temporal_frames.append(scored)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scored = hybrid.copy()
    scored["quality_score"] = np.mean(
        [frame["quality_score"].to_numpy(dtype=np.float64) for frame in aligned_quality],
        axis=0,
    )
    mean_delta_start = np.mean(
        [frame["delta_t_start"].to_numpy(dtype=np.float64) for frame in temporal_frames],
        axis=0,
    )
    mean_delta_end = np.mean(
        [frame["delta_t_end"].to_numpy(dtype=np.float64) for frame in temporal_frames],
        axis=0,
    )
    scored["reference_blend050_t_start"] = 0.5 * scored["t_start"] + 0.5 * mean_delta_start
    scored["reference_blend050_t_end"] = 0.5 * scored["t_end"] + 0.5 * mean_delta_end
    scored = add_test_expert_boundaries(scored, temporal_frames)
    scored.to_csv(out_dir / "scored.csv", index=False)
    prediction_args = SimpleNamespace(
        nms_boundary="reference_blend050",
        score_column="quality_score",
        min_score=0.10,
        duration_dmax=60.0,
        duration_sigma=20.0,
        pre_nms_topk_per_roi=1000,
        soft_nms_sigma=0.25,
        soft_nms_score_threshold=0.001,
        vote_tiou=0.5,
        multi_vote_topk=100,
    )
    prediction = build_multi_expert_prediction(scored, prediction_args, 0.5, "median")
    save_predictions(
        prediction_path,
        prediction,
        f"{PROTOCOL_VERSION}:complete-local-expert-multi-median-blend050",
    )
    detections = sum(
        len(items)
        for recording in prediction["results"].values()
        for items in recording.values()
    )
    report = {
        "base_stage1_proposals": int(len(base_proposals)),
        "lattice_proposals": int(len(base)),
        "screened_prefix_proposals": int(len(prefix)),
        "hybrid_proposals": int(len(hybrid)),
        "detections": int(detections),
        "real_target_annotations_accessed": False,
        "empty_annotation_sha256": sha256_file(annotation_path),
        "recording_info_sha256": sha256_file(info_path),
        "prediction_sha256": sha256_file(prediction_path),
        "recipe": LOCAL_EXPERT_RECIPE,
    }
    (out_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def proposal_csv_prediction(path: Path, recordings: Iterable[str]) -> dict:
    frame = proposals_in_seconds(pd.read_csv(path))
    result: dict[str, dict[str, list]] = {str(recording): {} for recording in recordings}
    for (recording, roi_id), group in frame.groupby(["rec_name", "roi_id"], sort=False):
        result[str(recording)][str(int(roi_id))] = [
            {
                "label": "ed",
                "segment": [float(row.t_start), float(row.t_end)],
                "score": float(row.score),
            }
            for row in group.itertuples(index=False)
        ]
    return {"version": f"{PROTOCOL_VERSION}:raw-stage1-proposals", "results": result}


def run_full_fusion(args: argparse.Namespace) -> None:
    manifest = verify_manifest(resolve(args.manifest))
    source = manifest["source_artifacts"]
    device = torch.device(args.device)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fit the source-only QFL model before any target feature is loaded.
    source_frame = pd.read_csv(source["qfl_source_candidates"]["path"])
    qfl_model = fit_linear_qfl(source_frame, device, args.steps, args.learning_rate)
    mean, std, weight, bias = qfl_model
    qfl_record = {
        "source_sha256": source["qfl_source_candidates"]["sha256"],
        "feature_columns": list(FEATURE_COLUMNS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weight": weight.tolist(),
        "bias": bias,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "score_blend": 0.5,
    }
    (out_dir / "frozen_source_qfl.json").write_text(
        json.dumps(qfl_record, indent=2), encoding="utf-8"
    )

    feature_dir = resolve(args.feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv")
    target_recordings = sorted(manifest["target"]["recordings"])
    feature_recordings = sorted(sequences["rec_name"].astype(str).unique().tolist())
    if feature_recordings != target_recordings:
        raise ValueError(
            f"Target feature recordings {feature_recordings} != frozen {target_recordings}"
        )
    if Path(metadata["data_path"]).resolve() != Path(manifest["target"]["path"]).resolve():
        raise ValueError("Target feature cache was not built from the frozen HDF5")

    continuous_paths = [
        Path(source[f"continuous_fold_{fold:02d}"]["path"]) for fold in range(5)
    ]
    models = load_models(
        continuous_paths[0].parents[1],
        int(metadata["feature_dim"]),
        device,
        continuous_paths,
    )
    loader = make_loader(feature_dir, sequences, args.batch_size, args.num_workers, device)
    actionness = extract_actionness(models, loader, device)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw_proposal = json.loads(resolve(args.proposal_prediction).read_text(encoding="utf-8"))
    proposal_recordings = sorted(str(value) for value in raw_proposal.get("results", {}))
    if proposal_recordings != target_recordings:
        raise ValueError(
            f"Local expert recordings {proposal_recordings} != frozen {target_recordings}"
        )
    local_detection_count = sum(
        len(items)
        for recording in raw_proposal["results"].values()
        for items in recording.values()
    )
    if local_detection_count:
        target_features = candidate_features(
            raw_proposal,
            actionness,
            float(metadata["grid_stride_s"]),
            annotations={},
            fold=-1,
        )
    else:
        target_features = pd.DataFrame(
            columns=[
                "rec_name",
                "roi_id",
                "t_start",
                "t_end",
                *FEATURE_COLUMNS,
                "target_tiou",
            ]
        )
    if target_features.empty:
        proposal_frame = pd.DataFrame(
            columns=["rec_name", "roi_id", "t_start", "t_end", "raw_score", "rank_score", "model"]
        )
    else:
        proposal_frame = score_quality_head(target_features, qfl_model, blend=0.5)
    target_features.drop(columns=["target_tiou"], errors="ignore").to_csv(
        out_dir / "target_candidate_features_no_labels.csv", index=False
    )
    proposal_prediction = frame_prediction(proposal_frame, sigma=0.5, max_predictions=200)
    save_predictions(
        out_dir / "proposal_qfl_predictions.json",
        proposal_prediction,
        f"{PROTOCOL_VERSION}:source-qfl-proposal",
    )

    continuous = json.loads(resolve(args.continuous_prediction).read_text(encoding="utf-8"))
    event = json.loads(resolve(args.event_prediction).read_text(encoding="utf-8"))
    frames = [
        prediction_rows(continuous, "continuous"),
        prediction_rows(event, "event"),
        proposal_frame,
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("All three frozen experts produced empty predictions")
    fused = build_prediction(
        frames,
        {"continuous": 0.2, "event": 0.4, "proposal": 0.4},
        sigma=0.5,
        per_model_topk=100,
        max_predictions=200,
    )
    save_predictions(
        out_dir / "predictions.json",
        fused,
        f"{PROTOCOL_VERSION}:continuous-event-stage1-qfl-fusion",
    )
    report = {
        "prediction": str(out_dir / "predictions.json"),
        "prediction_sha256": sha256_file(out_dir / "predictions.json"),
        "qfl_model": str(out_dir / "frozen_source_qfl.json"),
        "target_candidates": int(len(target_features)),
        "local_input_detections": int(local_detection_count),
        "transductive_global_ranking": True,
        "local_expert": "complete lattice/GroupDRO/TemporalMaxer/voting proposal branch",
        "local_prediction_sha256": sha256_file(resolve(args.proposal_prediction)),
    }
    (out_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def normalize_roi(value: object) -> int:
    text = str(value)
    return int(text[1:]) if text.startswith("N") else int(text)


def load_generic_ground_truth(annotation_path: Path, recordings: Iterable[str]) -> pd.DataFrame:
    selected = set(recordings)
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    for recording, entry in raw["database"].items():
        if recording not in selected:
            continue
        for roi, annotations in entry["annotations"].items():
            if str(roi) == "null":
                continue
            for annotation in annotations:
                start, end = map(float, annotation["segment"])
                if end <= start:
                    raise ValueError(f"Invalid target segment in {recording}/{roi}")
                rows.append(
                    {
                        "rec_name": recording,
                        "roi_id": normalize_roi(roi),
                        "t_start": start,
                        "t_end": end,
                        "source_label": annotation.get("source_label", annotation.get("label")),
                    }
                )
    if not rows:
        raise ValueError("No target actions found for the frozen recording set")
    return pd.DataFrame(rows)


def temporal_iou(segments: np.ndarray, target: np.ndarray) -> np.ndarray:
    intersection = np.maximum(
        np.minimum(segments[:, 1], target[1]) - np.maximum(segments[:, 0], target[0]), 0.0
    )
    union = (segments[:, 1] - segments[:, 0]) + (target[1] - target[0]) - intersection
    return intersection / np.maximum(union, 1e-12)


def proposals_in_seconds(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["roi_id"] = output["roi_id"].map(normalize_roi)
    output["t_start"] = output["t_start"].astype(float) / 1e6
    output["t_end"] = output["t_end"].astype(float) / 1e6
    return output


def proposal_recall(
    proposals: pd.DataFrame,
    ground_truth: pd.DataFrame,
    thresholds: Iterable[float] = TIOU_THRESHOLDS,
    budgets: Iterable[int] = PROPOSAL_BUDGETS,
) -> dict:
    thresholds = tuple(float(value) for value in thresholds)
    output = {}
    for budget in budgets:
        selected = (
            proposals.sort_values("score", ascending=False)
            .groupby(["rec_name", "roi_id"], sort=False)
            .head(int(budget))
        )
        best_overlaps = []
        groups = {
            key: group[["t_start", "t_end"]].to_numpy(dtype=np.float64)
            for key, group in selected.groupby(["rec_name", "roi_id"])
        }
        for row in ground_truth.itertuples(index=False):
            segments = groups.get((row.rec_name, int(row.roi_id)))
            best_overlaps.append(
                0.0
                if segments is None or len(segments) == 0
                else float(temporal_iou(segments, np.array([row.t_start, row.t_end])).max())
            )
        best = np.asarray(best_overlaps)
        recalls = {
            f"AR@{threshold:.1f}": float(np.mean(best >= threshold))
            for threshold in thresholds
        }
        output[str(int(budget))] = {
            **recalls,
            "mean_AR": float(np.mean(list(recalls.values()))),
        }
    return output


def prediction_frame(prediction: dict, recordings: Iterable[str]) -> pd.DataFrame:
    selected = set(recordings)
    rows = []
    for recording, rois in prediction["results"].items():
        if recording not in selected:
            continue
        for roi, detections in rois.items():
            for detection in detections:
                rows.append(
                    {
                        "rec_name": recording,
                        "roi_id": normalize_roi(roi),
                        "t_start": float(detection["segment"][0]),
                        "t_end": float(detection["segment"][1]),
                        "score": float(detection["score"]),
                    }
                )
    return pd.DataFrame(rows, columns=["rec_name", "roi_id", "t_start", "t_end", "score"])


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def detection_ap(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    thresholds: Iterable[float] = THUMOS_TIOU_THRESHOLDS,
) -> dict:
    if len(ground_truth) == 0:
        raise ValueError("AP requires at least one target action")
    ordered = detections.sort_values("score", ascending=False).reset_index(drop=True)
    gt_groups = {
        key: group[["t_start", "t_end"]].to_numpy(dtype=np.float64)
        for key, group in ground_truth.groupby(["rec_name", "roi_id"])
    }
    aps = {}
    for threshold in thresholds:
        matched = {key: np.zeros(len(value), dtype=bool) for key, value in gt_groups.items()}
        true_positive = np.zeros(len(ordered), dtype=np.float64)
        false_positive = np.zeros(len(ordered), dtype=np.float64)
        for index, row in enumerate(ordered.itertuples(index=False)):
            key = (row.rec_name, int(row.roi_id))
            targets = gt_groups.get(key)
            if targets is None or len(targets) == 0:
                false_positive[index] = 1.0
                continue
            overlaps = temporal_iou(targets, np.array([row.t_start, row.t_end]))
            best = int(np.argmax(overlaps))
            if overlaps[best] >= threshold and not matched[key][best]:
                true_positive[index] = 1.0
                matched[key][best] = True
            else:
                false_positive[index] = 1.0
        tp = np.cumsum(true_positive)
        fp = np.cumsum(false_positive)
        recall = tp / len(ground_truth)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        aps[f"AP@{float(threshold):.1f}"] = interpolated_ap(recall, precision)
    aps["mAP"] = float(np.mean(list(aps.values())))
    return aps


def proposal_oracle_detections(
    proposals: pd.DataFrame, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    groups = {
        key: group[["t_start", "t_end"]].to_numpy(dtype=np.float64)
        for key, group in ground_truth.groupby(["rec_name", "roi_id"])
    }
    output = proposals.copy()
    scores = []
    for row in output.itertuples(index=False):
        targets = groups.get((row.rec_name, int(row.roi_id)))
        scores.append(
            0.0
            if targets is None or len(targets) == 0
            else float(temporal_iou(targets, np.array([row.t_start, row.t_end])).max())
        )
    output["score"] = scores
    return output


def evaluate(args: argparse.Namespace) -> None:
    manifest = verify_manifest(resolve(args.manifest))
    recordings = list(manifest["target"]["recordings"])
    ground_truth = load_generic_ground_truth(resolve(args.ann_path), recordings)
    negative_recordings = len(set(recordings) - set(ground_truth["rec_name"].unique()))
    reportable = len(recordings) > 1 and negative_recordings > 0 and not args.exploratory
    results = {
        "protocol": PROTOCOL_VERSION,
        "semantic_scope": "class-agnostic foreground action localization",
        "recordings": len(recordings),
        "negative_recordings": negative_recordings,
        "ground_truth_instances": int(len(ground_truth)),
        "reportable_ap": reportable,
        "warning": None
        if reportable
        else "Exploratory pilot: AP is diagnostic and must not be reported as benchmark evidence.",
        "variants": {},
    }
    for name, directory in (("retag", args.retag_dir), ("eventpenguins_stage1", args.ours_dir)):
        root = resolve(directory)
        proposals = proposals_in_seconds(pd.read_csv(root / "proposals.csv"))
        prediction = json.loads((root / "predictions.json").read_text(encoding="utf-8"))
        detections = prediction_frame(prediction, recordings)
        oracle = proposal_oracle_detections(proposals, ground_truth)
        results["variants"][name] = {
            "proposal_count": int(len(proposals)),
            "detection_count": int(len(detections)),
            "proposal_recall": proposal_recall(proposals, ground_truth),
            "frozen_source_ap": detection_ap(detections, ground_truth),
            "iou_ranked_oracle_ap": detection_ap(oracle, ground_truth),
            "iou_ranked_oracle_definition": (
                "Ranks every proposal by its maximum target tIoU. This is a transparent "
                "coverage/ranking ceiling, not the literal reTAG Perfect Classifier: the "
                "paper describes thresholding by ground-truth tIoU followed by NMS, but its "
                "released code does not specify the binary-score tie order."
            ),
        }
    if args.full_prediction:
        full = json.loads(resolve(args.full_prediction).read_text(encoding="utf-8"))
        full_detections = prediction_frame(full, recordings)
        results["variants"]["eventpenguins_continuous_stack"] = {
            "detection_count": int(len(full_detections)),
            "frozen_source_ap": detection_ap(full_detections, ground_truth),
            "scope_note": (
                "Continuous ATSN + event-stat experts, complete lattice/GroupDRO/"
                "TemporalMaxer/voting local branch, completeness/QFL, and frozen "
                "0.20/0.40/0.40 fusion."
            ),
        }
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--prototype-path", default="tmp/prototype/ed_prototype.npy")
    parser.add_argument(
        "--continuous-root", default="tmp/temporalmaxer_continuous/cv_recipe_v1"
    )
    parser.add_argument(
        "--event-root", default="tmp/temporalmaxer_continuous/cv_eventstats_v1"
    )
    parser.add_argument(
        "--quality-root", default="tmp/quality_head/family_groupdro_cv"
    )
    parser.add_argument(
        "--dense-root", default="tmp/temporalmaxer_dense/screened_cv_blend075_erm"
    )
    parser.add_argument(
        "--qfl-source-features",
        default="tmp/temporalmaxer_continuous/actionness_qfl_cv_v1/candidate_features.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="Hash source artifacts before target inference")
    freeze.add_argument("--data-path", required=True)
    freeze.add_argument("--split", default="test")
    freeze.add_argument("--out", required=True)
    freeze.add_argument(
        "--exploratory-post-smoke",
        action="store_true",
        help="Record honestly that an earlier exploratory target evaluation already occurred.",
    )
    add_source_arguments(freeze)

    stage1 = commands.add_parser("stage1", help="Run frozen reTAG and full proposal stage")
    stage1.add_argument("--manifest", required=True)
    stage1.add_argument("--out-dir", required=True)
    stage1.add_argument("--device", default="cuda")
    stage1.add_argument("--batch-size", type=int, default=64)
    stage1.add_argument("--num-workers", type=int, default=8)

    local = commands.add_parser(
        "local", help="Run the complete label-free local proposal expert"
    )
    local.add_argument("--manifest", required=True)
    local.add_argument("--proposal-csv", required=True)
    local.add_argument("--out-dir", required=True)
    local.add_argument("--device", default="cuda")
    local.add_argument("--qhead-repr-batch-size", type=int, default=16)
    local.add_argument("--dense-repr-batch-size", type=int, default=32)
    local.add_argument("--dense-batch-size", type=int, default=512)
    local.add_argument("--num-workers", type=int, default=8)

    full = commands.add_parser(
        "full", help="Run label-free continuous/event/QFL fusion with frozen source artifacts"
    )
    full.add_argument("--manifest", required=True)
    full.add_argument("--feature-dir", required=True)
    full.add_argument("--proposal-prediction", required=True)
    full.add_argument("--continuous-prediction", required=True)
    full.add_argument("--event-prediction", required=True)
    full.add_argument("--out-dir", required=True)
    full.add_argument("--steps", type=int, default=500)
    full.add_argument("--learning-rate", type=float, default=0.03)
    full.add_argument("--batch-size", type=int, default=8)
    full.add_argument("--num-workers", type=int, default=2)
    full.add_argument("--device", default="cuda")

    metric = commands.add_parser(
        "evaluate", help="Read target labels only after frozen predictions exist"
    )
    metric.add_argument("--manifest", required=True)
    metric.add_argument("--ann-path", required=True)
    metric.add_argument("--retag-dir", required=True)
    metric.add_argument("--ours-dir", required=True)
    metric.add_argument("--out", required=True)
    metric.add_argument("--exploratory", action="store_true")
    metric.add_argument("--full-prediction", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        manifest = create_manifest(args)
        out_path = resolve(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"manifest": str(out_path), "target": manifest["target"]}, indent=2))
    elif args.command == "stage1":
        run_stage1(args)
    elif args.command == "local":
        run_local_expert(args)
    elif args.command == "full":
        run_full_fusion(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
