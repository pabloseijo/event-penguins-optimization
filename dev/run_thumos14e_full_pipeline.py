#!/usr/bin/env python3
"""Run the complete supervised EventPenguins arm on THUMOS14-E.

The script preserves the article topology: complete stage-one proposal expert,
continuous ATSN TemporalMaxer, event-statistics TemporalMaxer, global rank
fusion, context-relative completeness, linear QFL, and Gaussian Soft-NMS.
Every learned component uses only the 200 official validation videos through
video-disjoint folds. Test annotations are first read by the final evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.eval_multi_expert_boundary_voting_cv import (  # noqa: E402
    build_multi_expert_prediction,
)
from dev.eval_multi_expert_boundary_voting_test import (  # noqa: E402
    add_test_expert_boundaries,
)
from dev.prepare_thumos14_event_corpus import THUMOS_CLASSES, sha256_file  # noqa: E402
from dev.run_thumos14_event_generalization import (  # noqa: E402
    LOCAL_EXPERT_RECIPE,
    prefix_order,
    unique_proposals,
)
from dev.run_thumos14e_supervised import (  # noqa: E402
    PROTOCOL_VERSION,
    annotation_config_dir,
    atomic_json,
    fold_recordings,
    load_plan,
    run_logged,
)
from dev.train_temporalmaxer_dense import (  # noqa: E402
    load_cache as load_dense_cache,
    make_model as make_dense_model,
    score_model as score_dense_model,
    stable_proposal_index,
)


THUMOS_TIOU = (0.3, 0.4, 0.5, 0.6, 0.7)
QUALITY_CONFIG = {
    "epochs": 18,
    "batch_size": 4096,
    "max_train_samples": 140000,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "group_dro_eta": 0.01,
    "eval_every": 6,
}
DENSE_CONFIG = {
    "epochs": 25,
    "patience": 7,
    "batch_size": 512,
    "max_train_samples": 140000,
    "hidden_dim": 128,
    "pyramid_levels": 3,
    "dropout": 0.15,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
}
CONTINUOUS_CONFIG = {
    "epochs": 30,
    "patience": 8,
    "batch_size": 4,
    "hidden_dim": 128,
    "pyramid_levels": 6,
    "head_layers": 3,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
}
SHARED_FEATURE_RECIPE = {
    "grid_stride_s": 0.5,
    "window_duration_s": 1.0,
    "feature_dim": 512,
    "event_spectral_bins": 8,
}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def class_root(plan: dict[str, object], label: str, seed: int) -> Path:
    return (
        Path(plan["paths"]["out_root"])
        / "eventpenguins_full"
        / f"seed_{seed}"
        / label
    )


def class_annotations(plan: dict[str, object], label: str, trainable: bool) -> Path:
    work_dir = Path(plan["paths"]["work_dir"])
    filename = "annotations_trainable.json" if trainable else "annotations.json"
    return annotation_config_dir(work_dir) / "by_class" / label / filename


def run_if_missing(command: list[str], expected: Path, log_path: Path) -> None:
    if expected.exists():
        return
    run_logged(command, log_path)
    if not expected.exists():
        raise RuntimeError(f"Command completed without expected artifact {expected}")


def shared_feature_dirs(plan: dict[str, object]) -> tuple[Path, Path]:
    root = Path(plan["paths"]["out_root"]) / "shared_features"
    return root / "continuous", root / "event_stats"


def expected_corpus_recordings(plan: dict[str, object]) -> set[str]:
    manifest = pd.read_csv(
        plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False
    )
    recordings = set(manifest["video_id"].astype(str))
    if len(recordings) != 413:
        raise ValueError(f"Shared feature cache requires 413 recordings; got {len(recordings)}")
    return recordings


def validate_shared_feature_cache(
    plan: dict[str, object], *, num_shards: int | None = None
) -> dict[str, object]:
    feature_dir, event_dir = shared_feature_dirs(plan)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(feature_dir / "sequences.csv", keep_default_na=False)
    expected = expected_corpus_recordings(plan)
    actual = set(sequences["rec_name"].astype(str))
    if actual != expected or len(sequences) != 413:
        raise ValueError(
            "Shared feature cache does not contain exactly one ROI for each official video"
        )
    if set(metadata.get("recordings", [])) != expected:
        raise ValueError("Shared feature metadata recording universe differs from THUMOS14-E")
    if (
        float(metadata.get("grid_stride_s", -1))
        != SHARED_FEATURE_RECIPE["grid_stride_s"]
        or float(metadata.get("window_duration_s", -1))
        != SHARED_FEATURE_RECIPE["window_duration_s"]
        or int(metadata.get("feature_dim", -1))
        != SHARED_FEATURE_RECIPE["feature_dim"]
    ):
        raise ValueError(
            "Shared ATSN feature recipe differs from the frozen article recipe"
        )
    if Path(metadata["data_path"]).resolve() != Path(
        plan["inputs"]["event_hdf5"]["path"]
    ).resolve():
        raise ValueError("Shared features were indexed from a different event corpus")

    statuses = []
    if num_shards is not None:
        metadata_sha256 = sha256_file(feature_dir / "metadata.json")
        for shard in range(num_shards):
            path = feature_dir / f"shard_{shard:02d}_of_{num_shards:02d}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            status = json.loads(path.read_text(encoding="utf-8"))
            if (
                status.get("complete") is not True
                or int(status.get("shard_index", -1)) != shard
                or int(status.get("num_shards", -1)) != num_shards
                or status.get("model_sha256") != plan["inputs"]["source_atsn"]["sha256"]
                or status.get("data_sha256") != plan["inputs"]["event_hdf5"]["sha256"]
                or status.get("index_metadata_sha256") != metadata_sha256
            ):
                raise ValueError(f"Invalid or foreign ATSN extraction shard {shard}")
            statuses.append(status)
        if sum(int(status["num_sequences"]) for status in statuses) != 413:
            raise ValueError("Completed extraction shards do not cover the 413 videos")
        if sum(int(status["num_points"]) for status in statuses) != int(
            metadata["num_points"]
        ):
            raise ValueError("Completed extraction shards do not cover the feature matrix")

    event_metadata = json.loads((event_dir / "metadata.json").read_text(encoding="utf-8"))
    event_stats = np.load(event_dir / "event_stats.npy", mmap_mode="r")
    if (
        event_stats.shape[0] != int(metadata["num_points"])
        or int(event_metadata.get("num_points", -1)) != int(metadata["num_points"])
        or float(event_metadata.get("grid_stride_s", -1))
        != SHARED_FEATURE_RECIPE["grid_stride_s"]
        or int(event_metadata.get("spectral_bins", -1))
        != SHARED_FEATURE_RECIPE["event_spectral_bins"]
        or event_metadata.get("data_sha256")
        != plan["inputs"]["event_hdf5"]["sha256"]
        or event_metadata.get("feature_metadata_sha256")
        != sha256_file(feature_dir / "metadata.json")
        or event_metadata.get("sequence_index_sha256")
        != sha256_file(feature_dir / "sequences.csv")
    ):
        raise ValueError("Event-statistics cache is not aligned with the ATSN feature grid")
    return {
        "recordings": len(actual),
        "points": int(metadata["num_points"]),
        "num_shards": num_shards,
        "atsn_features_sha256": sha256_file(feature_dir / "frame_features.npy"),
        "event_stats_sha256": sha256_file(event_dir / "event_stats.npy"),
    }


def run_shared_index(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    feature_dir, _ = shared_feature_dirs(plan)
    command = [
        sys.executable,
        str(ROOT / "dev" / "extract_continuous_features.py"),
        "index",
        "--data-path", str(plan["inputs"]["event_hdf5"]["path"]),
        "--out-dir", str(feature_dir),
        "--splits", "train", "val", "test",
        "--grid-stride", str(SHARED_FEATURE_RECIPE["grid_stride_s"]),
        "--window-duration", str(SHARED_FEATURE_RECIPE["window_duration_s"]),
        "--feature-dim", str(SHARED_FEATURE_RECIPE["feature_dim"]),
    ]
    if args.force:
        command.append("--force")
    run_if_missing(command, feature_dir / "metadata.json", feature_dir / "index.log")
    expected = expected_corpus_recordings(plan)
    sequences = pd.read_csv(feature_dir / "sequences.csv", keep_default_na=False)
    if set(sequences["rec_name"].astype(str)) != expected or len(sequences) != 413:
        raise ValueError("Indexed feature universe differs from the 413 official videos")


def run_shared_extract(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    feature_dir, _ = shared_feature_dirs(plan)
    command = [
        sys.executable,
        str(ROOT / "dev" / "extract_continuous_features.py"),
        "extract",
        "--data-path", str(plan["inputs"]["event_hdf5"]["path"]),
        "--model-path", str(plan["inputs"]["source_atsn"]["path"]),
        "--out-dir", str(feature_dir),
        "--shard-index", str(args.shard_index),
        "--num-shards", str(args.num_shards),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]
    if args.overwrite_shard:
        command.append("--overwrite-shard")
    run_logged(
        command,
        feature_dir / f"extract_{args.shard_index:02d}_of_{args.num_shards:02d}.log",
    )
    status_path = feature_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("model_sha256") != plan["inputs"]["source_atsn"]["sha256"]
        or status.get("data_sha256") != plan["inputs"]["event_hdf5"]["sha256"]
        or status.get("index_metadata_sha256")
        != sha256_file(feature_dir / "metadata.json")
    ):
        raise ValueError("ATSN extraction shard provenance does not match the frozen plan")


def run_shared_event_stats(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    feature_dir, event_dir = shared_feature_dirs(plan)
    command = [
        sys.executable,
        str(ROOT / "dev" / "extract_continuous_event_stats.py"),
        "--feature-dir", str(feature_dir),
        "--data-path", str(plan["inputs"]["event_hdf5"]["path"]),
        "--out-dir", str(event_dir),
        "--spectral-bins", str(SHARED_FEATURE_RECIPE["event_spectral_bins"]),
    ]
    if args.force:
        command.append("--force")
    run_if_missing(command, event_dir / "metadata.json", event_dir / "extract.log")


def run_shared_verify(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    feature_dir, _ = shared_feature_dirs(plan)
    run_logged(
        [
            sys.executable,
            str(ROOT / "dev" / "extract_continuous_features.py"),
            "verify",
            "--out-dir", str(feature_dir),
        ],
        feature_dir / "verify.log",
    )
    report = validate_shared_feature_cache(plan, num_shards=args.num_shards)
    report.update(
        {
            "protocol": PROTOCOL_VERSION,
            "source_encoder_frozen": True,
            "source_recordings_mixed_into_target": False,
            "recipe": SHARED_FEATURE_RECIPE,
        }
    )
    atomic_json(feature_dir.parent / "report.json", report)


def split_proposals(
    proposals: pd.DataFrame,
    train_recordings: set[str],
    val_recordings: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = proposals["rec_name"].astype(str)
    unexpected = set(names) - train_recordings - val_recordings
    if unexpected:
        raise ValueError(f"Proposal rows escape the validation pool: {sorted(unexpected)}")
    train = proposals.loc[names.isin(train_recordings)].reset_index(drop=True)
    validation = proposals.loc[names.isin(val_recordings)].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("A local fold has an empty train or validation proposal set")
    return train, validation


def aligned_quality_frames(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if not frames:
        raise ValueError("At least one quality frame is required")
    expected = stable_proposal_index(frames[0])
    for index, frame in enumerate(frames[1:], start=1):
        if len(frame) != len(frames[0]) or not stable_proposal_index(frame).equals(expected):
            raise ValueError(f"Quality frame {index} is not aligned")
    return [frame.reset_index(drop=True) for frame in frames]


def screen_lattice(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = aligned_quality_frames(frames)
    mean_quality = np.mean(
        [frame["quality_score"].to_numpy(dtype=np.float64) for frame in frames],
        axis=0,
    )
    base = frames[0]
    columns = ["rec_name", "roi_id", "t_start", "t_end", "score"]
    prefix = unique_proposals(
        base.loc[
            base["cnn_score"] >= LOCAL_EXPERT_RECIPE["prefix_screen"]["cnn_threshold"],
            columns,
        ]
    )
    keep = (
        base["cnn_score"].to_numpy(dtype=np.float64)
        >= LOCAL_EXPERT_RECIPE["hybrid_screen"]["cnn_threshold"]
    ) | (
        mean_quality >= LOCAL_EXPERT_RECIPE["hybrid_screen"]["quality_threshold"]
    )
    return prefix, prefix_order(base.loc[keep, columns], prefix)


def local_prediction_from_scores(
    hybrid: pd.DataFrame,
    quality_frames: list[pd.DataFrame],
    dense_frames: list[pd.DataFrame],
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    quality_frames = aligned_quality_frames(quality_frames)
    quality_index = stable_proposal_index(quality_frames[0])
    positions = quality_index.get_indexer(stable_proposal_index(hybrid))
    if np.any(positions < 0):
        raise ValueError("Hybrid proposals cannot be aligned with quality scores")
    expected = stable_proposal_index(hybrid)
    for index, frame in enumerate(dense_frames):
        if len(frame) != len(hybrid) or not stable_proposal_index(frame).equals(expected):
            raise ValueError(f"Dense frame {index} is not aligned with hybrid proposals")

    scored = hybrid.reset_index(drop=True).copy()
    scored["quality_score"] = np.mean(
        [
            frame.iloc[positions]["quality_score"].to_numpy(dtype=np.float64)
            for frame in quality_frames
        ],
        axis=0,
    )
    mean_start = np.mean(
        [frame["delta_t_start"].to_numpy(dtype=np.float64) for frame in dense_frames],
        axis=0,
    )
    mean_end = np.mean(
        [frame["delta_t_end"].to_numpy(dtype=np.float64) for frame in dense_frames],
        axis=0,
    )
    scored["reference_blend050_t_start"] = 0.5 * scored["t_start"] + 0.5 * mean_start
    scored["reference_blend050_t_end"] = 0.5 * scored["t_end"] + 0.5 * mean_end
    scored = add_test_expert_boundaries(scored, dense_frames)
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
        min_action_duration=0.0,
    )
    prediction = build_multi_expert_prediction(scored, prediction_args, 0.5, "median")
    prediction["target_class"] = label
    prediction["minimum_action_duration_s"] = 0.0
    return scored, prediction


def build_lattice(data_path: Path, proposal_path: Path, output: Path, log: Path) -> None:
    recipe = LOCAL_EXPERT_RECIPE["lattice"]
    command = [
        sys.executable,
        str(ROOT / "dev" / "build_proposal_lattice.py"),
        "--proposals",
        str(proposal_path),
        "--out-proposals",
        str(output),
        "--data-path",
        str(data_path),
        "--min-duration-s",
        "0.0",
        "--max-duration-s",
        str(recipe["max_duration_s"]),
        "--top-k-per-roi",
        str(recipe["top_k_per_roi"]),
        "--score-quantile",
        str(recipe["score_quantile"]),
        "--max-per-roi",
        str(recipe["max_per_roi"]),
        "--nms-threshold",
        str(recipe["nms_threshold"]),
    ]
    run_if_missing(command, output, log)


def quality_train_command(
    data_path: Path,
    annotation_path: Path,
    source_model: Path,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "dev" / "train_quality_head.py"),
        "--data-path", str(data_path),
        "--ann-path", str(annotation_path),
        "--model-path", str(source_model),
        "--train-proposals", str(train_path),
        "--val-proposals", str(val_path),
        "--out-dir", str(out_dir),
        "--configs", "qhead_qfl_only",
        "--epochs", str(QUALITY_CONFIG["epochs"]),
        "--batch-size", str(QUALITY_CONFIG["batch_size"]),
        "--repr-batch-size", str(args.qhead_repr_batch_size),
        "--num-workers", str(args.num_workers),
        "--max-train-samples", str(QUALITY_CONFIG["max_train_samples"]),
        "--group-dro",
        "--group-dro-eta", str(QUALITY_CONFIG["group_dro_eta"]),
        "--lr", str(QUALITY_CONFIG["learning_rate"]),
        "--weight-decay", str(QUALITY_CONFIG["weight_decay"]),
        "--eval-every", str(QUALITY_CONFIG["eval_every"]),
        "--min-gt-duration", "0.0",
        "--min-score", "0.1",
        "--pre-nms-topk-per-roi", "0",
        "--tiou", *(str(value) for value in THUMOS_TIOU),
        "--seed", str(args.seed),
        "--device", args.device,
        "--quiet-progress",
    ]


def quality_score_command(
    data_path: Path,
    annotation_path: Path,
    source_model: Path,
    proposals: Path,
    representation: Path,
    checkpoint: Path,
    out_dir: Path,
    label: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "dev" / "train_quality_head.py"),
        "--data-path", str(data_path),
        "--ann-path", str(annotation_path),
        "--model-path", str(source_model),
        "--val-proposals", str(proposals),
        "--val-repr", str(representation),
        "--out-dir", str(out_dir),
        "--eval-checkpoint", str(checkpoint),
        "--eval-label", label,
        "--skip-evaluation",
        "--repr-batch-size", str(args.qhead_repr_batch_size),
        "--num-workers", str(args.num_workers),
        "--min-gt-duration", "0.0",
        "--device", args.device,
        "--quiet-progress",
    ]


def dense_train_command(
    data_path: Path,
    annotation_path: Path,
    source_model: Path,
    master: Path,
    train_path: Path,
    val_path: Path,
    cache_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "dev" / "train_temporalmaxer_dense.py"),
        "--data-path", str(data_path),
        "--ann-path", str(annotation_path),
        "--model-path", str(source_model),
        "--master-proposals", str(master),
        "--train-proposals", str(train_path),
        "--val-proposals", str(val_path),
        "--cache-dir", str(cache_dir),
        "--timestamp-cache-dir", str(cache_dir.parent / "roi_timestamps"),
        "--out-dir", str(out_dir),
        "--epochs", str(DENSE_CONFIG["epochs"]),
        "--patience", str(DENSE_CONFIG["patience"]),
        "--batch-size", str(DENSE_CONFIG["batch_size"]),
        "--max-train-samples", str(DENSE_CONFIG["max_train_samples"]),
        "--hidden-dim", str(DENSE_CONFIG["hidden_dim"]),
        "--pyramid-levels", str(DENSE_CONFIG["pyramid_levels"]),
        "--dropout", str(DENSE_CONFIG["dropout"]),
        "--lr", str(DENSE_CONFIG["learning_rate"]),
        "--weight-decay", str(DENSE_CONFIG["weight_decay"]),
        "--repr-batch-size", str(args.dense_repr_batch_size),
        "--num-workers", str(args.num_workers),
        "--boundary-blend", "0.75",
        "--min-action-duration", "0.0",
        "--tiou", *(str(value) for value in THUMOS_TIOU),
        "--seed", str(args.seed),
        "--device", args.device,
        "--quiet-progress",
    ]


def run_local_fold(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    label = args.target_class
    root = class_root(plan, label, args.seed)
    fold_dir = root / "local" / f"fold_{args.fold:02d}"
    prediction_path = fold_dir / "predictions" / "multi_median_blend050.json"
    if prediction_path.exists():
        print(f"[SKIP] {prediction_path}")
        return
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    source_model = Path(plan["inputs"]["source_atsn"]["path"])
    annotations = class_annotations(plan, label, trainable=True)
    folds = pd.read_csv(plan["inputs"]["fold_manifest"]["path"], keep_default_na=False)
    stage1 = (
        Path(plan["paths"]["out_root"])
        / "proposals" / "eventpenguins_stage1" / label
        / f"fold_{args.fold:02d}" / "validation" / "proposals.csv"
    )
    lattice_path = fold_dir / "lattice.csv"
    build_lattice(data_path, stage1, lattice_path, fold_dir / "lattice.log")
    lattice = pd.read_csv(lattice_path)
    train, validation = split_proposals(
        lattice,
        fold_recordings(folds, args.fold, "train"),
        fold_recordings(folds, args.fold, "val"),
    )
    train_path = fold_dir / "lattice_train.csv"
    val_path = fold_dir / "lattice_val.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(val_path, index=False)

    qhead_dir = fold_dir / "quality_head"
    qhead_checkpoint = qhead_dir / "qhead_qfl_only.pt"
    run_if_missing(
        quality_train_command(
            data_path, annotations, source_model, train_path, val_path, qhead_dir, args
        ),
        qhead_checkpoint,
        qhead_dir / "run.log",
    )
    qhead_train_eval = fold_dir / "quality_train_scores"
    train_scores_path = qhead_train_eval / "cache" / "train_scores_qhead_qfl_only.csv"
    run_if_missing(
        quality_score_command(
            data_path,
            annotations,
            source_model,
            train_path,
            qhead_dir / "cache" / "train_repr.npz",
            qhead_checkpoint,
            qhead_train_eval,
            "train",
            args,
        ),
        train_scores_path,
        qhead_train_eval / "run.log",
    )
    val_scores_path = qhead_dir / "cache" / "val_scores_qhead_qfl_only.csv"
    if not val_scores_path.exists():
        raise FileNotFoundError(val_scores_path)
    _, hybrid_train = screen_lattice([pd.read_csv(train_scores_path)])
    _, hybrid_val = screen_lattice([pd.read_csv(val_scores_path)])
    hybrid_train_path = fold_dir / "hybrid_train.csv"
    hybrid_val_path = fold_dir / "hybrid_val.csv"
    hybrid_train.to_csv(hybrid_train_path, index=False)
    hybrid_val.to_csv(hybrid_val_path, index=False)
    master_path = fold_dir / "hybrid_master.csv"
    pd.concat((hybrid_train, hybrid_val), ignore_index=True).to_csv(master_path, index=False)

    dense_dir = fold_dir / "dense"
    dense_checkpoint = dense_dir / "best.pt"
    run_if_missing(
        dense_train_command(
            data_path,
            annotations,
            source_model,
            master_path,
            hybrid_train_path,
            hybrid_val_path,
            fold_dir / "dense_cache",
            dense_dir,
            args,
        ),
        dense_checkpoint,
        dense_dir / "run.log",
    )
    dense_scores = dense_dir / "scored_best.csv"
    scored, prediction = local_prediction_from_scores(
        hybrid_val,
        [pd.read_csv(val_scores_path)],
        [pd.read_csv(dense_scores)],
        label,
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(fold_dir / "scored.csv", index=False)
    atomic_json(prediction_path, prediction)
    atomic_json(
        fold_dir / "report.json",
        {
            "protocol": PROTOCOL_VERSION,
            "target_class": label,
            "fold": args.fold,
            "seed": args.seed,
            "train_recordings": len(fold_recordings(folds, args.fold, "train")),
            "val_recordings": len(fold_recordings(folds, args.fold, "val")),
            "lattice_proposals": len(lattice),
            "hybrid_train_proposals": len(hybrid_train),
            "hybrid_val_proposals": len(hybrid_val),
            "prediction_sha256": sha256_file(prediction_path),
        },
    )


def write_empty_test_metadata(
    plan: dict[str, object], label: str, output_dir: Path
) -> Path:
    source_dir = (
        annotation_config_dir(Path(plan["paths"]["work_dir"]))
        / "by_class"
        / label
    )
    info = pd.read_csv(source_dir / "recording_info.csv", keep_default_na=False)
    test_info = info.loc[info["official_subset"].astype(str).str.lower() == "test"].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_info.to_csv(output_dir / "recording_info.csv", index=False)
    annotations = {
        "version": "THUMOS14-E-empty-test-inference-v1",
        "database": {
            str(recording): {"annotations": {"1": []}}
            for recording in test_info["timestamp"].astype(str)
        },
    }
    path = output_dir / "annotations.json"
    atomic_json(path, annotations)
    return path


def score_dense_checkpoints(
    hybrid: pd.DataFrame,
    cache_dir: Path,
    checkpoints: list[Path],
    device_name: str,
    batch_size: int,
) -> list[pd.DataFrame]:
    _, logits, metadata = load_dense_cache(cache_dir)
    indices = np.arange(len(hybrid), dtype=np.int64)
    device = torch.device(device_name)
    frames = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved = checkpoint["args"]
        dense_args = SimpleNamespace(
            hidden_dim=int(saved.get("hidden_dim", 128)),
            pyramid_levels=int(saved.get("pyramid_levels", 3)),
            dropout=float(saved.get("dropout", 0.15)),
            trident_bins=int(saved.get("trident_bins") or 0),
            tanp_sigma=float(saved.get("tanp_sigma", 0.0)),
            event_feature_cache_dir=None,
            event_features_only=False,
            batch_size=batch_size,
            augment_factor=int(saved.get("augment_factor", 5)),
            max_boundary_delta=float(saved.get("max_boundary_delta", 0.75)),
            boundary_blend=float(saved.get("boundary_blend", 0.75)),
            min_action_duration=0.0,
        )
        model = make_dense_model(metadata, dense_args).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        frames.append(
            score_dense_model(
                model,
                hybrid,
                indices,
                cache_dir / "frame_features.npy",
                logits,
                dense_args,
                device,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return frames


def run_local_test(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    label = args.target_class
    root = class_root(plan, label, args.seed)
    out_dir = root / "local_test"
    prediction_path = out_dir / "predictions.json"
    if prediction_path.exists():
        print(f"[SKIP] {prediction_path}")
        return
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    source_model = Path(plan["inputs"]["source_atsn"]["path"])
    stage1 = (
        Path(plan["paths"]["out_root"])
        / "proposals" / "eventpenguins_stage1" / label / "test" / "proposals.csv"
    )
    lattice_path = out_dir / "lattice.csv"
    build_lattice(data_path, stage1, lattice_path, out_dir / "lattice.log")
    empty_annotations = write_empty_test_metadata(plan, label, out_dir / "empty_metadata")
    representation = out_dir / "quality_repr.npz"
    quality_frames = []
    checkpoints = []
    for fold in range(5):
        fold_root = root / "local" / f"fold_{fold:02d}"
        qcheckpoint = fold_root / "quality_head" / "qhead_qfl_only.pt"
        dcheckpoint = fold_root / "dense" / "best.pt"
        checkpoints.append(dcheckpoint)
        score_dir = out_dir / "quality" / f"fold_{fold:02d}"
        score_path = score_dir / "cache" / f"fold_{fold:02d}_scores_qhead_qfl_only.csv"
        run_if_missing(
            quality_score_command(
                data_path,
                empty_annotations,
                source_model,
                lattice_path,
                representation,
                qcheckpoint,
                score_dir,
                f"fold_{fold:02d}",
                args,
            ),
            score_path,
            score_dir / "run.log",
        )
        quality_frames.append(pd.read_csv(score_path))
    prefix, hybrid = screen_lattice(quality_frames)
    prefix.to_csv(out_dir / "screened_prefix.csv", index=False)
    hybrid_path = out_dir / "hybrid.csv"
    hybrid.to_csv(hybrid_path, index=False)

    dense_cache = out_dir / "dense_cache"
    extract_command = [
        sys.executable,
        str(ROOT / "dev" / "train_temporalmaxer_dense.py"),
        "--data-path", str(data_path),
        "--ann-path", str(empty_annotations),
        "--model-path", str(source_model),
        "--master-proposals", str(hybrid_path),
        "--cache-dir", str(dense_cache),
        "--timestamp-cache-dir", str(out_dir / "dense_roi_timestamps"),
        "--out-dir", str(out_dir / "dense_extract"),
        "--extract-only",
        "--repr-batch-size", str(args.dense_repr_batch_size),
        "--num-workers", str(args.num_workers),
        "--min-action-duration", "0.0",
        "--device", args.device,
        "--quiet-progress",
    ]
    run_if_missing(
        extract_command,
        dense_cache / "frame_features.npy",
        out_dir / "dense_extract" / "run.log",
    )
    dense_frames = score_dense_checkpoints(
        hybrid, dense_cache, checkpoints, args.device, args.dense_batch_size
    )
    for fold, frame in enumerate(dense_frames):
        frame.to_csv(out_dir / f"dense_scored_fold_{fold:02d}.csv", index=False)
    scored, prediction = local_prediction_from_scores(
        hybrid, quality_frames, dense_frames, label
    )
    scored.to_csv(out_dir / "scored.csv", index=False)
    atomic_json(prediction_path, prediction)
    atomic_json(
        out_dir / "report.json",
        {
            "protocol": PROTOCOL_VERSION,
            "target_class": label,
            "seed": args.seed,
            "test_annotations_accessed_for_inference": False,
            "empty_annotations_sha256": sha256_file(empty_annotations),
            "lattice_proposals": len(pd.read_csv(lattice_path)),
            "hybrid_proposals": len(hybrid),
            "prediction_sha256": sha256_file(prediction_path),
        },
    )


def continuous_command(
    feature_dir: Path,
    event_dir: Path,
    annotations: Path,
    fold_manifest: Path,
    out_dir: Path,
    fold: int,
    seed: int,
    device: str,
    num_workers: int,
    event_branch: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "dev" / "train_temporalmaxer_continuous.py"),
        "--feature-dir", str(feature_dir),
        "--ann-path", str(annotations),
        "--fold-manifest", str(fold_manifest),
        "--fold", str(fold),
        "--out-dir", str(out_dir),
        "--epochs", str(CONTINUOUS_CONFIG["epochs"]),
        "--patience", str(CONTINUOUS_CONFIG["patience"]),
        "--batch-size", str(CONTINUOUS_CONFIG["batch_size"]),
        "--num-workers", str(num_workers),
        "--hidden-dim", str(CONTINUOUS_CONFIG["hidden_dim"]),
        "--pyramid-levels", str(CONTINUOUS_CONFIG["pyramid_levels"]),
        "--head-layers", str(CONTINUOUS_CONFIG["head_layers"]),
        "--dropout", str(CONTINUOUS_CONFIG["dropout"]),
        "--lr", str(CONTINUOUS_CONFIG["learning_rate"]),
        "--weight-decay", str(CONTINUOUS_CONFIG["weight_decay"]),
        "--quality-weight", "0.5",
        "--quality-power", "1.0",
        "--score-threshold", "0.005",
        "--pre-nms-topk", "200",
        "--soft-nms-sigma", "0.25",
        "--soft-nms-score-threshold", "0.001",
        "--max-predictions-per-roi", "200",
        "--min-action-duration", "0.0",
        "--tiou", *(str(value) for value in THUMOS_TIOU),
        "--seed", str(seed),
        "--device", device,
        "--quiet-progress",
    ]
    if event_branch:
        command.extend(("--auxiliary-feature-dir", str(event_dir)))
    return command


def run_continuous_fold(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    label = args.target_class
    root = class_root(plan, label, args.seed)
    shared = Path(plan["paths"]["out_root"]) / "shared_features"
    feature_dir = shared / "continuous"
    event_dir = shared / "event_stats"
    annotations = class_annotations(plan, label, trainable=True)
    fold_manifest = (
        annotation_config_dir(Path(plan["paths"]["work_dir"]))
        / "by_class"
        / label
        / "fold_manifest.csv"
    )
    for branch, auxiliary in (("continuous", False), ("event", True)):
        out_dir = root / branch / f"fold_{args.fold:02d}"
        run_if_missing(
            continuous_command(
                feature_dir,
                event_dir,
                annotations,
                fold_manifest,
                out_dir,
                args.fold,
                args.seed,
                args.device,
                args.num_workers,
                auxiliary,
            ),
            out_dir / "best.pt",
            out_dir / "run.log",
        )


def run_qfl_cv(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    label = args.target_class
    root = class_root(plan, label, args.seed)
    shared = Path(plan["paths"]["out_root"]) / "shared_features"
    annotations = class_annotations(plan, label, trainable=True)
    manifest = (
        annotation_config_dir(Path(plan["paths"]["work_dir"]))
        / "by_class"
        / label
        / "fold_manifest.csv"
    )
    out_dir = root / "qfl_cv"
    command = [
        sys.executable,
        str(ROOT / "dev" / "eval_actionness_quality_head_cv.py"),
        "--feature-dir", str(shared / "continuous"),
        "--continuous-root", str(root / "continuous"),
        "--event-root", str(root / "event"),
        "--proposal-root", str(root / "local"),
        "--proposal-variant", "multi_median_blend050",
        "--manifest", str(manifest),
        "--out-dir", str(out_dir),
        "--blends", "0.5",
        "--steps", "500",
        "--learning-rate", "0.03",
        "--batch-size", "8",
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--min-action-duration", "0.0",
        "--tiou", *(str(value) for value in THUMOS_TIOU),
    ]
    run_if_missing(
        command,
        out_dir / "candidate_features.csv",
        out_dir / "run.log",
    )


def run_full_test(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    label = args.target_class
    root = class_root(plan, label, args.seed)
    shared = Path(plan["paths"]["out_root"]) / "shared_features"
    ensemble_dir = root / "test_ensembles"
    ensemble_command = [
        sys.executable,
        str(ROOT / "dev" / "eval_continuous_multi_rep_fusion_test.py"),
        "--feature-dir", str(shared / "continuous"),
        "--auxiliary-feature-dir", str(shared / "event_stats"),
        "--continuous-root", str(root / "continuous"),
        "--event-root", str(root / "event"),
        "--proposal-prediction", str(root / "local_test" / "predictions.json"),
        "--out-dir", str(ensemble_dir),
        "--target-class", label,
        "--recording-manifest", str(plan["inputs"]["corpus_manifest"]["path"]),
        "--recording-subset", "test",
        "--min-action-duration", "0.0",
        "--tiou", *(str(value) for value in THUMOS_TIOU),
        "--batch-size", "8",
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--ensemble-only",
    ]
    run_if_missing(
        ensemble_command,
        ensemble_dir / "event_ensemble.json",
        ensemble_dir / "run.log",
    )
    out_dir = root / "test"
    qfl_command = [
        sys.executable,
        str(ROOT / "dev" / "eval_actionness_quality_head_test.py"),
        "--feature-dir", str(shared / "continuous"),
        "--continuous-root", str(root / "continuous"),
        "--source-features", str(root / "qfl_cv" / "candidate_features.csv"),
        "--continuous-prediction", str(ensemble_dir / "continuous_ensemble.json"),
        "--event-prediction", str(ensemble_dir / "event_ensemble.json"),
        "--proposal-prediction", str(root / "local_test" / "predictions.json"),
        "--out-dir", str(out_dir),
        "--target-class", label,
        "--recording-manifest", str(plan["inputs"]["corpus_manifest"]["path"]),
        "--recording-subset", "test",
        "--prediction-only",
        "--training-scope", "THUMOS14-E validation OOF only",
        "--score-blend", "0.5",
        "--steps", "500",
        "--learning-rate", "0.03",
        "--batch-size", "8",
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--min-action-duration", "0.0",
        "--tiou", *(str(value) for value in THUMOS_TIOU),
        "--weights", "0.2", "0.4", "0.4",
    ]
    run_if_missing(qfl_command, out_dir / "predictions.json", out_dir / "run.log")
    prediction = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    if prediction.get("target_class") != label:
        raise ValueError("Final prediction does not declare its THUMOS target class")
    atomic_json(
        root / "report.json",
        {
            "protocol": PROTOCOL_VERSION,
            "target_class": label,
            "seed": args.seed,
            "method": "eventpenguins_full",
            "test_annotations_accessed_for_inference": False,
            "components": [
                "complete stage-one lattice/quality/GroupDRO/boundary-voting expert",
                "continuous ATSN TemporalMaxer expert",
                "event-statistics TemporalMaxer expert",
                "global percentile-rank fusion",
                "context-relative completeness and linear QFL",
                "Gaussian Soft-NMS sigma=0.5",
            ],
            "prediction_sha256": sha256_file(out_dir / "predictions.json"),
        },
    )


def run_evaluate_all(args: argparse.Namespace) -> None:
    plan = load_plan(resolve(args.plan))
    out_root = Path(plan["paths"]["out_root"])
    prediction_root = out_root / "eventpenguins_full" / f"seed_{args.seed}"
    manifest = pd.read_csv(
        plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False
    )
    expected_test = set(
        manifest.loc[
            manifest["official_subset"].astype(str).str.lower() == "test",
            "video_id",
        ].astype(str)
    )
    if len(expected_test) != 213:
        raise ValueError(f"Expected 213 test files; got {len(expected_test)}")
    frozen_predictions = {}
    for label in THUMOS_CLASSES:
        path = prediction_root / label / "test" / "predictions.json"
        if not path.exists():
            raise FileNotFoundError(path)
        prediction = json.loads(path.read_text(encoding="utf-8"))
        if prediction.get("target_class") != label:
            raise ValueError(f"{path} does not declare target_class={label}")
        if set(prediction.get("results", {})) != expected_test:
            raise ValueError(f"{path} does not cover exactly the 213 test files")
        frozen_predictions[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    evaluation_dir = out_root / "evaluation" / "eventpenguins_full" / f"seed_{args.seed}"
    atomic_json(
        evaluation_dir / "frozen_predictions.json",
        {
            "protocol": PROTOCOL_VERSION,
            "seed": args.seed,
            "classes": len(frozen_predictions),
            "test_files": len(expected_test),
            "test_annotations_accessed_before_freeze": False,
            "predictions": frozen_predictions,
        },
    )
    command = [
        sys.executable,
        str(ROOT / "dev" / "evaluate_thumos14e_ovr.py"),
        "--actionformer-root", str(resolve(args.actionformer_root)),
        "--canonical-annotations",
        str(plan["inputs"]["canonical_annotations"]["path"]),
        "--predictions-root", str(prediction_root),
        "--prediction-name", "test/predictions.json",
        "--out-dir", str(evaluation_dir),
        "--num-workers", str(args.num_workers),
    ]
    run_logged(command, evaluation_dir / "evaluate.log")
    if not (evaluation_dir / "summary.json").exists():
        raise RuntimeError("Canonical evaluation did not produce summary.json")


def add_common(parser: argparse.ArgumentParser, *, fold: bool = False) -> None:
    parser.add_argument("--plan", required=True)
    parser.add_argument("--target-class", choices=THUMOS_CLASSES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    if fold:
        parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--qhead-repr-batch-size", type=int, default=16)
    parser.add_argument("--dense-repr-batch-size", type=int, default=32)
    parser.add_argument("--dense-batch-size", type=int, default=512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shared_index = commands.add_parser("shared-index")
    shared_index.add_argument("--plan", required=True)
    shared_index.add_argument("--force", action="store_true")
    shared_extract = commands.add_parser("shared-extract")
    shared_extract.add_argument("--plan", required=True)
    shared_extract.add_argument("--shard-index", type=int, required=True)
    shared_extract.add_argument("--num-shards", type=int, required=True)
    shared_extract.add_argument("--batch-size", type=int, default=256)
    shared_extract.add_argument("--num-workers", type=int, default=8)
    shared_extract.add_argument("--device", default="cuda")
    shared_extract.add_argument("--overwrite-shard", action="store_true")
    shared_stats = commands.add_parser("shared-event-stats")
    shared_stats.add_argument("--plan", required=True)
    shared_stats.add_argument("--force", action="store_true")
    shared_verify = commands.add_parser("shared-verify")
    shared_verify.add_argument("--plan", required=True)
    shared_verify.add_argument("--num-shards", type=int, required=True)
    local_fold = commands.add_parser("local-fold")
    add_common(local_fold, fold=True)
    local_test = commands.add_parser("local-test")
    add_common(local_test)
    continuous = commands.add_parser("continuous-fold")
    add_common(continuous, fold=True)
    qfl = commands.add_parser("qfl-cv")
    add_common(qfl)
    full = commands.add_parser("full-test")
    add_common(full)
    evaluate_all = commands.add_parser("evaluate-all")
    evaluate_all.add_argument("--plan", required=True)
    evaluate_all.add_argument("--seed", type=int, required=True)
    evaluate_all.add_argument("--actionformer-root", required=True)
    evaluate_all.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {
        "shared-index": run_shared_index,
        "shared-extract": run_shared_extract,
        "shared-event-stats": run_shared_event_stats,
        "shared-verify": run_shared_verify,
        "local-fold": run_local_fold,
        "local-test": run_local_test,
        "continuous-fold": run_continuous_fold,
        "qfl-cv": run_qfl_cv,
        "full-test": run_full_test,
        "evaluate-all": run_evaluate_all,
    }[args.command](args)


if __name__ == "__main__":
    main()
