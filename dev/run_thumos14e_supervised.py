#!/usr/bin/env python3
"""Orchestrate the literature-audited supervised THUMOS14-E comparison.

The script never treats the 20 classes as separate datasets. It uses one
converted corpus, one official validation/test split, and writes independent
per-class artifacts before any macro aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.prepare_thumos14_event_corpus import THUMOS_CLASSES, sha256_file  # noqa: E402
from dev.run_thumos14_event_generalization import (  # noqa: E402
    FULL_STAGE1_PROPOSAL_RECIPE,
    RETAG_PROPOSAL_RECIPE,
)
from src.proposals import ProposalGenerator  # noqa: E402
from src.prototype import build_ed_prototype  # noqa: E402


PROTOCOL_VERSION = "THUMOS14-E-supervised-paired-v1"
DEFAULT_SEEDS = (1337, 2027, 4099)
REQUIRED_FULL_COMPONENTS = (
    "stage1_actionness_reshape",
    "stage1_lattice_quality_ranking_groupdro_boundary_voting",
    "continuous_atsn_temporalmaxer",
    "event_statistics_temporalmaxer",
    "global_percentile_rank_fusion",
    "context_relative_completeness",
    "quality_focal_head",
    "gaussian_soft_nms",
)

DECISION_TRACEABILITY = {
    "official_split_and_metric": {
        "status": "published_rule",
        "evidence": "ActionFormer ECCV 2022 and the official THUMOS14 evaluator",
        "decision": "train on 200 validation videos; report per-class AP on canonical test",
    },
    "rgb_to_asynchronous_events": {
        "status": "published_precedent",
        "evidence": "Vid2E CVPR 2020 and v2e CVPRW 2021",
        "decision": "convert the official RGB videos once and share the event corpus",
    },
    "v2e_timing_profile": {
        "status": "predeclared_adaptation",
        "evidence": "Vid2E uses adaptive interpolation; v2e documents fixed and adaptive timing",
        "decision": (
            "use fixed 3 ms interpolation only for the primary row; keep adaptive and "
            "original-rate variants explicitly labelled as sensitivities"
        ),
    },
    "frozen_encoder_target_heads": {
        "status": "published_precedent",
        "evidence": "ActionFormer frozen I3D features with target-trained TAL detector",
        "decision": "freeze the shared ATSN encoder and train target heads on validation",
    },
    "twenty_binary_tasks": {
        "status": "predeclared_adaptation",
        "evidence": "reTAG has a closed binary ED/background head; THUMOS AP is per class",
        "decision": "run 20 one-vs-rest tasks without removing other classes",
    },
    "video_disjoint_target_folds": {
        "status": "predeclared_adaptation",
        "evidence": "model selection must remain inside THUMOS14 validation",
        "decision": "use five deterministic video-disjoint folds covering the same 200 videos",
    },
    "zero_second_target_prior": {
        "status": "predeclared_adaptation",
        "evidence": "reTAG declares its 2 s filter as ED-specific domain knowledge",
        "decision": "retain every official THUMOS instance in target training/evaluation",
    },
    "complete_eventpenguins_arm": {
        "status": "method_under_test",
        "evidence": "EventPenguins article Sections 3-4",
        "decision": "require all three experts, fusion, completeness, QFL and Soft-NMS",
    },
}

CONVERSION_ROLES = {
    "primary_fixed_interpolated": "fixed_interpolated",
    "sensitivity_adaptive_interpolated": "adaptive_interpolated",
    "sensitivity_original_rate": "original_rate_ablation",
}

NATIVE_MODEL_TOPOLOGY = {
    "retag": {
        "primary": "one target ATSN head per class and seed",
        "source_basis": "reTAG publishes one binary ATSN classifier",
    },
    "eventpenguins_full": {
        "primary": "five fold experts plus cross-fitted linear QFL per class and seed",
        "source_basis": "the article's complete evaluated topology",
    },
    "capacity_control": {
        "secondary_only": "five-head reTAG ensemble",
        "purpose": "separate method gain from ensemble capacity without replacing reTAG",
    },
}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def annotation_config_dir(work_dir: Path) -> Path:
    return work_dir / "config" / "annotations"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"1", "true", "yes"})


def class_selection(values: Iterable[str] | None) -> list[str]:
    selected = list(values) if values else list(THUMOS_CLASSES)
    unknown = set(selected) - set(THUMOS_CLASSES)
    if unknown:
        raise ValueError(f"Unknown THUMOS14 classes: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("Class selection contains duplicates")
    return selected


def validate_corpus_protocol(
    manifest: pd.DataFrame, fold_manifest: pd.DataFrame
) -> dict[str, object]:
    required = {"video_id", "official_subset", "evaluation_included"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Corpus manifest lacks columns {sorted(missing)}")
    if manifest["video_id"].duplicated().any():
        raise ValueError("Corpus manifest contains duplicate video IDs")
    subset = manifest["official_subset"].astype(str).str.lower()
    validation = set(manifest.loc[subset == "validation", "video_id"].astype(str))
    test = set(manifest.loc[subset == "test", "video_id"].astype(str))
    canonical_test = set(
        manifest.loc[(subset == "test") & parse_bool(manifest["evaluation_included"]), "video_id"]
        .astype(str)
    )
    if (len(validation), len(test), len(canonical_test)) != (200, 213, 212):
        raise ValueError(
            "Expected 200 validation, 213 test and 212 canonical test videos; "
            f"got {len(validation)}, {len(test)}, {len(canonical_test)}"
        )
    fold_required = {"fold", "train_record_names", "val_record_names"}
    fold_missing = fold_required - set(fold_manifest.columns)
    if fold_missing:
        raise ValueError(f"Fold manifest lacks columns {sorted(fold_missing)}")
    if set(fold_manifest["fold"].astype(int)) != set(range(5)):
        raise ValueError("Fold manifest must contain folds 0..4")
    declared: set[str] = set()
    for row in fold_manifest.itertuples(index=False):
        train_names = set(str(row.train_record_names).split())
        val_names = set(str(row.val_record_names).split())
        if train_names & val_names:
            raise ValueError(f"Fold {row.fold} has train/validation overlap")
        if train_names | val_names != validation:
            raise ValueError(f"Fold {row.fold} does not cover exactly the validation pool")
        declared.update(train_names | val_names)
    if declared & test:
        raise ValueError("Fold manifest leaks official test recordings")
    return {
        "validation_videos": len(validation),
        "test_files": len(test),
        "canonical_test_videos": len(canonical_test),
        "classes": len(THUMOS_CLASSES),
        "folds": len(fold_manifest),
    }


def validate_assembled_corpus(
    manifest: pd.DataFrame,
    assembly: dict[str, object],
    validation: dict[str, object],
    source_audit: dict[str, object],
    *,
    data_sha256: str,
    conversion_role: str,
) -> dict[str, object]:
    expected_ids = set(manifest["video_id"].astype(str))
    assembly_ids = {
        str(record["video_id"]) for record in assembly.get("recordings", [])
    }
    validation_ids = {
        str(record["video_id"]) for record in validation.get("recordings", [])
    }
    if assembly.get("partial") is not False:
        raise ValueError("The supervised protocol requires a complete, non-partial assembly")
    if assembly_ids != expected_ids or validation_ids != expected_ids:
        raise ValueError(
            "Assembly/validation recording universe differs from the 413-file manifest"
        )
    if assembly.get("index_sha256") != data_sha256:
        raise ValueError("Assembly manifest does not match the event HDF5 index")
    if validation.get("status") != "ok" or validation.get("index_sha256") != data_sha256:
        raise ValueError("The complete event HDF5 has not passed canonical validation")
    if (
        source_audit.get("videos") != 413
        or source_audit.get("hashes_verified") is not True
    ):
        raise ValueError("The 413 RGB source files have not passed hash-verified source audit")

    protocols = assembly.get("conversion_protocols", {})
    if not isinstance(protocols, dict) or len(protocols) != 1:
        raise ValueError("The corpus must contain exactly one frozen v2e conversion protocol")
    protocol_id, protocol = next(iter(protocols.items()))
    recipe = protocol.get("recipe", {})
    expected_timing = CONVERSION_ROLES[conversion_role]
    if recipe.get("timing") != expected_timing:
        raise ValueError(
            f"Conversion role {conversion_role} requires timing={expected_timing}; "
            f"got {recipe.get('timing')}"
        )
    if recipe.get("dvs_profile") != "clean":
        raise ValueError("The locked comparison requires the official v2e clean profile")
    if expected_timing == "fixed_interpolated":
        if (
            recipe.get("timestamp_resolution_s") != 0.003
            or recipe.get("disable_slomo") is not False
        ):
            raise ValueError("The primary conversion must use SuperSloMo at fixed 3 ms")

    summary = validation.get("class_summary", {}).get("instances_by_split_and_class", {})
    for split in ("train", "val", "test"):
        counts = summary.get(split, {})
        if set(counts) != set(THUMOS_CLASSES) or any(
            int(counts[label]) <= 0 for label in THUMOS_CLASSES
        ):
            raise ValueError(f"Validated corpus lacks one or more classes in split {split}")
    return {
        "conversion_protocol_id": protocol_id,
        "conversion_role": conversion_role,
        "timing": expected_timing,
        "dvs_profile": recipe["dvs_profile"],
        "validated_recordings": len(validation_ids),
    }


def target_recordings(manifest: pd.DataFrame, split: str) -> list[str]:
    subset = manifest["official_subset"].astype(str).str.lower()
    if split == "validation":
        selected = manifest.loc[subset == "validation", "video_id"]
    elif split == "test":
        selected = manifest.loc[subset == "test", "video_id"]
    else:
        raise ValueError("split must be validation or test")
    return sorted(selected.astype(str).tolist())


def fold_recordings(
    fold_manifest: pd.DataFrame, fold: int, selection: str
) -> set[str]:
    if selection not in {"train", "val"}:
        raise ValueError("selection must be train or val")
    rows = fold_manifest.loc[fold_manifest["fold"].astype(int) == fold]
    if len(rows) != 1:
        raise ValueError(f"Expected one manifest row for fold {fold}; got {len(rows)}")
    column = f"{selection}_record_names"
    recordings = set(str(rows.iloc[0][column]).split())
    if not recordings:
        raise ValueError(f"Fold {fold} has no {selection} recordings")
    return recordings


def target_prototype_path(out_root: Path, label: str, cv_fold: int | None) -> Path:
    variant = "final" if cv_fold is None else f"fold_{cv_fold:02d}"
    return out_root / "target_prototypes" / label / variant / "prototype.npy"


def full_stack_status(component_paths: dict[str, Path]) -> dict[str, object]:
    missing = [name for name in REQUIRED_FULL_COMPONENTS if not component_paths[name].exists()]
    return {
        "reportable_as_eventpenguins_full": not missing,
        "required_components": list(REQUIRED_FULL_COMPONENTS),
        "missing_components": missing,
    }


def load_plan(path: Path) -> dict[str, object]:
    plan = json.loads(resolve(path).read_text(encoding="utf-8"))
    if plan.get("protocol") != PROTOCOL_VERSION:
        raise ValueError(f"Unexpected protocol manifest: {plan.get('protocol')}")
    for name, record in plan["inputs"].items():
        input_path = Path(record["path"])
        if not input_path.exists() or sha256_file(input_path) != record["sha256"]:
            raise RuntimeError(f"Frozen protocol input changed: {name}")
    return plan


def build_plan(args: argparse.Namespace) -> None:
    work_dir = resolve(args.work_dir)
    out_root = resolve(args.out_root)
    inputs = {
        "corpus_manifest": work_dir / "manifest.csv",
        "assembly_manifest": work_dir / "corpus_manifest.json",
        "validation_report": work_dir / "validation_report.json",
        "source_audit": work_dir / "source_audit.json",
        "fold_manifest": annotation_config_dir(work_dir) / "fold_manifest.csv",
        "event_hdf5": work_dir / "preprocessed.h5",
        "source_atsn": resolve(args.source_model),
        "source_ed_prototype": resolve(args.source_prototype),
        "canonical_annotations": resolve(args.canonical_annotations),
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    manifest = pd.read_csv(inputs["corpus_manifest"], keep_default_na=False)
    folds = pd.read_csv(inputs["fold_manifest"], keep_default_na=False)
    audit = validate_corpus_protocol(manifest, folds)
    data_sha256 = sha256_file(inputs["event_hdf5"])
    conversion_audit = validate_assembled_corpus(
        manifest,
        json.loads(inputs["assembly_manifest"].read_text(encoding="utf-8")),
        json.loads(inputs["validation_report"].read_text(encoding="utf-8")),
        json.loads(inputs["source_audit"].read_text(encoding="utf-8")),
        data_sha256=data_sha256,
        conversion_role=args.conversion_role,
    )
    for label in THUMOS_CLASSES:
        class_dir = annotation_config_dir(work_dir) / "by_class" / label
        for filename in (
            "annotations_trainable.json",
            "recording_info.csv",
            "fold_manifest.csv",
        ):
            if not (class_dir / filename).exists():
                raise FileNotFoundError(class_dir / filename)
    plan = {
        "protocol": PROTOCOL_VERSION,
        "scientific_scope": {
            "corpora": 1,
            "class_tasks": 20,
            "source_recordings_mixed_into_target": False,
            "encoder": "same frozen source ATSN checkpoint in both arms",
            "heads": "independent per method, class and seed",
            "target_training": "official THUMOS14 validation videos only",
            "target_evaluation": "official canonical THUMOS14 test videos once",
            "conversion_role": args.conversion_role,
        },
        "literature_and_adaptations": {
            "published": [
                "THUMOS14 validation-to-test split and per-class AP",
                "reTAG proposal AR and binary ATSN training recipe",
                "Vid2E/v2e RGB-to-event simulation",
                "frozen encoder plus target-trained TAL head precedent",
            ],
            "predeclared_adaptations": [
                "twenty one-vs-rest binary tasks",
                "one target spatial prototype per class for the complete method",
                "zero-second minimum duration for all official THUMOS14 instances",
            ],
            "sensitivities": [
                "source ED 2-second prior",
                "source ED periodicity prior disabled",
                "original-rate versus interpolated v2e timing",
            ],
        },
        "decision_traceability": DECISION_TRACEABILITY,
        "native_model_topology": NATIVE_MODEL_TOPOLOGY,
        "non_reportable_partial_method": "eventpenguins_stage1_diagnostic",
        "full_method_required_components": list(REQUIRED_FULL_COMPONENTS),
        "seeds": list(args.seeds),
        "audit": audit,
        "conversion_audit": conversion_audit,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "paths": {"work_dir": str(work_dir), "out_root": str(out_root)},
    }
    atomic_json(out_root / "protocol_manifest.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


def build_prototypes(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    work_dir = Path(plan["paths"]["work_dir"])
    out_root = Path(plan["paths"]["out_root"])
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    manifest = pd.read_csv(plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False)
    folds = pd.read_csv(plan["inputs"]["fold_manifest"]["path"], keep_default_na=False)
    validation = set(target_recordings(manifest, "validation"))
    for label in class_selection(args.classes):
        annotation_path = (
            annotation_config_dir(work_dir)
            / "by_class"
            / label
            / "annotations_trainable.json"
        )
        training_pools = [
            (fold, fold_recordings(folds, fold, "train")) for fold in range(5)
        ] + [(None, validation)]
        for fold, recordings in training_pools:
            prototype = build_ed_prototype(
                str(data_path),
                str(annotation_path),
                split="train",
                min_duration=0.0,
                recordings=recordings,
            )
            if not np.isfinite(prototype).all() or float(np.linalg.norm(prototype)) == 0.0:
                raise ValueError(f"Empty or invalid target prototype for {label}, fold={fold}")
            prototype_path = target_prototype_path(out_root, label, fold)
            prototype_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(prototype_path, prototype)
            atomic_json(
                prototype_path.parent / "metadata.json",
                {
                    "protocol": PROTOCOL_VERSION,
                    "target_class": label,
                    "cv_fold": fold,
                    "training_recordings": len(recordings),
                    "training_recording_names": sorted(recordings),
                    "minimum_action_duration_s": 0.0,
                    "event_hdf5_sha256": sha256_file(data_path),
                    "annotations_sha256": sha256_file(annotation_path),
                    "prototype_sha256": sha256_file(prototype_path),
                },
            )
            print(f"[PROTOTYPE] {label} fold={fold}: {prototype_path}")


def proposal_configuration(
    branch: str, out_root: Path, target_class: str | None, cv_fold: int | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    if branch == "retag":
        return (
            {**RETAG_PROPOSAL_RECIPE, "minimum_proposal_duration_s": 0.0},
            {"status": "published reTAG recipe plus declared removal of ED 2 s prior"},
        )
    if branch != "eventpenguins_stage1":
        raise ValueError(f"Unknown branch {branch}")
    if target_class is None:
        raise ValueError("eventpenguins_stage1 requires --target-class")
    prototype_path = target_prototype_path(out_root, target_class, cv_fold)
    if not prototype_path.exists():
        raise FileNotFoundError(prototype_path)
    return (
        {
            **FULL_STAGE1_PROPOSAL_RECIPE,
            "prototype": np.load(prototype_path),
            "prototype_weight": 0.3,
            "minimum_proposal_duration_s": 0.0,
        },
        {
            "status": "complete stage-1 diagnostic; not the complete EventPenguins method",
            "cv_fold": cv_fold,
            "target_prototype_sha256": sha256_file(prototype_path),
        },
    )


def generate_proposals(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    work_dir = Path(plan["paths"]["work_dir"])
    out_root = Path(plan["paths"]["out_root"])
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    manifest = pd.read_csv(plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False)
    if args.cv_fold is not None and (args.branch != "eventpenguins_stage1" or args.split != "validation"):
        raise ValueError("--cv-fold is only valid for EventPenguins stage-1 validation proposals")
    recordings = target_recordings(manifest, args.split)
    recipe, declaration = proposal_configuration(
        args.branch, out_root, args.target_class, args.cv_fold
    )
    branch_name = args.branch if args.target_class is None else f"{args.branch}/{args.target_class}"
    if args.cv_fold is not None:
        branch_name += f"/fold_{args.cv_fold:02d}"
    output_dir = out_root / "proposals" / branch_name / args.split
    generator = ProposalGenerator(
        data_path=str(data_path),
        output_dir=str(output_dir / "logs"),
        **recipe,
    )
    proposals = generator.run(split=None, recordings=recordings)
    unexpected = set(proposals.get("rec_name", pd.Series(dtype=str)).astype(str)) - set(recordings)
    if unexpected:
        raise RuntimeError(f"Proposal generation escaped the declared split: {sorted(unexpected)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = output_dir / "proposals.csv"
    proposals.to_csv(proposal_path, index=False)
    atomic_json(
        output_dir / "report.json",
        {
            "protocol": PROTOCOL_VERSION,
            "branch": args.branch,
            "target_class": args.target_class,
            "cv_fold": args.cv_fold,
            "split": args.split,
            "recording_universe": len(recordings),
            "proposal_count": len(proposals),
            "proposal_sha256": sha256_file(proposal_path),
            "minimum_proposal_duration_s": 0.0,
            "declaration": declaration,
        },
    )
    print(f"[PROPOSALS] {args.branch} {args.target_class or 'shared'} {args.split}: {len(proposals)}")


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")


def proposal_path(out_root: Path, branch: str, label: str, split: str) -> Path:
    if branch == "retag":
        return out_root / "proposals" / branch / split / "proposals.csv"
    return out_root / "proposals" / branch / label / split / "proposals.csv"


def train_heads(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    work_dir = Path(plan["paths"]["work_dir"])
    out_root = Path(plan["paths"]["out_root"])
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    source_model = Path(plan["inputs"]["source_atsn"]["path"])
    classes = class_selection(args.classes)
    seeds = args.seeds or plan["seeds"]
    branch_label = (
        "retag" if args.branch == "retag" else "eventpenguins_stage1_diagnostic"
    )
    for label in classes:
        cache_key = "retag" if args.branch == "retag" else f"{args.branch}/{label}"
        feature_dirs = {
            split: out_root / "features" / cache_key / split
            for split in ("validation", "test")
        }
        for split, feature_dir in feature_dirs.items():
            command = [
                sys.executable,
                str(ROOT / "dev" / "train_thumos14_ovr_atsn.py"),
                "extract",
                "--data-path",
                str(data_path),
                "--proposals",
                str(proposal_path(out_root, args.branch, label, split)),
                "--source-model",
                str(source_model),
                "--out-dir",
                str(feature_dir),
                "--batch-size",
                str(args.feature_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
            ]
            run_logged(command, feature_dir / "extract.log")
        annotation_path = (
            annotation_config_dir(work_dir)
            / "by_class"
            / label
            / "annotations_trainable.json"
        )
        for seed in seeds:
            head_dir = out_root / "heads" / branch_label / f"seed_{seed}" / label
            train_command = [
                sys.executable,
                str(ROOT / "dev" / "train_thumos14_ovr_atsn.py"),
                "train",
                "--train-features-dir",
                str(feature_dirs["validation"]),
                "--annotations",
                str(annotation_path),
                "--source-model",
                str(source_model),
                "--out-dir",
                str(head_dir),
                "--train-all-validation",
                "--seed",
                str(seed),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
            ]
            run_logged(train_command, head_dir / "train.log")
            prediction_dir = out_root / "predictions" / branch_label / f"seed_{seed}" / label
            score_command = [
                sys.executable,
                str(ROOT / "dev" / "train_thumos14_ovr_atsn.py"),
                "score",
                "--features-dir",
                str(feature_dirs["test"]),
                "--checkpoint",
                str(head_dir / "final.pt"),
                "--out-dir",
                str(prediction_dir),
                "--device",
                args.device,
            ]
            run_logged(score_command, prediction_dir / "score.log")
            print(f"[HEAD] {branch_label} {label} seed={seed}")
    if set(classes) != set(THUMOS_CLASSES):
        print("[DIAGNOSTIC] Class subset complete; canonical macro evaluation intentionally skipped")
        return
    for seed in seeds:
        prediction_root = out_root / "predictions" / branch_label / f"seed_{seed}"
        evaluation_dir = out_root / "evaluation" / branch_label / f"seed_{seed}"
        command = [
            sys.executable,
            str(ROOT / "dev" / "evaluate_thumos14e_ovr.py"),
            "--actionformer-root",
            str(resolve(args.actionformer_root)),
            "--canonical-annotations",
            str(Path(plan["inputs"]["canonical_annotations"]["path"])),
            "--predictions-root",
            str(prediction_root),
            "--out-dir",
            str(evaluation_dir),
        ]
        run_logged(command, evaluation_dir / "evaluate.log")


def audit_full(args: argparse.Namespace) -> None:
    root = resolve(args.components_root)
    paths = {name: root / f"{name}.json" for name in REQUIRED_FULL_COMPONENTS}
    status = full_stack_status(paths)
    status["component_paths"] = {name: str(path) for name, path in paths.items()}
    atomic_json(resolve(args.out), status)
    print(json.dumps(status, indent=2, sort_keys=True))
    if not status["reportable_as_eventpenguins_full"]:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--work-dir", required=True)
    plan.add_argument("--out-root", required=True)
    plan.add_argument("--source-model", default="models/model.pk")
    plan.add_argument("--source-prototype", default="tmp/prototype/ed_prototype.npy")
    plan.add_argument("--canonical-annotations", required=True)
    plan.add_argument(
        "--conversion-role",
        choices=tuple(CONVERSION_ROLES),
        default="primary_fixed_interpolated",
    )
    plan.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))

    prototypes = commands.add_parser("prototypes")
    prototypes.add_argument("--plan", type=Path, required=True)
    prototypes.add_argument("--classes", nargs="+", default=None)

    proposals = commands.add_parser("proposals")
    proposals.add_argument("--plan", type=Path, required=True)
    proposals.add_argument("--branch", choices=("retag", "eventpenguins_stage1"), required=True)
    proposals.add_argument("--target-class", choices=THUMOS_CLASSES, default=None)
    proposals.add_argument("--split", choices=("validation", "test"), required=True)
    proposals.add_argument("--cv-fold", type=int, choices=range(5), default=None)

    heads = commands.add_parser("heads")
    heads.add_argument("--plan", type=Path, required=True)
    heads.add_argument("--branch", choices=("retag", "eventpenguins_stage1"), required=True)
    heads.add_argument("--classes", nargs="+", default=None)
    heads.add_argument("--seeds", type=int, nargs="+", default=None)
    heads.add_argument("--actionformer-root", required=True)
    heads.add_argument("--device", default="cuda")
    heads.add_argument("--feature-batch-size", type=int, default=64)
    heads.add_argument("--num-workers", type=int, default=8)

    full = commands.add_parser("audit-full")
    full.add_argument("--components-root", required=True)
    full.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {
        "plan": build_plan,
        "prototypes": build_prototypes,
        "proposals": generate_proposals,
        "heads": train_heads,
        "audit-full": audit_full,
    }[args.command](args)


if __name__ == "__main__":
    main()
