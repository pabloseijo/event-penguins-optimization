#!/usr/bin/env python3
"""Run the predeclared single-class THUMOS14-E integration pilot.

The pilot freezes one complete corpus and executes both native pipelines for
one class. It writes predictions but deliberately does not open test labels or
compute test AP. Every stage is restartable from its declared artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def append_journal(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_stage(
    name: str,
    command: list[str],
    expected: Path,
    log_dir: Path,
    journal: Path,
) -> None:
    if expected.exists():
        append_journal(
            journal,
            {"stage": name, "status": "skipped-existing", "artifact": str(expected)},
        )
        print(f"[SKIP] {name}: {expected}")
        return
    log_path = log_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    append_journal(
        journal,
        {"stage": name, "status": "started", "command": command, "time": started},
    )
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode or not expected.exists():
        append_journal(
            journal,
            {
                "stage": name,
                "status": "failed",
                "returncode": completed.returncode,
                "elapsed_s": elapsed,
                "log": str(log_path),
            },
        )
        raise RuntimeError(f"Stage {name} failed; see {log_path}")
    append_journal(
        journal,
        {
            "stage": name,
            "status": "complete",
            "artifact": str(expected),
            "elapsed_s": elapsed,
            "log": str(log_path),
        },
    )
    print(f"[DONE] {name}: {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--target-class", default="Diving")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--source-model", default="models/model.pk")
    parser.add_argument("--source-prototype", default="tmp/prototype/ed_prototype.npy")
    parser.add_argument("--canonical-annotations", required=True)
    parser.add_argument("--actionformer-root", required=True)
    parser.add_argument(
        "--conversion-role",
        choices=(
            "primary_fixed_interpolated",
            "sensitivity_adaptive_interpolated",
            "sensitivity_original_rate",
        ),
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--continuous-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = resolve(args.work_dir)
    out_root = resolve(args.out_root)
    plan = out_root / "protocol_manifest.json"
    logs = out_root / "pilot_logs"
    journal = logs / "journal.jsonl"
    label = args.target_class
    seed = args.seed
    supervised = str(ROOT / "dev" / "run_thumos14e_supervised.py")
    full = str(ROOT / "dev" / "run_thumos14e_full_pipeline.py")
    corpus = str(ROOT / "dev" / "prepare_thumos14_event_corpus.py")

    run_stage(
        "assemble",
        [PYTHON, corpus, "assemble", "--work-dir", str(work_dir)],
        work_dir / "corpus_manifest.json",
        logs,
        journal,
    )
    run_stage(
        "validate",
        [PYTHON, corpus, "validate", "--work-dir", str(work_dir)],
        work_dir / "validation_report.json",
        logs,
        journal,
    )
    run_stage(
        "plan",
        [
            PYTHON,
            supervised,
            "plan",
            "--work-dir", str(work_dir),
            "--out-root", str(out_root),
            "--source-model", str(resolve(args.source_model)),
            "--source-prototype", str(resolve(args.source_prototype)),
            "--canonical-annotations", str(resolve(args.canonical_annotations)),
            "--conversion-role", args.conversion_role,
            "--seeds", str(seed),
        ],
        plan,
        logs,
        journal,
    )
    run_stage(
        "prototypes",
        [PYTHON, supervised, "prototypes", "--plan", str(plan), "--classes", label],
        out_root / "target_prototypes" / label / "final" / "prototype.npy",
        logs,
        journal,
    )
    for split in ("validation", "test"):
        run_stage(
            f"retag-proposals-{split}",
            [
                PYTHON,
                supervised,
                "proposals",
                "--plan", str(plan),
                "--branch", "retag",
                "--split", split,
            ],
            out_root / "proposals" / "retag" / split / "proposals.csv",
            logs,
            journal,
        )
    for fold in range(5):
        run_stage(
            f"eventpenguins-proposals-fold-{fold:02d}",
            [
                PYTHON,
                supervised,
                "proposals",
                "--plan", str(plan),
                "--branch", "eventpenguins_stage1",
                "--target-class", label,
                "--split", "validation",
                "--cv-fold", str(fold),
            ],
            (
                out_root
                / "proposals"
                / "eventpenguins_stage1"
                / label
                / f"fold_{fold:02d}"
                / "validation"
                / "proposals.csv"
            ),
            logs,
            journal,
        )
    run_stage(
        "eventpenguins-proposals-test",
        [
            PYTHON,
            supervised,
            "proposals",
            "--plan", str(plan),
            "--branch", "eventpenguins_stage1",
            "--target-class", label,
            "--split", "test",
        ],
        (
            out_root
            / "proposals"
            / "eventpenguins_stage1"
            / label
            / "test"
            / "proposals.csv"
        ),
        logs,
        journal,
    )
    run_stage(
        "retag-head",
        [
            PYTHON,
            supervised,
            "heads",
            "--plan", str(plan),
            "--branch", "retag",
            "--classes", label,
            "--seeds", str(seed),
            "--actionformer-root", str(resolve(args.actionformer_root)),
            "--device", args.device,
            "--feature-batch-size", str(args.feature_batch_size),
            "--num-workers", str(args.num_workers),
        ],
        out_root / "predictions" / "retag" / f"seed_{seed}" / label / "predictions.json",
        logs,
        journal,
    )
    run_stage(
        "shared-index",
        [PYTHON, full, "shared-index", "--plan", str(plan)],
        out_root / "shared_features" / "continuous" / "metadata.json",
        logs,
        journal,
    )
    run_stage(
        "shared-event-stats",
        [PYTHON, full, "shared-event-stats", "--plan", str(plan)],
        out_root / "shared_features" / "event_stats" / "metadata.json",
        logs,
        journal,
    )
    run_stage(
        "shared-atsn",
        [
            PYTHON,
            full,
            "shared-extract",
            "--plan", str(plan),
            "--shard-index", "0",
            "--num-shards", "1",
            "--batch-size", str(args.continuous_batch_size),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
        ],
        out_root / "shared_features" / "continuous" / "shard_00_of_01.json",
        logs,
        journal,
    )
    run_stage(
        "shared-verify",
        [PYTHON, full, "shared-verify", "--plan", str(plan), "--num-shards", "1"],
        out_root / "shared_features" / "report.json",
        logs,
        journal,
    )
    class_args = [
        "--plan", str(plan),
        "--target-class", label,
        "--seed", str(seed),
        "--device", args.device,
        "--num-workers", str(args.num_workers),
    ]
    class_root = out_root / "eventpenguins_full" / f"seed_{seed}" / label
    for fold in range(5):
        run_stage(
            f"local-fold-{fold:02d}",
            [PYTHON, full, "local-fold", *class_args, "--fold", str(fold)],
            class_root / "local" / f"fold_{fold:02d}" / "report.json",
            logs,
            journal,
        )
    run_stage(
        "local-test",
        [PYTHON, full, "local-test", *class_args],
        class_root / "local_test" / "report.json",
        logs,
        journal,
    )
    for fold in range(5):
        run_stage(
            f"continuous-fold-{fold:02d}",
            [PYTHON, full, "continuous-fold", *class_args, "--fold", str(fold)],
            class_root / "event" / f"fold_{fold:02d}" / "best.pt",
            logs,
            journal,
        )
    run_stage(
        "qfl-cv",
        [PYTHON, full, "qfl-cv", *class_args],
        class_root / "qfl_cv" / "candidate_features.csv",
        logs,
        journal,
    )
    run_stage(
        "full-test-prediction",
        [PYTHON, full, "full-test", *class_args],
        class_root / "test" / "predictions.json",
        logs,
        journal,
    )
    report = {
        "protocol": "THUMOS14-E-single-class-integration-pilot-v1",
        "conversion_role": args.conversion_role,
        "target_class": label,
        "seed": seed,
        "test_ap_computed": False,
        "corpus_files": 413,
        "source_recordings_mixed_into_target": False,
        "retag_prediction": str(
            out_root / "predictions" / "retag" / f"seed_{seed}" / label / "predictions.json"
        ),
        "eventpenguins_prediction": str(class_root / "test" / "predictions.json"),
    }
    (logs / "pilot_complete.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
