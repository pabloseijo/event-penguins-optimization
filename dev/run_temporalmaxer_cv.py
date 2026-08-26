"""Run recording-disjoint TemporalMaxer-lite folds across available GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense TemporalMaxer CV folds")
    parser.add_argument("--fold-dir", default="tmp/cv/recording_folds_r5")
    parser.add_argument("--master-proposals", default="tmp/temporalmaxer_dense/source_master.csv")
    parser.add_argument("--cache-dir", default="tmp/temporalmaxer_dense/source_cache")
    parser.add_argument("--event-feature-cache-dir", default=None)
    parser.add_argument("--event-features-only", action="store_true")
    parser.add_argument("--corrupted-event-feature-cache-dir", default=None)
    parser.add_argument("--corrupted-event-probability", type=float, default=0.5)
    parser.add_argument("--out-root", default="tmp/temporalmaxer_dense/cv")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--group-dro", action="store_true")
    parser.add_argument("--tanp-sigma", type=float, default=0.0)
    parser.add_argument("--trc-weight", type=float, default=0.0)
    parser.add_argument("--trc-topk", type=int, default=3)
    parser.add_argument("--trident-bins", type=int, default=0)
    parser.add_argument("--trident-weight", type=float, default=0.5)
    parser.add_argument(
        "--selection-score",
        choices=["cnn_score", "dense_score", "brem_score"],
        default="cnn_score",
    )
    parser.add_argument(
        "--selection-boundary", choices=["raw", "blend", "trident"], default="blend"
    )
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.5)
    parser.add_argument("--distribution-weight", type=float, default=0.25)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--fast-selection-eval", action="store_true")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=20,
        help="Resume interrupted folds from last.pt up to this many times.",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def run_fold(fold: int, gpu: str, args: argparse.Namespace) -> tuple[int, int]:
    fold_path = resolve(args.fold_dir) / f"fold_{fold:02d}"
    out_dir = resolve(args.out_root) / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "best.pt").exists() and (out_dir / "metrics_best.csv").exists():
        return fold, 0
    command = [
        sys.executable,
        str(ROOT / "dev" / "train_temporalmaxer_dense.py"),
        "--master-proposals", str(resolve(args.master_proposals)),
        "--train-proposals", str(fold_path / "train_proposals.csv"),
        "--val-proposals", str(fold_path / "val_proposals.csv"),
        "--cache-dir", str(resolve(args.cache_dir)),
        "--out-dir", str(out_dir),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--max-train-samples", str(args.max_train_samples),
        "--hidden-dim", str(args.hidden_dim),
        "--seed", str(args.seed + fold),
        "--tanp-sigma", str(args.tanp_sigma),
        "--trc-weight", str(args.trc_weight),
        "--trc-topk", str(args.trc_topk),
        "--trident-bins", str(args.trident_bins),
        "--trident-weight", str(args.trident_weight),
        "--selection-score", args.selection_score,
        "--selection-boundary", args.selection_boundary,
        "--quality-weight", str(args.quality_weight),
        "--action-weight", str(args.action_weight),
        "--distribution-weight", str(args.distribution_weight),
        "--boundary-weight", str(args.boundary_weight),
        "--num-workers", "0",
    ]
    if args.event_feature_cache_dir:
        command.extend(
            ["--event-feature-cache-dir", str(resolve(args.event_feature_cache_dir))]
        )
    if args.corrupted_event_feature_cache_dir:
        command.extend(
            [
                "--corrupted-event-feature-cache-dir",
                str(resolve(args.corrupted_event_feature_cache_dir)),
                "--corrupted-event-probability",
                str(args.corrupted_event_probability),
            ]
        )
    if args.event_features_only:
        command.append("--event-features-only")
    if args.group_dro:
        command.append("--group-dro")
    if args.quiet_progress:
        command.append("--quiet-progress")
    if args.fast_selection_eval:
        command.append("--fast-selection-eval")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = out_dir / "run.log"
    returncode = 1
    for attempt in range(1, max(args.max_retries, 1) + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[RUNNER] attempt={attempt}/{args.max_retries}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = completed.returncode
        if returncode == 0:
            break
    return fold, returncode


def aggregate(args: argparse.Namespace) -> None:
    manifest_path = resolve(args.fold_dir) / "manifest.csv"
    manifest = pd.read_csv(manifest_path).set_index("fold")
    frames = []
    for fold in args.folds:
        path = resolve(args.out_root) / f"fold_{fold:02d}" / "metrics_best.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold metrics: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "fold", fold)
        frame["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        frames.append(frame)
    all_metrics = pd.concat(frames, ignore_index=True)
    out_root = resolve(args.out_root)
    all_metrics.to_csv(out_root / "all_fold_metrics.csv", index=False)
    metric_columns = [column for column in all_metrics.columns if column == "mAP" or column.startswith("AP@")]
    rows = []
    for (score, boundary), group in all_metrics.groupby(["score_column", "boundary_mode"]):
        row = {
            "score_column": score,
            "boundary_mode": boundary,
            "folds": len(group),
        }
        weights = group["val_ed_instances"].to_numpy(dtype=np.float64)
        for column in metric_columns:
            values = group[column].to_numpy(dtype=np.float64)
            row[f"mean_{column}"] = float(values.mean())
            row[f"weighted_{column}"] = float(np.average(values, weights=weights))
            row[f"worst_{column}"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mean_mAP", ascending=False)
    summary.to_csv(out_root / "cv_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    if not args.gpus:
        raise ValueError("At least one GPU is required")
    assignments = [(fold, args.gpus[index % len(args.gpus)]) for index, fold in enumerate(args.folds)]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(run_fold, fold, gpu, args) for fold, gpu in assignments]
        for future in concurrent.futures.as_completed(futures):
            fold, returncode = future.result()
            print(f"[FOLD {fold:02d}] returncode={returncode}")
            if returncode != 0:
                failures.append(fold)
    if failures:
        raise RuntimeError(f"Failed folds: {failures}")
    aggregate(args)


if __name__ == "__main__":
    main()
