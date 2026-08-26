"""Run recording-disjoint quality-head folds across available GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", default="tmp/cv/lattice_trainval_family_folds")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--weight-average-start-epoch", type=int, default=0)
    parser.add_argument("--weight-average-interval", type=int, default=1)
    parser.add_argument("--rank-sort-weight", type=float, default=0.0)
    parser.add_argument("--rank-sort-delta", type=float, default=0.5)
    parser.add_argument("--rank-sort-max-positives", type=int, default=128)
    parser.add_argument("--rank-sort-max-negatives", type=int, default=384)
    parser.add_argument("--oof-hardness-csv", default=None)
    parser.add_argument("--oof-hardness-threshold", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def run_fold(fold: int, gpu: str, args: argparse.Namespace) -> tuple[int, int]:
    fold_dir = resolve(args.fold_dir) / f"fold_{fold:02d}"
    out_dir = resolve(args.out_root) / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "qhead_qfl_only.pt"
    if checkpoint_path.exists() and (out_dir / "summary.csv").exists():
        return fold, 0

    command = [
        sys.executable,
        str(ROOT / "dev" / "train_quality_head.py"),
        "--train-proposals", str(fold_dir / "train_proposals.csv"),
        "--val-proposals", str(fold_dir / "val_proposals.csv"),
        "--train-repr", str(fold_dir / "train_repr.npz"),
        "--val-repr", str(fold_dir / "val_repr.npz"),
        "--out-dir", str(out_dir),
        "--configs", "qhead_qfl_only",
        "--epochs", str(args.epochs),
        "--batch-size", "4096",
        "--max-train-samples", "140000",
        "--group-dro",
        "--group-dro-eta", "0.01",
        "--weight-average-start-epoch", str(args.weight_average_start_epoch),
        "--weight-average-interval", str(args.weight_average_interval),
        "--rank-sort-weight", str(args.rank_sort_weight),
        "--rank-sort-delta", str(args.rank_sort_delta),
        "--rank-sort-max-positives", str(args.rank_sort_max_positives),
        "--rank-sort-max-negatives", str(args.rank_sort_max_negatives),
        "--eval-every", str(args.eval_every),
        "--min-score", "0.1",
        "--score-cols", "quality_score",
        "--seed", str(1337 + fold),
        "--quiet-progress",
        "--reuse-labeled-cache",
        "--resume-training",
        "--skip-baseline-evaluation",
    ]
    if args.oof_hardness_csv:
        command.extend(
            [
                "--oof-hardness-csv",
                str(resolve(args.oof_hardness_csv)),
                "--oof-hardness-threshold",
                str(args.oof_hardness_threshold),
            ]
        )
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
    rows = []
    for fold in args.folds:
        checkpoint = torch.load(
            resolve(args.out_root) / f"fold_{fold:02d}" / "qhead_qfl_only.pt",
            map_location="cpu",
            weights_only=False,
        )
        row = {"fold": fold, **checkpoint["best_metrics"]}
        rows.append(row)
    metrics = pd.DataFrame(rows)
    out_root = resolve(args.out_root)
    metrics.to_csv(out_root / "fold_metrics.csv", index=False)
    metric_columns = ["mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7"]
    summary = {"folds": len(metrics)}
    for column in metric_columns:
        values = metrics[column].to_numpy(dtype=np.float64)
        summary[f"mean_{column}"] = float(values.mean())
        summary[f"worst_{column}"] = float(values.min())
    pd.DataFrame([summary]).to_csv(out_root / "cv_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


def main() -> None:
    args = parse_args()
    assignments = [
        (fold, args.gpus[index % len(args.gpus)])
        for index, fold in enumerate(args.folds)
    ]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(run_fold, fold, gpu, args)
            for fold, gpu in assignments
        ]
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
