"""Run and aggregate full-ROI TemporalMaxer recording-disjoint CV folds."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTROL_MEAN_MAP = 0.8077965
CONTROL_WORST_MAP = 0.6909280
CONTROL_MEAN_AP07 = 0.6240949


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="tmp/temporalmaxer_continuous/source_features")
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--out-root", default="tmp/temporalmaxer_continuous/cv_baseline")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--disable-quality", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--epochs-per-process", type=int, default=0)
    return parser.parse_args()


def run_fold(fold: int, gpu: str, args: argparse.Namespace) -> tuple[int, int]:
    out_dir = resolve(args.out_root) / f"fold_{fold:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "best.pt").exists() and (out_dir / "metrics_best.json").exists():
        return fold, 0
    command = [
        sys.executable,
        str(ROOT / "dev" / "train_temporalmaxer_continuous.py"),
        "--feature-dir", str(resolve(args.feature_dir)),
        "--fold-manifest", str(resolve(args.fold_manifest)),
        "--fold", str(fold),
        "--out-dir", str(out_dir),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--quality-weight", str(args.quality_weight),
        "--seed", str(args.seed),
        "--epochs-per-process", str(args.epochs_per_process),
    ]
    if args.disable_quality:
        command.append("--disable-quality")
    if args.quiet_progress:
        command.append("--quiet-progress")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = out_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return fold, completed.returncode


def aggregate(args: argparse.Namespace) -> dict:
    manifest = pd.read_csv(resolve(args.fold_manifest)).set_index("fold")
    rows = []
    for fold in args.folds:
        metric_path = resolve(args.out_root) / f"fold_{fold:02d}" / "metrics_best.json"
        if not metric_path.exists():
            raise FileNotFoundError(metric_path)
        row = json.loads(metric_path.read_text(encoding="utf-8"))
        row["fold"] = fold
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("fold")
    weights = frame["val_ed_instances"].to_numpy(dtype=np.float64)
    summary = {
        "mean_mAP": float(frame["mAP"].mean()),
        "weighted_mAP": float(np.average(frame["mAP"], weights=weights)),
        "worst_fold_mAP": float(frame["mAP"].min()),
        "mean_AP@0.1": float(frame["AP@0.1"].mean()),
        "mean_AP@0.3": float(frame["AP@0.3"].mean()),
        "mean_AP@0.5": float(frame["AP@0.5"].mean()),
        "mean_AP@0.7": float(frame["AP@0.7"].mean()),
        "control_mean_mAP": CONTROL_MEAN_MAP,
        "control_worst_fold_mAP": CONTROL_WORST_MAP,
        "control_mean_AP@0.7": CONTROL_MEAN_AP07,
    }
    summary["passes_source_gate"] = bool(
        summary["mean_mAP"] > CONTROL_MEAN_MAP
        and summary["worst_fold_mAP"] > CONTROL_WORST_MAP
        and summary["mean_AP@0.7"] > CONTROL_MEAN_AP07
    )
    out_root = resolve(args.out_root)
    frame.to_csv(out_root / "fold_metrics.csv", index=False)
    (out_root / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(run_fold, fold, args.gpus[index % len(args.gpus)], args)
            for index, fold in enumerate(args.folds)
        ]
        failures = []
        for future in concurrent.futures.as_completed(futures):
            fold, returncode = future.result()
            print(f"[FOLD] {fold}: returncode={returncode}")
            if returncode != 0:
                failures.append(fold)
    if failures:
        raise SystemExit(f"Failed folds: {failures}")
    aggregate(args)


if __name__ == "__main__":
    main()
