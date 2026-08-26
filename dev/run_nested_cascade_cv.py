"""Run leakage-free nested CV for the recomputed-feature temporal cascade.

For every outer fold, each inner first-stage head is trained without either the
outer validation fold or its own inner validation fold. The second-stage
refiner is then trained on those inner out-of-fold predictions and evaluated on
the untouched outer fold. Frozen ATSN representations and deterministic label
frames are reused because they contain no learned fold-specific predictions.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dev.train_quality_head import CONFIGS, ROOT, set_seed, train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested CV for the temporal cascade.")
    parser.add_argument("--fold-root", default="tmp/cv/lattice_trainval_family_folds")
    parser.add_argument("--stage1-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--out-dir", default="tmp/quality_head/cascade_groupdro_nested_cv")
    parser.add_argument("--outer-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fold_parts(
    fold_root: Path,
    stage1_root: Path,
) -> tuple[list[pd.DataFrame], list[np.ndarray]]:
    frames = []
    embeddings = []
    for fold in range(5):
        frame_path = stage1_root / f"fold_{fold:02d}" / "cache" / "val_quality_labels.csv"
        repr_path = fold_root / f"fold_{fold:02d}" / "val_repr.npz"
        frame = pd.read_csv(frame_path).reset_index(drop=True)
        representation = np.load(repr_path, allow_pickle=False)["embeddings"]
        if len(frame) != len(representation):
            raise ValueError(f"Fold {fold} labels/repr mismatch: {len(frame)} != {len(representation)}")
        frames.append(frame)
        embeddings.append(representation)
        print(f"[INFO] Fold {fold}: n={len(frame)} emb={representation.shape}")
    return frames, embeddings


def stage1_args(template: dict, out_dir: Path, seed: int) -> argparse.Namespace:
    values = copy.deepcopy(template)
    values.update(
        {
            "out_dir": str(out_dir),
            "seed": seed,
            "decoupled_boundary_head": False,
            "skip_evaluation": False,
        }
    )
    return argparse.Namespace(**values)


def train_inner_stage1(
    outer: int,
    inner: int,
    frames: list[pd.DataFrame],
    embeddings: list[np.ndarray],
    template_args: dict,
    args: argparse.Namespace,
    device: torch.device,
    out_root: Path,
) -> Path:
    inner_dir = out_root / f"outer_{outer:02d}" / f"inner_{inner:02d}_stage1"
    cache_dir = inner_dir / "cache"
    pred_dir = inner_dir / "predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_path = cache_dir / "val_scores_qhead_qfl_only.csv"
    if score_path.exists():
        print(f"[INFO] Reusing nested stage1 outer={outer} inner={inner}: {score_path}")
        return score_path

    train_folds = [fold for fold in range(5) if fold not in {outer, inner}]
    train_frame = pd.concat([frames[fold] for fold in train_folds], ignore_index=True)
    train_embeddings = np.concatenate([embeddings[fold] for fold in train_folds], axis=0)
    val_frame = frames[inner].reset_index(drop=True)
    val_embeddings = embeddings[inner]
    run_args = stage1_args(template_args, inner_dir, args.seed)
    if args.quiet_progress:
        run_args.quiet_progress = True
    set_seed(run_args.seed)
    config = next(config for config in CONFIGS if config.name == "qhead_qfl_only")
    print(
        f"[NESTED STAGE1] outer={outer} inner={inner} train_folds={train_folds} "
        f"train={len(train_frame)} val={len(val_frame)}"
    )
    scored, metrics = train_config(
        config,
        train_frame,
        val_frame,
        train_embeddings,
        val_embeddings,
        run_args,
        device,
        inner_dir,
        pred_dir,
    )
    scored.to_csv(score_path, index=False)
    (inner_dir / "best.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    del train_frame, train_embeddings, val_frame, val_embeddings, scored
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_path


def run_outer_cascade(
    outer: int,
    inner_scores: list[Path],
    stage1_root: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> Path:
    cascade_dir = out_root / f"outer_{outer:02d}" / "cascade"
    summary_path = cascade_dir / "eval_summary.csv"
    if summary_path.exists():
        print(f"[INFO] Reusing nested cascade outer={outer}: {summary_path}")
        return summary_path
    outer_score = (
        stage1_root
        / f"fold_{outer:02d}"
        / "cache"
        / "val_scores_qhead_qfl_only.csv"
    )
    command = [
        sys.executable,
        str(ROOT / "dev" / "train_cascade_refiner.py"),
        "--train-stage1-csv",
        *map(str, inner_scores),
        "--eval-stage1-csv",
        str(outer_score),
        "--eval-score-col",
        "quality_score",
        "--out-dir",
        str(cascade_dir),
        "--cache-dir",
        str(cascade_dir / "cache"),
        "--epochs",
        "20",
        "--eval-epochs",
        "20",
        "--alpha",
        "0.2",
        "--residual-gate",
        "score",
        "--batch-size",
        "2048",
        "--max-train-samples",
        "100000",
        "--repr-batch-size",
        "32",
        "--num-workers",
        "8",
        "--seed",
        str(args.seed),
        "--quiet-progress",
    ]
    print(f"[NESTED CASCADE] outer={outer} inner_scores={len(inner_scores)}")
    subprocess.run(command, cwd=ROOT, check=True)
    return summary_path


def aggregate(out_root: Path, outer_folds: list[int]) -> pd.DataFrame:
    rows = []
    for outer in outer_folds:
        summary_path = out_root / f"outer_{outer:02d}" / "cascade" / "eval_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        baseline = summary[(summary["epoch"] == 0) & (summary["alpha"] == 0.0)].iloc[0]
        cascade = summary[(summary["epoch"] == 20) & (summary["alpha"] == 0.2)].iloc[0]
        for variant, row in (("stage1", baseline), ("nested_cascade", cascade)):
            values = row.to_dict()
            values.update({"outer_fold": outer, "variant": variant})
            rows.append(values)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    detail.to_csv(out_root / "nested_detail.csv", index=False)
    metrics = ["mAP", "AP@0.1", "AP@0.3", "AP@0.5", "AP@0.7", "recall@0.7"]
    summary = detail.groupby("variant", as_index=False)[metrics].mean()
    summary.to_csv(out_root / "nested_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    invalid = sorted(set(args.outer_folds) - set(range(5)))
    if invalid:
        raise ValueError(f"Invalid outer folds: {invalid}")
    fold_root = resolve(args.fold_root)
    stage1_root = resolve(args.stage1_root)
    out_root = resolve(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    template = load_checkpoint(stage1_root / "fold_00" / "qhead_qfl_only.pt", device)["args"]
    frames, embeddings = load_fold_parts(fold_root, stage1_root)

    inner_outputs: dict[int, list[Path]] = {}
    for outer in args.outer_folds:
        outputs = []
        for inner in range(5):
            if inner == outer:
                continue
            outputs.append(
                train_inner_stage1(
                    outer,
                    inner,
                    frames,
                    embeddings,
                    template,
                    args,
                    device,
                    out_root,
                )
            )
        inner_outputs[outer] = outputs

    del frames, embeddings
    gc.collect()
    for outer in args.outer_folds:
        run_outer_cascade(outer, inner_outputs[outer], stage1_root, out_root, args)
    summary = aggregate(out_root, args.outer_folds)
    print("\n[NESTED CV SUMMARY]")
    print(summary.to_string(index=False) if not summary.empty else "No completed outer folds")


if __name__ == "__main__":
    main()
