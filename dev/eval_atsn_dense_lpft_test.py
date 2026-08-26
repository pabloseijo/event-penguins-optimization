"""Evaluate the source-CV-approved surgical ATSN adaptation once on test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dev.eval_temporalmaxer_groupdro_cv import add_ranking_fusions
from dev.train_atsn_temporalmaxer_lpft import (
    load_model,
    outputs_to_scored,
    score_raw_model,
)
from dev.train_temporalmaxer_dense import (
    evaluate_variant,
    load_cache,
    map_to_master,
    softmax_ed,
    stable_proposal_index,
)


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-proposals",
        default="tmp/temporalmaxer_dense/hybrid_test_proposals.csv",
    )
    parser.add_argument(
        "--cache-dir", default="tmp/temporalmaxer_dense/hybrid_test_cache"
    )
    parser.add_argument(
        "--base-checkpoint-root",
        default="tmp/temporalmaxer_dense/screened_cv_blend075_erm",
    )
    parser.add_argument(
        "--fold-zero-root",
        default="tmp/temporalmaxer_dense/atsn_dense_firstblock_pilot/fold_00",
    )
    parser.add_argument(
        "--cv-root", default="tmp/temporalmaxer_dense/atsn_dense_firstblock_cv"
    )
    parser.add_argument(
        "--groupdro-root", default="tmp/quality_head/family_groupdro_test"
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_dense/atsn_dense_firstblock_test"
    )
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-block", default="first")
    parser.add_argument("--freeze-detector", action="store_true", default=True)
    parser.add_argument("--event-drop-prob", type=float, default=0.0)
    parser.add_argument("--sample-duration-jitter", type=float, default=0.0)
    parser.add_argument("--max-boundary-delta", type=float, default=0.75)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--pre-nms-topk-per-roi", type=int, default=1000)
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument(
        "--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7]
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_numpy(array: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array)
    temporary.replace(path)


def assert_aligned(frame: pd.DataFrame, proposals: pd.DataFrame, label: str) -> None:
    if len(frame) != len(proposals) or not stable_proposal_index(frame).equals(
        stable_proposal_index(proposals)
    ):
        raise ValueError(f"{label} is not aligned with the test proposal master")


def fold_checkpoint(args: argparse.Namespace, fold: int) -> Path:
    root = resolve(args.fold_zero_root) if fold == 0 else resolve(args.cv_root) / f"fold_{fold:02d}"
    return root / f"epoch_{args.epoch:02d}.pt"


def score_fold(
    fold: int,
    proposals: pd.DataFrame,
    master_logits: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> None:
    scored_path = out_dir / f"scored_fold_{fold:02d}.csv"
    if scored_path.exists():
        scored = pd.read_csv(scored_path)
        assert_aligned(scored, proposals, f"Fine-tuned fold {fold}")
        print(f"[FOLD {fold:02d}] reutilizado", flush=True)
        return
    args.fold = fold
    model, _ = load_model(args, device)
    checkpoint = torch.load(
        fold_checkpoint(args, fold), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    outputs = score_raw_model(
        model,
        proposals,
        args,
        out_dir / f"raw_fold_{fold:02d}",
        device,
    )
    scored = outputs_to_scored(
        proposals,
        outputs,
        softmax_ed(np.asarray(master_logits)),
        args,
    )
    atomic_csv(scored, scored_path)
    print(f"[FOLD {fold:02d}] puntuado", flush=True)


def load_groupdro_ensemble(
    proposals: pd.DataFrame, args: argparse.Namespace, cache_dir: Path
) -> np.ndarray:
    frames = []
    for fold in range(5):
        cache_path = cache_dir / f"groupdro_quality_fold_{fold:02d}.npy"
        if cache_path.exists():
            quality = np.load(cache_path)
            if len(quality) != len(proposals):
                raise ValueError(f"Invalid cached GroupDRO fold {fold}")
            frames.append(quality)
            continue
        path = (
            resolve(args.groupdro_root)
            / f"fold_{fold:02d}"
            / "cache"
            / f"test_groupdro_fold_{fold:02d}_scores_qhead_qfl_only.csv"
        )
        frame = pd.read_csv(path, usecols=KEY_COLUMNS + ["quality_score"])
        positions = map_to_master(frame, proposals)
        quality = frame.iloc[positions]["quality_score"].to_numpy(dtype=np.float64)
        atomic_numpy(quality, cache_path)
        frames.append(
            quality
        )
    return np.mean(frames, axis=0)


def ensemble_test_frames(
    proposals: pd.DataFrame,
    frames: list[pd.DataFrame],
    quality_score: np.ndarray,
) -> pd.DataFrame:
    output = proposals.copy()
    output["quality_score"] = quality_score
    for column in ("dense_score", "brem_score", "dense_point"):
        output[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )
    for column in ("delta_t_start", "delta_t_end"):
        output[column] = np.mean(
            [frame[column].to_numpy(dtype=np.float64) for frame in frames], axis=0
        )
    output = add_ranking_fusions(output)
    output["blend050_t_start"] = 0.5 * (
        output["t_start"].to_numpy(dtype=np.float64)
        + output["delta_t_start"].to_numpy(dtype=np.float64)
    )
    output["blend050_t_end"] = 0.5 * (
        output["t_end"].to_numpy(dtype=np.float64)
        + output["delta_t_end"].to_numpy(dtype=np.float64)
    )
    return output


def aggregate_and_evaluate(
    proposals: pd.DataFrame, args: argparse.Namespace, out_dir: Path
) -> None:
    frames = []
    for fold in range(5):
        path = out_dir / f"scored_fold_{fold:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        assert_aligned(frame, proposals, f"Fine-tuned fold {fold}")
        frames.append(frame)
    ensemble = ensemble_test_frames(
        proposals,
        frames,
        load_groupdro_ensemble(proposals, args, out_dir),
    )
    atomic_csv(ensemble, out_dir / "scored_ensemble.csv")
    row = evaluate_variant(
        ensemble,
        "qhead_brem_score",
        "blend050",
        "source_cv_approved_firstblock_test",
        args,
        out_dir / "predictions",
    )
    atomic_csv(pd.DataFrame([row]), out_dir / "metrics.csv")
    print(pd.DataFrame([row]).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.score_only and args.aggregate_only:
        raise ValueError("Choose at most one of --score-only and --aggregate-only")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    proposals = pd.read_csv(resolve(args.master_proposals)).reset_index(drop=True)
    _, master_logits, metadata = load_cache(resolve(args.cache_dir))
    if len(master_logits) != len(proposals):
        raise ValueError("Test cache and proposal master have different lengths")
    args.num_segments = int(metadata["num_segments"])
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        for fold in args.folds:
            score_fold(fold, proposals, master_logits, args, out_dir, device)
    if not args.score_only:
        aggregate_and_evaluate(proposals, args, out_dir)


if __name__ == "__main__":
    main()
