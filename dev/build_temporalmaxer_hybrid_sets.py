"""Build source-CV and test sets selected by frozen CNN or GroupDRO scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]
OUTPUT_COLUMNS = KEY_COLUMNS + ["score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groupdro-cv-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--groupdro-test-root", default="tmp/quality_head/family_groupdro_test")
    parser.add_argument("--source-prefix", default="tmp/temporalmaxer_dense/screened_folds_r5/master_proposals.csv")
    parser.add_argument("--test-prefix", default="tmp/temporalmaxer_dense/screened_test_proposals.csv")
    parser.add_argument("--source-out", default="tmp/temporalmaxer_dense/hybrid_source")
    parser.add_argument("--test-out", default="tmp/temporalmaxer_dense/hybrid_test_proposals.csv")
    parser.add_argument("--cnn-threshold", type=float, default=0.09)
    parser.add_argument("--quality-threshold", type=float, default=0.1)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def stable_index(frame: pd.DataFrame) -> pd.MultiIndex:
    keys = frame[KEY_COLUMNS].copy()
    keys["rec_name"] = keys["rec_name"].astype(str)
    keys["roi_id"] = keys["roi_id"].astype(str)
    for column in ("t_start", "t_end"):
        keys[column] = np.rint(keys[column].to_numpy(dtype=np.float64) * 1_000.0).astype(np.int64)
    return pd.MultiIndex.from_frame(keys)


def unique(frame: pd.DataFrame) -> pd.DataFrame:
    index = stable_index(frame)
    return frame.loc[~index.duplicated()].reset_index(drop=True)


def prefix_order(frame: pd.DataFrame, prefix_path: Path) -> pd.DataFrame:
    frame = unique(frame)
    index = stable_index(frame)
    prefix = pd.read_csv(prefix_path)
    positions = index.get_indexer(stable_index(prefix))
    if np.any(positions < 0):
        raise ValueError(f"Prefix {prefix_path} is not contained in the hybrid set")
    remaining = np.ones(len(frame), dtype=bool)
    remaining[positions] = False
    return pd.concat((frame.iloc[positions], frame.loc[remaining]), ignore_index=True)


def build_source(args: argparse.Namespace) -> None:
    root = resolve(args.groupdro_cv_root)
    out = resolve(args.source_out)
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for fold in range(5):
        path = root / f"fold_{fold:02d}" / "cache" / "val_scores_qhead_qfl_only.csv"
        frame = pd.read_csv(path, usecols=OUTPUT_COLUMNS + ["cnn_score", "quality_score"])
        keep = (frame["cnn_score"] >= args.cnn_threshold) | (
            frame["quality_score"] >= args.quality_threshold
        )
        frame = unique(frame.loc[keep, OUTPUT_COLUMNS])
        fold_dir = out / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(fold_dir / "val_proposals.csv", index=False)
        parts.append(frame)
        print(f"[SOURCE {fold:02d}] val={len(frame)}")
    master = prefix_order(pd.concat(parts, ignore_index=True), resolve(args.source_prefix))
    master.to_csv(out / "master_proposals.csv", index=False)
    print(f"[SOURCE] master={len(master)}")


def build_test(args: argparse.Namespace) -> None:
    root = resolve(args.groupdro_test_root)
    frames = []
    for fold in range(5):
        path = (
            root
            / f"fold_{fold:02d}"
            / "cache"
            / f"test_groupdro_fold_{fold:02d}_scores_qhead_qfl_only.csv"
        )
        frame = pd.read_csv(path, usecols=OUTPUT_COLUMNS + ["cnn_score", "quality_score"])
        frames.append(frame)
    base = frames[0]
    base_index = stable_index(base)
    for fold, frame in enumerate(frames[1:], start=1):
        if not stable_index(frame).equals(base_index):
            raise ValueError(f"Test GroupDRO fold {fold} is misaligned")
    quality = np.mean([frame["quality_score"].to_numpy(dtype=np.float64) for frame in frames], axis=0)
    keep = (base["cnn_score"].to_numpy(dtype=np.float64) >= args.cnn_threshold) | (
        quality >= args.quality_threshold
    )
    test = prefix_order(base.loc[keep, OUTPUT_COLUMNS], resolve(args.test_prefix))
    out = resolve(args.test_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(out, index=False)
    print(f"[TEST] proposals={len(test)}")


def main() -> None:
    args = parse_args()
    build_source(args)
    build_test(args)


if __name__ == "__main__":
    main()
