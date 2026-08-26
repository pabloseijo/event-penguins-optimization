"""Build compact lattice folds using an operational frozen-CNN score screen."""

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
    parser.add_argument("--qhead-cv-root", default="tmp/quality_head/family_groupdro_cv")
    parser.add_argument("--source-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--cnn-threshold", type=float, default=0.1)
    parser.add_argument("--out-dir", default="tmp/temporalmaxer_dense/screened_folds_r5")
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


def read_screened(path: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=OUTPUT_COLUMNS + ["cnn_score"])
    frame = frame.loc[frame["cnn_score"] >= threshold, OUTPUT_COLUMNS].reset_index(drop=True)
    index = stable_index(frame)
    if index.has_duplicates:
        frame = frame.loc[~index.duplicated()].reset_index(drop=True)
    return frame


def main() -> None:
    args = parse_args()
    source_root = resolve(args.qhead_cv_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.source_manifest)).copy()
    folds: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for fold in manifest["fold"].astype(int):
        cache = source_root / f"fold_{fold:02d}" / "cache"
        train = read_screened(cache / "train_quality_labels.csv", args.cnn_threshold)
        val = read_screened(cache / "val_quality_labels.csv", args.cnn_threshold)
        if set(train["rec_name"].astype(str)) & set(val["rec_name"].astype(str)):
            raise ValueError(f"Fold {fold} is not recording-disjoint")
        fold_dir = out_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "train_proposals.csv"
        val_path = fold_dir / "val_proposals.csv"
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        folds[fold] = (train, val)
        row = manifest["fold"].astype(int) == fold
        manifest.loc[row, "train_proposals"] = len(train)
        manifest.loc[row, "val_proposals"] = len(val)
        manifest.loc[row, "train_path"] = str(train_path)
        manifest.loc[row, "val_path"] = str(val_path)
        print(f"[FOLD {fold:02d}] train={len(train)} val={len(val)}")

    master = pd.concat(folds[0], ignore_index=True)
    master_index = stable_index(master)
    master = master.loc[~master_index.duplicated()].reset_index(drop=True)
    master_index = stable_index(master)
    for fold, (train, val) in folds.items():
        for split, frame in (("train", train), ("val", val)):
            missing = int((master_index.get_indexer(stable_index(frame)) < 0).sum())
            if missing:
                raise ValueError(f"Fold {fold} {split}: {missing} proposals missing")

    master.to_csv(out_dir / "master_proposals.csv", index=False)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(
        f"[RESULTADO] master={len(master)} cnn_threshold={args.cnn_threshold:g} "
        f"path={out_dir / 'master_proposals.csv'}"
    )


if __name__ == "__main__":
    main()
