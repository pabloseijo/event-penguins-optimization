"""Build compact recording-disjoint TemporalMaxer folds from qhead lattice caches."""

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
    parser.add_argument(
        "--qhead-cv-root",
        default="tmp/quality_head/family_groupdro_cv",
    )
    parser.add_argument(
        "--source-manifest",
        default="tmp/cv/recording_folds_r5/manifest.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp/temporalmaxer_dense/lattice_folds_r5",
    )
    parser.add_argument(
        "--prefix-master",
        default="tmp/temporalmaxer_dense/source_master.csv",
        help="Optional cached proposal master to place first for feature-cache reuse.",
    )
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


def read_lattice(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=OUTPUT_COLUMNS)
    if frame[OUTPUT_COLUMNS].isna().any().any():
        raise ValueError(f"Non-finite lattice fields in {path}")
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
    fold_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for fold in manifest["fold"].astype(int):
        cache = source_root / f"fold_{fold:02d}" / "cache"
        fold_dir = out_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = fold_dir / "train_proposals.csv"
        val_path = fold_dir / "val_proposals.csv"
        train = read_lattice(
            train_path if train_path.exists() else cache / "train_quality_labels.csv"
        )
        val = read_lattice(
            val_path if val_path.exists() else cache / "val_quality_labels.csv"
        )
        train_records = set(train["rec_name"].astype(str))
        val_records = set(val["rec_name"].astype(str))
        overlap = train_records & val_records
        if overlap:
            raise ValueError(f"Fold {fold} shares recordings: {sorted(overlap)}")
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        fold_frames[fold] = (train, val)
        print(f"[FOLD {fold:02d}] compactado train={len(train)} val={len(val)}")

    master = pd.concat(fold_frames[0], ignore_index=True)
    master_index = stable_index(master)
    master = master.loc[~master_index.duplicated()].reset_index(drop=True)
    master_index = stable_index(master)
    if not master_index.is_unique:
        raise ValueError("Master lattice is not unique")
    prefix_path = resolve(args.prefix_master)
    if prefix_path.exists():
        prefix = read_lattice(prefix_path)
        prefix_positions = master_index.get_indexer(stable_index(prefix))
        prefix_positions = prefix_positions[prefix_positions >= 0]
        if len(np.unique(prefix_positions)) != len(prefix_positions):
            raise ValueError("Prefix master contains duplicate proposal identities")
        remaining = np.ones(len(master), dtype=bool)
        remaining[prefix_positions] = False
        master = pd.concat(
            (master.iloc[prefix_positions], master.loc[remaining]),
            ignore_index=True,
        )
        master_index = stable_index(master)
        print(
            f"[INFO] Prefix reutilizable: {len(prefix_positions)}/{len(prefix)} propostas"
        )

    for fold, (train, val) in fold_frames.items():
        for split, frame in (("train", train), ("val", val)):
            missing = int((master_index.get_indexer(stable_index(frame)) < 0).sum())
            if missing:
                raise ValueError(f"Fold {fold} {split}: {missing} proposals missing from master")
        train_path = out_dir / f"fold_{fold:02d}" / "train_proposals.csv"
        val_path = out_dir / f"fold_{fold:02d}" / "val_proposals.csv"
        row = manifest["fold"].astype(int) == fold
        manifest.loc[row, "train_proposals"] = len(train)
        manifest.loc[row, "val_proposals"] = len(val)
        manifest.loc[row, "train_path"] = str(train_path)
        manifest.loc[row, "val_path"] = str(val_path)
        print(f"[FOLD {fold:02d}] validado train={len(train)} val={len(val)}")

    master_path = out_dir / "master_proposals.csv"
    master.to_csv(master_path, index=False)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"[RESULTADO] master={len(master)} path={master_path}")


if __name__ == "__main__":
    main()
