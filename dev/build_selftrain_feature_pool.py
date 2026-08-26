"""Merge source features with a few pseudo-labeled target recordings for self-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="tmp/temporalmaxer_continuous/source_features_v1")
    parser.add_argument("--target-dir", default="tmp/temporalmaxer_continuous/test_features_v1")
    parser.add_argument(
        "--target-recordings", nargs="+", default=["22-01-15_05-58-00", "22-01-15_11-48-00"]
    )
    parser.add_argument(
        "--out-dir", default="tmp/temporalmaxer_continuous/source_selftrain_v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve(args.source_dir)
    target_dir = resolve(args.target_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_meta = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    target_meta = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert source_meta["feature_dim"] == target_meta["feature_dim"]
    assert source_meta["grid_stride_s"] == target_meta["grid_stride_s"]

    source_features = np.load(source_dir / "frame_features.npy", mmap_mode="r")
    target_features = np.load(target_dir / "frame_features.npy", mmap_mode="r")

    source_sequences = pd.read_csv(source_dir / "sequences.csv")
    target_sequences = pd.read_csv(target_dir / "sequences.csv")
    selected = target_sequences[target_sequences["rec_name"].isin(args.target_recordings)].copy()
    if selected.empty:
        raise ValueError("No matching target recordings found in target sequences.csv")

    offset_shift = int(len(source_features))
    selected["offset"] = selected["offset"].astype(int) + offset_shift
    selected["split"] = "train"

    merged_features = np.concatenate([np.asarray(source_features), np.asarray(target_features)], axis=0)
    merged_sequences = pd.concat([source_sequences, selected], ignore_index=True)
    merged_sequences["sequence_index"] = range(len(merged_sequences))

    np.save(out_dir / "frame_features.npy", merged_features)
    merged_sequences.to_csv(out_dir / "sequences.csv", index=False)
    metadata = dict(source_meta)
    metadata["num_points"] = int(len(merged_features))
    metadata["num_sequences"] = int(len(merged_sequences))
    metadata["recordings"] = sorted(set(source_meta["recordings"]) | set(args.target_recordings))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[DONE] {out_dir}: {len(merged_sequences)} sequences "
        f"({len(source_sequences)} source + {len(selected)} pseudo-labeled target), "
        f"{len(merged_features)} points"
    )


if __name__ == "__main__":
    main()
