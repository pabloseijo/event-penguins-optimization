"""Build an operational test proposal set using only a frozen-CNN score screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels", required=True)
    parser.add_argument("--cnn-threshold", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    args = parse_args()
    columns = KEY_COLUMNS + ["score", "cnn_score"]
    frame = pd.read_csv(resolve(args.quality_labels), usecols=columns)
    frame = frame.loc[frame["cnn_score"] >= args.cnn_threshold, KEY_COLUMNS + ["score"]]
    keys = frame[KEY_COLUMNS].copy()
    for column in ("t_start", "t_end"):
        keys[column] = np.rint(keys[column].to_numpy(dtype=np.float64) * 1_000.0).astype(np.int64)
    keep = ~pd.MultiIndex.from_frame(keys).duplicated()
    frame = frame.loc[keep].reset_index(drop=True)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[RESULTADO] test={len(frame)} cnn_threshold={args.cnn_threshold:g} path={out}")


if __name__ == "__main__":
    main()
