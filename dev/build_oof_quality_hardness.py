"""Assemble recording-disjoint quality scores for offline hard-example mining."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["rec_name", "roi_id", "t_start", "t_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-root",
        default="tmp/quality_head/family_groupdro_cv",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--out",
        default="tmp/quality_head/family_groupdro_cv/oof_quality_hardness.csv",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    args = parse_args()
    frames = []
    for fold in args.folds:
        path = (
            resolve(args.cv_root)
            / f"fold_{fold:02d}"
            / "cache"
            / "val_scores_qhead_qfl_only.csv"
        )
        frame = pd.read_csv(path, usecols=[*KEYS, "quality_score"])
        frame["oof_fold"] = fold
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True).rename(
        columns={"quality_score": "oof_quality_score"}
    )
    if output.duplicated(KEYS).any():
        raise ValueError("OOF folds contain duplicate proposal keys")
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"[INFO] OOF hardness rows={len(output)} path={out_path}")


if __name__ == "__main__":
    main()
