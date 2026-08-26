"""Assemble recording-disjoint fold predictions in a reference proposal order."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["rec_name", "roi_id", "t_start", "t_end"]
PREDICTION_COLUMNS = ["quality_score", "refined_t_start", "refined_t_end"]
TIME_QUANTUM_US = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aligned out-of-fold scored proposal file.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--fold-scored", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = pd.read_csv(args.reference).reset_index(drop=True)
    fold_frames = [pd.read_csv(path) for path in args.fold_scored]
    predictions = pd.concat(
        [frame[KEY_COLUMNS + PREDICTION_COLUMNS] for frame in fold_frames],
        ignore_index=True,
    )
    merge_keys = ["rec_name", "roi_id", "_start_key", "_end_key"]
    for frame in (reference, predictions):
        frame["_start_key"] = (frame["t_start"] / TIME_QUANTUM_US).round().astype("int64")
        frame["_end_key"] = (frame["t_end"] / TIME_QUANTUM_US).round().astype("int64")
    duplicate = predictions.duplicated(merge_keys, keep=False)
    if duplicate.any():
        raise ValueError(f"Fold predictions overlap on {int(duplicate.sum())} proposal rows")

    aligned = reference[merge_keys].merge(
        predictions[merge_keys + PREDICTION_COLUMNS],
        on=merge_keys,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    missing = aligned["quality_score"].isna()
    if missing.any():
        raise ValueError(f"Missing OOF predictions for {int(missing.sum())} reference rows")
    out = reference.drop(columns=["_start_key", "_end_key"]).copy()
    for column in PREDICTION_COLUMNS:
        out[column] = aligned[column].to_numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"[RESULTADO] rows={len(out)} folds={len(fold_frames)} output={output}")


if __name__ == "__main__":
    main()
