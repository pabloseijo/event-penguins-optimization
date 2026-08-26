"""Build recording-level train/validation folds for proposal experiments.

This utility avoids choosing scoring thresholds or checkpoints on a single tiny
validation split. It groups by recording, keeps the official test split out of
the folds by default, and balances folds by the number of ED instances.

Run from event_penguins/:
    python dev/build_recording_folds.py \
        --proposals tmp/debug/proposals_all.csv \
        --out-dir tmp/cv/recording_folds
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create recording-level folds from a proposal CSV.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--recording-info", default="config/annotations/recording_info.csv")
    parser.add_argument("--repr", default=None, help="Optional .npz with embeddings/logits aligned with --proposals.")
    parser.add_argument("--out-dir", default="tmp/cv/recording_folds")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_recording_splits(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["timestamp"]: row["split"] for row in csv.DictReader(f)}


def count_ed_instances(ann_path: Path, min_duration_s: float = 2.0) -> dict[str, int]:
    with open(ann_path, encoding="utf-8") as f:
        db = json.load(f)["database"]
    counts = {}
    for rec, value in db.items():
        total = 0
        for roi, anns in value.get("annotations", {}).items():
            if roi == "null":
                continue
            for ann in anns:
                if ann["label"] != "ed":
                    continue
                start, end = map(float, ann["segment"])
                if end - start >= min_duration_s:
                    total += 1
        counts[rec] = total
    return counts


def balanced_recording_folds(records: list[str], weights: dict[str, int], n_folds: int, seed: int) -> list[list[str]]:
    shuffled = pd.Series(records).sample(frac=1.0, random_state=seed).tolist()
    ordered = sorted(shuffled, key=lambda rec: weights.get(rec, 0), reverse=True)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    fold_weights = [0 for _ in range(n_folds)]
    for rec in ordered:
        idx = min(range(n_folds), key=lambda i: (fold_weights[i], len(folds[i])))
        folds[idx].append(rec)
        fold_weights[idx] += int(weights.get(rec, 0))
    return [sorted(fold) for fold in folds]


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proposals = pd.read_csv(resolve(args.proposals))
    proposals["source_index"] = proposals.index.to_numpy(dtype=np.int64)
    repr_data = None
    if args.repr:
        repr_data = np.load(resolve(args.repr), allow_pickle=True)
        if len(repr_data["embeddings"]) != len(proposals) or len(repr_data["logits"]) != len(proposals):
            raise ValueError("--repr arrays must be aligned with --proposals length.")
    splits = load_recording_splits(resolve(args.recording_info))
    ed_counts = count_ed_instances(resolve(args.ann_path))

    proposals["official_split"] = proposals["rec_name"].map(splits)
    fold_df = proposals[proposals["official_split"].isin(args.include_splits)].copy()
    if fold_df.empty:
        raise ValueError("No proposals remain after --include-splits filtering.")

    records = sorted(fold_df["rec_name"].unique())
    n_folds = min(max(2, args.folds), len(records))
    folds = balanced_recording_folds(records, ed_counts, n_folds, args.seed)

    manifest_rows = []
    for fold_idx, val_records in enumerate(folds):
        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        val_set = set(val_records)
        train = fold_df[~fold_df["rec_name"].isin(val_set)].drop(columns=["official_split"])
        val = fold_df[fold_df["rec_name"].isin(val_set)].drop(columns=["official_split"])
        train_path = fold_dir / "train_proposals.csv"
        val_path = fold_dir / "val_proposals.csv"
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        train_repr_path = ""
        val_repr_path = ""
        if repr_data is not None:
            train_idx = train["source_index"].to_numpy(dtype=np.int64)
            val_idx = val["source_index"].to_numpy(dtype=np.int64)
            train_repr_path = str(fold_dir / "train_repr.npz")
            val_repr_path = str(fold_dir / "val_repr.npz")
            np.savez(
                train_repr_path,
                embeddings=repr_data["embeddings"][train_idx],
                logits=repr_data["logits"][train_idx],
            )
            np.savez(
                val_repr_path,
                embeddings=repr_data["embeddings"][val_idx],
                logits=repr_data["logits"][val_idx],
            )
        manifest_rows.append(
            {
                "fold": fold_idx,
                "train_records": len(train["rec_name"].unique()),
                "val_records": len(val_records),
                "train_proposals": len(train),
                "val_proposals": len(val),
                "val_ed_instances": sum(ed_counts.get(rec, 0) for rec in val_records),
                "val_record_names": " ".join(val_records),
                "train_path": str(train_path),
                "val_path": str(val_path),
                "train_repr_path": train_repr_path,
                "val_repr_path": val_repr_path,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(manifest.to_string(index=False))
    print(f"[INFO] Folds written to {out_dir}")


if __name__ == "__main__":
    main()
