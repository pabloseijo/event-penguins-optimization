"""Aggregate fixed-epoch adaptation metrics across recording folds."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    manifest = pd.read_csv(resolve(args.manifest)).set_index("fold")
    rows = []
    for fold in args.folds:
        history = pd.read_csv(resolve(args.root) / f"fold_{fold:02d}" / "history.csv")
        selected = history[history["epoch"] == args.epoch]
        if len(selected) != 1:
            raise ValueError(f"Expected epoch {args.epoch} once for fold {fold}")
        row = selected.iloc[0].to_dict()
        row["fold"] = fold
        row["val_ed_instances"] = int(manifest.loc[fold, "val_ed_instances"])
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("fold")
    weights = frame["val_ed_instances"].to_numpy(np.float64)
    summary = {
        "epoch": args.epoch,
        "mean_mAP": float(frame["mAP"].mean()),
        "weighted_mAP": float(np.average(frame["mAP"], weights=weights)),
        "worst_fold_mAP": float(frame["mAP"].min()),
        "mean_AP@0.1": float(frame["AP@0.1"].mean()),
        "mean_AP@0.3": float(frame["AP@0.3"].mean()),
        "mean_AP@0.5": float(frame["AP@0.5"].mean()),
        "mean_AP@0.7": float(frame["AP@0.7"].mean()),
    }
    out_root = resolve(args.root)
    frame.to_csv(out_root / f"metrics_epoch_{args.epoch:03d}.csv", index=False)
    (out_root / f"summary_epoch_{args.epoch:03d}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(frame[["fold", "mAP", "AP@0.7", "val_ed_instances"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
