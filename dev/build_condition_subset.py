"""Build aligned proposal/representation subsets from acquisition metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_quality_head import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter aligned proposals and cached representations by recording metadata."
    )
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--representations", required=True)
    parser.add_argument("--recording-info", default="config/annotations/recording_info.csv")
    parser.add_argument("--out-proposals", required=True)
    parser.add_argument("--out-representations", required=True)
    parser.add_argument("--recordings", nargs="*", default=[])
    parser.add_argument("--roi-group-id", type=int, default=None)
    parser.add_argument("--precipitation", action="store_true")
    parser.add_argument("--night", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def selected_recordings(info: pd.DataFrame, args: argparse.Namespace) -> set[str]:
    selected = info.copy()
    if args.recordings:
        selected = selected[selected["timestamp"].isin(args.recordings)]
    if args.roi_group_id is not None:
        selected = selected[selected["roi_group_id"].eq(args.roi_group_id)]
    if args.precipitation:
        selected = selected[selected["precipitation"].fillna(0).eq(1)]
    if args.night:
        selected = selected[selected["night"].fillna(0).eq(1)]
    if not (args.recordings or args.roi_group_id is not None or args.precipitation or args.night):
        raise ValueError("At least one recording condition is required")
    return set(selected["timestamp"].astype(str))


def main() -> None:
    args = parse_args()
    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    representations = np.load(resolve(args.representations), allow_pickle=False)
    embeddings = representations["embeddings"]
    logits = representations["logits"]
    if len(proposals) != len(embeddings) or len(proposals) != len(logits):
        raise ValueError(
            "Proposals and representations are not aligned: "
            f"{len(proposals)}, {len(embeddings)}, {len(logits)}"
        )

    info = pd.read_csv(resolve(args.recording_info))
    recordings = selected_recordings(info, args)
    mask = proposals["rec_name"].astype(str).isin(recordings).to_numpy()
    if not mask.any():
        raise ValueError(f"No proposals match recordings: {sorted(recordings)}")

    out_proposals = resolve(args.out_proposals)
    out_representations = resolve(args.out_representations)
    out_proposals.parent.mkdir(parents=True, exist_ok=True)
    out_representations.parent.mkdir(parents=True, exist_ok=True)
    proposals.loc[mask].reset_index(drop=True).to_csv(out_proposals, index=False)
    np.savez(
        out_representations,
        embeddings=embeddings[mask],
        logits=logits[mask],
    )
    counts = proposals.loc[mask, "rec_name"].value_counts().sort_index()
    print(f"[RESULTADO] proposals={int(mask.sum())} recordings={len(counts)}")
    print(counts.to_string())
    print(f"[RESULTADO] proposals_path={out_proposals}")
    print(f"[RESULTADO] representations_path={out_representations}")


if __name__ == "__main__":
    main()
