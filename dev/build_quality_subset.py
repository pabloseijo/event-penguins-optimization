"""Materialize a stratified proposal/representation subset for quality-head training.

The subset follows the same sampling policy as ``train_quality_head.py`` and
keeps proposal rows aligned with cached ATSN embeddings and logits. It is useful
when an experiment needs additional expensive per-proposal features but the
trainer will only consume a bounded stratified sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_quality_head import CONFIGS, ROOT, sample_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aligned quality-head training subset.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--representations", required=True)
    parser.add_argument("--out-proposals", required=True)
    parser.add_argument("--out-representations", required=True)
    parser.add_argument("--out-indices", default=None)
    parser.add_argument("--config", default="qhead_qfl_only", choices=[cfg.name for cfg in CONFIGS])
    parser.add_argument("--max-samples", type=int, default=140000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hard-neg-top-frac", type=float, default=1.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    labels = pd.read_csv(
        resolve(args.labels),
        usecols=["sample_kind", "hardness_score"],
    ).reset_index(drop=True)
    representations = np.load(resolve(args.representations), allow_pickle=False)
    embeddings = representations["embeddings"]
    logits = representations["logits"]
    lengths = {len(proposals), len(labels), len(embeddings), len(logits)}
    if len(lengths) != 1:
        raise ValueError(
            "Proposals, labels, embeddings and logits must have the same length: "
            f"{len(proposals)}, {len(labels)}, {len(embeddings)}, {len(logits)}"
        )

    config = next(cfg for cfg in CONFIGS if cfg.name == args.config)
    indices = sample_indices(
        labels,
        config,
        max_samples=args.max_samples,
        seed=args.seed,
        hard_neg_top_frac=args.hard_neg_top_frac,
    )
    subset = proposals.iloc[indices].reset_index(drop=True)

    out_proposals = resolve(args.out_proposals)
    out_representations = resolve(args.out_representations)
    out_proposals.parent.mkdir(parents=True, exist_ok=True)
    out_representations.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_proposals, index=False)
    np.savez(
        out_representations,
        embeddings=embeddings[indices],
        logits=logits[indices],
    )
    if args.out_indices:
        out_indices = resolve(args.out_indices)
        out_indices.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_indices, indices)

    counts = labels.iloc[indices]["sample_kind"].value_counts().to_dict()
    print(f"[RESULTADO] n={len(indices)} counts={counts}")
    print(f"[RESULTADO] proposals={out_proposals}")
    print(f"[RESULTADO] representations={out_representations}")


if __name__ == "__main__":
    main()
