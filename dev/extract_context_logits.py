"""Extract and cache frozen-ATSN logits for transformed temporal windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from dev.train_quality_head import ROOT, collect_or_load_context_features, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache ATSN logits for temporal context windows.")
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--context-dir", required=True)
    parser.add_argument(
        "--context-roles",
        nargs="+",
        choices=["previous", "next", "expanded", "center", "fixed_center"],
    )
    parser.add_argument("--context-window-scale", type=float, default=1.0)
    parser.add_argument("--context-feature-mode", choices=["logits", "embeddings"], default="logits")
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--repr-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    embeddings, logits = collect_or_load_context_features(
        proposals,
        resolve(args.context_dir),
        args,
        device,
    )
    print(
        f"[RESULTADO] proposals={len(proposals)} mode={args.context_feature_mode} "
        f"embeddings={list(embeddings)} logits={list(logits)}"
    )


if __name__ == "__main__":
    main()
