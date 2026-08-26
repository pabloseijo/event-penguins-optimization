"""Apply a label-free recording-level safety guard to cached TTA scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dev.train_atsn_lpft import evaluate_scored, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropy guard for episodic TTA scores.")
    parser.add_argument("--scored-proposals", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-score-col", default="cnn_score_base")
    parser.add_argument("--adapted-score-col", default="cnn_score")
    parser.add_argument("--entropy-margin", type=float, default=0.0)
    parser.add_argument(
        "--guard-only",
        action="store_true",
        help="Evaluate only the guarded scores when base and adapted metrics are already cached.",
    )
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--min-ed-score", type=float, nargs="+", default=[0.02])
    parser.add_argument("--soft-nms-sigma", type=float, default=0.25)
    parser.add_argument("--soft-nms-score-threshold", type=float, default=0.001)
    parser.add_argument("--duration-dmax", type=float, default=60.0)
    parser.add_argument("--duration-sigma", type=float, default=20.0)
    parser.add_argument("--tiou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    return parser.parse_args()


def binary_entropy(scores: np.ndarray) -> float:
    probabilities = np.clip(scores.astype(np.float64), 1e-8, 1.0 - 1e-8)
    return float(
        np.mean(
            -probabilities * np.log(probabilities)
            - (1.0 - probabilities) * np.log(1.0 - probabilities)
        )
    )


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(resolve_path(args.scored_proposals)).reset_index(drop=True)
    for column in (args.base_score_col, args.adapted_score_col):
        if column not in scored:
            raise ValueError(f"Missing score column: {column}")

    guarded = np.zeros(len(scored), dtype=np.float64)
    decisions = []
    for recording, group in scored.groupby("rec_name", sort=True):
        base = group[args.base_score_col].to_numpy(dtype=np.float64)
        adapted = group[args.adapted_score_col].to_numpy(dtype=np.float64)
        entropy_base = binary_entropy(base)
        entropy_adapted = binary_entropy(adapted)
        accept = entropy_adapted <= entropy_base - args.entropy_margin
        guarded[group.index.to_numpy(dtype=np.int64)] = adapted if accept else base
        decisions.append(
            {
                "rec_name": recording,
                "entropy_base": entropy_base,
                "entropy_adapted": entropy_adapted,
                "entropy_delta": entropy_adapted - entropy_base,
                "accept_adaptation": bool(accept),
                "proposals": int(len(group)),
            }
        )

    scored["cnn_score_guarded"] = guarded
    scored.to_csv(out_dir / "scored_guarded.csv", index=False)
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(out_dir / "decisions.csv", index=False)
    ann_path = resolve_path(args.ann_path)
    rows = []
    score_variants = [("guarded", "cnn_score_guarded")]
    if not args.guard_only:
        score_variants = [
            ("base", args.base_score_col),
            ("adapted", args.adapted_score_col),
            *score_variants,
        ]
    for label, column in score_variants:
        frame = scored.copy()
        frame["cnn_score"] = frame[column]
        metrics = evaluate_scored(frame, ann_path, args, pred_dir, label)
        rows.append({"variant": label, **metrics})
    summary = pd.DataFrame(rows).sort_values("mAP", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    payload = {
        "accepted_recordings": int(decision_frame["accept_adaptation"].sum()),
        "total_recordings": int(len(decision_frame)),
        "entropy_margin": args.entropy_margin,
        "best": summary.iloc[0].to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(decision_frame.to_string(index=False))
    print(summary[["variant", "mAP", "AP@0.5", "AP@0.7"]].to_string(index=False))


if __name__ == "__main__":
    main()
