"""Consensus pseudo-labels for self-training on the hard day-15 test recordings.

Only keeps detections where at least 2 of the 3 independently trained experts
(continuous, event, proposal) agree with tIoU >= --iou-threshold. Cross-expert
agreement is used instead of single-model confidence specifically because the
diagnosed failure mode in this domain is background false positives scoring
*higher* than true positives in every individual model — a single-model
confidence threshold would reinforce exactly that error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuous-pred",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_hardneg_v1/continuous_ensemble.json",
    )
    parser.add_argument(
        "--event-pred",
        default="tmp/temporalmaxer_continuous/multi_rep_fusion_test_hardneg_v1/event_ensemble.json",
    )
    parser.add_argument(
        "--proposal-pred",
        default=(
            "tmp/temporalmaxer_dense/multi_expert_boundary_voting_test/"
            "predictions/multi_median_blend050.json"
        ),
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["22-01-15_05-58-00", "22-01-15_11-48-00"],
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-agree", type=int, default=2, help="Minimum number of experts that must agree.")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Discard per-expert detections below this score before clustering.",
    )
    parser.add_argument(
        "--dedup-iou",
        type=float,
        default=0.3,
        help="NMS threshold applied to accepted consensus clusters to remove near-duplicates.",
    )
    parser.add_argument(
        "--out-path",
        default="tmp/temporalmaxer_continuous/selftrain_pseudolabels_v1/pseudo_annotations.json",
    )
    return parser.parse_args()


def segments_for(pred: dict, rec: str) -> dict[str, list[tuple[float, float, float]]]:
    return {
        roi: [(float(d["segment"][0]), float(d["segment"][1]), float(d["score"])) for d in dets]
        for roi, dets in pred["results"].get(rec, {}).items()
    }


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def consensus_segments(
    expert_dets: list[list[tuple[float, float, float]]],
    iou_threshold: float,
    min_agree: int,
    min_duration: float,
    min_score: float,
    dedup_iou: float,
) -> list[tuple[float, float]]:
    filtered = [[seg for seg in dets if seg[2] >= min_score] for dets in expert_dets]
    all_dets = [(idx, seg) for idx, dets in enumerate(filtered) for seg in dets]
    used = [False] * len(all_dets)
    clusters: list[tuple[int, float, float, float]] = []  # (n_experts, score, start, end)
    all_dets.sort(key=lambda item: item[1][2], reverse=True)
    for i, (expert_i, seg_i) in enumerate(all_dets):
        if used[i]:
            continue
        cluster = [(expert_i, seg_i)]
        used[i] = True
        for j in range(i + 1, len(all_dets)):
            if used[j]:
                continue
            expert_j, seg_j = all_dets[j]
            if expert_j == expert_i:
                continue
            if temporal_iou(seg_i[:2], seg_j[:2]) >= iou_threshold:
                cluster.append((expert_j, seg_j))
                used[j] = True
        experts_present = {e for e, _ in cluster}
        if len(experts_present) < min_agree:
            continue
        start = float(np.mean([s[0] for _, s in cluster]))
        end = float(np.mean([s[1] for _, s in cluster]))
        if end - start < min_duration:
            continue
        mean_score = float(np.mean([s[2] for _, s in cluster]))
        clusters.append((len(experts_present), mean_score, start, end))

    # NMS-style dedup: prefer clusters with more agreeing experts, then higher score.
    clusters.sort(key=lambda c: (c[0], c[1]), reverse=True)
    kept: list[tuple[float, float]] = []
    for _, _, start, end in clusters:
        if all(temporal_iou((start, end), (ks, ke)) < dedup_iou for ks, ke in kept):
            kept.append((start, end))
    return kept


def main() -> None:
    args = parse_args()
    continuous = json.loads(resolve(args.continuous_pred).read_text(encoding="utf-8"))
    event = json.loads(resolve(args.event_pred).read_text(encoding="utf-8"))
    proposal = json.loads(resolve(args.proposal_pred).read_text(encoding="utf-8"))

    database: dict[str, dict] = {}
    total_kept = 0
    for rec in args.recordings:
        cont_rois = segments_for(continuous, rec)
        event_rois = segments_for(event, rec)
        prop_rois = segments_for(proposal, rec)
        roi_ids = sorted(set(cont_rois) | set(event_rois) | set(prop_rois), key=int)
        annotations = {}
        for roi in roi_ids:
            experts = [
                cont_rois.get(roi, []),
                event_rois.get(roi, []),
                prop_rois.get(roi, []),
            ]
            kept = consensus_segments(
                experts,
                args.iou_threshold,
                args.min_agree,
                args.min_duration,
                args.min_score,
                args.dedup_iou,
            )
            annotations[roi] = [{"label": "ed", "segment": [s, e]} for s, e in kept]
            total_kept += len(kept)
        database[rec] = {"annotations": annotations}

    out_path = resolve(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"database": database}, indent=2), encoding="utf-8")
    print(f"[DONE] {out_path} pseudo-segments={total_kept}")
    for rec in args.recordings:
        n = sum(len(v) for v in database[rec]["annotations"].values())
        print(f"  {rec}: {n} pseudo-segments across {len(database[rec]['annotations'])} ROIs")


if __name__ == "__main__":
    main()
