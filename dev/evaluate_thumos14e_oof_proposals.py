#!/usr/bin/env python3
"""Avalía reTAG e o xerador CoTAD con propostas OOF en THUMOS14-E."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.prepare_thumos14_event_corpus import THUMOS_CLASSES


TIOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.7)
BUDGETS = (20, 30, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--target-class", default="Diving")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_proposals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["rec_name", "t_start", "t_end", "score"])
    frame["rec_name"] = frame["rec_name"].astype(str)
    frame[["t_start", "t_end"]] = frame[["t_start", "t_end"]].astype(float) / 1e6
    frame["score"] = frame["score"].astype(float)
    return frame


def temporal_iou(segments: np.ndarray, target: np.ndarray) -> np.ndarray:
    intersection = np.maximum(
        np.minimum(segments[:, 1], target[1]) - np.maximum(segments[:, 0], target[0]),
        0.0,
    )
    union = segments[:, 1] - segments[:, 0] + target[1] - target[0] - intersection
    return intersection / np.maximum(union, 1e-12)


def class_metrics(
    proposals: pd.DataFrame, manifest: pd.DataFrame, label: str
) -> tuple[int, list[float]]:
    ground_truth: list[tuple[str, tuple[float, float]]] = []
    for row in manifest.itertuples(index=False):
        for annotation in json.loads(row.annotations_json):
            if annotation["label"] == label:
                ground_truth.append((str(row.video_id), tuple(annotation["segment"])))

    ranked = {
        name: group.sort_values("score", ascending=False)
        for name, group in proposals.groupby("rec_name", sort=False)
    }
    values = []
    for budget in BUDGETS:
        recalls = []
        for threshold in TIOU_THRESHOLDS:
            hits = 0
            for video_id, target in ground_truth:
                candidates = ranked.get(video_id)
                if candidates is None:
                    continue
                segments = candidates[["t_start", "t_end"]].head(budget).to_numpy(float)
                hits += bool(len(segments) and temporal_iou(segments, np.asarray(target)).max() >= threshold)
            recalls.append(hits / len(ground_truth))
        values.append(float(np.mean(recalls)))
    return len(ground_truth), values


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.expanduser().resolve()
    proposal_root = args.proposal_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = work_dir / "manifest.csv"
    fold_path = work_dir / "config" / "annotations" / "fold_manifest.csv"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest = manifest.loc[manifest["official_subset"].eq("validation")].copy()
    folds = pd.read_csv(fold_path, keep_default_na=False)

    retag_dir = proposal_root / "retag" / "validation"
    retag_path = retag_dir / "proposals.csv"
    retag_raw_path = retag_dir / "proposals.full.csv"
    retag = load_proposals(retag_path)
    retag_raw_count = len(pd.read_csv(retag_raw_path, usecols=["rec_name"]))
    oof_frames = []
    raw_oof_count = 0
    oof_videos: set[str] = set()
    fold_inputs = []
    for fold in range(5):
        fold_row = folds.loc[folds["fold"].astype(int).eq(fold)]
        if len(fold_row) != 1:
            raise ValueError(f"Esperábase unha fila para o fold {fold}")
        held_out = set(str(fold_row.iloc[0]["val_record_names"]).split())
        overlap = oof_videos & held_out
        if overlap:
            raise ValueError(f"Vídeos repetidos entre folds: {sorted(overlap)[:3]}")
        oof_videos.update(held_out)

        fold_dir = proposal_root / "eventpenguins_stage1" / args.target_class / f"fold_{fold:02d}" / "validation"
        ranked_path = fold_dir / "proposals.csv"
        raw_path = fold_dir / "proposals.full.csv"
        ranked = load_proposals(ranked_path)
        oof_frames.append(ranked.loc[ranked["rec_name"].isin(held_out)])
        raw_names = pd.read_csv(raw_path, usecols=["rec_name"])["rec_name"].astype(str)
        raw_oof_count += int(raw_names.isin(held_out).sum())
        fold_inputs.append(
            {
                "fold": fold,
                "held_out_videos": len(held_out),
                "ranked_sha256": sha256_file(ranked_path),
                "raw_sha256": sha256_file(raw_path),
            }
        )

    expected_videos = set(manifest["video_id"].astype(str))
    if oof_videos != expected_videos:
        raise ValueError(
            f"A unión OOF non coincide con validation: {len(oof_videos)} fronte a {len(expected_videos)}"
        )
    cotad = pd.concat(oof_frames, ignore_index=True)

    rows = []
    for label in THUMOS_CLASSES:
        gt_count, retag_values = class_metrics(retag, manifest, label)
        _, cotad_values = class_metrics(cotad, manifest, label)
        rows.append(
            {
                "action": label,
                "gt": gt_count,
                **{f"retag_ar{budget}": value for budget, value in zip(BUDGETS, retag_values)},
                **{f"cotad_ar{budget}": value for budget, value in zip(BUDGETS, cotad_values)},
                "delta_ar50": cotad_values[-1] - retag_values[-1],
            }
        )
    per_class = pd.DataFrame(rows)
    metric_columns = [column for column in per_class if column not in {"action", "gt"}]
    macro = {column: float(per_class[column].mean()) for column in metric_columns}
    output = pd.concat(
        [per_class, pd.DataFrame([{"action": "Macro", "gt": int(per_class["gt"].sum()), **macro}])],
        ignore_index=True,
    )
    output.to_csv(out_dir / "thumos14e_ar_validation_oof.csv", index=False, float_format="%.9f")

    deltas = per_class["delta_ar50"]
    report = {
        "protocol": "THUMOS14-E-proposal-AR-OOF-v1",
        "thresholds": TIOU_THRESHOLDS,
        "budgets": BUDGETS,
        "target_prototype_class": args.target_class,
        "validation_videos": len(oof_videos),
        "ground_truth_instances": int(per_class["gt"].sum()),
        "retag_raw_candidates": retag_raw_count,
        "cotad_raw_oof_candidates": raw_oof_count,
        "improve_tie_lose_at_50": [
            int((deltas > 0).sum()),
            int((deltas == 0).sum()),
            int((deltas < 0).sum()),
        ],
        "manifest_sha256": sha256_file(manifest_path),
        "retag_ranked_sha256": sha256_file(retag_path),
        "retag_raw_sha256": sha256_file(retag_raw_path),
        "fold_inputs": fold_inputs,
        "macro": macro,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
