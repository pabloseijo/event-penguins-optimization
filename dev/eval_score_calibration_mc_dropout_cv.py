"""Calibration pilot: does MC Dropout give a better-calibrated confidence score?

Follows the TFM shared by Pardo (biomedical segmentation, MC Dropout/TTA/Noisy +
uncertainty-aware fusion improves ECE/Brier/NLL without moving IoU/Dice). This
script translates that idea to our continuous detector: it does NOT change which
detections are produced (mAP is unaffected by design), it only asks whether the
deterministic score or the MC Dropout mean score is a better-calibrated estimate
of "this detection is a true positive".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.train_temporalmaxer_continuous import (
    ContinuousSequenceDataset,
    apply_soft_nms,
    collate_sequences,
    load_annotations,
)
from src.evaluation import segment_iou
from src.temporalmaxer_continuous import TemporalMaxerContinuous


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", default="tmp/temporalmaxer_continuous/cv_pal_consistency_pilot_v1")
    parser.add_argument("--fold-manifest", default="tmp/cv/recording_folds_r5/manifest.csv")
    parser.add_argument("--ann-path", default="config/annotations/annotations.json")
    parser.add_argument("--out-dir", default="tmp/temporalmaxer_continuous/score_calibration_mc_dropout_v1")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--tp-iou", type=float, default=0.5)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def build_model_and_dataset(
    checkpoint: dict, args: argparse.Namespace, device: torch.device
) -> tuple[TemporalMaxerContinuous, ContinuousSequenceDataset, dict, argparse.Namespace]:
    ckpt_args = argparse.Namespace(**checkpoint["args"])
    metadata = checkpoint["metadata"]
    feature_dir = resolve(ckpt_args.feature_dir)
    sequences_path = (
        resolve(ckpt_args.sequences_path) if ckpt_args.sequences_path else feature_dir / "sequences.csv"
    )
    sequences = pd.read_csv(sequences_path)
    fold_manifest = pd.read_csv(resolve(args.fold_manifest)).set_index("fold")
    val_recordings = str(fold_manifest.loc[ckpt_args.fold, "val_record_names"]).split()
    val_sequences = sequences[sequences["rec_name"].isin(val_recordings)].copy()

    annotations = load_annotations(resolve(ckpt_args.ann_path))
    feature_path = feature_dir / ckpt_args.feature_array_name
    auxiliary_path = None
    auxiliary_mean = None
    auxiliary_std = None
    if ckpt_args.auxiliary_feature_dir:
        auxiliary_dir = resolve(ckpt_args.auxiliary_feature_dir)
        auxiliary_metadata = json.loads((auxiliary_dir / "metadata.json").read_text(encoding="utf-8"))
        auxiliary_path = auxiliary_dir / "event_stats.npy"
        auxiliary_mean = np.asarray(auxiliary_metadata["mean"], dtype=np.float32)
        auxiliary_std = np.asarray(auxiliary_metadata["std"], dtype=np.float32)
    feature_mean = None
    feature_std = None
    if ckpt_args.standardize_features:
        feature_mean = np.asarray(metadata["mean"], dtype=np.float32)
        feature_std = np.asarray(metadata["std"], dtype=np.float32)

    val_dataset = ContinuousSequenceDataset(
        feature_path,
        val_sequences,
        annotations,
        auxiliary_path,
        auxiliary_mean,
        auxiliary_std,
        ckpt_args.feature_normalization,
        feature_channel_mean=feature_mean,
        feature_channel_std=feature_std,
    )

    model = TemporalMaxerContinuous(
        input_dim=int(metadata["feature_dim"]) + int(metadata.get("auxiliary_feature_dim", 0)),
        hidden_dim=ckpt_args.hidden_dim,
        pyramid_levels=ckpt_args.pyramid_levels,
        head_layers=ckpt_args.head_layers,
        dropout=ckpt_args.dropout,
        use_quality=not ckpt_args.disable_quality,
        reg_max=ckpt_args.reg_max,
        trident_bins=ckpt_args.trident_bins,
        center_sampling_radius=ckpt_args.center_sampling_radius,
        use_boundary_heads=ckpt_args.use_boundary_heads,
        boundary_refine_radius_seconds=(
            ckpt_args.boundary_refine_radius if ckpt_args.use_boundary_heads else 0.0
        ),
        boundary_refine_blend=ckpt_args.boundary_refine_blend,
        tanp_std=0.0,
        tanp_probability=ckpt_args.tanp_probability,
        use_temporal_order=ckpt_args.temporal_order,
        temporal_order_chunks=ckpt_args.temporal_order_chunks,
        classification_input_dim=(
            int(metadata["feature_dim"]) if ckpt_args.cross_layer_task_decoupling else None
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, val_dataset, metadata, ckpt_args


def decode_batch(
    model: TemporalMaxerContinuous,
    output: dict,
    metadata: dict,
    batch: dict,
    ckpt_args: argparse.Namespace,
) -> list[np.ndarray]:
    candidates = model.decode(
        output,
        grid_stride_seconds=float(metadata["grid_stride_s"]),
        durations_seconds=batch["duration_s"],
        score_threshold=ckpt_args.score_threshold,
        pre_nms_topk=ckpt_args.pre_nms_topk,
        quality_power=ckpt_args.quality_power,
    )
    detections = []
    for roi_candidates in candidates:
        detections.append(apply_soft_nms(roi_candidates, ckpt_args))
    return detections


def assign_tp_labels(rows: pd.DataFrame, annotations: dict, iou_threshold: float) -> pd.Series:
    """Greedy TP/FP assignment per (rec_name, roi_id), sorted by raw_score desc."""
    labels = pd.Series(0, index=rows.index, dtype=np.int64)
    for (rec_name, roi_id), group in rows.groupby(["rec_name", "roi_id"]):
        gt = annotations.get((str(rec_name), int(roi_id)))
        if gt is None or len(gt) == 0:
            continue
        ordered = group.sort_values("raw_score", ascending=False)
        locked = np.zeros(len(gt), dtype=bool)
        for idx, row in ordered.iterrows():
            segment = np.asarray([row["t_start"], row["t_end"]], dtype=np.float64)
            ious = segment_iou(segment, np.asarray(gt, dtype=np.float64))
            order = ious.argsort()[::-1]
            for gt_idx in order:
                if ious[gt_idx] < iou_threshold:
                    break
                if locked[gt_idx]:
                    continue
                labels.loc[idx] = 1
                locked[gt_idx] = True
                break
    return labels


def calibration_metrics(scores: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, float]:
    scores = np.clip(scores, 1e-6, 1.0 - 1e-6)
    labels = labels.astype(np.float64)
    brier = float(np.mean((scores - labels) ** 2))
    nll = float(-np.mean(labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores)))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    total = len(scores)
    for bin_index in range(n_bins):
        mask = bin_idx == bin_index
        if not mask.any():
            continue
        bin_conf = float(scores[mask].mean())
        bin_acc = float(labels[mask].mean())
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return {"brier": brier, "nll": nll, "ece": float(ece), "n": int(total)}


def run_fold(
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    partial_csv: Path,
    done_keys: set[tuple[str, int]],
) -> pd.DataFrame:
    checkpoint = torch.load(
        resolve(args.cv_root) / f"fold_{fold:02d}" / "best.pt", map_location=device, weights_only=False
    )
    model, val_dataset, metadata, ckpt_args = build_model_and_dataset(checkpoint, args, device)
    annotations = load_annotations(resolve(ckpt_args.ann_path))
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )

    write_header = not partial_csv.exists()
    all_rows: list[dict] = []
    if not write_header:
        all_rows = pd.read_csv(partial_csv).to_dict("records")
    partial_file = open(partial_csv, "a", encoding="utf-8")
    done_path = partial_csv.with_suffix(".done_rois.csv")
    done_file = open(done_path, "a", encoding="utf-8")
    torch.manual_seed(args.seed + fold)
    for batch in tqdm(loader, desc=f"fold{fold:02d}", disable=args.quiet_progress):
        if all(
            (str(rec_name), int(roi_id)) in done_keys
            for rec_name, roi_id in zip(batch["rec_name"], batch["roi_id"])
        ):
            continue
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        model.eval()
        with torch.no_grad():
            det_output = model(features, mask)
        det_detections = decode_batch(model, det_output, metadata, batch, ckpt_args)

        mc_pass_detections: list[list[np.ndarray]] = []
        model.train()
        with torch.no_grad():
            for _ in range(args.mc_passes):
                mc_output = model(features, mask)
                mc_pass_detections.append(decode_batch(model, mc_output, metadata, batch, ckpt_args))
        model.eval()

        for sample_index, (rec_name, roi_id) in enumerate(zip(batch["rec_name"], batch["roi_id"])):
            key = (str(rec_name), int(roi_id))
            if key in done_keys:
                continue
            det = det_detections[sample_index]
            new_rows = []
            if det.shape[0] > 0:
                mc_scores = np.zeros((args.mc_passes, det.shape[0]), dtype=np.float64)
                for pass_index, pass_detections in enumerate(mc_pass_detections):
                    candidates = pass_detections[sample_index]
                    if candidates.shape[0] == 0:
                        continue
                    for det_index in range(det.shape[0]):
                        ious = segment_iou(det[det_index, :2], candidates[:, :2])
                        best = int(ious.argmax()) if len(ious) else -1
                        if best >= 0 and ious[best] >= args.match_iou:
                            mc_scores[pass_index, det_index] = candidates[best, 2]
                for det_index in range(det.shape[0]):
                    new_rows.append(
                        {
                            "fold": fold,
                            "rec_name": rec_name,
                            "roi_id": int(roi_id),
                            "t_start": float(det[det_index, 0]),
                            "t_end": float(det[det_index, 1]),
                            "raw_score": float(det[det_index, 2]),
                            "mc_mean_score": float(mc_scores[:, det_index].mean()),
                            "mc_std_score": float(mc_scores[:, det_index].std()),
                        }
                    )
            all_rows.extend(new_rows)
            done_keys.add(key)
            if new_rows:
                pd.DataFrame(new_rows).to_csv(partial_file, header=write_header, index=False)
                write_header = False
            done_file.write(f"{rec_name},{int(roi_id)}\n")
            partial_file.flush()
            done_file.flush()

    partial_file.close()
    done_file.close()
    rows = pd.DataFrame(all_rows)
    if rows.empty:
        return rows
    rows["correct"] = assign_tp_labels(rows, annotations, args.tp_iou)
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries = []
    all_rows = []
    for fold in range(5):
        fold_csv = out_dir / f"fold_{fold:02d}_detections.csv"
        if fold_csv.exists():
            rows = pd.read_csv(fold_csv)
        else:
            partial_csv = out_dir / f"fold_{fold:02d}_partial.csv"
            done_path = partial_csv.with_suffix(".done_rois.csv")
            done_keys = set()
            if done_path.exists():
                with open(done_path, encoding="utf-8") as handle:
                    for line in handle:
                        rec_name, roi_id = line.strip().split(",")
                        done_keys.add((rec_name, int(roi_id)))
            rows = run_fold(fold, args, device, partial_csv, done_keys)
            if rows.empty:
                continue
            rows.to_csv(fold_csv, index=False)
            partial_csv.unlink(missing_ok=True)
            done_path.unlink(missing_ok=True)
        if rows.empty:
            continue
        all_rows.append(rows)
        det_metrics = calibration_metrics(
            rows["raw_score"].to_numpy(), rows["correct"].to_numpy(), args.calibration_bins
        )
        mc_metrics = calibration_metrics(
            rows["mc_mean_score"].to_numpy(), rows["correct"].to_numpy(), args.calibration_bins
        )
        std_tp = float(rows.loc[rows["correct"] == 1, "mc_std_score"].mean())
        std_fp = float(rows.loc[rows["correct"] == 0, "mc_std_score"].mean())
        fold_summaries.append(
            {
                "fold": fold,
                "n_detections": len(rows),
                "precision": float(rows["correct"].mean()),
                "det_ece": det_metrics["ece"],
                "det_brier": det_metrics["brier"],
                "det_nll": det_metrics["nll"],
                "mc_ece": mc_metrics["ece"],
                "mc_brier": mc_metrics["brier"],
                "mc_nll": mc_metrics["nll"],
                "mc_std_tp": std_tp,
                "mc_std_fp": std_fp,
            }
        )

    summary = pd.DataFrame(fold_summaries)
    summary.to_csv(out_dir / "fold_summary.csv", index=False)
    print(summary.to_string(index=False))

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(out_dir / "all_detections.csv", index=False)
        overall_det = calibration_metrics(
            combined["raw_score"].to_numpy(), combined["correct"].to_numpy(), args.calibration_bins
        )
        overall_mc = calibration_metrics(
            combined["mc_mean_score"].to_numpy(), combined["correct"].to_numpy(), args.calibration_bins
        )
        overall = {
            "n_detections": len(combined),
            "precision": float(combined["correct"].mean()),
            "det_ece": overall_det["ece"],
            "det_brier": overall_det["brier"],
            "det_nll": overall_det["nll"],
            "mc_ece": overall_mc["ece"],
            "mc_brier": overall_mc["brier"],
            "mc_nll": overall_mc["nll"],
        }
        (out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
        print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
