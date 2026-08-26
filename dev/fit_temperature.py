"""Fit temperature scaling parameter T on the val split.

T is chosen to minimise NLL of the calibrated model on (logit, GT_label) pairs.
Each proposal is labelled 1 if its tIoU with any GT segment >= tiou_pos, else 0.

Usage (from event_penguins/ with pyenv active):
    ~/event_penguins/pyenv/bin/python3 dev/fit_temperature.py
    ~/event_penguins/pyenv/bin/python3 dev/fit_temperature.py --tiou-pos 0.5 --out tmp/temperature.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from src.proposals import ProposalGenerator
from src.prototype import build_ed_prototype
from src.classification import ProposalClassifier


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",  default="data/preprocessed.h5")
    p.add_argument("--ann-path",   default="config/annotations/annotations.json")
    p.add_argument("--model-path", default="models/model.pk")
    p.add_argument("--proto-path", default="tmp/prototype/ed_prototype.npy")
    p.add_argument("--tiou-pos",   type=float, default=0.5,
                   help="Min tIoU to label a proposal as positive (default 0.5)")
    p.add_argument("--out",        default="tmp/temperature.json",
                   help="Output JSON with fitted T")
    return p.parse_args()


def compute_tiou(t_start: float, t_end: float, gt_segments: list) -> float:
    """Max tIoU between a proposal (in µs) and GT segments (in seconds)."""
    t_s = t_start / 1e6
    t_e = t_end   / 1e6
    best = 0.0
    for seg in gt_segments:
        g_s, g_e = seg[0], seg[1]
        inter = max(0.0, min(t_e, g_e) - max(t_s, g_s))
        union = max(t_e, g_e) - min(t_s, g_s)
        if union > 0:
            best = max(best, inter / union)
    return best


def build_gt_index(ann_path: str, split: str) -> dict:
    """Build {rec_name: {roi_id_str: [segments]}} for a given split.

    Split membership is read from recording_info.csv (sibling of annotations.json).
    Annotation roi_ids are numeric strings ("16"); proposals use "r16" format.
    The returned dict uses the numeric string as key so callers can look up by
    stripping the leading "r" from the proposal roi_id.
    """
    ann_dir = Path(ann_path).parent
    info_path = ann_dir / "recording_info.csv"

    import csv
    val_recs = set()
    with open(info_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                val_recs.add(row["timestamp"])

    with open(ann_path) as f:
        ann = json.load(f)

    gt = {}
    for rec_name, rec_data in ann["database"].items():
        if rec_name not in val_recs:
            continue
        gt[rec_name] = {}
        # annotations is a dict: {roi_id_str: [list of {label, segment}]}
        for roi_id_str, roi_anns in rec_data.get("annotations", {}).items():
            ed_segs = [a["segment"] for a in roi_anns if a.get("label") == "ed"]
            if ed_segs:
                gt[rec_name][roi_id_str] = ed_segs
    return gt


def nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of temperature-scaled binary classifier."""
    scaled = logits / T
    log_p_ed = scaled[:, 1] - np.log(np.exp(scaled[:, 0]) + np.exp(scaled[:, 1]))
    log_p_bg = scaled[:, 0] - np.log(np.exp(scaled[:, 0]) + np.exp(scaled[:, 1]))
    return -float(np.mean(labels * log_p_ed + (1 - labels) * log_p_bg))


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Build val proposals using best R4 config ──────────────────────────────
    proto_path = Path(args.proto_path)
    if proto_path.exists():
        prototype = np.load(proto_path)
    else:
        print("[INFO] Construíndo prototipo desde train split...")
        prototype = build_ed_prototype(args.data_path, args.ann_path, split="train")
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proto_path, prototype)

    gen = ProposalGenerator(
        bin_width=0.033,
        percentile=1.0,
        nms_threshold=0.95,
        data_path=args.data_path,
        output_dir="tmp/fit_temperature_proposals",
        use_adaptive_lambda=True,
        use_spatial_compactness=True,
        use_noise_penalization=True,
        use_dispersed_noise=True,
        use_periodicity=True,
        prototype=prototype,
    )
    proposals = gen.run(split="val")
    print(f"[INFO] Val proposals: {len(proposals)}")

    # ── Collect logits ────────────────────────────────────────────────────────
    clf = ProposalClassifier(
        device=device,
        model_path=args.model_path,
        num_tsn_samples=7,
        augment_factor=5,
        data_path=args.data_path,
        sample_duration=1,
        decay=5e-6,
        nms_threshold=0.5,
        batch_size=8,
        use_soft_nms=True,
        soft_nms_sigma=0.25,
        min_ed_score=0.0,  # collect ALL proposals
    )
    logits, meta = clf.collect_logits(proposals)
    print(f"[INFO] Logits recollidos: {logits.shape}")

    # ── Assign binary GT labels ───────────────────────────────────────────────
    gt_index = build_gt_index(args.ann_path, split="val")
    labels = np.zeros(len(meta), dtype=np.float32)
    n_pos = 0
    for idx, (rec_name, roi_id, t_start, t_end) in enumerate(meta):
        # roi_id from proposals is "N01"; annotation keys are "1", "13", etc.
        # Strip the leading letter and remove leading zeros via int conversion.
        roi_key = str(int(roi_id[1:]))
        segs = gt_index.get(rec_name, {}).get(roi_key, [])
        if segs and compute_tiou(t_start, t_end, segs) >= args.tiou_pos:
            labels[idx] = 1.0
            n_pos += 1

    n_neg = len(labels) - n_pos
    print(f"[INFO] Positivos (tIoU≥{args.tiou_pos}): {n_pos}  |  Negativos: {n_neg}")
    print(f"[INFO] NLL sen calibrar (T=1.0): {nll(1.0, logits, labels):.4f}")

    # ── Fit T via scalar minimisation ─────────────────────────────────────────
    result = minimize_scalar(
        lambda T: nll(T, logits, labels),
        bounds=(0.1, 20.0),
        method="bounded",
    )
    T_opt = float(result.x)
    print(f"\n[RESULTADO] T óptimo = {T_opt:.4f}")
    print(f"[INFO] NLL con T={T_opt:.4f}: {nll(T_opt, logits, labels):.4f}")

    # NLL at common values for reference
    for T_ref in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, T_opt]:
        print(f"  T={T_ref:.2f}  NLL={nll(T_ref, logits, labels):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "T_opt": T_opt,
            "nll_T1": nll(1.0, logits, labels),
            "nll_Topt": nll(T_opt, logits, labels),
            "n_proposals": len(meta),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "tiou_pos": args.tiou_pos,
        }, f, indent=2)
    print(f"[INFO] Gardado en {out_path}")


if __name__ == "__main__":
    main()
