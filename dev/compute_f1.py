"""
Computa F1 score para o sistema final do TFG (R5) e compara co paper de Fourier.

Dúas variantes de F1:
  1. Instance-level F1  — igual que mAP pero en vez de AP calcula F1 ao mellor
                          score threshold para cada tIoU. Estándar en TAD.
  2. Segment-level F1   — discretiza o tempo en bins e compara ao nivel de bin.
                          Máis comparable co protocolo de ventás fixas do paper
                          Fourier (Hamann et al., 2024, arXiv:2410.06698).

Uso:
    cd event_penguins/
    conda activate eventpenguins
    python dev/compute_f1.py
    python dev/compute_f1.py --split test --temperature 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.proposals import ProposalGenerator
from src.prototype import build_ed_prototype
from src.classification import ProposalClassifier


# ─── Configuración final R5 (mellor sistema do TFG) ──────────────────────────
BEST_PROPOSAL_CFG = dict(
    bin_width=0.033,
    percentile=1.0,
    nms_threshold=0.95,
    use_adaptive_lambda=True,
    use_spatial_compactness=True,
    use_noise_penalization=True,
    use_dispersed_noise=True,
    use_periodicity=True,
)

BEST_CLASSIFIER_CFG = dict(
    num_tsn_samples=7,
    augment_factor=5,
    sample_duration=1,
    decay=5e-6,
    nms_threshold=0.5,
    batch_size=8,
    use_soft_nms=True,
    soft_nms_sigma=0.25,
    min_ed_score=0.3,
    temperature=2.0,
    duration_penalty_dmax=60,
    duration_penalty_sigma=20,
)

# ─── F1 papers de referencia (Hamann et al., 2024, Advanced Intelligent Systems)
FOURIER_RESULTS = {
    "Energy-band FFT (54 params)":  0.543,
    "rate + conv1D (1700 params)":  0.578,
    "ResNet18 (11.4M params)":      0.723,
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",  default="data/preprocessed.h5")
    p.add_argument("--ann-path",   default="config/annotations/annotations.json")
    p.add_argument("--model-path", default="models/model.pk")
    p.add_argument("--proto-path", default="tmp/prototype/ed_prototype.npy")
    p.add_argument("--split",      default="test")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Temperature scaling T (default: 2.0)")
    p.add_argument("--out",        default="tmp/f1_comparison.json")
    return p.parse_args()


def tiou(a_start, a_end, b_start, b_end) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def compute_instance_f1(
    predictions: dict,
    annotations: dict,
    tiou_threshold: float = 0.5,
    split: str = "test",
    min_duration: float = 2.0,
) -> tuple[float, float, float, float]:
    """
    Instance-level F1 (máximo sobre todos os score thresholds).

    Retorna: (best_f1, precision_at_best, recall_at_best, best_score_threshold)
    """
    # Filtramos GT polas mesmas secuencias que aparecen nas prediccions
    # (igual que fai DetectionsEvaluator con valid_sequences)
    valid_sequences = set(predictions["results"].keys())

    gt_list = []  # (video_roi_id, t_start, t_end)
    ann_db = annotations["database"]
    for seq_id, seq_data in ann_db.items():
        if seq_id not in valid_sequences:
            continue
        for roi_id, roi_anns in seq_data["annotations"].items():
            for ann in roi_anns:
                if ann["label"] != "ed":
                    continue
                dur = ann["segment"][1] - ann["segment"][0]
                if dur < min_duration:
                    continue
                gt_list.append((f"{seq_id}_{roi_id}", ann["segment"][0], ann["segment"][1]))

    # Recollemos todas as deteccions
    det_list = []  # (video_roi_id, t_start, t_end, score)
    for seq_id, roi_dict in predictions["results"].items():
        for roi_id, dets in roi_dict.items():
            if roi_id == "null":
                continue
            for d in dets:
                if d["label"] != "ed":
                    continue
                dur = d["segment"][1] - d["segment"][0]
                if dur < min_duration:
                    continue
                det_list.append((f"{seq_id}_{roi_id}", d["segment"][0], d["segment"][1], d["score"]))

    if not gt_list or not det_list:
        return 0.0, 0.0, 0.0, 0.0

    # Score thresholds: todos os scores únicos + extremos
    scores = sorted(set(d[3] for d in det_list), reverse=True)
    n_gt = len(gt_list)

    best_f1 = 0.0
    best_p = best_r = best_thr = 0.0

    for thr in scores:
        active = [d for d in det_list if d[3] >= thr]
        if not active:
            continue

        gt_matched = [False] * n_gt
        tp = 0
        fp = 0

        for vid, ds, de, sc in sorted(active, key=lambda x: -x[3]):
            matched = False
            for gi, (gvid, gs, ge) in enumerate(gt_list):
                if gvid != vid:
                    continue
                if gt_matched[gi]:
                    continue
                if tiou(ds, de, gs, ge) >= tiou_threshold:
                    gt_matched[gi] = True
                    matched = True
                    break
            if matched:
                tp += 1
            else:
                fp += 1

        fn = n_gt - sum(gt_matched)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1, best_p, best_r, best_thr = f1, prec, rec, thr

    return best_f1, best_p, best_r, best_thr


def compute_segment_f1(
    predictions: dict,
    annotations: dict,
    bin_width: float = 0.033,
    split: str = "test",
    score_threshold: float = 0.3,
    min_duration: float = 2.0,
) -> tuple[float, float, float]:
    """
    Segment-level F1: compara ao nivel de bin temporal.
    Máis comparable co protocolo de ventás fixas do paper de Fourier.

    Retorna: (f1, precision, recall)
    """
    ann_db = annotations["database"]
    valid_sequences = set(predictions["results"].keys())

    # Por secuencia+roi, rexistramos os GT bins e os detection bins
    tp_total = fp_total = fn_total = 0

    for seq_id, seq_data in ann_db.items():
        if seq_id not in valid_sequences:
            continue

        seq_preds = predictions["results"].get(seq_id, {})

        for roi_id, roi_anns in seq_data["annotations"].items():
            # GT intervals
            gt_intervals = [
                (ann["segment"][0], ann["segment"][1])
                for ann in roi_anns
                if ann["label"] == "ed"
                and (ann["segment"][1] - ann["segment"][0]) >= min_duration
            ]
            if not gt_intervals:
                continue

            # roi_id nas prediccions pode ser int ou str — normalizamos
            try:
                roi_dets = seq_preds.get(int(roi_id), seq_preds.get(roi_id, []))
            except (ValueError, TypeError):
                roi_dets = seq_preds.get(roi_id, [])
            det_intervals = [
                (d["segment"][0], d["segment"][1])
                for d in roi_dets
                if d["label"] == "ed"
                and d["score"] >= score_threshold
                and (d["segment"][1] - d["segment"][0]) >= min_duration
            ]

            # Time range to evaluate
            t_max = max(max(e for _, e in gt_intervals), max((e for _, e in det_intervals), default=0))
            t_max = t_max + 1.0
            bins = int(t_max / bin_width) + 1

            gt_mask  = np.zeros(bins, dtype=bool)
            det_mask = np.zeros(bins, dtype=bool)

            for ts, te in gt_intervals:
                b0 = int(ts / bin_width)
                b1 = int(te / bin_width)
                gt_mask[b0:b1+1] = True

            for ts, te in det_intervals:
                b0 = int(ts / bin_width)
                b1 = int(te / bin_width)
                det_mask[b0:b1+1] = True

            tp_total += int(np.sum(gt_mask & det_mask))
            fp_total += int(np.sum(det_mask & ~gt_mask))
            fn_total += int(np.sum(gt_mask & ~det_mask))

    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def main():
    args = _parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Split: {args.split}")

    # Prototipo
    proto_path = Path(args.proto_path)
    if proto_path.exists():
        prototype = np.load(proto_path)
    else:
        print("[INFO] Construíndo prototipo...")
        prototype = build_ed_prototype(args.data_path, args.ann_path, split="train")
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proto_path, prototype)

    # Run pipeline R5
    print("\n[PIPELINE] Executando configuración R5 (mellor sistema TFG)...")
    prop_cfg = {**BEST_PROPOSAL_CFG, "data_path": args.data_path, "prototype": prototype}
    clf_cfg  = {**BEST_CLASSIFIER_CFG, "data_path": args.data_path,
                "model_path": args.model_path, "temperature": args.temperature}

    gen = ProposalGenerator(**prop_cfg)
    proposals = gen.run(split=args.split)
    print(f"  Propostas: {len(proposals)}")

    classifier = ProposalClassifier(device, **clf_cfg)
    results = classifier.run(proposals)
    print(f"  Deteccions: {sum(len(v) for roi in results['results'].values() for v in roi.values())}")

    # Cargar anotacións
    ann = json.load(open(args.ann_path))

    # ─── Instance-level F1 ────────────────────────────────────────
    print("\n[F1 INSTANCE-LEVEL]")
    tiou_thresholds = [0.1, 0.3, 0.5, 0.7]
    instance_results = {}
    for thr in tiou_thresholds:
        f1, p, r, score_thr = compute_instance_f1(results, ann, tiou_threshold=thr, split=args.split)
        instance_results[thr] = {"f1": f1, "precision": p, "recall": r, "score_threshold": score_thr}
        print(f"  tIoU={thr:.1f}  →  F1={f1:.4f}  (P={p:.4f}, R={r:.4f}, score_thr={score_thr:.3f})")

    # ─── Segment-level F1 ─────────────────────────────────────────
    print("\n[F1 SEGMENT-LEVEL (comparable co paper Fourier)]")
    seg_f1, seg_p, seg_r = compute_segment_f1(results, ann, split=args.split, score_threshold=0.3)
    print(f"  Segment F1={seg_f1:.4f}  (P={seg_p:.4f}, R={seg_r:.4f})")

    # ─── Comparación co paper Fourier ─────────────────────────────
    print("\n" + "="*65)
    print("COMPARACIÓN CO PAPER DE FOURIER (Hamann et al., Adv. Int. Sys. 2024)")
    print("="*65)
    print(f"\n{'Método':<40} {'F1':>8}  {'Nota'}")
    print("-"*65)
    for name, f1_val in FOURIER_RESULTS.items():
        note = "(ventás fixas 5s, non require localización)"
        print(f"  {name:<38} {f1_val:.3f}  {note}")
    print("-"*65)
    print(f"  {'Noso (instance F1 @ tIoU=0.1)':<38} {instance_results[0.1]['f1']:.3f}  (localización esixida tIoU≥0.1)")
    print(f"  {'Noso (instance F1 @ tIoU=0.5)':<38} {instance_results[0.5]['f1']:.3f}  (localización esixida tIoU≥0.5)")
    print(f"  {'Noso (segment-level F1)':<38} {seg_f1:.3f}  (nivel bin 33ms, máis comparable)")
    print("="*65)
    print("""
NOTA METODOLÓXICA:
  O paper de Fourier usa F1 sobre ventás fixas de 5s (clasificación binaria).
  O noso instance F1 require ademais acertar os límites temporais (tIoU≥α).
  A comparación máis xusta é: noso segment-level F1 vs Fourier F1=0.543.
""")

    # Gardar resultados
    out = {
        "system": "reTAG + TFG R5 (mellor configuración)",
        "split": args.split,
        "instance_f1": {str(k): v for k, v in instance_results.items()},
        "segment_f1": {"f1": seg_f1, "precision": seg_p, "recall": seg_r},
        "fourier_reference": FOURIER_RESULTS,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[INFO] Resultados gardados en {args.out}")


if __name__ == "__main__":
    main()
