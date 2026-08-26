"""F1 a nivel de segmento do sistema actual, comparable co paper de Fourier.

O paper de Fourier (Hamann et al., Advanced Intelligent Systems 2025) clasifica ventás
fixas e reporta F1: 0,723 co seu ResNet18 e 0,54 co clasificador de banda de enerxía.
A nosa métrica é mAP sobre localización, que non é comparable directamente.

Este script discretiza as nosas deteccións en ventás do mesmo tamaño que usan eles e
calcula F1 binario por ventá, que si é comparable. Barre o limiar de score e devolve o
mellor F1, igual que fan eles.

Non necesita GPU: só as predicións xa calculadas e as anotacións.

    python dev/compute_segment_f1_current.py \
        --predictions tmp/temporalmaxer_continuous/actionness_qfl_test_v1/predictions.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--ann-path", default="config/annotations/annotations.json")
    p.add_argument("--info-path", default="config/annotations/recording_info.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--window", type=float, default=5.0,
                   help="Window length in seconds; 5 s is the Fourier paper's setting.")
    p.add_argument("--min-duration", type=float, default=2.0)
    p.add_argument("--out", default="tmp/segment_f1_current.json")
    return p.parse_args()


def windows_from_segments(segments, window, n_windows):
    """Mark a window positive if any segment overlaps it at all."""
    labels = np.zeros(n_windows, dtype=bool)
    for start, end in segments:
        lo = max(0, int(np.floor(start / window)))
        hi = min(n_windows, int(np.ceil(end / window)))
        labels[lo:hi] = True
    return labels


def main() -> None:
    args = parse_args()
    ann = json.loads(Path(args.ann_path).read_text())["database"]
    info = {r["timestamp"]: r for r in csv.DictReader(open(args.info_path))}
    preds = json.loads(Path(args.predictions).read_text())

    # Predictions are keyed by "recording/roi"; group them the same way as the ground truth.
    by_key: dict[tuple[str, str], list] = {}
    for entry in (preds["results"] if isinstance(preds, dict) and "results" in preds else preds):
        if isinstance(entry, str):
            continue
        rec = entry.get("recording") or entry.get("video-id", "").split("/")[0]
        roi = str(entry.get("roi") or entry.get("video-id", "").split("/")[-1])
        seg = entry.get("segment") or [entry.get("t-start"), entry.get("t-end")]
        by_key.setdefault((rec, roi), []).append((float(seg[0]), float(seg[1]), float(entry["score"])))

    thresholds = np.quantile([s for v in by_key.values() for *_, s in v], np.linspace(0, 0.999, 200))
    best = {"f1": -1.0}

    for thr in thresholds:
        tp = fp = fn = 0
        for rec, payload in ann.items():
            if info.get(rec, {}).get("split") != args.split:
                continue
            for roi, segs in payload["annotations"].items():
                if roi == "null":
                    continue
                truth = [s["segment"] for s in segs
                         if s["label"] == "ed"
                         and s["segment"][1] - s["segment"][0] >= args.min_duration]
                got = [(a, b) for a, b, s in by_key.get((rec, roi), []) if s >= thr]
                horizon = max([b for _, b in truth] + [b for _, b in got] + [0.0])
                if horizon <= 0:
                    continue
                n = int(np.ceil(horizon / args.window))
                y = windows_from_segments(truth, args.window, n)
                p = windows_from_segments(got, args.window, n)
                tp += int((y & p).sum()); fp += int((~y & p).sum()); fn += int((y & ~p).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": precision, "recall": recall,
                    "threshold": float(thr), "tp": tp, "fp": fp, "fn": fn}

    best["window_seconds"] = args.window
    best["reference_fourier_resnet18"] = 0.723
    best["reference_fourier_energy_band"] = 0.54
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=2))

    print(f"Segment-level F1 at {args.window:.0f} s windows: {best['f1']:.4f}")
    print(f"  precision {best['precision']:.4f}  recall {best['recall']:.4f}  "
          f"at score >= {best['threshold']:.4f}")
    print(f"  Fourier paper: 0.723 (ResNet18), 0.54 (energy band)")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
