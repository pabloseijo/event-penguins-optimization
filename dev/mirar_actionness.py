#!/usr/bin/env python3
"""¿O actionness de THUMOS14-E ten sinal, ou só mide o movemento da cámara?

É a comprobación que decide se paga a pena todo o experimento. En EventPenguins
a cámara está fixa e `a(t) = nº de eventos` mide o paxaro. En THUMOS hai paneo,
zoom e cortes, e a hipótese pesimista é que o actionness mida iso.

Compara o actionness medio **dentro** das anotacións fronte a **fóra**, que é
exactamente o que fai a nosa contribución `C(p)`. Se as dúas distribucións se
solapan, `C(p)` non ten nada que medir.

    python dev/mirar_actionness.py --n-videos 12
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/pablo.garcia.seijo/event_penguins")
CORPUS = ROOT / "data/thumos14_events/thumos14e_original_rate_v1"


def actionness(h5: Path, bin_s: float, dur_s: float) -> np.ndarray:
    """Eventos por bin, normalizado a eventos/segundo."""
    with h5py.File(h5, "r") as h:
        ts = h["events"][:, 0].astype(np.float64)
    if ts.size == 0:
        return np.zeros(1)
    # v2e escribe timestamps en µs desde o inicio do clip
    t = (ts - ts.min()) / 1e6
    nbins = max(1, int(dur_s / bin_s))
    conta, _ = np.histogram(t, bins=nbins, range=(0, dur_s))
    return conta / bin_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--bin", type=float, default=0.033, help="ancho de bin, como reTAG")
    args = ap.parse_args()

    filas = [r for r in csv.DictReader(open(CORPUS / "manifest.csv"))
             if r["official_subset"] == "validation" and json.loads(r["annotations_json"])]

    print(f"{'vídeo':<26} {'accións':>8} {'dentro':>10} {'fóra':>10} {'ratio':>7}")
    ratios, dentros, foras = [], [], []
    feitos = 0
    for r in filas:
        if feitos >= args.n_videos:
            break
        vid = r["video_id"]
        h5 = CORPUS / "v2e" / vid / "events.h5"
        if not h5.exists():
            continue
        meta = json.load(open(CORPUS / "source_metadata" / f"{vid}.json"))
        dur = float(meta.get("duration") or meta.get("duration_s"))
        a = actionness(h5, args.bin, dur)

        anots = json.loads(r["annotations_json"])
        mask = np.zeros(len(a), dtype=bool)
        for an in anots:
            i0 = int(an["segment"][0] / args.bin)
            i1 = min(len(a), int(an["segment"][1] / args.bin) + 1)
            mask[i0:i1] = True
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue

        d, f = float(a[mask].mean()), float(a[~mask].mean())
        ratio = d / f if f > 0 else float("nan")
        print(f"{vid:<26} {len(anots):>8} {d:>10.0f} {f:>10.0f} {ratio:>7.2f}")
        ratios.append(ratio); dentros.append(d); foras.append(f)
        feitos += 1

    if not ratios:
        print("sen vídeos convertidos con anotacións")
        return 1

    print(f"\n{'MEDIA':<26} {'':>8} {statistics.mean(dentros):>10.0f} "
          f"{statistics.mean(foras):>10.0f} {statistics.mean(ratios):>7.2f}")
    print(f"{'MEDIANA do ratio':<26} {'':>8} {'':>10} {'':>10} "
          f"{statistics.median(ratios):>7.2f}")

    r = statistics.median(ratios)
    print()
    if r > 1.5:
        print(f"✓ O actionness SEPARA: dentro das accións hai {r:.1f}× máis eventos.")
        print("  C(p) ten sinal que medir. O experimento ten sentido.")
    elif r > 1.15:
        print(f"~ Separación DÉBIL ({r:.2f}×). C(p) medirá algo, pero pouco.")
    else:
        print(f"✗ NON separa ({r:.2f}×). O actionness non distingue acción de fondo:")
        print("  domínao o movemento de cámara. C(p) non ten nada que medir aquí.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
