"""Estima o desprazamento temporal real -> v2e por correlacion cruzada.

O corpus v2e derivase do video RGB orixinal (prepare_thumos14_event_corpus.py:1334
toma duration_s de ffprobe sobre o MP4), asi que o seu eixe de tempo E o das
anotacions canonicas. Correlacionando a taxa de eventos por bin de 33 ms das
duas gravacions do MESMO video, o pico da o desprazamento a aplicar as
anotacions. Non fai falla nin o video RGB nin preguntarlle a ninguen.

Saida: temporal_alignment.json, un rexistro por video con lag, correlacion no
pico, correlacion sen aliñar, marxe sobre o segundo pico local e un status.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np

BIN_US = 33333
XANELA_BINS = 300          # +-10 s de busca
MARXE_MIN = 0.10           # o pico ten que destacar isto sobre o fondo
R_MIN = 0.35               # correlacion minima para fiarse do aliñamento


def taxa_real(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        g = f["recording"]["N01"] if "recording" in f else f["N01"]
        t = np.asarray(g["events"][:, 2]).astype(np.int64)
    if len(t) == 0:
        return np.zeros(1)
    return np.bincount(t // BIN_US, minlength=int(t[-1] // BIN_US) + 1).astype(np.float64)


def z(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    s = v.std()
    return v / s if s > 0 else v


def aliñar(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    if n < 30:
        return {"status": "curto_de_mais"}
    A, B = z(a[:n]), z(b[:n])
    c = np.correlate(A, B, mode="full") / n
    lags = np.arange(-n + 1, n)
    m = np.abs(lags) <= XANELA_BINS
    c_w, l_w = c[m], lags[m]
    k = int(np.argmax(c_w))
    lag, pico = int(l_w[k]), float(c_w[k])
    base = float(c_w[l_w == 0][0]) if (l_w == 0).any() else float("nan")
    # marxe: canto destaca o pico sobre o mellor lag afastado del
    lonxe = np.abs(l_w - lag) > 15
    segundo = float(c_w[lonxe].max()) if lonxe.any() else 0.0
    status = "ok"
    if pico < R_MIN:
        status = "correlacion_baixa"
    elif pico - segundo < MARXE_MIN:
        status = "pico_ambiguo"
    return {
        "lag_bins": lag,
        "desprazamento_s": round(lag * BIN_US / 1e6, 4),
        "r_pico": round(pico, 4),
        "r_sen_alinar": round(base, 4),
        "r_segundo": round(segundo, 4),
        "marxe": round(pico - segundo, 4),
        "bins_comparados": int(n),
        "status": status,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--rates", default="rates")
    ap.add_argument("--saida", default="temporal_alignment.json")
    args = ap.parse_args()

    saida = {}
    for p in sorted(glob.glob(os.path.join(args.rates, "*.npy"))):
        vid = os.path.basename(p)[:-4]
        real = os.path.join(args.corpus, vid + ".h5")
        if not os.path.exists(real):
            continue
        r = aliñar(taxa_real(real), np.load(p).astype(np.float64))
        r["video_id"] = vid
        saida[vid] = r

    with open(args.saida, "w", encoding="utf-8") as fh:
        json.dump(saida, fh, indent=2, sort_keys=True)

    filas = list(saida.values())
    ok = [f for f in filas if f.get("status") == "ok"]
    print(f"{len(filas)} vídeos · {len(ok)} con aliñamento fiable\n")
    print(f"{'video':<28}{'lag':>9}{'r pico':>9}{'r sen al.':>11}{'marxe':>8}  status")
    for f in filas:
        if "lag_bins" not in f:
            print(f"{f['video_id']:<28}{'--':>9}{'--':>9}{'--':>11}{'--':>8}  {f['status']}")
            continue
        print(f"{f['video_id']:<28}{f['desprazamento_s']:>+8.3f}s{f['r_pico']:>9.3f}"
              f"{f['r_sen_alinar']:>11.3f}{f['marxe']:>8.3f}  {f['status']}")
    if ok:
        d = np.array([f["desprazamento_s"] for f in ok])
        rp = np.array([f["r_pico"] for f in ok])
        rs = np.array([f["r_sen_alinar"] for f in ok])
        print(f"\ndesprazamento: mediana {np.median(d):+.3f} s · "
              f"rango [{d.min():+.3f}, {d.max():+.3f}] · sd {d.std():.3f}")
        print(f"correlacion:   aliñada {rp.mean():.3f} · sen aliñar {rs.mean():.3f}")


if __name__ == "__main__":
    main()
