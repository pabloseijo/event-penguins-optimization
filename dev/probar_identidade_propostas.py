"""Proba que a version por bloques da EXACTAMENTE o mesmo que a monolitica.

Compara, sobre gravacions reais:
  get_event_rate          novo (dataset, por bloques)  vs  vello (array enteiro)
  get_spatial_compactness novo (bin_idx por bloque)    vs  vello (bin_idx enteiro)
  get_prototype_score     novo (bloque lido unha vez)  vs  vello (tres slices)

As formulas vellas reimplantanse aqui inline para non depender de ter as duas
versions do modulo importables a vez. Se algunha comparacion non e exacta, o
script sae con erro e NON se despreza nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.proposals import get_event_rate, get_spatial_compactness  # noqa: E402
from src.prototype import get_prototype_score  # noqa: E402


# ---------------- versions VELLAS, copiadas literalmente ----------------

def vello_event_rate(events, bin_width):
    t_min, t_max = events[0, 2], events[-1, 2]
    bin_num = int((t_max - t_min) / bin_width)
    counts, bins = np.histogram(events[:, 2], bins=bin_num)
    return counts, bins


def vello_compactness(events, bins, roi_height, roi_width, k=10.0, d0=0.5):
    bin_num = len(bins) - 1
    bin_idx = np.searchsorted(bins[1:], events[:, 2], side="right")
    bin_idx = np.clip(bin_idx, 0, bin_num - 1)
    _CHUNK = 50_000_000
    count = np.bincount(bin_idx, minlength=bin_num).astype(np.float64)
    sum_x = np.zeros(bin_num); sum_y = np.zeros(bin_num)
    sum_x2 = np.zeros(bin_num); sum_y2 = np.zeros(bin_num)
    for ini in range(0, len(bin_idx), _CHUNK):
        sl = slice(ini, ini + _CHUNK)
        bi = bin_idx[sl]
        x = events[sl, 0].astype(np.float64)
        y = events[sl, 1].astype(np.float64)
        sum_x += np.bincount(bi, weights=x, minlength=bin_num)
        sum_y += np.bincount(bi, weights=y, minlength=bin_num)
        sum_x2 += np.bincount(bi, weights=x * x, minlength=bin_num)
        sum_y2 += np.bincount(bi, weights=y * y, minlength=bin_num)
    safe = np.maximum(count, 1.0)
    mx = sum_x / safe; my = sum_y / safe
    vx = np.maximum(sum_x2 / safe - mx ** 2, 0.0)
    vy = np.maximum(sum_y2 / safe - my ** 2, 0.0)
    spread = np.sqrt(vx + vy)
    mspread = np.sqrt((roi_width / 2.0) ** 2 + (roi_height / 2.0) ** 2) + 1e-9
    sn = np.clip(spread / mspread, 0.0, 1.0)
    comp = 1.0 / (1.0 + np.exp(k * (sn - d0)))
    return np.where(count >= 2, comp, 0.0)


def vello_prototype(events, bins, prototype, roi_height, roi_width, min_ev=5):
    grid_h, grid_w = prototype.shape
    bin_num = len(bins) - 1
    n_cells = grid_h * grid_w
    CHUNK = 25_000_000
    n = len(events)
    grid = np.zeros((bin_num, n_cells), dtype=np.float64)
    cpb = np.zeros(bin_num, dtype=np.int64)
    for ini in range(0, n, CHUNK):
        sl = slice(ini, min(ini + CHUNK, n))
        bi = np.searchsorted(bins[1:], events[sl, 2], side="right")
        np.clip(bi, 0, bin_num - 1, out=bi)
        gy = np.clip((events[sl, 1] / roi_height * grid_h).astype(np.int64), 0, grid_h - 1)
        gx = np.clip((events[sl, 0] / roi_width * grid_w).astype(np.int64), 0, grid_w - 1)
        comb = bi * n_cells + gy * grid_w + gx
        grid += np.bincount(comb, minlength=bin_num * n_cells).reshape(bin_num, n_cells)
        cpb += np.bincount(bi, minlength=bin_num)
    proto_flat = prototype.ravel()
    scores = np.zeros(bin_num, dtype=np.float64)
    normas = np.linalg.norm(grid, axis=1)
    # Copiado literalmente de src/prototype.py: divide DESPOIS do produto
    # escalar, e recorta a [0,1]. Facelo na outra orde da a mesma matematica
    # pero difire nun ulp, e iso xa me deu un falso "DIFIRE" de 3,3e-16.
    validos = (cpb >= min_ev) & (normas > 0)
    scores[validos] = np.clip((grid[validos] @ proto_flat) / normas[validos], 0.0, 1.0)
    return scores


# ---------------------------------- proba ----------------------------------

def main() -> None:
    data_path = sys.argv[1]
    bin_width = float(sys.argv[2]) if len(sys.argv) > 2 else 33333.0
    rng = np.random.default_rng(0)
    with h5py.File(data_path, "r") as f:
        recs = sorted(f.keys())
        tamanos = []
        for r in recs:
            roi = list(f[r].keys())[0]
            tamanos.append((f[r][roi]["events"].shape[0], r, roi))
        tamanos.sort()
        # unha pequena, unha mediana e a maior que caiba en memoria para a version vella
        escollidas = [tamanos[len(tamanos) // 2], tamanos[-40], tamanos[-12]]
        proto = rng.random((8, 8))
        proto /= np.linalg.norm(proto)

        fallos = 0
        for n, rec, roi in escollidas:
            ds = f[rec][roi]
            h = int(ds.attrs["height"]); w = int(ds.attrs["width"])
            arr = np.asarray(ds["events"])
            print(f"\n== {rec} · {n:,} eventos · {n * 16 / 2**30:.2f} GB ==", flush=True)

            c_v, b_v = vello_event_rate(arr, bin_width)
            c_n, b_n = get_event_rate(ds["events"], bin_width)
            ok1 = np.array_equal(c_v, c_n) and np.allclose(b_v, b_n, rtol=0, atol=0)
            print(f"  get_event_rate .......... {'IDENTICO' if ok1 else 'DIFIRE'}"
                  f"  (bins {len(b_n) - 1}, suma {c_n.sum():,})")

            k_v = vello_compactness(arr, b_v, h, w)
            k_n = get_spatial_compactness(ds["events"], b_n, h, w)
            ok2 = np.array_equal(k_v, k_n)
            print(f"  get_spatial_compactness . {'IDENTICO' if ok2 else 'DIFIRE'}"
                  f"  (max |dif| {np.abs(k_v - k_n).max():.3e})")

            p_v = vello_prototype(arr, b_v, proto, h, w)
            p_n = get_prototype_score(ds["events"], b_n, proto, h, w)
            ok3 = np.array_equal(p_v, p_n)
            print(f"  get_prototype_score ..... {'IDENTICO' if ok3 else 'DIFIRE'}"
                  f"  (max |dif| {np.abs(p_v - p_n).max():.3e})")

            if not (ok1 and ok2 and ok3):
                fallos += 1
            del arr

    print()
    if fallos:
        print(f"FALLO: {fallos} gravacions con diferenzas. NON despregar.")
        sys.exit(1)
    print("TODAS AS COMPARACIONS SON EXACTAS. Seguro despregar.")


if __name__ == "__main__":
    main()
