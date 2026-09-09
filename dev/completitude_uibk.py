"""Contribucion da completitude contexto-relativa no corpus REAL de Innsbruck.

C(p) = a_in - 0.5*(a_esq + a_der) e aritmetica sobre o actionness, asi que mide
a achega do artigo SEN a extraccion de features nin a cadea do detector.

Adaptacion de dev/ar_completitude.py, que estaba cableado ao corpus v2e
(thumos14e_original_rate_v1). Aqui:
  - actionness desde a cache de timestamps xa construida (190 GB), non do HDF5
  - as duas ramas: propostas de reTAG e de CoTAD
  - AR@20/30/50 antes e despois de reordenar, co protocolo publicado de reTAG

Receita do artigo: score_fusionado = 0.75*rank(score) + 0.25*rank(C).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "dev"))
from dev.prepare_thumos14_event_corpus import THUMOS_CLASSES  # noqa: E402
from dev.run_thumos14_event_generalization import (  # noqa: E402
    load_generic_ground_truth,
    proposal_recall,
)

W = R / "data/thumos14_real"
P = R / "tmp/thumos14e_real/v1"
TC = P / "shared_features/continuous/timestamp_cache"
CACHE = P / "actionness_cache"
CACHE.mkdir(parents=True, exist_ok=True)
BIN = 0.033
TIOU = (0.1, 0.3, 0.5, 0.7)
BUDGETS = (20, 30, 50)
W_SCORE, W_COMP = 0.75, 0.25
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"


def actionness(vid: str) -> np.ndarray | None:
    """Histograma de eventos en bins de 33 ms. Cacheado: uns 20.000 floats."""
    f = CACHE / f"{vid}.npy"
    if f.exists():
        return np.load(f)
    src = TC / vid / "N01.npy"
    if not src.exists():
        return None
    ts = np.load(src, mmap_mode="r")
    t0, t1 = float(ts[0]), float(ts[-1])
    dur = (t1 - t0) / 1e6
    if dur <= 0:
        return None
    n = max(1, int(dur / BIN))
    c = np.zeros(n, dtype=np.float64)
    paso = 50_000_000
    for i in range(0, len(ts), paso):
        bloque = (np.asarray(ts[i:i + paso], dtype=np.float64) - t0) / 1e6
        h, _ = np.histogram(bloque, bins=n, range=(0.0, dur))
        c += h
    c /= BIN
    np.save(f, c)
    return c


def completitude(a: np.ndarray, ini: float, fin: float) -> float:
    """C(p) co contexto de 0.5|p| a cada lado, como no artigo."""
    n = len(a)
    i0, i1 = int(ini / BIN), min(n, int(fin / BIN) + 1)
    if i1 <= i0:
        return 0.0
    ctx = max(1, (i1 - i0) // 2)
    dentro = a[i0:i1].mean()
    esq = a[max(0, i0 - ctx):i0]
    der = a[i1:min(n, i1 + ctx)]
    fora = 0.5 * ((esq.mean() if len(esq) else 0.0) + (der.mean() if len(der) else 0.0))
    return float(dentro - fora)


def rangos(v: np.ndarray) -> np.ndarray:
    o = np.argsort(np.argsort(v))
    return o / max(1, len(v) - 1)


def reordenar(df: pd.DataFrame, act: dict) -> pd.DataFrame:
    """Engade a columna score_fusionado co C(p) de cada proposta."""
    out = df.copy()
    comp = np.zeros(len(out))
    for vid, idx in out.groupby("rec_name").indices.items():
        a = act.get(vid)
        if a is None:
            continue
        s = out["t_start"].to_numpy()[idx]
        e = out["t_end"].to_numpy()[idx]
        comp[idx] = [completitude(a, s[k], e[k]) for k in range(len(idx))]
    out["completitude"] = comp
    novo = np.zeros(len(out))
    for _, idx in out.groupby("rec_name").indices.items():
        novo[idx] = (W_SCORE * rangos(out["score"].to_numpy()[idx])
                     + W_COMP * rangos(comp[idx]))
    out["score_orixinal"] = out["score"]
    out["score"] = novo
    return out


man = pd.read_csv(W / "manifest.csv", keep_default_na=False)
inc = man["evaluation_included"].astype(str).str.lower().isin({"1", "true", "yes"})
recs = sorted(man.loc[(man["official_subset"].str.lower() == SPLIT) & inc, "video_id"].astype(str))
print(f"protocolo: {len(recs)} gravacions de {SPLIT}")

print("precalculando actionness (cacheado por gravacion)...")
act = {}
for i, v in enumerate(recs, 1):
    a = actionness(v)
    if a is not None:
        act[v] = a
    if i % 40 == 0:
        print(f"  {i}/{len(recs)}")
print(f"  actionness dispoñible en {len(act)} de {len(recs)} gravacions\n")

retag = pd.read_csv(P / f"proposals/retag/{SPLIT}/proposals_completo.csv")
retag["roi_id"] = 1
retag["t_start"] /= 1e6
retag["t_end"] /= 1e6
retag_c = reordenar(retag, act)

filas = []
for label in THUMOS_CLASSES:
    ann = W / "config/annotations/by_class" / label / "annotations.json"
    if SPLIT == "test":
        paths = [P / f"proposals/eventpenguins_stage1/{label}/test/proposals.csv"]
    else:
        paths = sorted((P / f"proposals/eventpenguins_stage1/{label}").glob("fold_0*/validation/proposals.csv"))
    if not ann.exists() or not paths or not all(q.exists() for q in paths):
        continue
    gt = load_generic_ground_truth(ann, recs)
    gt = gt[gt["source_label"] == label]
    if gt.empty:
        continue
    ours = pd.concat([pd.read_csv(q) for q in paths], ignore_index=True)
    ours = ours.drop_duplicates(subset=["rec_name", "t_start", "t_end"])
    ours["roi_id"] = 1
    ours["t_start"] /= 1e6
    ours["t_end"] /= 1e6
    ours_c = reordenar(ours, act)

    def ar(d, b):
        return float((d.get(b, d.get(str(b))))["mean_AR"])

    m = retag["rec_name"].isin(recs)
    mo = ours["rec_name"].isin(recs)
    r0 = proposal_recall(retag[m], gt, TIOU, BUDGETS)
    r1 = proposal_recall(retag_c[m], gt, TIOU, BUDGETS)
    o0 = proposal_recall(ours[mo], gt, TIOU, BUDGETS)
    o1 = proposal_recall(ours_c[mo], gt, TIOU, BUDGETS)
    filas.append({
        "clase": label, "GT": len(gt),
        "reTAG_base": ar(r0, 50), "reTAG_comp": ar(r1, 50),
        "CoTAD_base": ar(o0, 50), "CoTAD_comp": ar(o1, 50),
    })
    f = filas[-1]
    print(f"  {label:<19} reTAG {f['reTAG_base']:.4f}->{f['reTAG_comp']:.4f}   "
          f"CoTAD {f['CoTAD_base']:.4f}->{f['CoTAD_comp']:.4f}")

t = pd.DataFrame(filas)
out = P / "comparacion_ar_uibk"
t.to_csv(out / f"completitude_{SPLIT}.csv", index=False)
print("\n" + "=" * 70)
print(f"AR@50 macro · {len(t)} clases · corpus de Innsbruck · {SPLIT}")
print(f"{'':12}{'sen C(p)':>12}{'con C(p)':>12}{'delta':>10}")
for nome in ("reTAG", "CoTAD"):
    b, c = t[f"{nome}_base"].mean(), t[f"{nome}_comp"].mean()
    print(f"{nome:<12}{b:>12.4f}{c:>12.4f}{c-b:>+10.4f}")
print(f"\nC(p) mellora CoTAD en {(t['CoTAD_comp'] > t['CoTAD_base']).sum()} de {len(t)} clases")
print(f"C(p) mellora reTAG en {(t['reTAG_comp'] > t['reTAG_base']).sum()} de {len(t)} clases")
