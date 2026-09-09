"""Teito de recall do conxunto de propostas de reTAG: existen as boas?

Se existen propostas con tIoU>=0.5 contra o GT, o problema e de SELECCION e
arranxase reordenando/filtrando co que xa esta extraido. Se non existen, hai que
volver xerar as propostas con outros parametros de duracion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
P = R / "tmp/thumos14e_real/v1"
CANON = Path("/home/pablo.garcia.seijo/actionformer_release/data/thumos/annotations/thumos14.json")
db = json.loads(CANON.read_text())
db = db.get("database", db)

props = pd.read_csv(P / "proposals/retag/test/proposals.csv")
props["dur"] = (props["t_end"] - props["t_start"]) / 1e6
print(f"propostas de test: {len(props):,} en {props['rec_name'].nunique()} gravacions")
print(f"  duracion: mediana={props['dur'].median():.3f}s  "
      f"p75={props['dur'].quantile(.75):.2f}s  p95={props['dur'].quantile(.95):.2f}s")
print()

por_rec: dict = {}
for _, r in props.iterrows():
    por_rec.setdefault(r["rec_name"], []).append((r["t_start"] / 1e6, r["t_end"] / 1e6))

print(f"{'clase':<19}{'GT':>5}{'cub@0.5':>9}{'cub@0.3':>9}{'teito@0.5':>11}")
print("-" * 53)
tot_gt = tot_c5 = tot_c3 = 0
for clase in sorted({a["label"] for v in db.values() for a in v.get("annotations", [])}):
    gt = [(vid, a["segment"]) for vid, info in db.items()
          if (info.get("subset") or "").lower() == "test"
          for a in info.get("annotations", []) if a.get("label") == clase
          and vid in por_rec]
    if not gt:
        continue
    c5 = c3 = 0
    for vid, (g0, g1) in gt:
        arr = np.array(por_rec[vid], dtype=float)
        inter = np.maximum(0, np.minimum(g1, arr[:, 1]) - np.maximum(g0, arr[:, 0]))
        union = (g1 - g0) + (arr[:, 1] - arr[:, 0]) - inter
        t = np.max(np.where(union > 0, inter / union, 0))
        c5 += t >= 0.5
        c3 += t >= 0.3
    tot_gt += len(gt); tot_c5 += c5; tot_c3 += c3
    print(f"{clase:<19}{len(gt):>5}{c5:>9}{c3:>9}{c5/len(gt):>11.3f}")
print("-" * 53)
print(f"{'TOTAL':<19}{tot_gt:>5}{tot_c5:>9}{tot_c3:>9}{tot_c5/max(tot_gt,1):>11.3f}")
print()
print(f"TEITO DE RECALL do conxunto de propostas: {tot_c5/max(tot_gt,1)*100:.1f}% a tIoU 0.5, "
      f"{tot_c3/max(tot_gt,1)*100:.1f}% a tIoU 0.3")
