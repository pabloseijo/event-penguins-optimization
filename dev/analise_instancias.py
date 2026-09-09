"""Cantas instancias reais hai por fold, e canto do corpus estamos usando."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "dev"))
from dev.train_quality_head import build_annotation_index  # noqa: E402

CLASE = sys.argv[1] if len(sys.argv) > 1 else "BaseballPitch"
ANN = R / f"data/thumos14_real/config/annotations/by_class/{CLASE}/annotations.json"

print(f"=== {CLASE} ===")
for split in ("train", "val", "test"):
    idx, recs = build_annotation_index(ANN, split, 0.0)
    n_seg = sum(len(v.get("ed", [])) for rec in idx.values() for v in rec.values())
    print(f"  split {split:5s}: {len(recs):3d} gravacions, {n_seg:4d} instancias 'ed'")

print()
print("=== canto do corpus estamos a usar ===")
sub = json.loads((R / "tmp/thumos14e_real/v1/subconxunto2.json").read_text())
for s in ("validation", "test"):
    d = sub["splits"][s]
    print(f"  {s}: {d['n']} gravacions de {d['de']} "
          f"({d['fraccion_eventos']*100:.1f}% dos eventos)")
    print(f"     {CLASE}: {d['instancias_por_clase'].get(CLASE, 0)} instancias no subconxunto")

print()
print("=== e no corpus completo? ===")
man = pd.read_csv(R / "data/thumos14_real/manifest.csv", keep_default_na=False)
for s in ("validation", "test"):
    sel = man[man["official_subset"] == s]
    total = 0
    for _, fila in sel.iterrows():
        bruto = fila.get("annotations_json") or ""
        if bruto:
            try:
                for it in json.loads(bruto):
                    if isinstance(it, dict) and it.get("label") == CLASE:
                        total += 1
            except (json.JSONDecodeError, TypeError):
                pass
    print(f"  {s}: {total} instancias de {CLASE} en {len(sel)} gravacions")
