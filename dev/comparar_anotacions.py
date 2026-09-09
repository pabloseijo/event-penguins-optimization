"""Compara as anotacions do noso corpus real coas canonicas de THUMOS14."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
CANON = Path("/home/pablo.garcia.seijo/actionformer_release/data/thumos/annotations/thumos14.json")

canon = json.loads(CANON.read_text())
db = canon.get("database", canon)
por_clase_canon: dict[str, Counter] = {}
subset_de = {}
for vid, info in db.items():
    sub = (info.get("subset") or "").lower()
    subset_de[vid] = sub
    for a in info.get("annotations", []):
        lab = a.get("label")
        por_clase_canon.setdefault(lab, Counter())[sub] += 1

man = pd.read_csv(R / "data/thumos14_real/manifest.csv", keep_default_na=False)
por_clase_noso: dict[str, Counter] = {}
gravacions = Counter()
for _, fila in man.iterrows():
    s = fila["official_subset"]
    gravacions[s] += 1
    bruto = fila.get("annotations_json") or ""
    if not bruto:
        continue
    try:
        for it in json.loads(bruto):
            if isinstance(it, dict) and it.get("label"):
                por_clase_noso.setdefault(it["label"], Counter())[s] += 1
    except (json.JSONDecodeError, TypeError):
        pass

print(f"gravacions no noso corpus: {dict(gravacions)}")
print()
print(f"{'clase':<20} {'val canon':>10} {'val noso':>9} {'test canon':>11} {'test noso':>10}")
print("-" * 64)
tv = tn = sv = sn = 0
for c in sorted(set(por_clase_canon) | set(por_clase_noso)):
    cv = por_clase_canon.get(c, Counter())
    nv = por_clase_noso.get(c, Counter())
    a = cv.get("validation", 0); b = nv.get("validation", 0)
    d = cv.get("test", 0); e = nv.get("test", 0)
    if a + b + d + e == 0:
        continue
    tv += a; tn += b; sv += d; sn += e
    print(f"{c:<20} {a:>10} {b:>9} {d:>11} {e:>10}")
print("-" * 64)
print(f"{'TOTAL':<20} {tv:>10} {tn:>9} {sv:>11} {sn:>10}")
