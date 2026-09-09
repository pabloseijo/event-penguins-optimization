"""Comproba se as predicions caen onde deben, antes de crer ningun mAP."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

R = Path("/home/pablo.garcia.seijo/event_penguins")
P = R / "tmp/thumos14e_real/v1"
CANON = Path("/home/pablo.garcia.seijo/actionformer_release/data/thumos/annotations/thumos14.json")

db = json.loads(CANON.read_text())
db = db.get("database", db)

sub = json.loads((P / "subconxunto2.json").read_text())
recs_test = set(sub["splits"]["test"]["gravacions"])
print(f"gravacions do subconxunto de test: {len(recs_test)}")

# instancias canonicas dentro e fora do subconxunto
dentro = fora = 0
for vid, info in db.items():
    if (info.get("subset") or "").lower() != "test":
        continue
    n = len(info.get("annotations", []))
    if vid in recs_test:
        dentro += n
    else:
        fora += n
print(f"instancias de test: {dentro} dentro do subconxunto, {fora} fora "
      f"({dentro/(dentro+fora)*100:.1f}% visible)")
print()

for f in sorted(glob.glob(str(P / "predictions/retag/seed_1337/*/predictions.json")))[:3]:
    clase = Path(f).parent.name
    d = json.load(open(f))
    res = d.get("results", d)
    if not isinstance(res, dict):
        continue
    npred = sum(len(v) for v in res.values())
    todos = [s for v in res.values() for s in v]
    if not todos:
        print(f"{clase}: sen predicions"); continue
    segs = np.array([[s["segment"][0], s["segment"][1]] for s in todos], dtype=float)
    scores = np.array([s.get("score", 0.0) for s in todos], dtype=float)
    dur = segs[:, 1] - segs[:, 0]
    print(f"{clase}: {npred} predicions")
    print(f"   duracion pred:  min={dur.min():.2f}s  mediana={np.median(dur):.2f}s  max={dur.max():.2f}s")
    print(f"   score:          min={scores.min():.4f}  mediana={np.median(scores):.4f}  max={scores.max():.4f}")
    print(f"   rango temporal: {segs.min():.1f}s a {segs.max():.1f}s")
    # duracion das anotacions desa clase
    gd = [a["segment"][1] - a["segment"][0] for vid, info in db.items()
          if (info.get("subset") or "").lower() == "test"
          for a in info.get("annotations", []) if a.get("label") == clase]
    if gd:
        print(f"   duracion GT:    min={min(gd):.2f}s  mediana={np.median(gd):.2f}s  max={max(gd):.2f}s")
    # cantas predicions caen en videos que teñen esa clase
    con = {vid for vid, info in db.items()
           if (info.get("subset") or "").lower() == "test"
           and any(a.get("label") == clase for a in info.get("annotations", []))}
    acertan = sum(len(v) for k, v in res.items() if k in con)
    print(f"   predicions en videos que SI conteñen a clase: {acertan} de {npred}")
    print()
