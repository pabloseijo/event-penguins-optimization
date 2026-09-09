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


def segs_de(res: dict) -> list:
    saida = []
    for vid, roles in res.items():
        if not isinstance(roles, dict):
            continue
        for _, lista in roles.items():
            for s in lista:
                if isinstance(s, dict) and "segment" in s:
                    saida.append((vid, float(s["segment"][0]), float(s["segment"][1]),
                                  float(s.get("score", 0.0))))
    return saida


for f in sorted(glob.glob(str(P / "predictions/retag/seed_1337/*/predictions.json"))):
    clase = Path(f).parent.name
    d = json.load(open(f))
    todos = segs_de(d.get("results", {}))
    if not todos:
        print(f"{clase:<19} sen predicions")
        continue
    a = np.array([[t[1], t[2], t[3]] for t in todos])
    dur = a[:, 1] - a[:, 0]
    gt = [x["segment"] for vid, info in db.items()
          if (info.get("subset") or "").lower() == "test"
          for x in info.get("annotations", []) if x.get("label") == clase]
    gdur = [g[1] - g[0] for g in gt] or [0]
    # mellor tIoU de cada predicion contra o GT do seu propio video
    porvid: dict = {}
    for vid, info in db.items():
        if (info.get("subset") or "").lower() != "test":
            continue
        porvid[vid] = np.array([x["segment"] for x in info.get("annotations", [])
                                if x.get("label") == clase], dtype=float)
    mellores = []
    for vid, s0, s1, _ in todos:
        g = porvid.get(vid, np.empty((0, 2)))
        if len(g) == 0:
            mellores.append(0.0); continue
        inter = np.maximum(0, np.minimum(s1, g[:, 1]) - np.maximum(s0, g[:, 0]))
        union = (s1 - s0) + (g[:, 1] - g[:, 0]) - inter
        mellores.append(float(np.max(np.where(union > 0, inter / union, 0))))
    m = np.array(mellores)
    print(f"{clase:<19} n={len(todos):>4}  dur_pred={np.median(dur):>6.2f}s  "
          f"dur_GT={np.median(gdur):>6.2f}s  tIoU_max={m.max():.3f}  "
          f"con_tIoU>0.5={int((m >= 0.5).sum()):>3}  >0.1={int((m >= 0.1).sum()):>3}")
