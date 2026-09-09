"""Reordena as predicions de reTAG cun prior de duracion tomado de VALIDATION.

Diagnostico do 2026-09-08: as predicions tiñan mediana de 0,07-0,17 s fronte a
accions de 1,4-7,6 s, asi que o tIoU era ~0,02 e o mAP 0,0004. Pero o conxunto de
propostas ten un teito de recall do 38,6 % a tIoU 0,5: as boas ESTAN, so que a
cabeza rankea por riba os segmentos curtos.

O prior sae das anotacions de VALIDATION (o split de adestramento), nunca do de
test. Reutilizase o proposal_scores.csv que xa gardou o paso `score`, asi que non
hai que volver pasar o modelo: e reordenacion pura.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
P = R / "tmp/thumos14e_real/v1"
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "dev"))
from dev.train_thumos14_ovr_atsn import temporal_nms  # noqa: E402

CANON = Path("/home/pablo.garcia.seijo/actionformer_release/data/thumos/annotations/thumos14.json")
db = json.loads(CANON.read_text()); db = db.get("database", db)

NMS = 0.6
TOPK = 200
SAIDA = P / "predictions_reordenadas"
Q_BAIXO, Q_ALTO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10, \
                  float(sys.argv[2]) if len(sys.argv) > 2 else 0.90


def prior_duracion(clase: str) -> tuple[float, float]:
    d = [a["segment"][1] - a["segment"][0] for info in db.values()
         if (info.get("subset") or "").lower() == "validation"
         for a in info.get("annotations", []) if a.get("label") == clase]
    if not d:
        return 0.0, 1e9
    return float(np.quantile(d, Q_BAIXO)), float(np.quantile(d, Q_ALTO))


print(f"prior de duracion: cuantis {Q_BAIXO:.2f}-{Q_ALTO:.2f} das anotacions de VALIDATION")
print(f"{'clase':<19}{'lo(s)':>7}{'hi(s)':>8}{'propostas':>11}{'tras filtro':>12}{'predicions':>11}")
print("-" * 68)
SAIDA.mkdir(parents=True, exist_ok=True)
for d in sorted((P / "predictions/retag/seed_1337").iterdir()):
    f = d / "proposal_scores.csv"
    if not f.exists():
        continue
    clase = d.name
    lo, hi = prior_duracion(clase)
    sc = pd.read_csv(f)
    sc["dur"] = (sc["t_end"] - sc["t_start"]) / 1e6
    keep = sc[(sc["dur"] >= lo) & (sc["dur"] <= hi)]
    res: dict = {}
    npred = 0
    for (rec, roi), g in keep.groupby(["rec_name", "roi_id"], sort=True):
        g = g.nlargest(min(len(g), 5000), "cnn_score")
        vals = g[["t_start", "t_end", "cnn_score"]].to_numpy(dtype=np.float64)
        kept = temporal_nms(vals, NMS) if len(vals) else vals
        kept = kept[np.argsort(-kept[:, 2])][:TOPK] if len(kept) else kept
        res.setdefault(str(rec), {})["1"] = [
            {"label": "ed", "source_label": clase,
             "segment": [float(a) / 1e6, float(b) / 1e6], "score": float(c)}
            for a, b, c in kept
        ]
        npred += len(kept)
    od = SAIDA / clase
    od.mkdir(parents=True, exist_ok=True)
    (od / "predictions.json").write_text(json.dumps(
        {"version": "reordenado-duracion-v1", "target_class": clase, "results": res}))
    print(f"{clase:<19}{lo:>7.2f}{hi:>8.2f}{len(sc):>11,}{len(keep):>12,}{npred:>11,}")
print(f"\nescrito en {SAIDA}")
