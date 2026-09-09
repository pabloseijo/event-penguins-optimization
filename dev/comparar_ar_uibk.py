"""Comparacion emparellada reTAG vs CoTAD sobre o corpus REAL de Innsbruck.

Protocolo: o publicado por reTAG -- AR@20/30/50 promediando tIoU {0,1;0,3;0,5;0,7},
mellores N candidatos por (gravacion, ROI). E o mesmo co que se comparan as duas
ramas en EventPenguins, asi que as filas son directamente comparables con aquelas.

Por que AR e non mAP: e a metrica publicada por reTAG e a que xa se usa na
comparacion de EventPenguins. Ademais normaliza o numero de propostas, o que
importa aqui porque reTAG produce un conxunto AXENO A CLASE (65.354 no test) e
CoTAD un POR CLASE (~400.000), e sen normalizar a comparacion non valeria.

Usanse as funcions do avaliador establecido (proposal_recall,
load_generic_ground_truth) en vez de reimplementalas.
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
TIOU = (0.1, 0.3, 0.5, 0.7)
BUDGETS = (20, 30, 50)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"

man = pd.read_csv(W / "manifest.csv", keep_default_na=False)
inc = man["evaluation_included"].astype(str).str.lower().isin({"1", "true", "yes"})
recs = sorted(man.loc[(man["official_subset"].str.lower() == SPLIT) & inc, "video_id"].astype(str))
print(f"protocolo: {len(recs)} gravacions canonicas de {SPLIT}")

retag_path = P / f"proposals/retag/{SPLIT}/proposals_completo.csv"
retag = pd.read_csv(retag_path)
retag["roi_id"] = 1
# As anotacions veñen en SEGUNDOS e as propostas en MICROSEGUNDOS. O avaliador
# orixinal converte co seu --time-unit; aqui hai que facelo a man ou todos os
# solapamentos dan cero.
retag["t_start"] = retag["t_start"] / 1e6
retag["t_end"] = retag["t_end"] / 1e6
print(f"reTAG: {len(retag):,} propostas en {retag['rec_name'].nunique()} gravacions\n")

filas = []
for label in THUMOS_CLASSES:
    ann = W / "config/annotations/by_class" / label / "annotations.json"
    if SPLIT == "test":
        ours_paths = [P / f"proposals/eventpenguins_stage1/{label}/test/proposals.csv"]
    else:
        # En validation as propostas de CoTAD son POR FOLD. A avaliacion correcta
        # e out-of-fold: cada gravacion puntuase co fold que a retivo, e despois
        # concatenase. Sen isto avaliariase con propostas vistas no adestramento.
        ours_paths = sorted(
            (P / f"proposals/eventpenguins_stage1/{label}").glob("fold_0*/validation/proposals.csv")
        )
    if not ann.exists() or not ours_paths or not all(q.exists() for q in ours_paths):
        print(f"  {label}: faltan datos, saltada")
        continue
    gt = load_generic_ground_truth(ann, recs)
    # O ficheiro por clase inclue TODAS as accions: as da clase obxectivo con
    # label "ed" e as demais como "other_action". Sen filtrar, as 20 clases
    # daban a mesma GT de 3.454 instancias.
    gt = gt[gt["source_label"] == label]
    if gt.empty:
        continue
    ours = pd.concat([pd.read_csv(q) for q in ours_paths], ignore_index=True)
    ours = ours.drop_duplicates(subset=["rec_name", "t_start", "t_end"])
    ours["roi_id"] = 1
    ours["t_start"] = ours["t_start"] / 1e6
    ours["t_end"] = ours["t_end"] / 1e6
    r_ret = proposal_recall(retag[retag["rec_name"].isin(recs)], gt, TIOU, BUDGETS)
    r_our = proposal_recall(ours[ours["rec_name"].isin(recs)], gt, TIOU, BUDGETS)

    def ar(d: dict, b: int) -> float:
        # proposal_recall devolve {orzamento: {"AR@0.1":..., "mean_AR":...}};
        # a chave pode vir como int ou como cadea segun a versión.
        sub = d.get(b, d.get(str(b)))
        return float(sub["mean_AR"])

    filas.append({
        "clase": label, "GT": len(gt),
        "reTAG_AR20": ar(r_ret, 20), "reTAG_AR30": ar(r_ret, 30), "reTAG_AR50": ar(r_ret, 50),
        "CoTAD_AR20": ar(r_our, 20), "CoTAD_AR30": ar(r_our, 30), "CoTAD_AR50": ar(r_our, 50),
    })
    f = filas[-1]
    print(f"  {label:<19} GT={f['GT']:>4}  reTAG {f['reTAG_AR20']:.4f}/{f['reTAG_AR30']:.4f}/"
          f"{f['reTAG_AR50']:.4f}   CoTAD {f['CoTAD_AR20']:.4f}/{f['CoTAD_AR30']:.4f}/{f['CoTAD_AR50']:.4f}")

t = pd.DataFrame(filas)
out = P / "comparacion_ar_uibk"
out.mkdir(parents=True, exist_ok=True)
t.to_csv(out / f"ar_por_clase_{SPLIT}.csv", index=False)
print("\n" + "=" * 72)
print(f"MACRO sobre {len(t)} clases · corpus real de Innsbruck · split {SPLIT}")
print(f"{'rama':<10}{'AR@20':>10}{'AR@30':>10}{'AR@50':>10}")
for nome, pre in (("reTAG", "reTAG"), ("CoTAD", "CoTAD")):
    print(f"{nome:<10}{t[pre+'_AR20'].mean():>10.4f}{t[pre+'_AR30'].mean():>10.4f}{t[pre+'_AR50'].mean():>10.4f}")
gana = (t["CoTAD_AR50"] > t["reTAG_AR50"]).sum()
empata = (t["CoTAD_AR50"] == t["reTAG_AR50"]).sum()
print(f"\nAR@50: CoTAD gaña en {gana}, empata en {empata}, perde en {len(t)-gana-empata} de {len(t)} clases")
print(f"escrito en {out}")
