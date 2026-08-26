#!/usr/bin/env python3
"""Average Recall nos dous dominios, para a táboa comparativa do artigo.

Mide o mesmo en EventPenguins (eventos reais, cámara fixa) e en THUMOS14-E
(eventos sintéticos, cámara móbil), coas dúas ramas. A degradación entre os dous
é o argumento: mide onde deixa de funcionar o actionness.

Usa o protocolo publicado por reTAG: N=20/30/50 e media sobre tIoU
{0.1, 0.3, 0.5, 0.7}.

    python dev/ar_dous_dominios.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Repository root holding the proposal artifacts of both domains. Defaults to the
# repository this file lives in; override with EVENT_PENGUINS_ROOT when the
# artifacts sit elsewhere, as they do on the experiment server.
ROOT = Path(os.environ.get("EVENT_PENGUINS_ROOT", Path(__file__).resolve().parent.parent))
csv.field_size_limit(sys.maxsize)


def tiou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


RETAG_TIOU = (0.1, 0.3, 0.5, 0.7)
RETAG_BUDGETS = (20, 30, 50)


def recall(props, gt, n: int, umbral: float) -> float:
    total = acertos = 0
    for chave, instancias in gt.items():
        cands = props.get(chave, [])[:n]
        for ins in instancias:
            total += 1
            if any(tiou(c, ins) >= umbral for c in cands):
                acertos += 1
    return acertos / total if total else 0.0


def average_recall(props, gt, n: int) -> tuple[float, list[float]]:
    per_tiou = [recall(props, gt, n, threshold) for threshold in RETAG_TIOU]
    return sum(per_tiou) / len(per_tiou), per_tiou


# ---------------------------------------------------------------- pingüíns

def penguins_gt(splits: set[str]) -> dict:
    info = {r["timestamp"]: r["split"]
            for r in csv.DictReader(open(ROOT / "config/annotations/recording_info.csv"))}
    db = json.loads((ROOT / "config/annotations/annotations.json").read_text())["database"]
    out = defaultdict(list)
    for rec, cont in db.items():
        if info.get(rec) not in splits:
            continue
        for roi, anots in cont.get("annotations", {}).items():
            if roi == "null":
                continue
            for a in anots:
                if a.get("label") == "ed" and a["segment"][1] - a["segment"][0] >= 2.0:
                    out[(rec, roi)].append(tuple(a["segment"]))
    return out


def penguins_props(ficheiro: str, splits: set[str]) -> dict:
    info = {r["timestamp"]: r["split"]
            for r in csv.DictReader(open(ROOT / "config/annotations/recording_info.csv"))}
    bruto = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "tmp/debug" / ficheiro)):
        if info.get(r["rec_name"]) not in splits:
            continue
        k = (r["rec_name"], str(int(r["roi_id"][1:])))
        bruto[k].append((float(r["t_start"]) / 1e6, float(r["t_end"]) / 1e6,
                         float(r.get("score", 0))))
    return {k: [(a, b) for a, b, _ in sorted(v, key=lambda x: -x[2])]
            for k, v in bruto.items()}


# ---------------------------------------------------------------- thumos

TH = ROOT / "tmp/thumos14e_supervised/original_rate_v1"
MAN = ROOT / "data/thumos14_events/thumos14e_original_rate_v1/manifest.csv"


def thumos_gt(clase: str, subset: str) -> dict:
    out = defaultdict(list)
    for r in csv.DictReader(open(MAN)):
        if r["official_subset"] != subset:
            continue
        for a in json.loads(r["annotations_json"]):
            if a["label"] == clase:
                out[r["video_id"]].append(tuple(a["segment"]))
    return out


def thumos_props(ruta: Path) -> dict:
    bruto = defaultdict(list)
    for r in csv.DictReader(open(ruta)):
        bruto[r["rec_name"]].append((float(r["t_start"]) / 1e6,
                                     float(r["t_end"]) / 1e6, float(r["score"])))
    return {k: [(a, b) for a, b, _ in sorted(v, key=lambda x: -x[2])]
            for k, v in bruto.items()}


def main() -> int:
    print(f"Average Recall sobre tIoU {RETAG_TIOU}\n")

    # --- EventPenguins, split de test, os N que publica reTAG
    gt_p = penguins_gt({"test"})
    n_inst = sum(len(v) for v in gt_p.values())
    print(f"EventPenguins · test · {n_inst} instancias ED")
    print(f"  {'rama':<10} {'#props':>10}  {'AR@20':>7} {'AR@30':>7} {'AR@50':>7}")
    for nome, fich in (("reTAG", "proposals_baseline.csv"),
                       ("noso", "proposals_adaptive_spatial_noise.csv")):
        p = penguins_props(fich, {"test"})
        tot = sum(len(v) for v in p.values())
        vals = [average_recall(p, gt_p, n)[0] for n in RETAG_BUDGETS]
        print(f"  {nome:<10} {tot:>10,}  " + " ".join(f"{v:>7.4f}" for v in vals))

    # --- THUMOS14-E
    gt_t = thumos_gt("Diving", "validation")
    n_inst = sum(len(v) for v in gt_t.values())
    print(f"\nTHUMOS14-E · validation · Diving · {n_inst} instancias")
    print(f"  {'rama':<10} {'#props':>10}  {'AR@20':>7} {'AR@30':>7} {'AR@50':>7}")
    ramas = {
        "reTAG": TH / "proposals/retag/validation/proposals.csv",
        "noso": TH / "proposals/eventpenguins_stage1/Diving/fold_00/validation/proposals.csv",
    }
    for nome, ruta in ramas.items():
        if not ruta.exists():
            print(f"  {nome:<10} sen datos")
            continue
        p = thumos_props(ruta)
        tot = sum(len(v) for v in p.values())
        vals = [average_recall(p, gt_t, n)[0] for n in RETAG_BUDGETS]
        print(f"  {nome:<10} {tot:>10,}  " + " ".join(f"{v:>7.4f}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
