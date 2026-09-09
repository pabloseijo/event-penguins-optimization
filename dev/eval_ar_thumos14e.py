#!/usr/bin/env python3
"""Average Recall en THUMOS14-E: propostas de reTAG fronte ás nosas.

Illa a xeración de candidatos, sen clasificador. É a métrica que di se o
actionness serve para propoñer nesta entrada: se AR xa está no chan, o mAP baixo
non se pode atribuír nin ao ranking nin á clasificación.

Os tempos das propostas veñen en microsegundos desde o inicio do vídeo; as
anotacións do manifesto, en segundos.

    python dev/eval_ar_thumos14e.py --clase Diving
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/pablo.garcia.seijo/event_penguins")
OUT = ROOT / "tmp/thumos14e_supervised/original_rate_v1"
MANIFEST = ROOT / "data/thumos14_events/thumos14e_original_rate_v1/manifest.csv"

csv.field_size_limit(sys.maxsize)


def tiou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def cargar_propostas(ruta: Path) -> dict[str, list[tuple[float, float]]]:
    """rec_name → [(inicio_s, fin_s)] ordenadas por score descendente."""
    bruto = defaultdict(list)
    with ruta.open() as fh:
        for r in csv.DictReader(fh):
            bruto[r["rec_name"]].append(
                (float(r["t_start"]) / 1e6, float(r["t_end"]) / 1e6, float(r["score"])))
    return {k: [(a, b) for a, b, _ in sorted(v, key=lambda x: -x[2])]
            for k, v in bruto.items()}


def cargar_gt(clase: str, subset: str) -> dict[str, list[tuple[float, float]]]:
    out = defaultdict(list)
    with MANIFEST.open() as fh:
        for r in csv.DictReader(fh):
            if r["official_subset"] != subset:
                continue
            for a in json.loads(r["annotations_json"]):
                if a["label"] == clase:
                    out[r["video_id"]].append(tuple(a["segment"]))
    return out


def ar(props, gt, n: int, umbral: float) -> tuple[float, int, int]:
    total = acertos = 0
    for vid, instancias in gt.items():
        cands = props.get(vid, [])[:n]
        for ins in instancias:
            total += 1
            if any(tiou(c, ins) >= umbral for c in cands):
                acertos += 1
    return (acertos / total if total else 0.0), acertos, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clase", default="Diving")
    ap.add_argument("--subset", default="validation")
    ap.add_argument("--fold", default="00")
    args = ap.parse_args()

    ramas = {
        "reTAG": OUT / "proposals/retag" / args.subset / "proposals.csv",
        "noso": (OUT / "proposals/eventpenguins_stage1" / args.clase
                 / f"fold_{args.fold}" / args.subset / "proposals.csv"),
    }
    for nome, ruta in ramas.items():
        if not ruta.exists():
            print(f"falta {nome}: {ruta}")
            return 1

    gt = cargar_gt(args.clase, args.subset)
    n_inst = sum(len(v) for v in gt.values())
    print(f"{args.clase} · {args.subset} · {len(gt)} vídeos · {n_inst} instancias\n")

    cargadas = {n: cargar_propostas(r) for n, r in ramas.items()}
    for nome, p in cargadas.items():
        tot = sum(len(v) for v in p.values())
        print(f"  {nome:<6} {tot:>10,} propostas en {len(p)} vídeos")

    for umbral in (0.3, 0.5, 0.7):
        print(f"\n  AR con tIoU {umbral}")
        print(f"    {'N':>6}  {'reTAG':>9}  {'noso':>9}  {'dif':>9}")
        for n in (10, 50, 100, 500, 1000):
            fila = {}
            for nome, p in cargadas.items():
                fila[nome], _, _ = ar(p, gt, n, umbral)
            d = fila["noso"] - fila["reTAG"]
            print(f"    {n:>6}  {fila['reTAG']:>9.4f}  {fila['noso']:>9.4f}  {d:>+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
