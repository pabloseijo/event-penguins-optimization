#!/usr/bin/env python3
"""Average Recall por fold: propostas de reTAG fronte ás nosas.

Esta comparación **non usa o encoder**, así que evita a fuga que bloquea o
experimento A completo: `models/model.pk` é o checkpoint publicado por reTAG,
adestrado co train oficial, e as gravacións de validación dos folds están nese
train. As propostas, en cambio, non teñen ningún parámetro aprendido.

Mide o que aporta a etapa 1: canta cobertura dá cada xerador de candidatos ao
mesmo orzamento de propostas.

    python dev/eval_ar_por_fold.py --variante proposals_adaptive_spatial_noise.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/pablo.garcia.seijo/event_penguins")
DEBUG = ROOT / "tmp/debug"
FOLDS = ROOT / "tmp/cv/recording_folds_r5"


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def cargar_propostas(ruta: Path) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
    """(rec_name, roi_id) → [(t_start, t_end, score)] ordenadas por score desc."""
    out = defaultdict(list)
    with ruta.open() as fh:
        for r in csv.DictReader(fh):
            out[(r["rec_name"], str(int(r["roi_id"][1:])))].append(
                (float(r["t_start"]), float(r["t_end"]), float(r.get("score", 0.0))))
    for k in out:
        out[k].sort(key=lambda x: -x[2])
    return out


def cargar_gt(ann: Path, etiqueta: str = "ed") -> dict[tuple[str, str], list[tuple[float, float]]]:
    """(rec_name, roi_id) -> [(inicio, fin)] da clase pedida."""
    db = json.loads(ann.read_text())["database"]
    out = defaultdict(list)
    for rec, contido in db.items():
        for roi, anots in contido.get("annotations", {}).items():
            for a in anots:
                if a.get("label") == etiqueta:
                    s = a["segment"]
                    out[(rec, str(roi))].append((float(s[0]), float(s[1])))
    return out


def ar_en(props, gt, recs: set[str], n: int, umbral: float) -> tuple[float, int]:
    """Average Recall con orzamento de N propostas por ROI."""
    total = acertos = 0
    for chave, instancias in gt.items():
        if chave[0] not in recs:
            continue
        candidatos = props.get(chave, [])[:n]
        for ins in instancias:
            total += 1
            if any(tiou((c[0], c[1]), ins) >= umbral for c in candidatos):
                acertos += 1
    return (acertos / total if total else 0.0), total


def folds_val() -> dict[str, set[str]]:
    out = {}
    with (FOLDS / "manifest.csv").open() as fh:
        for r in csv.DictReader(fh):
            out[r["fold"]] = set(r["val_record_names"].split())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="proposals_baseline.csv")
    ap.add_argument("--variante", default="proposals_adaptive_spatial_noise.csv")
    ap.add_argument("--ann", default=str(ROOT / "config/annotations/annotations.json"))
    ap.add_argument("--n", type=int, default=50, help="orzamento de propostas por ROI")
    ap.add_argument("--tiou", type=float, default=0.5)
    ap.add_argument("--etiqueta", default="ed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = cargar_propostas(DEBUG / args.baseline)
    nosa = cargar_propostas(DEBUG / args.variante)
    gt = cargar_gt(Path(args.ann), args.etiqueta)
    fv = folds_val()

    print(f"AR@{args.n} con tIoU {args.tiou}\n")
    print(f"{'fold':<6} {'gravacións':>11} {'inst':>6} {'reTAG':>9} {'nosa':>9} {'dif':>9}")
    ra, rb = {}, {}
    for f, recs in sorted(fv.items()):
        a, n_inst = ar_en(base, gt, recs, args.n, args.tiou)
        b, _ = ar_en(nosa, gt, recs, args.n, args.tiou)
        ra[f], rb[f] = a, b
        print(f"{f:<6} {len(recs):>11} {n_inst:>6} {a:>9.4f} {b:>9.4f} {b-a:>+9.4f}")

    ma = sum(ra.values()) / len(ra)
    mb = sum(rb.values()) / len(rb)
    print(f"{'media':<6} {'':>11} {'':>6} {ma:>9.4f} {mb:>9.4f} {mb-ma:>+9.4f}")
    print(f"{'peor':<6} {'':>11} {'':>6} {min(ra.values()):>9.4f} {min(rb.values()):>9.4f}")

    if args.out:
        Path(args.out + ".retag.json").write_text(json.dumps({"por_fold": ra}, indent=1))
        Path(args.out + ".nosa.json").write_text(json.dumps({"por_fold": rb}, indent=1))
        print(f"\ngardado en {args.out}.{{retag,nosa}}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
