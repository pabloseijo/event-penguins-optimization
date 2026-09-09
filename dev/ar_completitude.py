#!/usr/bin/env python3
"""Efecto da completitude contexto-relativa no ranking, sen clasificador.

C(p) = a_in - 0.5*(a_left + a_right) e aritmetica sobre o actionness, asi que
mide a contribucion do artigo sen necesitar as 368 h de extraccion de features.

Compara AR@N coas propostas ordenadas polo score orixinal fronte a reordenadas
co score fusionado 0.75*rank(score) + 0.25*rank(C), que e a receita do artigo.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, h5py

csv.field_size_limit(sys.maxsize)
R = Path("/home/pablo.garcia.seijo/event_penguins")
TH = R / "tmp/thumos14e_supervised/original_rate_v1"
CAN = R / "data/thumos14_events/thumos14e_original_rate_v1/canonical"
MAN = R / "data/thumos14_events/thumos14e_original_rate_v1/manifest.csv"
BIN = 0.033


def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def actionness(vid, dur):
    with h5py.File(CAN / f"{vid}.h5", "r") as h:
        ts = h["recording/N01/events"][:, 0].astype(np.float64)
    t = (ts - ts.min()) / 1e6
    n = max(1, int(dur / BIN))
    c, _ = np.histogram(t, bins=n, range=(0, dur))
    return c / BIN


def completitude(a, ini, fin):
    """C(p) co contexto de 0.5|p| a cada lado, como no artigo."""
    n = len(a)
    i0, i1 = int(ini / BIN), min(n, int(fin / BIN) + 1)
    if i1 <= i0:
        return 0.0
    ctx = max(1, (i1 - i0) // 2)
    dentro = a[i0:i1].mean()
    esq = a[max(0, i0 - ctx):i0]
    der = a[i1:min(n, i1 + ctx)]
    fora = 0.5 * ((esq.mean() if len(esq) else 0.0) + (der.mean() if len(der) else 0.0))
    return dentro - fora


def rangos(v):
    o = np.argsort(np.argsort(v))
    return o / max(1, len(v) - 1)


def ar(props, gt, n, u=0.5):
    t = h = 0
    for vid, ins in gt.items():
        c = props.get(vid, [])[:n]
        for x in ins:
            t += 1
            if any(tiou(y, x) >= u for y in c):
                h += 1
    return h / t if t else 0.0


def main():
    clase = sys.argv[1] if len(sys.argv) > 1 else "Diving"
    dur = {}
    gt = defaultdict(list)
    for r in csv.DictReader(open(MAN)):
        if r["official_subset"] != "validation":
            continue
        dur[r["video_id"]] = None
        for a in json.loads(r["annotations_json"]):
            if a["label"] == clase:
                gt[r["video_id"]].append(tuple(a["segment"]))
    for v in dur:
        m = json.load(open(R / "data/thumos14_events/thumos14e_original_rate_v1"
                           / "source_metadata" / f"{v}.json"))
        dur[v] = float(m.get("duration") or m.get("duration_s"))

    ruta = TH / "proposals/eventpenguins_stage1" / clase / "fold_00/validation/proposals.csv"
    bruto = defaultdict(list)
    for r in csv.DictReader(open(ruta)):
        bruto[r["rec_name"]].append((float(r["t_start"]) / 1e6,
                                     float(r["t_end"]) / 1e6, float(r["score"])))

    base, comp = {}, {}
    for vid in gt:                      # so os videos coa clase, que e onde se mide
        p = bruto.get(vid, [])
        if not p:
            continue
        a = actionness(vid, dur[vid])
        cs = np.array([completitude(a, x[0], x[1]) for x in p])
        sc = np.array([x[2] for x in p])
        fus = 0.75 * rangos(sc) + 0.25 * rangos(cs)
        base[vid] = [(x[0], x[1]) for x in sorted(p, key=lambda y: -y[2])]
        comp[vid] = [(p[i][0], p[i][1]) for i in np.argsort(-fus)]

    print(f"{clase} · {len(gt)} vídeos · {sum(len(v) for v in gt.values())} instancias\n")
    print(f"  {'N':>6}  {'score só':>10}  {'+ C(p)':>10}  {'dif':>9}")
    for n in (50, 100, 500, 1000):
        a1, a2 = ar(base, gt, n), ar(comp, gt, n)
        print(f"  {n:>6}  {a1:>10.4f}  {a2:>10.4f}  {a2-a1:>+9.4f}")


if __name__ == "__main__":
    main()
