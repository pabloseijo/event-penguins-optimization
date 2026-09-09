"""Escolle o subconxunto por DENSIDADE DE ANOTACIONS, non por tamaño.

A primeira version escollia as gravacions mais pequenas garantindo que cada
clase aparecese polo menos nunha. Resultado: 11 das 20 clases quedaron con
menos de 10 instancias anotadas, e tres cabezas de reTAG fallaron con
counts=[2907, 0] — cero positivos para adestrar. Cubrir unha clase non e o
mesmo que ter exemplos dela.

Criterio novo: para cada clase, engadir gravacions por orde de
    anotacions_desa_clase / eventos
ata acadar un obxectivo por clase. Iso maximiza os exemplos por hora de
extraccion, que e exactamente o que interesa cando o gargalo e o tempo.

O nesgo segue existindo e segue declarandose: agora favorece gravacions
densas en anotacions, non curtas.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.run_thumos14e_supervised import load_plan  # noqa: E402


def anotacions_por_clase(fila) -> Counter:
    c: Counter = Counter()
    bruto = fila.get("annotations_json") or ""
    if bruto:
        try:
            datos = json.loads(bruto)
            for item in datos if isinstance(datos, list) else []:
                if isinstance(item, dict) and item.get("label"):
                    c[item["label"]] += 1
        except (json.JSONDecodeError, TypeError):
            pass
    if not c and fila.get("labels_json"):
        for e in json.loads(fila["labels_json"]):
            c[e] += 1
    return c


def escoller(pool: pd.DataFrame, obxectivo: int, tope: int):
    """Greedy: en cada volta, a gravacion que mais aporta á clase mais feble."""
    info = {}
    for _, fila in pool.iterrows():
        info[fila["video_id"]] = (anotacions_por_clase(fila), int(fila["eventos"]))

    acumulado: Counter = Counter()
    elixidas: list[str] = []
    clases = sorted({c for a, _ in info.values() for c in a})

    while len(elixidas) < tope:
        febles = [c for c in clases if acumulado[c] < obxectivo]
        if not febles:
            break
        mellor, mellor_val = None, 0.0
        for vid, (a, ev) in info.items():
            if vid in elixidas or ev <= 0:
                continue
            aporte = sum(min(a[c], obxectivo - acumulado[c]) for c in febles)
            if aporte <= 0:
                continue
            val = aporte / (ev / 1e6)  # anotacions uteis por millon de eventos
            if val > mellor_val:
                mellor, mellor_val = vid, val
        if mellor is None:
            break
        elixidas.append(mellor)
        acumulado += info[mellor][0]
    return elixidas, acumulado, clases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--obxectivo", type=int, default=30)
    ap.add_argument("--tope", type=int, default=90)
    args = ap.parse_args()

    plan = load_plan(Path(args.plan))
    out_root = Path(plan["paths"]["out_root"])
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    manifest = pd.read_csv(plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False)

    with h5py.File(data_path, "r") as f:
        tam = {r: int(f[r]["N01"]["events"].shape[0]) for r in f.keys()}
    manifest["eventos"] = manifest["video_id"].map(tam)

    resultado = {}
    for split in ("validation", "test"):
        pool = manifest[manifest["official_subset"] == split].copy()
        elixidas, acumulado, clases = escoller(pool, args.obxectivo, args.tope)
        sel = pool[pool["video_id"].isin(elixidas)]
        febles = [c for c in clases if acumulado[c] < args.obxectivo]
        print(f"\n== {split} ==")
        print(f"  {len(elixidas)} gravacions de {len(pool)} · "
              f"{sel['eventos'].sum()/pool['eventos'].sum()*100:.1f}% dos eventos")
        print(f"  clases por debaixo de {args.obxectivo}: {len(febles)}")
        if febles:
            print("   ", ", ".join(f"{c}({acumulado[c]})" for c in febles))
        else:
            print(f"    todas as clases con >= {args.obxectivo} instancias")
        resultado[split] = {
            "gravacions": sorted(elixidas),
            "n": len(elixidas),
            "de": int(len(pool)),
            "instancias_por_clase": {c: int(acumulado[c]) for c in clases},
            "clases_febles": febles,
            "eventos": int(sel["eventos"].sum()),
            "fraccion_eventos": float(sel["eventos"].sum() / pool["eventos"].sum()),
        }

    for split in ("validation", "test"):
        p_in = out_root / "proposals" / "retag" / split / "proposals_completo.csv"
        if not p_in.exists():
            p_in = out_root / "proposals" / "retag" / split / "proposals_orixinal.csv"
        if not p_in.exists():
            print(f"  aviso: non atopo as propostas completas de {split}")
            continue
        p = pd.read_csv(p_in)
        sub = p[p["rec_name"].isin(resultado[split]["gravacions"])]
        sub.to_csv(p_in.parent / "proposals_sub2.csv", index=False)
        resultado[split]["propostas"] = int(len(sub))
        print(f"  {split}: {len(sub):,} de {len(p):,} propostas "
              f"({len(sub)/len(p)*100:.1f}%) -> proposals_sub2.csv")

    destino = out_root / "subconxunto2.json"
    destino.write_text(
        json.dumps(
            {
                "motivo": (
                    "Segunda version do subconxunto. A primeira escollia as gravacions "
                    "mais pequenas e deixou 11 das 20 clases con menos de 10 instancias "
                    "anotadas; tres cabezas de reTAG fallaron con cero positivos."
                ),
                "criterio": (
                    "greedy por anotacions utiles da clase mais feble por millon de "
                    "eventos, ata un obxectivo por clase. NESGO DECLARADO: favorece "
                    "gravacions densas en anotacions."
                ),
                "obxectivo_por_clase": args.obxectivo,
                "splits": resultado,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nescrito {destino}")


if __name__ == "__main__":
    main()
