"""Comproba o corpus real contra as MESMAS invariantes que esixe o pipeline.

Replica validate_event_dataset (dev/prepare_thumos14_event_corpus.py:1398-1428)
e as comprobacions de build_index (dev/extract_continuous_features.py:145-190),
para cazar problemas ANTES de gastar horas de transferencia:

  - xerarquia /recording/N01/events e attrs split/official_subset/cv_fold
  - events: ndim==2, shape[1]==4, len>0
  - timestamps non decrecentes (dentro de cada bloque e entre bloques)
  - 0 <= x_min <= x_max < width  e  o mesmo para y
  - polaridades subconxunto de {0,1}
  - duration_s finito e > 0

Por defecto le o ficheiro enteiro por bloques (--completo) ou so os extremos
(--rapido), que e o que se usa cando o disco e lento.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import h5py
import numpy as np

BLOQUE = 4_000_000


def revisar(path: str, completo: bool) -> dict:
    fallos: list[str] = []
    info: dict = {"video_id": os.path.basename(path)[:-3]}
    try:
        with h5py.File(path, "r") as f:
            if "recording" not in f:
                return {**info, "fallos": ["sen grupo /recording"]}
            rec = f["recording"]
            for a in ("split", "official_subset", "cv_fold", "duration_s"):
                if a not in rec.attrs:
                    fallos.append(f"falta attr /recording.{a}")
            if "N01" not in rec:
                return {**info, "fallos": fallos + ["sen grupo N01"]}
            g = rec["N01"]
            w = int(g.attrs.get("width", 0))
            h = int(g.attrs.get("height", 0))
            dur = float(rec.attrs.get("duration_s", 0.0))
            if not np.isfinite(dur) or dur <= 0:
                fallos.append(f"duration_s invalida: {dur}")
            e = g["events"]
            info.update({"eventos": int(e.shape[0]), "duration_s": round(dur, 4)})
            if e.ndim != 2 or e.shape[1] != 4:
                fallos.append(f"forma {e.shape}")
                return {**info, "fallos": fallos}
            if e.shape[0] == 0:
                fallos.append("dataset baleiro")
                return {**info, "fallos": fallos}

            xmin = ymin = 2**31
            xmax = ymax = -1
            pols: set[int] = set()
            anterior = -1
            n = e.shape[0]
            if completo or n <= 2 * BLOQUE:
                # ficheiro pequeno: lese enteiro, e asi os tramos nunca se solapan
                tramos = [(i, min(i + BLOQUE, n)) for i in range(0, n, BLOQUE)]
            else:
                # extremos, disxuntos por construcion porque n > 2*BLOQUE
                tramos = [(0, BLOQUE), (n - BLOQUE, n)]
            for a, z in tramos:
                b = np.asarray(e[a:z]).astype(np.int64)
                t = b[:, 2]
                if np.any(np.diff(t) < 0):
                    fallos.append(f"timestamps decrecentes en [{a},{z})")
                if anterior >= 0 and t[0] < anterior:
                    fallos.append(f"timestamp retrocede na costura de {a}")
                anterior = int(t[-1])
                xmin = min(xmin, int(b[:, 0].min())); xmax = max(xmax, int(b[:, 0].max()))
                ymin = min(ymin, int(b[:, 1].min())); ymax = max(ymax, int(b[:, 1].max()))
                pols |= set(np.unique(b[:, 3]).tolist())
            if not (0 <= xmin <= xmax < w):
                fallos.append(f"x fora de rango: [{xmin},{xmax}] con width={w}")
            if not (0 <= ymin <= ymax < h):
                fallos.append(f"y fora de rango: [{ymin},{ymax}] con height={h}")
            if not pols <= {0, 1}:
                fallos.append(f"polaridades {sorted(pols)}")
            info.update({"x": [xmin, xmax], "y": [ymin, ymax], "p": sorted(pols)})
    except Exception as exc:  # noqa: BLE001
        fallos.append(f"excepcion: {exc!r}")
    return {**info, "fallos": fallos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--lista", default="manifest413.csv")
    ap.add_argument("--completo", action="store_true")
    ap.add_argument("--saida", default="validacion_corpus.json")
    args = ap.parse_args()

    with open(args.lista, encoding="utf-8") as fh:
        esperados = {r["video_id"] for r in csv.DictReader(fh)}
    presentes = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(args.corpus, "*.h5"))}
    faltan = sorted(esperados - presentes)
    sobran = sorted(presentes - esperados)

    resultados = []
    malos = []
    for vid in sorted(presentes):
        r = revisar(os.path.join(args.corpus, vid + ".h5"), args.completo)
        resultados.append(r)
        if r["fallos"]:
            malos.append(r)
            print(f"MAL  {vid}: {'; '.join(r['fallos'])}", flush=True)

    total_ev = sum(r.get("eventos", 0) for r in resultados)
    with open(args.saida, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "modo": "completo" if args.completo else "extremos",
                "esperados": len(esperados),
                "presentes": len(presentes),
                "faltan": faltan,
                "sobran": sobran,
                "con_fallos": [r["video_id"] for r in malos],
                "eventos_totais": total_ev,
                "detalle": resultados,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"\nesperados {len(esperados)} · presentes {len(presentes)} · "
          f"faltan {len(faltan)} · con fallos {len(malos)}")
    if faltan:
        print("faltan:", ", ".join(faltan))
    print(f"eventos totais: {total_ev:,}")


if __name__ == "__main__":
    main()
