"""Converte os 413 videos do corpus a partir dos .aedat4 reais de UIBK.

Resumible: se o .h5 de saida existe e abre ben, saltase. Paralelo por ficheiro.
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py

from converter_aedat4 import converter

RAIZ = Path(os.environ.get("THUMOS_AEDAT_RAIZ", "E:/thumos14-events"))
DIRS = {"test": "TH14_TEST_COMPRESSED", "validation": "TH14_VALID_COMPRESSED"}


def entrada_de(video_id, subset):
    return RAIZ / DIRS[subset] / f"{video_id}.aedat4"


def xa_feito(path):
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if "recording" not in f:
                return False
            g = f["recording"]
            return "N01" in g and g["N01"]["events"].shape[0] > 0 and "split" in g.attrs
    except Exception:
        return False


def tarefa(args):
    video_id, subset, saida, fila = args
    t0 = time.time()
    try:
        n, dur = converter(entrada_de(video_id, subset), Path(saida), fila)
        return video_id, n, dur, os.path.getsize(saida), time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        return video_id, 0, 0.0, 0, time.time() - t0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lista", default="manifest413.csv")
    ap.add_argument("--saida", default="D:/thumos14-real/corpus")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limite", type=int, default=None)
    args = ap.parse_args()

    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=True)
    with open(args.lista, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    if len(filas) != 413:
        raise ValueError(f"Esperabanse 413 videos no manifesto; hai {len(filas)}")
    orfos = list(destino.glob("*.parcial"))
    for o in orfos:
        o.unlink()
    if orfos:
        print(f"[limpeza] borrados {len(orfos)} ficheiros .parcial orfos", flush=True)
    pendentes = []
    for r in filas:
        saida = destino / f"{r['video_id']}.h5"
        if xa_feito(saida):
            continue
        pendentes.append((r["video_id"], r["official_subset"], str(saida), r))
    xa = len(filas) - len(pendentes)
    if args.limite:
        pendentes = pendentes[: args.limite]

    print(f"[{time.strftime('%H:%M:%S')}] {len(filas)} videos, "
          f"{xa} xa feitos, {len(pendentes)} pendentes, "
          f"{args.workers} workers", flush=True)

    feitos = 0
    bytes_tot = 0
    ev_tot = 0
    fallos = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futuros = {pool.submit(tarefa, p): p[0] for p in pendentes}
        for fut in as_completed(futuros):
            vid, n, dur, tam, seg, erro = fut.result()
            feitos += 1
            if erro:
                fallos.append((vid, erro))
                print(f"[{time.strftime('%H:%M:%S')}] FALLO {vid}: {erro}", flush=True)
                continue
            bytes_tot += tam
            ev_tot += n
            if feitos % 10 == 0 or feitos <= 5:
                transc = time.time() - t0
                ritmo = feitos / transc * 3600
                resta = (len(pendentes) - feitos) / max(ritmo, 1e-9)
                print(
                    f"[{time.strftime('%H:%M:%S')}] {feitos}/{len(pendentes)} "
                    f"| {vid} {n:,} ev {tam/2**20:.0f} MB en {seg:.0f}s "
                    f"| acumulado {bytes_tot/2**30:.1f} GB "
                    f"| ritmo {ritmo:.0f} videos/h | restan ~{resta:.1f} h",
                    flush=True,
                )
    print(f"[{time.strftime('%H:%M:%S')}] REMATADO {feitos} videos, "
          f"{ev_tot:,} eventos, {bytes_tot/2**30:.1f} GB, "
          f"{len(fallos)} fallos en {(time.time()-t0)/3600:.2f} h", flush=True)
    for vid, erro in fallos:
        print("  fallo:", vid, erro, flush=True)


if __name__ == "__main__":
    main()
