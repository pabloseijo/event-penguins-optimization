"""Pre-constrúe a caché de timestamps das ROIs, en paralelo e con memoria acoutada.

Por que existe: ProposalDataset._get_roi_data (src/classification.py:174-186)
constrúe a caché a demanda facendo np.asarray(events[:, 2]) do ROI ENTEIRO. Nun
corpus de 51 G de eventos iso son varios GB por worker, e con moitos workers a
maquina pasa a swap. Medido: a extraccion sostida vai a 9 batches/100 s, e nese
mesmo intervalo so se constrúe UNHA cache. O gargalo non e a rede neuronal, e
construir estas 413 caches.

Este script fai o mesmo traballo por bloques (memoria acoutada), escribe co
mesmo convenio de rutas e coa mesma escritura atomica, e pode correr a carón da
extraccion: se os dous constrúen a mesma cache o resultado e identico e o
replace atomico resolve a carreira.

Percorre a lista ao REVES por defecto, para non pelexar coa extraccion, que vai
de principio a fin.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BLOQUE = 20_000_000  # eventos por lectura; fixa o pico de memoria (~0,3 GB)


def construir(args: tuple[str, str, str, str]) -> tuple[str, int, float, str | None]:
    rec, roi, data_path, cache_dir = args
    destino = Path(cache_dir) / rec / f"{roi}.npy"
    if destino.exists():
        return rec, 0, 0.0, "xa estaba"
    t0 = time.time()
    try:
        with h5py.File(data_path, "r") as f:
            dset = f[rec][roi]["events"]
            n = dset.shape[0]
            saida = np.empty(n, dtype=np.uint32)
            for a in range(0, n, BLOQUE):
                z = min(a + BLOQUE, n)
                saida[a:z] = dset[a:z, 2]
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_suffix(f".{os.getpid()}.npy.tmp")
        with temporal.open("wb") as fh:
            np.save(fh, saida)
        temporal.replace(destino)
        return rec, n, time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        return rec, 0, time.time() - t0, repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", required=True,
                    help="shared_features/continuous, onde vive sequences.csv")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--orde", choices=("directa", "inversa"), default="inversa")
    args = ap.parse_args()

    feature_dir = Path(args.feature_dir)
    cache_dir = feature_dir / "timestamp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seqs = pd.read_csv(feature_dir / "sequences.csv", keep_default_na=False)

    tarefas = [
        (str(r.rec_name), str(r.roi_key), args.data_path, str(cache_dir))
        for r in seqs.itertuples(index=False)
    ]
    if args.orde == "inversa":
        tarefas.reverse()
    pendentes = [t for t in tarefas if not (cache_dir / t[0] / f"{t[1]}.npy").exists()]

    print(f"[{time.strftime('%H:%M:%S')}] {len(tarefas)} ROIs, "
          f"{len(tarefas) - len(pendentes)} xa cacheadas, {len(pendentes)} pendentes, "
          f"{args.workers} workers, orde {args.orde}", flush=True)

    feitos = 0
    eventos = 0
    fallos = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futuros = [pool.submit(construir, t) for t in pendentes]
        for fut in as_completed(futuros):
            rec, n, seg, erro = fut.result()
            feitos += 1
            if erro and erro != "xa estaba":
                fallos += 1
                print(f"[{time.strftime('%H:%M:%S')}] FALLO {rec}: {erro}", flush=True)
                continue
            eventos += n
            if feitos % 10 == 0:
                transc = time.time() - t0
                resta = (len(pendentes) - feitos) * transc / max(feitos, 1)
                print(f"[{time.strftime('%H:%M:%S')}] {feitos}/{len(pendentes)} "
                      f"| {rec} {n:,} ev en {seg:.0f}s "
                      f"| restan ~{resta/60:.0f} min", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] REMATADO {feitos} ROIs, {eventos:,} eventos, "
          f"{fallos} fallos, en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
