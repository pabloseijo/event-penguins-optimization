#!/usr/bin/env python3
"""Conversión concorrente de THUMOS14-E cun pool de traballos e semáforo por GPU.

Segue o patrón de `Mimir/wiki/metodo/scraping-concurrente-raspberry.md`
(ThreadPoolExecutor + semáforo + backoff + log por worker), adaptado a GPU:
o semáforo limita procesos simultáneos **por tarxeta**, non por dominio.

Fronte ao sharding estático (`--shard-index 0/1`), unha cola compartida evita
que unha GPU quede parada mentres a outra remata os vídeos longos.

Delega cada vídeo no conversor canónico, así que `conversion.json`, os hashes e
a trazabilidade quedan exactamente igual.

    ./pyenv/bin/python dev/convert_pool.py --top-n 5 --per-gpu 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/pablo.garcia.seijo/event_penguins")
MAIN = ROOT / "data/thumos14_events/thumos14e_v1"
V2E = Path("/home/pablo.garcia.seijo/v2e/v2e.py")

LOCK = threading.Lock()
JOURNAL = MAIN / "logs" / "pool_journal.jsonl"


def dispoñibles() -> list[int]:
    """GPUs reservadas para nós, segundo o xestor do CiTIUS."""
    out = subprocess.run(["/usr/local/bin/gpu", "env"], capture_output=True, text=True).stdout
    for tok in out.replace("=", " ").split():
        if "," in tok or tok.isdigit():
            try:
                return [int(x) for x in tok.split(",")]
            except ValueError:
                continue
    return [0, 1]


def rexistrar(**campos) -> None:
    with LOCK:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with JOURNAL.open("a") as fh:
            fh.write(json.dumps({"ts": time.time(), **campos}) + "\n")


def converter(video_id: str, sems: dict[int, threading.Semaphore],
              args: argparse.Namespace, intentos: int = 3) -> tuple[str, bool, float]:
    """Converte un vídeo na primeira GPU con oco. Backoff entre reintentos."""
    if (MAIN / "v2e" / video_id / "conversion.json").exists():
        return video_id, True, 0.0

    for intento in range(intentos):
        gpu = None
        for g, sem in sems.items():                  # busca oco sen bloquear
            if sem.acquire(blocking=False):
                gpu = g
                break
        if gpu is None:                              # todas ocupadas: espera na primeira
            gpu = next(iter(sems))
            sems[gpu].acquire()

        t0 = time.time()
        try:
            cmd = [
                str(ROOT / "pyenv/bin/python"),
                "dev/prepare_thumos14_event_corpus.py", "convert",
                "--work-dir", str(MAIN),
                "--video-ids", video_id,
                "--v2e-entry", str(V2E),
                "--v2e-python", str(ROOT / "pyenv/bin/python"),
                "--v2e-pythonpath", str(ROOT / "dev/v2e_headless_stubs"),
                "--fixed-timestamp-resolution", "--timestamp-resolution", "0.003",
                "--dvs-profile", "clean",
            ]
            env = {"PYTHONPATH": ".:dev", "CUDA_VISIBLE_DEVICES": str(gpu),
                   "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/pablo.garcia.seijo"}
            log = MAIN / "logs" / f"pool_{video_id}.log"
            with log.open("w") as fh:
                r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            if r.returncode == 0 and (MAIN / "v2e" / video_id / "conversion.json").exists():
                rexistrar(video=video_id, gpu=gpu, segundos=round(dt, 1), ok=True)
                return video_id, True, dt
            rexistrar(video=video_id, gpu=gpu, segundos=round(dt, 1), ok=False,
                      intento=intento, rc=r.returncode)
        finally:
            sems[gpu].release()

        if intento < intentos - 1:
            time.sleep(20 * (intento + 1))           # backoff

    return video_id, False, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, required=True, help="tanda: top-N clases predeclaradas")
    ap.add_argument("--per-gpu", type=int, default=3, help="procesos simultáneos por tarxeta")
    args = ap.parse_args()

    ids = subprocess.run(
        [str(ROOT / "pyenv/bin/python"), "dev/thumos14e_batch_ids.py",
         "--top-n", str(args.top_n), "--only-new"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()

    gpus = dispoñibles()
    sems = {g: threading.Semaphore(args.per_gpu) for g in gpus}
    workers = len(gpus) * args.per_gpu

    print(f"tanda top-{args.top_n}: {len(ids)} vídeos por converter")
    print(f"GPUs {gpus} · {args.per_gpu} por GPU · {workers} workers")
    if not ids:
        return 0

    t0, feitos, fallos = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(converter, v, sems, args): v for v in ids}
        for fut in as_completed(futs):
            vid, ok, dt = fut.result()
            feitos += 1
            if not ok:
                fallos.append(vid)
            transcorrido = time.time() - t0
            ritmo = feitos / (transcorrido / 3600) if transcorrido > 0 else 0
            restante = (len(ids) - feitos) / ritmo if ritmo > 0 else 0
            print(f"  [{feitos:>3}/{len(ids)}] {'✓' if ok else '✗'} {vid} "
                  f"{dt/60:.1f} min · {ritmo:.1f}/h · faltan {restante:.1f} h", flush=True)

    print(f"\ntanda top-{args.top_n}: {feitos - len(fallos)} ok, {len(fallos)} fallos, "
          f"{(time.time() - t0)/3600:.1f} h")
    if fallos:
        print("fallaron:", " ".join(fallos))
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
