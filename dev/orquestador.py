#!/usr/bin/env python3
"""Orquestador autónomo do traballo pendente de WACV.

Deseñado para sobrevivir ao que rompeu antes: non depende de tmux, non encadea
sesións e non asume que ningún proceso pai siga vivo. Execútase con `setsid`, así
que non morre ao pechar SSH, e un cron relánzao se cae.

Propiedades:
  · **idempotente** — cada fase comproba o seu artefacto e sáltase se xa está feita
  · **secuencial por deseño** — as fases pesadas non compiten pola CPU, que foi o
    que degradou todo cando corrían tres cousas á vez (load 113 sobre 20 cores)
  · **con journal** — `orquestador.jsonl` rexistra inicio, fin e resultado

    setsid nohup ./pyenv/bin/python dev/orquestador.py > tmp/orquestador.out 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/pablo.garcia.seijo/event_penguins")
EXPA = ROOT / "tmp/experiment_a"
MAIN = ROOT / "data/thumos14_events/thumos14e_v1"
JOURNAL = ROOT / "tmp/orquestador.jsonl"
LOCK = ROOT / "tmp/orquestador.lock"


def rexistrar(**campos) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **campos}) + "\n")


def correr(cmd: list[str], log: Path, fase: str) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    rexistrar(fase=fase, estado="inicio")
    t0 = time.time()
    env = {"PYTHONPATH": ".:dev", "PATH": "/usr/local/bin:/usr/bin:/bin",
           "HOME": str(ROOT.parent)}
    with log.open("a") as fh:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    ok = r.returncode == 0
    rexistrar(fase=fase, estado="fin", ok=ok, rc=r.returncode,
              minutos=round((time.time() - t0) / 60, 1))
    return ok


# ---------------------------------------------------------------- fases

def fase_a_retag() -> bool:
    """Experimento A, rama reTAG: 5 folds disxuntos por gravación."""
    for f in ("00", "01", "02", "03", "04"):
        destino = EXPA / f"retag_fold_{f}"
        if (destino / "metrics.csv").exists() and (destino / "predictions").exists():
            rexistrar(fase=f"a_retag_{f}", estado="saltada", motivo="xa existe")
            continue
        d = ROOT / f"tmp/cv/recording_folds_r5/fold_{f}"
        ok = correr(
            [str(ROOT / "pyenv/bin/python"), "dev/train_atsn_lpft.py",
             "--train-proposals", str(d / "train_proposals.csv"),
             "--val-proposals", str(d / "val_proposals.csv"),
             "--out-dir", str(destino)],
            EXPA / "logs" / f"retag_fold_{f}.log", f"a_retag_{f}")
        if not ok:
            return False
    return True


def fase_thumos(top_n: int) -> bool:
    """Conversión a 3 ms da tanda top-N. O pool xa é idempotente por vídeo."""
    return correr(
        [str(ROOT / "pyenv/bin/python"), "dev/convert_pool.py",
         "--top-n", str(top_n), "--per-gpu", "2"],
        MAIN / "logs" / f"pool_top{top_n}.log", f"thumos_top{top_n}")


FASES = [
    # FUGA: models/model.pk viu o train oficial, e as gravacions de validacion dos folds estan nese train
#    ("A · reTAG nos 5 folds", fase_a_retag),
    ("THUMOS14-E · tanda 5 clases", lambda: fase_thumos(5)),
    ("THUMOS14-E · tanda 10 clases", lambda: fase_thumos(10)),
    ("THUMOS14-E · tanda 20 clases", lambda: fase_thumos(20)),
]


def main() -> int:
    if LOCK.exists():
        idade = time.time() - LOCK.stat().st_mtime
        if idade < 3600:                       # outro proceso vivo hai menos dunha hora
            print(f"xa hai un orquestador activo (lock de {idade/60:.0f} min)")
            return 0
        rexistrar(fase="lock", estado="caducado", minutos=round(idade / 60))

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(time.time()))
    rexistrar(fase="orquestador", estado="arranque")

    try:
        for nome, fn in FASES:
            print(f"\n{'='*60}\n{nome}\n{'='*60}", flush=True)
            LOCK.write_text(str(time.time()))   # heartbeat
            if not fn():
                rexistrar(fase=nome, estado="fallou", nota="continúo coas seguintes")
                print(f"  ✗ {nome} fallou, sigo", flush=True)
            else:
                print(f"  ✓ {nome}", flush=True)
    finally:
        LOCK.unlink(missing_ok=True)
        rexistrar(fase="orquestador", estado="remate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
