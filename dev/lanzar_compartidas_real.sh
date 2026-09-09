#!/bin/bash
# Etapas compartidas do corpus real: extraccion ATSN (2 shards, un por GPU) e
# estatisticas de eventos. Ambas se fan UNHA vez e sirven para as 20 clases.
#
# Require reserva viva: sen "gpu claim" o enforcer manda SIGTERM aos ~60 s.
# num-workers 6 por shard e nice 10 para non deixar a maquina de Antonio
# inutilizable: 2x6 workers + 2 mains son 14 dos 20 nucleos.
# OMP/MKL a 1 para evitar sobresubscricion de fios dentro de cada worker.
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
PLAN=$R/tmp/thumos14e_real/v1/protocol_manifest.json
eval "$(/usr/local/bin/gpu env --shell)"
BASE="env PYTHONPATH=.:dev OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"

for i in 0 1; do
  setsid nohup nice -n 10 $BASE $R/pyenv/bin/python \
    dev/run_thumos14e_full_pipeline.py shared-extract \
      --plan $PLAN --shard-index $i --num-shards 2 \
      --batch-size 64 --num-workers 8 --device cuda:$i \
    > $R/tmp/real_atsn_$i.log 2>&1 &
  echo "shard $i -> cuda:$i pid=$!"
  sleep 3
done

setsid nohup nice -n 15 $BASE $R/pyenv/bin/python \
  dev/run_thumos14e_full_pipeline.py shared-event-stats --plan $PLAN \
  > $R/tmp/real_event_stats.log 2>&1 &
echo "event-stats pid=$!"
