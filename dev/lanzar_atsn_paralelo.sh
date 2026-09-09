#!/bin/bash
# Extrae as features ATSN compartidas en 4 shards disxuntos, 2 por GPU.
# Require reserva viva (gpu claim); sen ela o enforcer manda SIGTERM aos ~60 s.
# Dimensionado sobre medidas do 2026-09-03: 20 nucleos, 62 GB, 1 shard con 8
# workers usaba 8 nucleos e 24 GB. 4 shards x 4 workers = 20 procesos.
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
PLAN=$R/tmp/thumos14e_supervised/original_rate_v1/protocol_manifest.json
eval "$(/usr/local/bin/gpu env --shell)"
for i in 0 1 2 3; do
  DEV=cuda:$((i % 2))
  setsid nohup env PYTHONPATH=.:dev $R/pyenv/bin/python \
    dev/run_thumos14e_full_pipeline.py shared-extract \
      --plan $PLAN \
      --shard-index $i --num-shards 4 \
      --batch-size 256 --num-workers 4 --device $DEV \
    > $R/tmp/atsn_shard_$i.log 2>&1 &
  echo "shard $i -> $DEV pid=$!"
  sleep 3
done
