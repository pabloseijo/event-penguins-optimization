#!/bin/bash
# Extrae as features ATSN compartidas (etapa shared-atsn) DENTRO dunha reserva de GPU.
# Sen "gpu exec" o enforcer do CiTIUS manda SIGTERM aos ~60 s: gpu_reservation_missing.
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
PLAN=$R/tmp/thumos14e_supervised/original_rate_v1/protocol_manifest.json
exec /usr/local/bin/gpu exec --numgpus 1 -- \
  env PYTHONPATH=.:dev $R/pyenv/bin/python dev/run_thumos14e_full_pipeline.py shared-extract \
    --plan $PLAN \
    --shard-index 0 --num-shards 1 \
    --batch-size 256 --num-workers 8 --device cuda
