#!/bin/bash
# THUMOS14-E principal (3 ms interpolado) por tandas de clases.
# Orde PREDECLARADO por instancias en validation, fixado antes de ver ningún resultado.
# Cada tanda converte validation+test das clases correspondentes, así queda avaliable.
set -u
ROOT=/home/pablo.garcia.seijo/event_penguins
ORIG=$ROOT/data/thumos14_events/thumos14e_original_rate_v1/v2e
MAIN=$ROOT/data/thumos14_events/thumos14e_v1
SHARD=$1
POLICY=$2

while [ $(find $ORIG -name conversion.json 2>/dev/null | wc -l) -lt 413 ]; do sleep 60; done
sleep 120

cd $ROOT
mkdir -p $MAIN/logs

for N in 5 10 20; do
  IDS=$(./pyenv/bin/python dev/thumos14e_batch_ids.py --top-n $N)
  env PYTHONPATH=.:dev /usr/local/bin/gpu exec $POLICY ./pyenv/bin/python \
    dev/prepare_thumos14_event_corpus.py convert \
    --work-dir $MAIN --video-ids $IDS \
    --v2e-entry /home/pablo.garcia.seijo/v2e/v2e.py \
    --v2e-python $ROOT/pyenv/bin/python \
    --v2e-pythonpath $ROOT/dev/v2e_headless_stubs \
    --fixed-timestamp-resolution --timestamp-resolution 0.003 \
    --dvs-profile clean \
    --shard-index $SHARD --num-shards 2 \
    >> $MAIN/logs/main_batch${N}_shard${SHARD}.log 2>&1
  echo "tanda $N rematada shard $SHARD: $(date -Is)" >> $MAIN/logs/batches.log
done
