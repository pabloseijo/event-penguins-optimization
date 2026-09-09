#!/bin/bash
# Conversión principal THUMOS14-E: validation a 3 ms interpolado (SuperSloMo, clean).
# Agarda a que remate a sensibilidade de cadencia orixinal para non competir por GPU.
set -u
ROOT=/home/pablo.garcia.seijo/event_penguins
ORIG=$ROOT/data/thumos14_events/thumos14e_original_rate_v1/v2e
MAIN=$ROOT/data/thumos14_events/thumos14e_v1
SHARD=$1

while [ $(find $ORIG -name conversion.json 2>/dev/null | wc -l) -lt 413 ]; do sleep 60; done
sleep 120   # marxe para que o piloto Diving reclame a súa GPU primeiro

cd $ROOT
mkdir -p $MAIN/logs
IDS=$(python3 -c "
import csv
rows=list(csv.DictReader(open('$MAIN/manifest.csv')))
print(' '.join(r['video_id'] for r in rows if r['official_subset']=='validation'))
")

env PYTHONPATH=.:dev /usr/local/bin/gpu exec $2 ./pyenv/bin/python \
  dev/prepare_thumos14_event_corpus.py convert \
  --work-dir $MAIN \
  --video-ids $IDS \
  --v2e-entry /home/pablo.garcia.seijo/v2e/v2e.py \
  --v2e-python $ROOT/pyenv/bin/python \
  --v2e-pythonpath $ROOT/dev/v2e_headless_stubs \
  --fixed-timestamp-resolution --timestamp-resolution 0.003 \
  --dvs-profile clean \
  --shard-index $SHARD --num-shards 2 \
  > $MAIN/logs/main_val_shard$SHARD.log 2>&1
