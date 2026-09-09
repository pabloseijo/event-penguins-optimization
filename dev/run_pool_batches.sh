#!/bin/bash
# THUMOS14-E principal por tandas, cun pool concorrente e semáforo por GPU.
set -u
ROOT=/home/pablo.garcia.seijo/event_penguins
ORIG=$ROOT/data/thumos14_events/thumos14e_original_rate_v1/v2e
PER_GPU=${1:-3}
cd $ROOT

while [ $(find $ORIG -name conversion.json 2>/dev/null | wc -l) -lt 413 ]; do sleep 60; done
sleep 180   # marxe para o piloto Diving

for N in 5 10 20; do
  echo "=== tanda top-$N · $(date -Is) ==="
  env PYTHONPATH=.:dev ./pyenv/bin/python dev/convert_pool.py --top-n $N --per-gpu $PER_GPU
  echo "tanda $N rematada: $(date -Is)"
done
