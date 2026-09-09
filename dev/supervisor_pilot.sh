#!/bin/bash
# Relanza o piloto THUMOS14-E se cae. Para cando remate: crea FEITO.
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
[ -f $R/tmp/thumos14e_supervised/original_rate_v1/FEITO ] && exit 0
[ -f $R/tmp/pilot.pid ] && kill -0 $(cat $R/tmp/pilot.pid 2>/dev/null) 2>/dev/null && exit 0
echo "[$(date -Is)] relanzando piloto" >> $R/tmp/supervisor_pilot.log
setsid nohup env PYTHONPATH=.:dev PROPOSALS_POOL=1 $R/pyenv/bin/python dev/run_thumos14e_pilot.py \
  --work-dir $R/data/thumos14_events/thumos14e_original_rate_v1 \
  --out-root $R/tmp/thumos14e_supervised/original_rate_v1 \
  --target-class Diving --seed 1337 \
  --source-model $R/models/model.pk \
  --source-prototype $R/tmp/prototype/ed_prototype.npy \
  --canonical-annotations /home/pablo.garcia.seijo/actionformer_release/data/thumos/annotations/thumos14.json \
  --actionformer-root /home/pablo.garcia.seijo/actionformer_release \
  --conversion-role sensitivity_original_rate --device cuda --num-workers 0 \
  >> $R/tmp/pilot_diving.log 2>&1 &
echo $! > /home/pablo.garcia.seijo/event_penguins/tmp/pilot.pid
