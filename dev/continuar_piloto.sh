#!/bin/bash
# Agarda a que remate o último fold e relanza o piloto, que é idempotente:
# salta os cinco folds xa feitos e segue por test, shared-atsn e as cabezas.
# PROPOSALS_POOL=1 obrigatorio: con máis, o OOM killer mata o proceso.
set -u
R=/home/pablo.garcia.seijo/event_penguins
OUT=$R/tmp/thumos14e_supervised/original_rate_v1
cd $R || exit 1

while pgrep -f 'cv-fold' > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] folds rematados, continúo o piloto"

env PYTHONPATH=.:dev PROPOSALS_POOL=1 /usr/local/bin/gpu exec --guaranteed $R/pyenv/bin/python dev/run_thumos14e_pilot.py \
  --work-dir $R/data/thumos14_events/thumos14e_original_rate_v1 \
  --out-root $OUT --target-class Diving --seed 1337 \
  --source-model $R/models/model.pk \
  --source-prototype $R/tmp/prototype/ed_prototype.npy \
  --canonical-annotations /home/pablo.garcia.seijo/actionformer_release/data/thumos/annotations/thumos14.json \
  --actionformer-root /home/pablo.garcia.seijo/actionformer_release \
  --conversion-role sensitivity_original_rate --device cuda --num-workers 8
echo "[$(date +%H:%M)] piloto rc=$?"
