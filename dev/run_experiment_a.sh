#!/bin/bash
# Experimento A: reTAG nos 5 folds disxuntos por gravación.
# Protocolo predeclarado en protocolo-comparacion-xusta-retag-eventpenguins-2026-08-11.
# Dous folds á vez, un por GPU, para non saturar os 20 cores.
set -u
ROOT=/home/pablo.garcia.seijo/event_penguins
cd $ROOT
mkdir -p tmp/experiment_a/logs

for F in 00 01 02 03 04; do
  D=tmp/cv/recording_folds_r5/fold_$F
  echo "=== fold $F · $(date -Is) ==="
  env PYTHONPATH=.:dev /usr/local/bin/gpu exec --guaranteed ./pyenv/bin/python \
    dev/train_atsn_lpft.py \
    --train-proposals $D/train_proposals.csv \
    --val-proposals $D/val_proposals.csv \
    --out-dir tmp/experiment_a/retag_fold_$F \
    > tmp/experiment_a/logs/retag_fold_$F.log 2>&1
  echo "fold $F rematado: $(date -Is) rc=$?"
done
echo "=== EXPERIMENTO A (rama reTAG) COMPLETO · $(date -Is) ==="
