#!/bin/bash
# Folds un tras outro. Medido: un só proceso chega a 28,9 GB nas gravacións
# grandes, así que en 62 GB non caben dous. Tres mataron o OOM killer (rc=137).
set -u
R=/home/pablo.garcia.seijo/event_penguins
OUT=$R/tmp/thumos14e_supervised/original_rate_v1
CLASE=${1:-Diving}
cd $R || exit 1
mkdir -p $OUT/paralelo_logs

for f in 1 2 3 4; do
  CSV=$OUT/proposals/eventpenguins_stage1/$CLASE/fold_0$f/validation/proposals.csv
  if [ -f "$CSV" ]; then echo "[$(date +%H:%M)] fold $f xa feito"; continue; fi
  # agarda a que non haxa outro fold corriendo
  while pgrep -f 'cv-fold' > /dev/null; do sleep 30; done
  echo "[$(date +%H:%M)] fold $f lanzado"
  env PYTHONPATH=.:dev PROPOSALS_POOL=1 $R/pyenv/bin/python \
    dev/run_thumos14e_supervised.py proposals --plan $OUT/protocol_manifest.json \
    --branch eventpenguins_stage1 --target-class $CLASE --split validation --cv-fold $f \
    > $OUT/paralelo_logs/fold_$f.log 2>&1
  echo "[$(date +%H:%M)] fold $f rc=$?"
done
echo "=== TODOS OS FOLDS · $(date -Is) ==="
