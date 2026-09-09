#!/bin/bash
# Propostas de EventPenguins por folds, en paralelo ENTRE folds e secuencial
# DENTRO de cada un.
#
# Por que así: o Pool de multiprocessing bloquéase con h5py sobre este corpus
# (parou de forma reproducible entre a gravación 139 e a 193 con 1, 3 e 5
# workers, e tamén con spawn). O paralelismo real vén de procesos separados,
# que o sistema operativo illa de verdade.
#
# Seguridade comprobada antes de escribir isto:
#   · cada fold escribe en proposals/<branch>/<clase>/fold_NN/, artefactos e logs
#   · non hai ningún ficheiro compartido entre folds
#   · run_stage e este script saltan o fold se o CSV xa existe
#
# Memoria: as gravacións grandes chegan a 10 GB. Con 62 GB na máquina, tres
# folds á vez deixan margen; cinco non.
set -u
R=/home/pablo.garcia.seijo/event_penguins
OUT=$R/tmp/thumos14e_supervised/original_rate_v1
PLAN=$OUT/protocol_manifest.json
CLASE=${1:-Diving}
MAX_PARALELO=${2:-3}

cd $R || exit 1
mkdir -p $OUT/paralelo_logs

lanzar_fold() {
  local f=$1
  local csv="$OUT/proposals/eventpenguins_stage1/$CLASE/fold_$(printf %02d $f)/validation/proposals.csv"
  if [ -f "$csv" ]; then
    echo "[$(date +%H:%M)] fold $f xa feito, sáltoo"
    return 0
  fi
  echo "[$(date +%H:%M)] fold $f lanzado"
  env PYTHONPATH=.:dev PROPOSALS_POOL=1 $R/pyenv/bin/python \
    dev/run_thumos14e_supervised.py proposals \
    --plan $PLAN --branch eventpenguins_stage1 \
    --target-class $CLASE --split validation --cv-fold $f \
    > $OUT/paralelo_logs/fold_$f.log 2>&1
  local rc=$?
  echo "[$(date +%H:%M)] fold $f rematado rc=$rc"
  return $rc
}

activos=0
for f in 0 1 2 3 4; do
  lanzar_fold $f &
  activos=$((activos + 1))
  if [ $activos -ge $MAX_PARALELO ]; then
    wait -n              # agarda a que remate calquera, e lanza o seguinte
    activos=$((activos - 1))
  fi
done
wait

echo "=== propostas por fold rematadas · $(date -Is) ==="
for f in 0 1 2 3 4; do
  csv="$OUT/proposals/eventpenguins_stage1/$CLASE/fold_$(printf %02d $f)/validation/proposals.csv"
  if [ -f "$csv" ]; then
    echo "  fold $f  ✓  $(wc -l < $csv) liñas"
  else
    echo "  fold $f  ✗  SEN CSV"
  fi
done
