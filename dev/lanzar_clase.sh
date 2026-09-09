#!/bin/bash
# Corre a cadea completa dunha clase de THUMOS14 nas DUAS ramas.
# Uso: lanzar_clase.sh <Clase> <cuda:0|cuda:1> [num_workers] [plan]
#
# A orde e a mesma que executa dev/run_thumos14e_pilot.py. As etapas dentro
# dunha clase son secuenciais porque dependen unhas doutras; o paralelismo vai
# entre clases (dev/lanzar_clases_paralelo.sh).
#
# RESUMIBLE: cada etapa declara o seu artefacto e saltase se xa existe. Fai
# falla porque build_prototypes e generate_proposals NON comproban iso por si
# mesmos (o piloto facia a comprobacion por fora). Sen isto, unha clase que
# falle na etapa 12 repetiria as 11 anteriores, que son horas de propostas.
#
# PROPOSALS_POOL=1 e obrigatorio: o valor por defecto e 16 e src/proposals.py
# usa fork con h5py, que segundo os seus propios comentarios bloquea os workers
# en futex entre a gravacion 139 e a 193 de 200, de forma reproducible.
set -u
CLASE=$1
DEV=$2
NW=${3:-3}
R=/home/pablo.garcia.seijo/event_penguins
PLAN=${4:-$R/tmp/thumos14e_real/v1/protocol_manifest.json}
cd $R || exit 1
# out_root e o directorio onde vive o propio manifesto: build_plan escribe alí o
# protocol_manifest.json. Usamos dirname en vez de parsear o JSON porque a
# version anterior facia python -c con comiñas simples, e ao mandar o script por
# ssh dentro de comiñas simples esas comiñas desapareceron: quedou
# json.load(open())[paths][out_root], OUT baleiro, e todas as comprobacions de
# artefacto mirando na raiz do sistema. As nove clases lanzadas deron FALLO
# nunha etapa que si rematara ben.
OUT=$(dirname "$PLAN")
AF=/home/pablo.garcia.seijo/actionformer_release
SEED=1337
CR=$OUT/eventpenguins_full/seed_$SEED/$CLASE
PY="env PYTHONPATH=.:dev PROPOSALS_POOL=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $R/pyenv/bin/python"
FULL="dev/run_thumos14e_full_pipeline.py"
SUP="dev/run_thumos14e_supervised.py"
CARGS="--plan $PLAN --target-class $CLASE --seed $SEED --device $DEV --num-workers $NW"

# Limita cantas clases poden estar A VEZ nunha etapa cara en memoria.
# Medido o 2026-09-06: dúas clases en retag-head consumían 27,6 GB (13,8 cada
# unha, 1 main mais 3 workers). Con doce á vez serían 165 GB nunha máquina de
# 62, e o swap xa estaba cheo. Emúlase un semáforo de N rañuras con flock sobre
# N ficheiros: quen non colle rañura agarda, non morre.
RANURAS_PESADAS=${RANURAS_PESADAS:-4}
con_ranura() {
  local i fd
  while true; do
    for i in $(seq 1 "$RANURAS_PESADAS"); do
      exec {fd}>"/tmp/ranura_pesada_$i.lock"
      if flock -n "$fd"; then
        "$@"
        local rc=$?
        flock -u "$fd"
        exec {fd}>&-
        return $rc
      fi
      exec {fd}>&-
    done
    sleep 30
  done
}

# retag-head non compite polas rañuras de memoria: espera polo peche das
# features, que ten a extraccion compartida mentres as calcula. Sen isto,
# varias clases lanzarian a sua propia extraccion sobre o mesmo directorio.
con_features() {
  local fd
  exec {fd}>/tmp/retag_features.lock
  flock "$fd"
  "$@"
  local rc=$?
  flock -u "$fd"
  exec {fd}>&-
  return $rc
}

etapa() {
  local nome=$1; local artefacto=$2; shift 2
  if [ -e "$artefacto" ]; then
    echo "[$(date -Is)] $CLASE :: SALTO $nome (xa existe)"
    return 0
  fi
  echo "[$(date -Is)] $CLASE :: $nome"
  local corredor=""
  [ "${PESADA:-0}" = "1" ] && corredor="con_ranura"
  [ "${FEATURES:-0}" = "1" ] && corredor="con_features"
  if ! $corredor $PY "$@"; then
    echo "[$(date -Is)] $CLASE :: FALLO en $nome"
    exit 1
  fi
  if [ ! -e "$artefacto" ]; then
    echo "[$(date -Is)] $CLASE :: FALLO en $nome (rematou sen producir $artefacto)"
    exit 1
  fi
}

# --- prototipo espacial da clase ---
etapa prototypes "$OUT/target_prototypes/$CLASE/final/prototype.npy" \
  $SUP prototypes --plan $PLAN --classes $CLASE

# --- propostas da rama CoTAD (stage 1), por fold e no test ---
for f in 0 1 2 3 4; do
  etapa propostas-fold-$f \
    "$OUT/proposals/eventpenguins_stage1/$CLASE/fold_0$f/validation/proposals.csv" \
    $SUP proposals --plan $PLAN --branch eventpenguins_stage1 \
      --target-class $CLASE --split validation --cv-fold $f
done
etapa propostas-test "$OUT/proposals/eventpenguins_stage1/$CLASE/test/proposals.csv" \
  $SUP proposals --plan $PLAN --branch eventpenguins_stage1 \
    --target-class $CLASE --split test

# --- rama reTAG: a cabeza (as propostas son compartidas por split) ---
# As DUAS ramas son independentes: CoTAD non usa as features de reTAG para
# nada. Pero esta etapa ia ANTES de local-fold, asi que ao borrar as cabezas
# do subconxunto vello as 20 clases quedaron presas detras da extraccion
# compartida, coa GPU 1 ao 0 % durante horas. Con SO_COTAD=1 saltase e o
# detector arranca xa; con SO_RETAG=1 corre so este brazo cando as features
# estean listas.
if [ "${SO_COTAD:-0}" != "1" ]; then
  FEATURES=1 etapa retag-head "$OUT/predictions/retag/seed_$SEED/$CLASE/predictions.json" \
    $SUP heads --plan $PLAN --branch retag --classes $CLASE \
      --seeds $SEED --actionformer-root $AF --device $DEV --num-workers $NW
fi
if [ "${SO_RETAG:-0}" = "1" ]; then
  echo "[$(date -Is)] $CLASE :: RAMA RETAG COMPLETA"
  exit 0
fi

# --- rama CoTAD: detector completo ---
for f in 0 1 2 3 4; do
  PESADA=1 etapa local-fold-$f "$CR/local/fold_0$f/report.json" $FULL local-fold $CARGS --fold $f
done
PESADA=1 etapa local-test "$CR/local_test/report.json" $FULL local-test $CARGS
for f in 0 1 2 3 4; do
  PESADA=1 etapa continuous-fold-$f "$CR/event/fold_0$f/best.pt" $FULL continuous-fold $CARGS --fold $f
done
PESADA=1 etapa qfl-cv "$CR/qfl_cv/candidate_features.csv" $FULL qfl-cv $CARGS
PESADA=1 etapa full-test "$CR/test/predictions.json" $FULL full-test $CARGS

echo "[$(date -Is)] $CLASE :: COMPLETA"
