#!/usr/bin/env bash
# Encadea, sen intervencion, o que queda do corpus real:
#
#   1. agarda a que rematen as tres etapas compartidas
#   2. shared-verify
#   3. as 20 clases, N en voo, nas duas GPUs
#   4. evaluate-all coa taboa final
#
# DELIBERADAMENTE NON RELANZA NADA. O 2026-09-03 un supervisor con cron cada 20
# min relanzaba un piloto condenado, mataba os lanzamentos manuais e enchia o
# correo. Se algo falla aqui, para e deixa o motivo no log.
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
PLAN=$R/tmp/thumos14e_real/v1/protocol_manifest.json
D=$R/tmp/thumos14e_real/v1
SF=$D/shared_features
LOG=$R/tmp/orquestra_real.log
CONC=${CONC:-6}
LIMITE_ESPERA=${LIMITE_ESPERA:-43200}   # 12 h de garda maxima

di() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

di "orquestrador arrancado · concorrencia $CONC"

# ---- 1. agardar polas compartidas -------------------------------------------
t0=$(date +%s)
while true; do
  shards=$(ls $SF/continuous/shard_0[01]_of_02.json 2>/dev/null | wc -l)
  ev=0; [ -f $SF/event_stats/metadata.json ] && ev=1
  rp=0; [ -f $D/proposals/retag/validation/proposals.csv ] && \
        [ -f $D/proposals/retag/test/proposals.csv ] && rp=1
  [ "$shards" = "2" ] && [ "$ev" = "1" ] && [ "$rp" = "1" ] && break

  vivos=$(pgrep -u $USER -c -f "shared-extrac[t]|extract_continuous_event_stat[s]" 2>/dev/null || echo 0)
  if [ "$vivos" = "0" ]; then
    # Pode ser unha carreira: o ultimo shard escribe o seu JSON e sae entre o
    # "ls" de arriba e este pgrep. Relese o estado antes de declarar fallo.
    sleep 10
    shards=$(ls $SF/continuous/shard_0[01]_of_02.json 2>/dev/null | wc -l)
    ev=0; [ -f $SF/event_stats/metadata.json ] && ev=1
    if [ "$shards" = "2" ] && [ "$ev" = "1" ] && [ "$rp" = "1" ]; then
      di "as compartidas remataron xusto agora"
      break
    fi
    di "PARADA: non queda ningun proceso das compartidas e faltan (shards=$shards event=$ev retag=$rp)"
    exit 1
  fi
  if [ $(( $(date +%s) - t0 )) -gt "$LIMITE_ESPERA" ]; then
    di "PARADA: superado o limite de espera polas compartidas"
    exit 1
  fi
  sleep 120
done
di "compartidas completas (2 shards ATSN, event-stats, retag-proposals)"

# ---- 2. verificacion da cache compartida ------------------------------------
if ! env PYTHONPATH=.:dev $R/pyenv/bin/python dev/run_thumos14e_full_pipeline.py \
        shared-verify --plan $PLAN --num-shards 2 >> "$LOG" 2>&1; then
  di "PARADA: shared-verify fallou"
  exit 1
fi
di "shared-verify OK"

# ---- 3. as 20 clases --------------------------------------------------------
di "lanzando as 20 clases, $CONC en voo"
if ! CONC=$CONC dev/lanzar_clases_paralelo.sh "$CONC" >> "$LOG" 2>&1; then
  di "aviso: o lanzador devolveu erro; reviso clases completas antes de decidir"
fi

completas=$(grep -l "COMPLETA" $R/tmp/clases_logs/*.log 2>/dev/null | wc -l)
di "clases completas: $completas de 20"
if [ "$completas" != "20" ]; then
  di "PARADA: faltan clases; evaluate-all esixe as 20. Fallos en:"
  grep -l "FALLO" $R/tmp/clases_logs/*.log 2>/dev/null | tee -a "$LOG"
  exit 1
fi

# ---- 4. avaliacion final ----------------------------------------------------
di "lanzando evaluate-all"
if env PYTHONPATH=.:dev $R/pyenv/bin/python dev/run_thumos14e_full_pipeline.py \
      evaluate-all --plan $PLAN --seed 1337 \
      --actionformer-root /home/pablo.garcia.seijo/actionformer_release >> "$LOG" 2>&1; then
  di "EVALUATE-ALL REMATADO"
else
  di "PARADA: evaluate-all fallou"
  exit 1
fi
di "TODO REMATADO"
