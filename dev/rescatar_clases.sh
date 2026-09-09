#!/usr/bin/env bash
# Relanza clases que caeron, e SO iso. Pensado para a noite, sen ninguen diante.
#
# Non e o supervisor con cron do 2026-09-03, que relanzaba un piloto condenado
# cada 20 min, mataba os lanzamentos manuais e enchia o correo. As diferenzas:
#   - so toca clases con FALLO que NON esten a correr
#   - un maximo de 3 intentos por clase, e despois desiste e dio
#   - o lanzador e resumible, asi que un reintento retoma na etapa que caeu
#   - a version nova do lanzador ten o semaforo de ranuras, asi que os
#     reintentos entran ordenados en retag-head en vez de amontoarse
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
LOG=$R/tmp/rescate.log
MAX=${MAX:-3}
declare -A intentos

di() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
di "rescatador arrancado (max $MAX intentos por clase)"

while true; do
  sleep 300
  vivas=$(pgrep -u "$USER" -af lanzar_clase.sh 2>/dev/null | grep -oE "lanzar_clase.sh [A-Za-z]+" | awk "{print \$2}" | sort -u)
  for f in $R/tmp/clases_logs/*.log; do
    [ -e "$f" ] || continue
    c=$(basename "$f" .log)
    grep -q COMPLETA "$f" 2>/dev/null && continue
    grep -q FALLO "$f" 2>/dev/null || continue
    echo "$vivas" | grep -qx "$c" && continue
    n=${intentos[$c]:-0}
    if [ "$n" -ge "$MAX" ]; then continue; fi
    intentos[$c]=$((n + 1))
    dev=cuda:$(( n % 2 ))
    di "rescatando $c (intento $((n+1))/$MAX) en $dev"
    setsid nohup dev/lanzar_clase.sh "$c" "$dev" 3 >> "$f" 2>&1 < /dev/null &
    sleep 10
  done
  comp=$(grep -l COMPLETA $R/tmp/clases_logs/*.log 2>/dev/null | wc -l)
  if [ "$comp" = "20" ]; then di "as 20 completas; rescatador retirase"; break; fi
done
