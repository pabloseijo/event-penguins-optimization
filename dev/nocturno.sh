#!/usr/bin/env bash
# Mantén as duas ramas a correr sen ninguen diante, ata onde chegue.
#
# Rama CoTAD (GPU 1): manten CONC cadeas vivas, saltando retag-head. Non depende
# das features compartidas, asi que pode correr desde xa.
# Rama reTAG (GPU 0): en canto a extraccion compartida remate, adestra as 20
# cabezas unha a unha. Son baratas comparadas co detector.
#
# NON relanza indefinidamente: tres intentos por clase e desiste, coma o
# rescatador. O 2026-09-03 un supervisor con cron relanzou un piloto condenado
# cada 20 min durante horas.
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
U=$(id -un)
CONC=${CONC:-2}
# Rañuras pesadas por cadea. Estaba fixado en 2 desde cando a quality head
# consumia 13,8 GB; cos arranxos do 2026-09-08 unha etapa completa baixou a
# ~10 GB, asi que en 62 GB caben cinco. Con 2 quedaban catro nucleos de
# vinte traballando e a carga en 3,7.
RANURAS=${RANURAS:-5}
lanzadas=0
LOG=$R/tmp/nocturno.log
FV=$R/tmp/thumos14e_real/v1/features/retag/validation/features.npy
FT=$R/tmp/thumos14e_real/v1/features/retag/test/features.npy
P=$R/tmp/thumos14e_real/v1
declare -A tentativas
di() { echo "[$(date -Is)] $*" >> "$LOG"; }
di "nocturno arrancado · $CONC cadeas CoTAD en voo"

CLASES=$(ls $R/tmp/clases_logs/*.log | xargs -n1 basename | sed "s/.log$//")
retag_feito=0

while true; do
  sleep 120

  # ---- rama reTAG: agarda polas features e despois as 20 cabezas ----------
  if [ "$retag_feito" = "0" ] && [ -f "$FV" ] && [ -f "$FT" ]; then
    di "as features compartidas estan listas; arrancan as cabezas de reTAG"
    for c in $CLASES; do
      [ -f "$P/predictions/retag/seed_1337/$c/predictions.json" ] && continue
      di "  cabeza reTAG: $c"
      env SO_RETAG=1 dev/lanzar_clase.sh "$c" cuda:0 3 >> $R/tmp/clases_logs/$c.log 2>&1
    done
    n=$(ls $P/predictions/retag/seed_1337/*/predictions.json 2>/dev/null | wc -l)
    di "rama reTAG rematada: $n / 20 cabezas"
    retag_feito=1
  fi

  # ---- rama CoTAD: manter CONC cadeas vivas ------------------------------
  vivas=$(pgrep -u "$U" -c -f "lanzar_clase.sh" || true)
  [ "$vivas" -ge "$CONC" ] && continue

  correndo=$(pgrep -u "$U" -af "lanzar_clase.sh" 2>/dev/null \
             | grep -oE "lanzar_clase.sh [A-Za-z]+" | awk "{print \$2}" | sort -u)
  for c in $CLASES; do
    [ "$vivas" -ge "$CONC" ] && break
    [ -f "$P/eventpenguins_full/seed_1337/$c/test/predictions.json" ] && continue
    echo "$correndo" | grep -qx "$c" && continue
    t=${tentativas[$c]:-0}
    [ "$t" -ge 3 ] && continue
    tentativas[$c]=$((t + 1))
    # A GPU 0 quedaba ao 0 % con todas as cadeas na 1. Esta carga e limitada
    # por CPU, asi que isto reparte memoria de video, non computo.
    # Consolidado nunha soa GPU o 2026-09-08 tras o aviso do CiTIUS: a
    # carga esta limitada por CPU e as GPUs estaban ao 0 %, asi que
    # ocupar dúas non aportaba nada e privaba a outros dunha 5090.
    dev=cuda:$(( lanzadas % 2 ))
    di "CoTAD: $c (intento $((t+1))/3) en $dev · $RANURAS rañuras"
    setsid nohup env SO_COTAD=1 RANURAS_PESADAS="$RANURAS" dev/lanzar_clase.sh "$c" "$dev" 3 \
      >> $R/tmp/clases_logs/$c.log 2>&1 < /dev/null &
    lanzadas=$((lanzadas + 1))
    vivas=$((vivas + 1))
    sleep 20
  done
done
