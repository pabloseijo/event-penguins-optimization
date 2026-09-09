#!/bin/bash
# Reparte as clases de THUMOS14-E entre as dúas GPUs, N clases en voo.
# Uso: lanzar_clases_paralelo.sh <concorrencia> [Clase1 Clase2 ...]
# Sen lista de clases, corre as 20. Require reserva viva (gpu claim).
set -u
N=${1:-2}
shift || true
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
TODAS=(BaseballPitch BasketballDunk Billiards CleanAndJerk CliffDiving
       CricketBowling CricketShot Diving FrisbeeCatch GolfSwing HammerThrow
       HighJump JavelinThrow LongJump PoleVault Shotput SoccerPenalty
       TennisSwing ThrowDiscus VolleyballSpiking)
if [ $# -gt 0 ]; then CLASES=("$@"); else CLASES=("${TODAS[@]}"); fi
mkdir -p $R/tmp/clases_logs
echo "[$(date -Is)] ${#CLASES[@]} clases, $N en voo, 2 GPUs"
i=0
for CLASE in "${CLASES[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do wait -n; done
  DEV=cuda:$((i % 2))
  echo "[$(date -Is)] lanzo $CLASE en $DEV"
  dev/lanzar_clase.sh "$CLASE" "$DEV" ${NW:-3} ${PLAN:-} > $R/tmp/clases_logs/$CLASE.log 2>&1 &
  i=$((i + 1))
  sleep 5
done
wait
echo "[$(date -Is)] remataron todas; revisa tmp/clases_logs/*.log"
grep -l FALLO $R/tmp/clases_logs/*.log 2>/dev/null && echo "^^ clases con fallo" || echo "sen fallos rexistrados"
