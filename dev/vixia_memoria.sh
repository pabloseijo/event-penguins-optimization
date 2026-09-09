#!/usr/bin/env bash
# Mata a cadea de clase MAIS NOVA se a memoria dispoñible baixa do limiar.
#
# Isto e o que faltaba. O semaforo de lanzar_clase.sh limita ETAPAS, pero cada
# etapa xera fillos (build_proposal_lattice, train_quality_head e os workers do
# seu DataLoader) e os fillos non se contaban: o 2026-09-07 habia 4 rañuras
# ocupadas e 18 procesos de quality_head. Nove clases morreron co OOM-killer.
#
# E conxelar non abonda: as 17:33 conxelaronse 16 grupos e a maquina seguiu con
# usuario=0% e sistema=100%, porque un proceso parado conserva o seu RSS e co
# swap cheo o nucleo non podia expulsar nada. Quedaba en reclamo directo. So ao
# matalos de verdade baixou o uso de 52 a 33 GB e o swap drenou de 7 a 1 GB.
#
# Por iso este vixia mata, e mata o mais novo: e o que menos traballo leva feito
# e as etapas son resumibles.
set -u
U=$(id -un)
LIMIAR_GB=${LIMIAR_GB:-8}
di() { echo "[$(date -Is)] vixía: $*"; }
di "activo · limiar $LIMIAR_GB GB dispoñibles"
while true; do
  sleep 45
  disp=$(free -g | awk "NR==2{print \$7}")
  [ "$disp" -ge "$LIMIAR_GB" ] && continue
  # a cadea mais nova = a de menor tempo transcorrido
  vitima=$(ps -u "$U" -o pid=,etimes=,args= | grep "[l]anzar_clase.sh" \
           | sort -k2 -n | head -1 | awk "{print \$1}")
  if [ -z "$vitima" ]; then
    di "só $disp GB dispoñibles e non hai cadea que sacrificar"
    continue
  fi
  clase=$(ps -o args= -p "$vitima" | grep -oE "lanzar_clase.sh [A-Za-z]+" | awk "{print \$2}")
  g=$(ps -o pgid= -p "$vitima" | tr -d " ")
  di "só $disp GB dispoñibles -> mato $clase (grupo $g, a máis nova)"
  kill -9 -"$g" 2>/dev/null
  sleep 30
done
