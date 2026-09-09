#!/usr/bin/env bash
# Reserva rañuras do semáforo mentres corre a extracción compartida de reTAG.
#
# Por que fai falla: ata hoxe a extracción tomaba 3 das 4 rañuras, así que
# contaba no orzamento de memoria da máquina. Ao darlle un peche propio (para
# que non quedase bloqueada agardando por etapas de local-fold que duran horas)
# deixou de contar, e pasou a sumarse enteira aos 4 detectores: 27 GB dela máis
# 22 GB de catro build_proposal_lattice sobre 62 GB totais.
#
# Ese foi o que matou nove clases o 2026-09-07 entre as 11:58 e as 12:00, todas
# con "Command failed (-9)" ou cun worker do DataLoader morto. Non era un bug do
# detector: era falta de marxe.
#
# Este gardián recupera a contabilidade sen tocar as cadeas xa lanzadas (que
# levan RANURAS_PESADAS=4 fixado no seu arranque): colle as rañuras el mesmo e
# só as solta cando a extracción remata.
set -u
N=${N:-2}
di() { echo "[$(date -Is)] gardián: $*"; }

if ! pgrep -u "$(id -un)" -f "extraer_features_retag.sh" >/dev/null 2>&1; then
  di "a extracción xa non corre; non reservo nada"
  exit 0
fi

for i in $(seq $((5 - N)) 4); do
  eval "exec {fd$i}>/tmp/ranura_pesada_$i.lock"
  if eval "flock -n \$fd$i"; then
    di "rañura $i reservada"
  else
    di "rañura $i xa ocupada por unha clase; agardo a que a solte"
    eval "flock \$fd$i" && di "rañura $i reservada (tras agardar)"
  fi
done

di "$N rañuras reservadas; o detector queda con $((4 - N))"
while pgrep -u "$(id -un)" -f "extraer_features_retag.sh" >/dev/null 2>&1; do
  sleep 60
done
di "a extracción rematou; solto as rañuras e o detector volve a 4"
