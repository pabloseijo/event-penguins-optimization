#!/usr/bin/env bash
# Conxela as cadeas do detector para que a extraccion compartida de reTAG teña
# a maquina. NON mata nada: SIGSTOP, non SIGKILL, asi que ningunha etapa perde
# o traballo feito.
#
# Por que: o 2026-09-07 as 17:25 a maquina estaba con usuario=0% e sistema=100%
# (tormenta de reclamo de paxinas: 6 GB dispoñibles, swap cheo) e a extraccion
# caera a 851 s/lote, ETA 259 h. Habia 17 cadeas vivas e 18 procesos de
# train_quality_head. O semaforo limita ETAPAS, pero cada etapa xera fillos e
# os fillos non se contaban.
#
# As features de reTAG son compartidas polas 20 clases, asi que a extraccion e
# o camino critico de toda a rama baseline: vai primeiro, soa.
set -u
U=$(id -un)

# 1. Parar os supervisores primeiro, ou volven relanzar o que conxelemos.
for s in rescatar_clases.sh gardian_memoria.sh orquestra_real.sh; do
  P=$(pgrep -u "$U" -f "$s" || true)
  if [ -n "$P" ]; then kill -9 $P 2>/dev/null; echo "  parado $s ($P)"; fi
done

# 2. O grupo de procesos da extraccion queda intacto. Identificase polo script,
#    non polo binario de python, que e o mesmo para todo.
EXTR=$(pgrep -u "$U" -f "extraer_features_retag.sh" | head -1)
if [ -z "$EXTR" ]; then echo "  AVISO: non atopo a extraccion; abortando por seguridade"; exit 1; fi
GEXTR=$(ps -o pgid= -p "$EXTR" | tr -d " ")
echo "  extraccion: pid $EXTR, grupo $GEXTR (INTACTA)"

# 3. Conxelar cada cadea de clase polo seu grupo enteiro (setsid deulle un
#    propio), o que colle tamen os fillos: lattice, quality_head e os seus
#    workers do DataLoader.
n=0
for p in $(pgrep -u "$U" -f "lanzar_clase.sh" || true); do
  g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d " ")
  [ -z "$g" ] && continue
  [ "$g" = "$GEXTR" ] && { echo "  SALTO grupo $g (e o da extraccion)"; continue; }
  kill -STOP -"$g" 2>/dev/null && n=$((n + 1))
done
echo "  grupos conxelados: $n"

sleep 5
echo "  procesos en estado T (parados): $(ps -u "$U" -o stat= | grep -c "^T" || true)"
