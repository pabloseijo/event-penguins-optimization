#!/usr/bin/env bash
# Reinicia a extraccion coas propostas reordenadas, sen que ninguen se cole.
#
# O problema que resolve: cando morre o script da extraccion, LIBERA as tres
# rañuras do semaforo, e as clases que estaban agardando en retag-head entran
# de inmediato e lanzan cada unha a sua propia extraccion sobre o mesmo
# directorio. Pasou o 2026-09-06: un intruso apareceu cun segundo de vida.
#
# A solucion e conxelar (SIGSTOP) as cadeas de clase mentres se fai o cambio, e
# retomalas (SIGCONT) cando a extraccion nova xa ten as rañuras.
set -u
R=/home/pablo.garcia.seijo/event_penguins
U=$(id -un)
D=$R/tmp/thumos14e_real/v1/features/retag

conta() { pgrep -u "$U" -c -f "$1" 2>/dev/null | head -1; }

echo "== 1. conxelar as cadeas de clase =="
CADEAS=$(pgrep -u "$U" -f "lanzar_clase.sh" || true)
[ -n "$CADEAS" ] && kill -STOP $CADEAS 2>/dev/null
echo "  conxeladas: $(echo $CADEAS | wc -w)"

echo "== 2. parar toda extraccion =="
for i in 1 2 3; do
  for pat in "extraer_features_retag.sh" "train_thumos14_ovr_atsn.py extract" "supervised.py heads"; do
    P=$(pgrep -u "$U" -f "$pat" || true)
    [ -n "$P" ] && kill -9 $P 2>/dev/null
  done
  sleep 3
  V=$(conta "train_thumos14_ovr_atsn.py extract")
  [ "${V:-0}" = "0" ] && break
done
V=$(conta "train_thumos14_ovr_atsn.py extract")
echo "  extract vivos: ${V:-0}"
if [ "${V:-0}" != "0" ]; then
  echo "  non se puido parar; retomo as clases e aborto"
  [ -n "$CADEAS" ] && kill -CONT $CADEAS 2>/dev/null
  exit 1
fi

echo "== 3. resetear o progreso (a orde das propostas cambiou) =="
for s in validation test; do
  for f in features.npy.building progress.json features.npy; do
    [ -e "$D/$s/$f" ] && { echo "  borrando $s/$f"; rm -f "$D/$s/$f"; }
  done
done

echo "== 4. lanzar a extraccion e agardar a que colla as rañuras =="
cd "$R" || exit 1
setsid nohup env NW=8 dev/extraer_features_retag.sh > tmp/features_retag.log 2>&1 < /dev/null &
for i in $(seq 1 20); do
  sleep 3
  if grep -q "rañura 3 tomada" tmp/features_retag.log 2>/dev/null; then break; fi
done
echo "  rañuras tomadas: $(grep -c "tomada" tmp/features_retag.log 2>/dev/null)"
echo "  script vivo: $(conta "extraer_features_retag.sh")"

echo "== 5. retomar as clases =="
[ -n "$CADEAS" ] && kill -CONT $CADEAS 2>/dev/null
echo "  retomadas: $(echo $CADEAS | wc -w)"
head -6 tmp/features_retag.log
