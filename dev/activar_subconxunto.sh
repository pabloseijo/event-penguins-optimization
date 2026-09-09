#!/usr/bin/env bash
# Pon o subconxunto como propostas ACTIVAS de reTAG, para que a extraccion
# remate e produza un features.npy utilizable.
#
# Por que fai falla: train_thumos14_ovr_atsn.py so fai
#   os.replace(building_path, features_path)
# cando cursor == len(proposals). Reordenar o ficheiro fai que o subconxunto se
# calcule primeiro, pero as suas filas NON se poden usar ata rematar as 448.324.
# Para ter features utilizables hai que darlle un ficheiro que remate.
#
# REVERSIBLE: o ficheiro completo queda en proposals_completo.csv e restaurase
# con restaurar_completo.sh. As features do subconxunto van a features/retag/,
# que e onde as busca a etapa heads.
set -u
R=/home/pablo.garcia.seijo/event_penguins
U=$(id -un)
OUT=$R/tmp/thumos14e_real/v1
D=$OUT/features/retag

conta() { pgrep -u "$U" -c -f "$1" 2>/dev/null | head -1; }

echo "== 1. conxelar as cadeas de clase (se non, colanse nas rañuras) =="
CADEAS=$(pgrep -u "$U" -f "lanzar_clase.sh" || true)
[ -n "$CADEAS" ] && kill -STOP $CADEAS 2>/dev/null
echo "  conxeladas: $(echo $CADEAS | wc -w)"

echo "== 2. parar toda extraccion =="
for i in 1 2 3; do
  for pat in "extraer_features_retag.sh" "train_thumos14_ovr_atsn.py extract" "supervised.py heads"; do
    P=$(pgrep -u "$U" -f "$pat" || true); [ -n "$P" ] && kill -9 $P 2>/dev/null
  done
  sleep 3
  [ "$(conta 'train_thumos14_ovr_atsn.py extract')" = "0" ] && break
done
V=$(conta "train_thumos14_ovr_atsn.py extract")
echo "  extract vivos: $V"
if [ "$V" != "0" ]; then
  echo "  non se puido parar; retomo e aborto"
  [ -n "$CADEAS" ] && kill -CONT $CADEAS 2>/dev/null
  exit 1
fi

echo "== 3. activar o subconxunto =="
for s in validation test; do
  P=$OUT/proposals/retag/$s
  [ -f "$P/proposals_sub.csv" ] || { echo "  falta $s/proposals_sub.csv"; exit 1; }
  # o completo garda-se unha soa vez; proposals.csv esta reordenado pero ten as
  # mesmas filas, asi que proposals_orixinal.csv e o de referencia
  [ -f "$P/proposals_completo.csv" ] || cp "$P/proposals_orixinal.csv" "$P/proposals_completo.csv"
  cp "$P/proposals_sub.csv" "$P/proposals.csv"
  echo "  $s: activo con $(( $(wc -l < "$P/proposals.csv") - 1 )) propostas"
done

echo "== 4. resetear o progreso (cambia o ficheiro de propostas) =="
for s in validation test; do
  for f in features.npy.building progress.json features.npy; do
    [ -e "$D/$s/$f" ] && { echo "  borrando $s/$f"; rm -f "$D/$s/$f"; }
  done
done

echo "== 5. lanzar e agardar polas rañuras =="
cd "$R" || exit 1
setsid nohup env NW=8 dev/extraer_features_retag.sh > tmp/features_retag.log 2>&1 < /dev/null &
for i in $(seq 1 20); do
  sleep 3
  grep -q "rañura 3 tomada" tmp/features_retag.log 2>/dev/null && break
done
echo "  rañuras: $(grep -c tomada tmp/features_retag.log 2>/dev/null)"
echo "  script vivo: $(conta 'extraer_features_retag.sh')"

echo "== 6. retomar as clases =="
[ -n "$CADEAS" ] && kill -CONT $CADEAS 2>/dev/null
echo "  retomadas: $(echo $CADEAS | wc -w)"
head -5 tmp/features_retag.log
