#!/usr/bin/env bash
# Activa o subconxunto2 (o bo) e prepara a reextraccion das features de reTAG.
#
# Por que se cambia: o subconxunto1 escollia as gravacions mais pequenas e
# deixaba 11 das 20 clases con menos de 10 instancias anotadas; tres cabezas
# fallaron con cero positivos. O subconxunto2 escolle por anotacions por custo
# de extraccion e garante >= 30 instancias en TODAS as clases.
#
# Borra tamen as cabezas xa adestradas: estan feitas sobre exemplos
# insuficientes e os seus numeros non significan nada.
set -u
R=/home/pablo.garcia.seijo/event_penguins
U=$(id -un)
O=$R/tmp/thumos14e_real/v1
D=$O/features/retag

conta() { pgrep -u "$U" -c -f "$1" 2>/dev/null | head -1; }

echo "== 1. conxelar as cadeas de clase =="
CADEAS=$(pgrep -u "$U" -f "lanzar_clase.sh" || true)
[ -n "$CADEAS" ] && kill -STOP $CADEAS 2>/dev/null
echo "  conxeladas: $(echo $CADEAS | wc -w)"

echo "== 2. parar extraccion e cabezas (usan as features vellas) =="
for i in 1 2 3; do
  for pat in "extraer_features_retag.sh" "train_thumos14_ovr_atsn.py" "supervised.py heads"; do
    P=$(pgrep -u "$U" -f "$pat" || true); [ -n "$P" ] && kill -9 $P 2>/dev/null
  done
  sleep 3
  [ "$(conta 'train_thumos14_ovr_atsn.py')" = "0" ] && break
done
echo "  extract/heads vivos: $(conta 'train_thumos14_ovr_atsn.py')"

echo "== 3. activar o subconxunto2 =="
for s in validation test; do
  P=$O/proposals/retag/$s
  [ -f "$P/proposals_sub2.csv" ] || { echo "  FALTA $s/proposals_sub2.csv"; exit 1; }
  cp "$P/proposals_sub2.csv" "$P/proposals.csv"
  echo "  $s: $(( $(wc -l < "$P/proposals.csv") - 1 )) propostas"
done

echo "== 4. borrar features e cabezas do subconxunto vello =="
for s in validation test; do
  for f in features.npy features.npy.building progress.json; do
    [ -e "$D/$s/$f" ] && { rm -f "$D/$s/$f"; echo "  borrado $s/$f"; }
  done
done
N=$(find $O/predictions/retag -name predictions.json 2>/dev/null | wc -l)
find $O/predictions/retag -name predictions.json -delete 2>/dev/null
find $O/heads/retag -mindepth 2 -maxdepth 2 -type d -exec rm -rf {} + 2>/dev/null
echo "  borradas $N cabezas adestradas sobre exemplos insuficientes"

echo "== 5. lanzar a reextraccion =="
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
