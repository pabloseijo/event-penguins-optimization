#!/usr/bin/env bash
# Para a extraccion de features de reTAG, TIRA o parcial contaminado e reinicia
# desde cero, verificando cada paso.
#
# Por que se tira: o extractor garda incrementalmente (features.npy.building
# mais progress.json, e retoma desde "completed"). O 2026-09-06 oito clases
# escribiron ese mesmo ficheiro a vez, cada unha coa sua idea de por onde ia.
# Non se pode verificar fila a fila que non este entrelazado, e unhas features
# corruptas envelenarian toda a rama de reTAG dando numeros plausibles.
#
# Un intento anterior de borralo non chegou a executarse e a extraccion seguiu
# CONSTRUINDO ENRIBA do parcial contaminado. Por iso aqui se verifica.
set -u
R=/home/pablo.garcia.seijo/event_penguins
U=$(id -un)
D=$R/tmp/thumos14e_real/v1/features/retag

echo "== 1. parar =="
for pat in "extraer_features_retag.sh" "train_thumos14_ovr_atsn.py extract"; do
  P=$(pgrep -u "$U" -f "$pat" || true)
  [ -n "$P" ] && { echo "  matando $pat: $(echo $P | wc -w)"; kill -TERM $P 2>/dev/null; }
done
sleep 6
P=$(pgrep -u "$U" -f "train_thumos14_ovr_atsn.py extract" || true)
[ -n "$P" ] && kill -9 $P 2>/dev/null
sleep 4
VIVOS=$(pgrep -u "$U" -c -f "train_thumos14_ovr_atsn.py extract" 2>/dev/null | head -1)
VIVOS=${VIVOS:-0}
echo "  extract vivos: $VIVOS"
if [ "$VIVOS" != "0" ]; then echo "  NON se puido parar; abortando sen borrar nada"; exit 1; fi

echo "== 2. tirar o parcial contaminado =="
for s in validation test; do
  for f in features.npy.building progress.json features.npy; do
    if [ -e "$D/$s/$f" ]; then
      echo "  borrando $s/$f ($(stat -c %s "$D/$s/$f") bytes)"
      rm -f "$D/$s/$f"
    fi
  done
done

echo "== 3. verificar que non queda nada =="
RESTO=$(find "$D" -name "features.npy*" -o -name "progress.json" 2>/dev/null | wc -l)
echo "  ficheiros de features restantes: $RESTO"
if [ "$RESTO" != "0" ]; then echo "  QUEDA ALGO; abortando"; exit 1; fi

echo "== 4. reiniciar desde cero =="
cd "$R" || exit 1
setsid nohup env NW=8 dev/extraer_features_retag.sh > tmp/features_retag.log 2>&1 < /dev/null &
sleep 20
echo "  script vivo: $(pgrep -u "$U" -c -f "extraer_features_retag.sh" 2>/dev/null | head -1)"
head -6 tmp/features_retag.log 2>/dev/null
