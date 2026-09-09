#!/usr/bin/env bash
# Deixa correndo UNHA soa extraccion de features de reTAG e para as demais.
#
# Situacion detectada o 2026-09-06: oito clases chegaran a retag-head e cada
# unha lanzara a sua propia extraccion, todas escribindo
# features/retag/validation. As features desa rama son compartidas polas 20
# clases (cache_key = "retag", sen etiqueta), asi que iso e traballo por vinte
# e, peor, unha carreira de escritura sobre os mesmos ficheiros.
#
# Preservase a extraccion lanzada por extraer_features_retag.sh e matase todo
# o que colgue das cadeas de clase. As clases non perden as propostas, que xa
# estan gravadas; so perden o intento de retag-head.
set -u
R=/home/pablo.garcia.seijo/event_penguins
U=$(id -un)

# 1. a arbore que hai que PRESERVAR
RAIZ=$(pgrep -u "$U" -f "extraer_features_retag.sh" | head -1)
if [ -z "$RAIZ" ]; then
  echo "AVISO: non atopo a extraccion compartida; non mato nada"
  exit 1
fi
PRESERVAR=$(pstree -p "$RAIZ" 2>/dev/null | grep -oE "\([0-9]+\)" | tr -d "()" | tr "\n" " ")
echo "preservando a arbore de $RAIZ: $(echo $PRESERVAR | wc -w) procesos"

# 2. parar as cadeas de clase e as suas cabezas
for pat in "rescatar_clases.sh" "supervised.py heads"; do
  P=$(pgrep -u "$U" -f "$pat" || true)
  [ -n "$P" ] && { echo "parando $pat ($(echo $P | wc -w))"; kill -TERM $P 2>/dev/null; }
done
sleep 4

# 3. matar as extraccions que NON son a compartida
MORTOS=0
for p in $(pgrep -u "$U" -f "train_thumos14_ovr_atsn.py extract"); do
  case " $PRESERVAR " in
    *" $p "*) ;;
    *) kill -TERM "$p" 2>/dev/null; MORTOS=$((MORTOS + 1)) ;;
  esac
done
echo "extraccions duplicadas paradas: $MORTOS"
sleep 6
for p in $(pgrep -u "$U" -f "train_thumos14_ovr_atsn.py extract"); do
  case " $PRESERVAR " in
    *" $p "*) ;;
    *) kill -9 "$p" 2>/dev/null ;;
  esac
done
sleep 3

VIVA=$(pgrep -u "$U" -c -f "extraer_features_retag.sh" 2>/dev/null || echo 0)
echo "extraccion compartida viva: $VIVA"
echo "procesos de extract restantes: $(pgrep -u "$U" -c -f "train_thumos14_ovr_atsn.py extract" 2>/dev/null || echo 0)"
echo "clases en voo: $(pgrep -u "$U" -c -f lanzar_clase.sh 2>/dev/null || echo 0)"
