#!/usr/bin/env bash
# Extrae UNHA soa vez as features ATSN conxeladas da rama reTAG, e impide que
# ninguen mais o intente mentres tanto.
#
# Por que fóra das clases: en run_thumos14e_supervised.py:579 o cache_key da
# rama retag é literalmente "retag", sen a etiqueta da clase, así que as
# features van a features/retag/{split} e son as MESMAS para as 20 clases.
#
# Por que importa tanto: o extractor garda INCREMENTALMENTE
# (features.npy.building máis progress.json, e retoma desde "completed"). O
# 2026-09-06 oito clases chegaron a retag-head á vez e escribiron todas ese
# mesmo ficheiro, cada unha coa súa idea de por onde ía. O parcial de 2,75 GB
# resultante non era fiable e houbo que tiralo. Aquí tómanse as TRES rañuras do
# semáforo de lanzar_clase.sh, así que ningunha clase pode entrar en retag-head
# mentres isto corre; libéranse soas ao saír.
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd "$R" || exit 1
OUT=$R/tmp/thumos14e_real/v1
PLAN=$OUT/protocol_manifest.json
DP=$R/data/thumos14_real/preprocessed.h5
TC=$OUT/shared_features/continuous/timestamp_cache
NW=${NW:-10}

cat > /tmp/ler_modelo.py <<'PYEOF'
import json
import sys

with open(sys.argv[1]) as fh:
    print(json.load(fh)["inputs"]["source_atsn"]["path"])
PYEOF
SM=$("$R/pyenv/bin/python" /tmp/ler_modelo.py "$PLAN") || exit 1
echo "[$(date -Is)] modelo: $SM"

# Peche PROPIO, non as rañuras do detector. Antes tomabanse as rañuras
# xerais e iso bloqueaba a extraccion agardando por etapas de local-fold que
# duran moito: dúas cousas distintas competindo polo mesmo semaforo. Este
# peche so gobierna a relacion extraccion <-> retag-head.
exec {fdr}>/tmp/retag_features.lock
flock -w 60 "$fdr" || { echo "[$(date -Is)] non puiden tomar o peche de features"; exit 1; }
echo "[$(date -Is)] peche de features tomado" 

eval "$(/usr/local/bin/gpu env --shell)"
for split in validation test; do
  P=$OUT/proposals/retag/$split/proposals.csv
  D=$OUT/features/retag/$split
  echo "[$(date -Is)] extraendo $split ($(wc -l < "$P") propostas) con $NW workers"
  env PYTHONPATH=.:dev OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$R/pyenv/bin/python" dev/train_thumos14_ovr_atsn.py extract \
      --data-path "$DP" --proposals "$P" --source-model "$SM" --out-dir "$D" \
      --batch-size 64 --num-workers "$NW" --device cuda:0 \
      --timestamp-cache-dir "$TC" || { echo "[$(date -Is)] FALLO en $split"; exit 1; }
  echo "[$(date -Is)] $split feito"
done
echo "[$(date -Is)] FEATURES RETAG COMPLETAS"
