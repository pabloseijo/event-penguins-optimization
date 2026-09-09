#!/usr/bin/env bash
# Reinicia os dous shards de extraccion ATSN co numero de workers indicado.
# Faise nun script e non nunha liña solta porque un pkill -f escrito na propia
# liña de comandos casa consigo mesmo e mata a sesion (pasou dúas veces).
set -u
NW=${1:-5}
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
PLAN=$R/tmp/thumos14e_real/v1/protocol_manifest.json

MEUS=$(pgrep -u "$USER" -f "extract_continuous_features.py extract" | grep -v "^$$\$" || true)
if [ -n "$MEUS" ]; then
  echo "matando: $(echo $MEUS | wc -w) procesos"
  kill -TERM $MEUS 2>/dev/null
  sleep 8
  kill -9 $MEUS 2>/dev/null
  sleep 3
fi

eval "$(/usr/local/bin/gpu env --shell)"
for i in 0 1; do
  setsid nohup nice -n 10 env PYTHONPATH=.:dev OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $R/pyenv/bin/python dev/run_thumos14e_full_pipeline.py shared-extract \
      --plan $PLAN --shard-index $i --num-shards 2 \
      --batch-size 64 --num-workers $NW --device cuda:$i \
    > $R/tmp/real_atsn_$i.log 2>&1 &
  echo "shard $i lanzado con $NW workers"
  sleep 3
done
sleep 12
ps -u "$USER" -o args | grep "extract_continuous_features.py extract" | grep -v grep | grep -o "num-workers [0-9]*" | sort | uniq -c
