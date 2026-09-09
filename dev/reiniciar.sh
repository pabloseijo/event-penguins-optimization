#!/usr/bin/env bash
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
U=$(id -un)
G=$R/tmp/thumos14e_real/v1/features/retag

echo "== estado antes =="
for s in validation test; do
  if [ -f $G/$s/features.npy ]; then echo "  $s: COMPLETA"
  elif [ -f $G/$s/progress.json ]; then
    echo "  $s: $($R/pyenv/bin/python -c "import json;print(json.load(open('$G/$s/progress.json'))['completed'])")"
  else echo "  $s: sen empezar"; fi
done

echo "== parando a calibracion e a extraccion vellas =="
for pat in calibrar_qhead train_quality_head extraer_features_retag "train_thumos14_ovr_atsn.py extract"; do
  for p in $(pgrep -u "$U" -f "$pat" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null
  done
done
sleep 10
echo "  vivos: $(pgrep -u "$U" -c -f train_thumos14_ovr_atsn 2>/dev/null || echo 0)"
free -g | sed -n '2p;3p' | sed 's/^/  /'

echo "== relanzando co codigo arranxado =="
mv -f $R/tmp/features_retag.log $R/tmp/features_retag.log.2 2>/dev/null
setsid nohup env NW=8 dev/extraer_features_retag.sh > $R/tmp/features_retag.log 2>&1 < /dev/null &
sleep 45
tr '\r' '\n' < $R/tmp/features_retag.log | grep -v '^$' | tail -3 | sed 's/^/  /'
