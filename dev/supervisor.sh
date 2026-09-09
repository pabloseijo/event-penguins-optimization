#!/bin/bash
# Relanza o orquestador se non está vivo. Usa PID file: buscar por patrón
# colisionaba co propio comando que invoca o supervisor.
ROOT=/home/pablo.garcia.seijo/event_penguins
PIDF=$ROOT/tmp/orquestador.pid
cd $ROOT || exit 1

if [ -f $PIDF ] && kill -0 $(cat $PIDF) 2>/dev/null; then exit 0; fi

rm -f tmp/orquestador.lock
echo "[$(date -Is)] relanzando orquestador" >> tmp/supervisor.log
setsid nohup ./pyenv/bin/python dev/orquestador.py >> tmp/orquestador.out 2>&1 &
echo $! > $PIDF
