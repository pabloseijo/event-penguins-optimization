#!/usr/bin/env bash
# Transfire o corpus real ao CiTIUS con N fluxos en paralelo.
#
# Medido o 2026-09-03 sobre esta VPN: 1 fluxo da 2,7 MB/s, 4 fluxos dan
# 13,7 MB/s, 8 fluxos dan os mesmos 13,7. O limite e POR FLUXO, non agregado,
# e o teito esta en ~14 MB/s. Por iso catro e o numero, non mais.
#
# Usa scp porque este Git Bash non ten rsync. A resumibilidade consegue-se
# comparando o tamano remoto co local antes de enviar cada ficheiro, asi que
# relanzar o script continua onde quedou.
set -u

ORIXE=${ORIXE:-/d/thumos14-real/corpus}
REMOTO=${REMOTO:-/home/pablo.garcia.seijo/event_penguins/data/thumos14_real/canonical}
DESTINO=${DESTINO:-ctheadless30}
FLUXOS=${FLUXOS:-4}
LOG=${LOG:-/d/thumos14-real/transferencia.log}
SSHOPT="-o ConnectTimeout=20 -o ServerAliveInterval=30 -o Compression=no"

ssh $SSHOPT "$DESTINO" "mkdir -p $REMOTO" || exit 1

cd "$ORIXE" || exit 1
ls *.h5 > /tmp/lista_corpus.txt || exit 1
TOTAL=$(wc -l < /tmp/lista_corpus.txt)

# unha soa consulta para saber que hai xa alá e con que tamano
ssh $SSHOPT "$DESTINO" "cd $REMOTO 2>/dev/null && stat -c '%n %s' *.h5 2>/dev/null" \
  > /tmp/remotos.txt || true
echo "[$(date -Is)] $TOTAL ficheiros locais, $(wc -l < /tmp/remotos.txt) xa no servidor, $FLUXOS fluxos" | tee -a "$LOG"

# reparte por roldas para que os grandes non caian todos no mesmo fluxo
split -n r/"$FLUXOS" -d /tmp/lista_corpus.txt /tmp/anaco_

for i in $(seq 0 $((FLUXOS - 1))); do
  A=/tmp/anaco_0$i
  [ -f "$A" ] || continue
  (
    enviados=0; saltados=0; fallos=0
    while read -r f; do
      [ -n "$f" ] || continue
      local_sz=$(stat -c %s "$f")
      remoto_sz=$(awk -v n="$f" '$1==n {print $2}' /tmp/remotos.txt)
      if [ "${remoto_sz:-0}" = "$local_sz" ]; then
        saltados=$((saltados+1)); continue
      fi
      if scp -q $SSHOPT "$f" "$DESTINO:$REMOTO/$f"; then
        enviados=$((enviados+1))
      else
        fallos=$((fallos+1))
        echo "[$(date -Is)] [fluxo $i] FALLO $f" >> "$LOG"
      fi
      if [ $(( (enviados + saltados) % 10 )) -eq 0 ]; then
        echo "[$(date -Is)] [fluxo $i] $enviados enviados, $saltados saltados, $fallos fallos" >> "$LOG"
      fi
    done < "$A"
    echo "[$(date -Is)] [fluxo $i] REMATA: $enviados enviados, $saltados saltados, $fallos fallos" >> "$LOG"
  ) &
done
wait

REMOTOS=$(ssh $SSHOPT "$DESTINO" "ls $REMOTO/*.h5 2>/dev/null | wc -l")
echo "[$(date -Is)] TRANSFERENCIA REMATADA: $REMOTOS de $TOTAL ficheiros no servidor" | tee -a "$LOG"
