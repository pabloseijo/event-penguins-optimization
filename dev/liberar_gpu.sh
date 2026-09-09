#!/usr/bin/env bash
# Correccions inmediatas tras o aviso de Jorge (tecnico do CiTIUS):
#   1. quitar manter_reserva_gpu.py: el confirma que mentres haxa procesos a
#      reserva non se perde, asi que ocupaba 630 MiB en cada GPU para nada
#   2. consolidar todo nunha soa GPU e liberar a outra: a carga esta limitada
#      por CPU, asi que non custa rendemento e deixa unha 5090 dispoñible
set -u
R=/home/pablo.garcia.seijo/event_penguins
cd $R || exit 1
U=$(id -un)

for p in $(pgrep -u "$U" -f 'manter_reserva_gp[u].py' 2>/dev/null); do
  kill -9 "$p" 2>/dev/null && echo "  parado manter_reserva_gpu.py (pid $p)"
done

# o orquestrador deixa de repartir: todo a cuda:1, e a 0 queda libre
python3 - <<'PY'
p = "dev/nocturno.sh"
s = open(p, encoding="utf-8").read()
vello = "    dev=cuda:$(( lanzadas % 2 ))"
novo = ("    # Consolidado nunha soa GPU o 2026-09-08 tras o aviso do CiTIUS: a\n"
        "    # carga esta limitada por CPU e as GPUs estaban ao 0 %, asi que\n"
        "    # ocupar dúas non aportaba nada e privaba a outros dunha 5090.\n"
        "    dev=cuda:1")
if vello in s:
    s = s.replace(vello, novo)
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("  nocturno: consolidado en cuda:1")
else:
    print("  aviso: non atopo o reparto de GPU en nocturno.sh")
PY

sleep 3
echo
echo "  estado das GPUs:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/    /'
echo "  procesos con CUDA:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/    /'
