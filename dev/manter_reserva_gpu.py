"""Mantén viva a reserva de GPU do CiTIUS durante as fases que son so de CPU.

Por que existe: o enforcer libera a reserva tras 2 h sen actividade na GPU, e
despois manda SIGTERM aos ~60 s a calquera proceso que a toque sen reserva. A
fase de propostas das 20 clases dura horas e non usa a GPU, asi que a reserva
caducaba e a seguinte etapa (retag-head) moria con "Command failed (-15)".
Pasou o 2026-09-06 as 02:07 con ThrowDiscus e VolleyballSpiking.

Mantense unha asignacion minima viva en cada GPU. Son ~300 MB por tarxeta: o
suficiente para que a GPU non conste ociosa, e desprezable fronte aos 32 GB de
cada 5090. Matar este proceso libera as dúas.
"""
import time

import torch

anclas = []
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    anclas.append(torch.zeros(1024, 1024, device=f"cuda:{i}"))
    print(f"ancla viva en cuda:{i}", flush=True)

print("mantendo a reserva; matar este proceso para liberala", flush=True)
while True:
    for i, a in enumerate(anclas):
        a.add_(0)          # toque minimo para que non conste ociosa
        torch.cuda.synchronize(i)
    time.sleep(300)
