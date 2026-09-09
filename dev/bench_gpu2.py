"""Port CORRECTO de create_time_map a GPU, e verificacion de identidade.

O port inxenuo usa index_put_(accumulate=False), que a documentacion de PyTorch
declara de comportamento INDEFINIDO con indices duplicados e non determinista en
CUDA. Daba maxdif=2,00, que e o rango enteiro.

A definicion de time surface e "conservar o timestamp do evento MAIS RECENTE por
pixel". Como os eventos veñen ordenados por tempo, iso e exactamente o que fai
scatter_reduce con amax, e e determinista.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R))
from src.classification import create_time_map  # noqa: E402

H, W, DECAY = 260, 346, 5e-6
dev = torch.device("cuda:0")
rng = np.random.default_rng(0)


def gpu_time_map(ev: torch.Tensor) -> torch.Tensor:
    x = ev[:, 0].long(); y = ev[:, 1].long()
    t = ev[:, 2].double(); p = ev[:, 3].long()
    idx = y * W + x

    # 1. timestamp do evento mais recente por pixel (determinista)
    tm = torch.zeros(H * W, dtype=torch.float64, device=dev)
    tm.scatter_reduce_(0, idx, t, reduce="amax", include_self=True)

    cur = t.max()
    tm = torch.exp(-DECAY * (cur - tm))

    # 2. polaridade do MESMO evento que gañou, non dun calquera: quedamos co de
    #    maior timestamp por pixel, que e o mesmo criterio.
    pol = torch.where(p == 0, -1.0, 1.0).double()
    gan = torch.zeros(H * W, dtype=torch.float64, device=dev)
    gan.scatter_reduce_(0, idx, t, reduce="amax", include_self=True)
    e_gana = gan[idx] == t                      # eventos que son o maximo do seu pixel
    polm = torch.ones(H * W, dtype=torch.float64, device=dev)
    polm.scatter_(0, idx[e_gana], pol[e_gana])  # ordenados: o ultimo escribe
    tocado = torch.zeros(H * W, dtype=torch.bool, device=dev)
    tocado.scatter_(0, idx, torch.ones_like(idx, dtype=torch.bool))
    tm = torch.where(tocado, tm * polm, tm)
    return tm.view(H, W)


print(f"{'eventos':>12} {'CPU':>10} {'GPU+transf':>12} {'GPU só':>10} {'ganancia':>9} {'maxdif':>11}")
for n in (500_000, 1_000_000, 2_000_000, 4_000_000):
    ev = np.empty((n, 4), dtype=np.uint32)
    ev[:, 0] = rng.integers(0, W, n)
    ev[:, 1] = rng.integers(0, H, n)
    ev[:, 2] = np.sort(rng.integers(0, 1_000_000, n))
    ev[:, 3] = rng.integers(0, 2, n)

    t0 = time.perf_counter()
    for _ in range(3):
        cpu = create_time_map(ev, DECAY, H, W)
    t_cpu = (time.perf_counter() - t0) / 3

    ev32 = ev.astype(np.int32)   # int32, non int64: metade de bytes no bus
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        g = torch.from_numpy(ev32).to(dev)
        out = gpu_time_map(g)
        torch.cuda.synchronize()
    t_all = (time.perf_counter() - t0) / 3

    g = torch.from_numpy(ev32).to(dev); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        out = gpu_time_map(g); torch.cuda.synchronize()
    t_only = (time.perf_counter() - t0) / 3

    dif = float(np.abs(out.cpu().numpy() - cpu).max())
    print(f"{n:>12,} {t_cpu*1000:>9.1f}m {t_all*1000:>11.1f}m {t_only*1000:>9.2f}m "
          f"{t_cpu/t_all:>8.1f}x {dif:>11.2e}")
