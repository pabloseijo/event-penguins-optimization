"""Port exacto: reducese sobre o INDICE do evento, non sobre o timestamp.

Con eventos ordenados por tempo, o indice maior por pixel e literalmente "o
ultimo que escribe", que e a semantica de time_map[y,x] = t en NumPy. Desempata
os casos de mesmo pixel e mesmo timestamp, que era onde fallaba a version con
amax sobre t.
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
    n = ev.shape[0]
    x = ev[:, 0].long(); y = ev[:, 1].long()
    t = ev[:, 2].double(); p = ev[:, 3].long()
    idx = y * W + x

    # indice do ULTIMO evento que toca cada pixel (-1 se ningun)
    ganador = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    ganador.scatter_reduce_(0, idx, torch.arange(n, device=dev), reduce="amax", include_self=True)
    tocado = ganador >= 0
    seguro = ganador.clamp(min=0)

    tm = torch.zeros(H * W, dtype=torch.float64, device=dev)
    tm[tocado] = t[seguro[tocado]]
    cur = t.max()
    tm = torch.exp(-DECAY * (cur - tm))

    pol = torch.where(p == 0, -1.0, 1.0).double()
    tm[tocado] = tm[tocado] * pol[seguro[tocado]]
    return tm.view(H, W)


print(f"{'eventos':>12} {'CPU':>10} {'GPU+transf':>12} {'GPU só':>10} {'ganancia':>9} {'maxdif':>11}")
for n in (500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000):
    ev = np.empty((n, 4), dtype=np.uint32)
    ev[:, 0] = rng.integers(0, W, n)
    ev[:, 1] = rng.integers(0, H, n)
    ev[:, 2] = np.sort(rng.integers(0, 1_000_000, n))
    ev[:, 3] = rng.integers(0, 2, n)

    t0 = time.perf_counter()
    for _ in range(3):
        cpu = create_time_map(ev, DECAY, H, W)
    t_cpu = (time.perf_counter() - t0) / 3

    ev32 = ev.astype(np.int32)
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
    ok = "IDENTICO" if dif < 1e-12 else "DIFIRE"
    print(f"{n:>12,} {t_cpu*1000:>9.1f}m {t_all*1000:>11.1f}m {t_only*1000:>9.2f}m "
          f"{t_cpu/t_all:>8.1f}x {dif:>11.2e}  {ok}")
