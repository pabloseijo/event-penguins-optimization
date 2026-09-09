"""CPU fronte a GPU para create_time_map, TRANSFERENCIA INCLUIDA.

Replica a semantica exacta de src/classification.py:
    time_map[y,x] = t              (ultimo gana nos duplicados)
    time_map = exp(-decay*(t_max - time_map))
    time_map[y,x] *= polaridade    (0 -> -1)

Os pixeles nunca tocados quedan en exp(-decay*t_max), non en cero: hai que
replicalo ou os numeros non son comparables.
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


def gpu_time_map(ev_t: torch.Tensor) -> torch.Tensor:
    x = ev_t[:, 0].long(); y = ev_t[:, 1].long()
    t = ev_t[:, 2].double(); p = ev_t[:, 3].long()
    flat = torch.zeros(H * W, dtype=torch.float64, device=dev)
    idx = y * W + x
    flat.index_put_((idx,), t, accumulate=False)
    cur = t.max()
    flat = torch.exp(-DECAY * (cur - flat))
    pol = torch.where(p == 0, -1, 1).double()
    flat.index_put_((idx,), flat[idx] * pol, accumulate=False)
    return flat.view(H, W)


print(f"{'eventos':>12} {'CPU':>10} {'GPU+transf':>12} {'GPU só':>10} {'ganancia':>10} {'maxdif':>10}")
for n in (100_000, 500_000, 1_000_000, 2_000_000, 4_000_000):
    ev = np.empty((n, 4), dtype=np.uint32)
    ev[:, 0] = rng.integers(0, W, n)
    ev[:, 1] = rng.integers(0, H, n)
    ev[:, 2] = np.sort(rng.integers(0, 1_000_000, n))
    ev[:, 3] = rng.integers(0, 2, n)

    t0 = time.perf_counter()
    for _ in range(3):
        cpu = create_time_map(ev, DECAY, H, W)
    t_cpu = (time.perf_counter() - t0) / 3

    ev_i64 = ev.astype(np.int64)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        g = torch.from_numpy(ev_i64).to(dev, non_blocking=False)
        out = gpu_time_map(g)
        torch.cuda.synchronize()
    t_gpu_all = (time.perf_counter() - t0) / 3

    g = torch.from_numpy(ev_i64).to(dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        out = gpu_time_map(g)
        torch.cuda.synchronize()
    t_gpu_only = (time.perf_counter() - t0) / 3

    dif = float(np.abs(out.cpu().numpy() - cpu).max())
    print(f"{n:>12,} {t_cpu*1000:>9.1f}m {t_gpu_all*1000:>11.1f}m {t_gpu_only*1000:>9.2f}m "
          f"{t_cpu/t_gpu_all:>9.1f}x {dif:>10.2e}")
