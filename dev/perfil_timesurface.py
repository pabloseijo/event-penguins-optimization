"""Onde se vai o tempo ao construir unha time surface, e canto pesaria en GPU.

Responde a dúas preguntas concretas:
  1. Que parte de create_img_representation custa: o scatter, a normalizacion
     ou o resize de PIL?
  2. Compensa levalo a GPU? Hai que contar a TRANSFERENCIA: agora enviase unha
     imaxe de 224x224x3 (150 KB); se o scatter fose en GPU habería que enviar os
     eventos crus, que son moitos mais bytes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R))
from src.classification import create_img_representation, create_time_map, range_norm  # noqa: E402
from PIL import Image  # noqa: E402

H, W = 260, 346
DECAY = 5e-6
rng = np.random.default_rng(0)

print(f"{'eventos/xanela':>15} {'scatter':>10} {'norm':>9} {'resize':>9} {'total':>9} {'bytes ev':>11} {'bytes img':>10}")
for n in (10_000, 100_000, 500_000, 2_000_000):
    ev = np.empty((n, 4), dtype=np.uint32)
    ev[:, 0] = rng.integers(0, W, n)
    ev[:, 1] = rng.integers(0, H, n)
    ev[:, 2] = np.sort(rng.integers(0, 1_000_000, n))
    ev[:, 3] = rng.integers(0, 2, n)

    t0 = time.perf_counter()
    for _ in range(5):
        m = create_time_map(ev, DECAY, H, W)
    t_scatter = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        u = range_norm(m, lower=-1, upper=1, dtype=np.uint8)
    t_norm = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        img = Image.fromarray(u).resize((224, 224), resample=Image.BILINEAR)
        a = np.repeat(np.array(img)[..., None], 3, axis=2)
    t_resize = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        create_img_representation(ev, DECAY, H, W)
    t_total = (time.perf_counter() - t0) / 5

    print(f"{n:>15,} {t_scatter*1000:>9.2f}m {t_norm*1000:>8.2f}m {t_resize*1000:>8.2f}m "
          f"{t_total*1000:>8.2f}m {ev.nbytes/1024:>10.0f}K {224*224*3/1024:>9.0f}K")

print()
print("Nota: por proposta constrúense 11 xanelas destas.")
