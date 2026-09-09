"""Proba que a consulta convertida da EXACTAMENTE os mesmos indices.

Comparase contra o comportamento ORIXINAL (consulta en float, que promociona o
array), en arrays pequenos onde ese camiño e barato. Cubrense os casos que
poderian romper o truncamento: fraccionarios, negativos, coincidencias exactas,
valores fora de rango polos dous lados e duplicados.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/pablo.garcia.seijo/event_penguins")
from src.classification import _consulta_no_dtype  # noqa: E402

rng = np.random.default_rng(1337)
fallos = 0
casos = 0

for proba in range(400):
    n = int(rng.integers(1, 500))
    ts = np.sort(rng.integers(0, 200000, size=n).astype(np.uint32))
    if proba % 4 == 0:                      # con duplicados
        ts = np.sort(np.concatenate([ts, ts[: n // 2]])).astype(np.uint32)

    consultas = np.concatenate([
        rng.uniform(-5000, 205000, size=8),          # xerais, incluidos negativos
        ts[rng.integers(0, len(ts), size=4)].astype(np.float64),        # exactos
        ts[rng.integers(0, len(ts), size=4)].astype(np.float64) + 0.5,  # xusto enriba
        ts[rng.integers(0, len(ts), size=4)].astype(np.float64) - 0.5,  # xusto debaixo
    ])
    q = torch.tensor(consultas, dtype=torch.float32)

    vello = np.searchsorted(ts, q)                       # como estaba
    novo = np.searchsorted(ts, _consulta_no_dtype(q, ts))  # como queda
    casos += len(consultas)
    if not np.array_equal(np.asarray(vello), np.asarray(novo)):
        fallos += 1
        if fallos <= 3:
            d = np.nonzero(np.asarray(vello) != np.asarray(novo))[0]
            print(f"  DIFIRE proba={proba} en {len(d)} consultas, ex: "
                  f"v={consultas[d[0]]} vello={np.asarray(vello)[d[0]]} novo={np.asarray(novo)[d[0]]}")

print(f"  {casos:,} consultas comparadas en 400 arrays")
print(f"  discrepancias: {fallos}")
print("  RESULTADO:", "IDENTICO" if fallos == 0 else "DIFIRE")
