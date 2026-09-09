"""Comproba se np.searchsorted materializa o array de timestamps.

Sospeita: roi_timestamps ven dun .npy con mmap_mode="r" (uint32), e a consulta
t_imgs_start e un TENSOR DE TORCH en float32. numpy ten que atopar un tipo
comun, e para iso pode CONVERTER o array enteiro a float64 -- materializando os
154 MB do mmap en cada chamada, dúas veces por proposta.
"""
import time
from pathlib import Path

import numpy as np
import torch

TC = Path("/home/pablo.garcia.seijo/event_penguins/tmp/thumos14e_real/v1/"
          "shared_features/continuous/timestamp_cache")
p = TC / "video_validation_0000666" / "N01.npy"
print(f"  ficheiro: {p.stat().st_size/1e6:.0f} MB")

ts = np.load(p, mmap_mode="r")
print(f"  timestamps: dtype={ts.dtype}  n={len(ts):,}  memmap={isinstance(ts, np.memmap)}")

# como se constrúe a consulta en src/classification.py
img_times = torch.linspace(1.0e8, 1.0e8 + 66000.0, 11)
consulta = img_times - 0.5 * 1.0e6
print(f"  consulta: tipo={type(consulta).__name__} dtype={consulta.dtype}")

print("\n  -- searchsorted TAL E COMO ESTA NO CODIGO (consulta = tensor torch) --")
t0 = time.perf_counter()
r1 = np.searchsorted(ts, consulta)
t1 = time.perf_counter()
print(f"     {t1-t0:.3f} s")

print("  -- searchsorted coa consulta convertida ao dtype do array --")
c2 = np.asarray(consulta, dtype=ts.dtype)
t0 = time.perf_counter()
r2 = np.searchsorted(ts, c2)
t1 = time.perf_counter()
print(f"     {t1-t0:.3f} s")

print(f"\n  mesmos indices: {np.array_equal(np.asarray(r1), np.asarray(r2))}")
