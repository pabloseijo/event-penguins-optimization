"""Mide ONDE vai o tempo ao construír as time surfaces dunha proposta.

Chama ao mesmo ProposalDataset.__getitem__ que usa a extracción real, sobre
propostas reais, e reparte o tempo entre as catro fases sospeitosas:
lectura do HDF5, scatter, exponencial e resize.

Sen isto, calquera optimizacion e a cegas: o informe estimaba 11,5 ms por
superficie segundo benchmarks publicados e nos estamos moi por riba.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classification import ProposalDataset  # noqa: E402

propostas_csv = sys.argv[1]
hdf5 = sys.argv[2]
cache_ts = sys.argv[3]
n_items = int(sys.argv[4]) if len(sys.argv) > 4 else 30

p = pd.read_csv(propostas_csv)
# propostas consecutivas dunha mesma gravacion, como as ve o extractor real
rec = p["rec_name"].value_counts().index[0]
sub = p[p["rec_name"] == rec].head(n_items).reset_index(drop=True)
print(f"gravacion {rec} · {len(sub)} propostas")
print(f"duracion mediana {((sub['t_end']-sub['t_start'])/1e6).median():.3f} s")

ds = ProposalDataset(
    proposals=sub,
    augment_fraction=0.0,
    data_path=hdf5,
    num_tsn_samples=21,
    sample_duration=1.0 * 1e6,
    decay=5e-6,
    cache_full_events=False,
    timestamp_cache_dir=cache_ts,
)

# quecemento: a primeira chamada abre o HDF5 e carga a cache de timestamps
_ = ds[0]

t0 = time.time()
pr = cProfile.Profile()
pr.enable()
for i in range(len(sub)):
    _ = ds[i]
pr.disable()
total = time.time() - t0

n_sup = len(sub) * 21
print(f"\ntotal {total:.2f} s para {len(sub)} propostas = {n_sup} superficies")
print(f"  {total/len(sub)*1000:.0f} ms por proposta · {total/n_sup*1000:.1f} ms por superficie")
print(f"  (referencia publicada para 420k eventos en NumPy: ~11,5 ms)\n")

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
saida = s.getvalue()
# quedamos coas liñas de datos, sen a cabeceira longa
for linha in saida.splitlines():
    if "{" in linha or "/" in linha or "function calls" in linha:
        print(linha[:150])
