"""Constrúe un conxunto de propostas que exercite as DUAS vías de __getitem__.

A vía nova (ler a union das xanelas dunha vez) so se activa cando a union cabe
en max_union_events. Interesa probar as duas: propostas curtas, que van pola
nova, e longas, que seguen indo pola de sempre.
"""
from __future__ import annotations

import sys

import h5py
import pandas as pd

propostas = sys.argv[1]
hdf5 = sys.argv[2]
saida = sys.argv[3]

p = pd.read_csv(propostas)
with h5py.File(hdf5, "r") as f:
    tam = {r: f[r]["N01"]["events"].shape[0] for r in f.keys()}

p["n"] = p["rec_name"].map(tam)
p["dur"] = (p["t_end"] - p["t_start"]) / 1e6

# gravacions pequenas, para que a proba sexa rapida
peq = p[p["n"] < 3_000_000]
curtas = peq[peq["dur"] < 0.5].head(16)
longas = peq[peq["dur"] > 5.0].head(4)
if len(longas) == 0:
    longas = p[p["dur"] > 5.0].head(2)

sub = pd.concat([curtas, longas])
sub[["rec_name", "roi_id", "t_start", "t_end", "score"]].to_csv(saida, index=False)

print(f"proba: {len(curtas)} curtas (via nova) + {len(longas)} longas (via antiga)")
print(f"duracions de {sub['dur'].min():.2f}s a {sub['dur'].max():.2f}s")
print(f"gravacions implicadas: {sub['rec_name'].nunique()}")
