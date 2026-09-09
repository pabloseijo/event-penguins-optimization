"""Compara os bounds da version vella e da nova sobre gravacions reais."""
import sys
import time

import h5py
import numpy as np

data_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12

with h5py.File(data_path, "r") as hf:
    recs = sorted(hf.keys())[:n]
    vello, novo = {}, {}
    t0 = time.time()
    for rec in recs:
        for roi in hf[rec].keys():
            ev = np.asarray(hf[rec][roi]["events"])
            if len(ev) == 0:
                continue
            vello[(rec, roi)] = (float(ev[0, 2]), float(ev[-1, 2]))
            del ev
    t_vello = time.time() - t0
    t0 = time.time()
    for rec in recs:
        for roi in hf[rec].keys():
            ds = hf[rec][roi]["events"]
            if ds.shape[0] == 0:
                continue
            novo[(rec, roi)] = (float(ds[0, 2]), float(ds[-1, 2]))
    t_novo = time.time() - t0

print(f"gravacions: {len(vello)}")
print(f"IDENTICOS: {vello == novo}")
print(f"vello {t_vello:.1f}s · novo {t_novo:.3f}s · {t_vello/max(t_novo,1e-6):.0f}x")
