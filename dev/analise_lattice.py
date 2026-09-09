"""De onde saen os 3 M de propostas do lattice, e canto rende cada parte."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "dev"))
from dev.train_quality_head import best_match_seconds, build_annotation_index, roi_to_ann_key  # noqa: E402

CLASE = sys.argv[1] if len(sys.argv) > 1 else "BaseballPitch"
FOLD = R / f"tmp/thumos14e_real/v1/eventpenguins_full/seed_1337/{CLASE}/local/fold_00"
ANN = R / f"data/thumos14_real/config/annotations/by_class/{CLASE}/annotations.json"

df = pd.read_csv(FOLD / "lattice_train.csv")
print(f"lattice_train: {len(df):,} propostas")
print(f"  gravacions: {df['rec_name'].nunique()}   ROIs: {df.groupby(['rec_name','roi_id']).ngroups}")
print()
print("  por 'source':")
print(df["source"].value_counts().to_string().replace("\n", "\n    "))
print()
print("  por 'variant' (top 12 de", df["variant"].nunique(), "):")
print(df["variant"].value_counts().head(12).to_string().replace("\n", "\n    "))

idx, _ = build_annotation_index(ANN, "train", 0.0)
cache: dict = {}
def tiou(sub: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        k = (r["rec_name"], r["roi_id"])
        if k not in cache:
            cache[k] = idx.get(r["rec_name"], {}).get(roi_to_ann_key(r["roi_id"]), {}).get("ed", np.empty((0, 2)))
        out[i] = best_match_seconds(r["t_start"], r["t_end"], cache[k])[0]
    return out

print()
print("  rendemento por variante (positivos tIoU>=0.5 por cada 1000 propostas):")
filas = []
for v, g in df.groupby("variant"):
    m = g if len(g) <= 4000 else g.sample(4000, random_state=1337)
    t = tiou(m)
    taxa = (t >= 0.5).mean() * 1000
    filas.append((v, len(g), taxa))
filas.sort(key=lambda x: -x[2])
print(f"    {'variante':<26}{'n':>10}{'pos/1000':>10}")
for v, n, taxa in filas[:10]:
    print(f"    {v:<26}{n:>10,}{taxa:>10.2f}")
print("    ...")
for v, n, taxa in filas[-5:]:
    print(f"    {v:<26}{n:>10,}{taxa:>10.2f}")
