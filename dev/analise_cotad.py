"""Analise do lattice de CoTAD: canto recall sobrevive a cada forma de recortar.

Hipotese a comprobar: limit_frame fai df.sample() UNIFORME, asi que quedarse con
50k de 3,0 M conserva o 1,6% de TODO, positivos incluidos. Se e iso, o recall
colapsa por construcion e a culpa e do recorte, non do metodo.

Usanse as funcions do propio modulo (best_match_seconds, build_annotation_index,
roi_to_ann_key) en vez de reimplementalas: xa houbo un test de identidade que
fallaba porque o erro estaba no test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "dev"))

from dev.train_quality_head import (  # noqa: E402
    best_match_seconds,
    build_annotation_index,
    roi_to_ann_key,
)

CLASE = sys.argv[1] if len(sys.argv) > 1 else "BaseballPitch"
FOLD = R / f"tmp/thumos14e_real/v1/eventpenguins_full/seed_1337/{CLASE}/local/fold_00"
ANN = R / f"data/thumos14_real/config/annotations/by_class/{CLASE}/annotations.json"

POS_TIOU = 0.5
SEMI_TIOU = 0.1


def mellor_tiou(df: pd.DataFrame, split: str) -> np.ndarray:
    idx, _ = build_annotation_index(ANN, split, 0.0)
    out = np.zeros(len(df), dtype=np.float64)
    recs = df["rec_name"].to_numpy()
    rois = df["roi_id"].to_numpy()
    ts = df["t_start"].to_numpy(dtype=np.float64)
    te = df["t_end"].to_numpy(dtype=np.float64)
    cache: dict = {}
    for i in range(len(df)):
        k = (recs[i], rois[i])
        if k not in cache:
            cache[k] = idx.get(recs[i], {}).get(roi_to_ann_key(rois[i]), {}).get(
                "ed", np.empty((0, 2))
            )
        out[i] = best_match_seconds(ts[i], te[i], cache[k])[0]
    return out


def informe(nome: str, ruta: Path, split: str, tope: int) -> None:
    df = pd.read_csv(ruta)
    print(f"\n=== {nome}: {len(df):,} propostas, tope do recorte {tope:,} "
          f"({tope/len(df)*100:.2f}%) ===")
    t = mellor_tiou(df, split)
    pos = t >= POS_TIOU
    semi = t >= SEMI_TIOU
    print(f"  no lattice COMPLETO: {pos.sum():,} positivos (tIoU>=0.5), "
          f"{semi.sum():,} con tIoU>=0.1")

    rng = np.random.default_rng(1337)
    sel = rng.choice(len(df), size=min(tope, len(df)), replace=False)
    print(f"  recorte ALEATORIO (o que fixen eu): {pos[sel].sum():,} positivos "
          f"({pos[sel].sum()/max(pos.sum(),1)*100:.1f}% dos que habia)")

    if "score" in df.columns:
        orde = np.argsort(-df["score"].to_numpy(dtype=np.float64), kind="stable")[:tope]
        print(f"  recorte por SCORE (top-k):        {pos[orde].sum():,} positivos "
              f"({pos[orde].sum()/max(pos.sum(),1)*100:.1f}% dos que habia)")

    # teito de recall: cantas instancias de verdade quedan cubertas
    def cobertas(masc: np.ndarray) -> int:
        sub = df[masc & pos]
        return len(sub.groupby(["rec_name", "roi_id"]).size()) if len(sub) else 0

    todo = np.ones(len(df), dtype=bool)
    m_alea = np.zeros(len(df), dtype=bool); m_alea[sel] = True
    print(f"  ROIs con algun positivo -> completo: {cobertas(todo)}  "
          f"aleatorio: {cobertas(m_alea)}")


informe("TRAIN", FOLD / "lattice_train.csv", "train", 50000)
informe("VAL", FOLD / "lattice_val.csv", "val", 20000)
