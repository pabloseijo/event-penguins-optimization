"""Discrimina a cabeza conxelada entre proposta boa e ruido nos eventos reais?

Mide o AUC de cnn_score para separar propostas con tIoU>=0.5 do resto, no split
de TEST. AUC 0.5 = a cabeza non aprendeu nada; 1.0 = perfecta. E a pregunta de
fondo: funciona o encoder ATSN conxelado sobre eventos reais de DVXplorer?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/pablo.garcia.seijo/event_penguins")
P = R / "tmp/thumos14e_real/v1"
CANON = Path("/home/pablo.garcia.seijo/actionformer_release/data/thumos/annotations/thumos14.json")
db = json.loads(CANON.read_text()); db = db.get("database", db)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    todo = np.concatenate([pos, neg])
    r = pd.Series(todo).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


print(f"{'clase':<19}{'n_pos':>7}{'n_neg':>9}{'AUC':>8}{'score_pos':>11}{'score_neg':>11}")
print("-" * 65)
aucs = []
for d in sorted((P / "predictions/retag/seed_1337").iterdir()):
    f = d / "proposal_scores.csv"
    if not f.exists():
        continue
    clase = d.name
    sc = pd.read_csv(f)
    gt = {}
    for vid, info in db.items():
        if (info.get("subset") or "").lower() != "test":
            continue
        a = [x["segment"] for x in info.get("annotations", []) if x.get("label") == clase]
        if a:
            gt[vid] = np.array(a, dtype=float)
    t = np.zeros(len(sc))
    s0 = sc["t_start"].to_numpy() / 1e6
    s1 = sc["t_end"].to_numpy() / 1e6
    recs = sc["rec_name"].to_numpy()
    for vid, g in gt.items():
        m = np.nonzero(recs == vid)[0]
        if len(m) == 0:
            continue
        inter = np.maximum(0, np.minimum(s1[m][:, None], g[:, 1]) - np.maximum(s0[m][:, None], g[:, 0]))
        union = (s1[m] - s0[m])[:, None] + (g[:, 1] - g[:, 0]) - inter
        t[m] = np.max(np.where(union > 0, inter / union, 0), axis=1)
    cs = sc["cnn_score"].to_numpy()
    pos, neg = cs[t >= 0.5], cs[t < 0.1]
    a = auc(pos, neg)
    aucs.append(a)
    print(f"{clase:<19}{len(pos):>7}{len(neg):>9,}{a:>8.3f}"
          f"{np.mean(pos) if len(pos) else float('nan'):>11.4f}{np.mean(neg):>11.4f}")
print("-" * 65)
v = [x for x in aucs if not np.isnan(x)]
print(f"AUC medio sobre {len(v)} clases: {np.mean(v):.3f}   (0.5 = azar)")
