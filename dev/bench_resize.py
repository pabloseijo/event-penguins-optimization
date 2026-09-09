"""PIL fronte a F.interpolate: canto difire o redimensionado?

E o unico paso do pipeline que non ten equivalente exacto en torch. PIL aplica
un filtro con soporte escalado (antialiasing) ao reducir; F.interpolate so o fai
con antialias=True, e nin asi e garantia de identidade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R))
from src.classification import create_time_map, range_norm  # noqa: E402

H, W, DECAY = 260, 346, 5e-6
rng = np.random.default_rng(0)
dev = torch.device("cuda:0")

for n in (200_000, 1_000_000):
    ev = np.empty((n, 4), dtype=np.uint32)
    ev[:, 0] = rng.integers(0, W, n); ev[:, 1] = rng.integers(0, H, n)
    ev[:, 2] = np.sort(rng.integers(0, 1_000_000, n)); ev[:, 3] = rng.integers(0, 2, n)

    m = create_time_map(ev, DECAY, H, W)
    u8 = range_norm(m, lower=-1, upper=1, dtype=np.uint8)

    pil = np.array(Image.fromarray(u8).resize((224, 224), resample=Image.BILINEAR))

    t = torch.from_numpy(u8).to(dev).double()[None, None]
    sen_aa = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False, antialias=False)
    con_aa = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)

    for nome, x in (("sen antialias", sen_aa), ("con antialias", con_aa)):
        a = x[0, 0].cpu().numpy()
        d = np.abs(a - pil.astype(np.float64))
        print(f"  n={n:>9,}  {nome:<14} maxdif={d.max():>7.2f}  media={d.mean():>6.3f}  "
              f"pixeis>1: {100*(d>1).mean():>5.1f}%")
    print(f"  (rango dos valores: {pil.min()}-{pil.max()})")
    print()
