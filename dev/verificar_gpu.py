"""Verifica a cadea completa GPU fronte a CPU: eventos -> tensor normalizado."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path("/home/pablo.garcia.seijo/event_penguins")
sys.path.insert(0, str(R))
from src.classification import ProposalDataset, create_img_representation, create_time_map  # noqa: E402

H, W, DECAY = 260, 346, 5e-6
dev = torch.device("cuda:0")
rng = np.random.default_rng(7)


def tm_gpu(ev, decay, height, width):
    n = ev.shape[0]
    d = ev.device
    if n == 0:
        return torch.zeros((height, width), dtype=torch.float64, device=d)
    x = ev[:, 0].long(); y = ev[:, 1].long(); t = ev[:, 2].double(); p = ev[:, 3].long()
    idx = y * width + x
    gan = torch.full((height * width,), -1, dtype=torch.long, device=d)
    gan.scatter_reduce_(0, idx, torch.arange(n, device=d), reduce="amax", include_self=True)
    toc = gan >= 0; seg = gan.clamp(min=0)
    tm = torch.zeros(height * width, dtype=torch.float64, device=d)
    tm[toc] = t[seg[toc]]
    tm = torch.exp(-decay * (t.max() - tm))
    pol = torch.where(p == 0, -1.0, 1.0).double()
    tm[toc] = tm[toc] * pol[seg[toc]]
    return tm.view(height, width)


def img_gpu(ev, decay, height, width, redondear_resize: bool):
    tm = tm_gpu(ev, decay, height, width)
    # range_norm(lower=-1, upper=1, dtype=uint8): trunca, non redondea
    u8 = torch.floor(255.0 * (torch.clamp(tm, -1.0, 1.0) + 1.0) / 2.0)
    out = F.interpolate(u8[None, None].float(), size=(224, 224),
                        mode="bilinear", align_corners=False, antialias=True)[0, 0]
    if redondear_resize:
        out = torch.round(out).clamp(0, 255)   # PIL devolve uint8
    img = out[None].repeat(3, 1, 1) / 255.0
    m = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    return (img - m) / s


for n in (200_000, 1_000_000, 3_000_000):
    ev = np.empty((n, 4), dtype=np.uint32)
    ev[:, 0] = rng.integers(0, W, n); ev[:, 1] = rng.integers(0, H, n)
    ev[:, 2] = np.sort(rng.integers(0, 3_000_000, n)); ev[:, 3] = rng.integers(0, 2, n)

    cpu = create_img_representation(ev, DECAY, H, W, ProposalDataset._transform)
    g = torch.from_numpy(ev.astype(np.int32)).to(dev)
    for redondear in (False, True):
        gpu = img_gpu(g, DECAY, H, W, redondear).cpu().numpy()
        d = np.abs(gpu - cpu.numpy())
        print(f"  n={n:>9,}  resize {'redondeado' if redondear else 'sen redondear':<14} "
              f"maxdif={d.max():.4f}  media={d.mean():.5f}  "
              f">0.01: {100*(d>0.01).mean():.2f}%")
    print()
