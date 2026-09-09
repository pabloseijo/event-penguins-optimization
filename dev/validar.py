"""Comproba esquema e a propiedade de dedup (queda o ULTIMO do bin)."""
import sys
import h5py, numpy as np

p = sys.argv[1]
BIN = 33333
with h5py.File(p, "r") as f:
    print("grupos:", list(f.keys()))
    g = f["N01"]
    print("datasets:", list(g.keys()))
    print("attrs:", {k: g.attrs[k] for k in ("duration_s", "height", "width")})
    e = g["events"]
    print("shape", e.shape, "dtype", e.dtype, "chunks", e.chunks, "compr", e.compression)
    a = np.asarray(e[:3_000_000]).astype(np.int64)
    x, y, t, pol = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    print("rango x:", x.min(), x.max(), "| y:", y.min(), y.max(), "| p:", np.unique(pol))
    print("t monotono non decrecente:", bool(np.all(np.diff(t) >= 0)))
    clave = (((t // BIN) * 260 + y) * 346 + x) * 2 + pol
    print("claves unicas == filas:", len(np.unique(clave)) == len(clave))
