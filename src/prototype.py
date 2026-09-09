from __future__ import annotations

import json

import h5py
import numpy as np


def _events_to_grid(
    events: np.ndarray,
    roi_height: int,
    roi_width: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    if len(events) == 0:
        return np.zeros((grid_h, grid_w), dtype=np.float64)

    gy = np.clip((events[:, 1] / roi_height * grid_h).astype(int), 0, grid_h - 1)
    gx = np.clip((events[:, 0] / roi_width * grid_w).astype(int), 0, grid_w - 1)

    grid = np.zeros((grid_h, grid_w), dtype=np.float64)
    np.add.at(grid, (gy, gx), 1)

    max_val = grid.max()
    return grid / max_val if max_val > 0 else grid


def _limite_binario(dataset, obxectivo: float, dereita: bool) -> int:
    """Indice do primeiro evento con t >= obxectivo (ou t > obxectivo se dereita).

    Equivale a np.searchsorted sobre a columna de timestamps, pero sen
    materializala: fai ~log2(N) lecturas dun so elemento sobre o dataset HDF5.
    Require que os timestamps sexan non decrecentes, cousa que a etapa validate
    do pipeline comproba en todo o corpus.
    """
    baixo, alto = 0, len(dataset)
    while baixo < alto:
        medio = (baixo + alto) // 2
        valor = float(dataset[medio, 2])
        se_avanza = valor <= obxectivo if dereita else valor < obxectivo
        if se_avanza:
            baixo = medio + 1
        else:
            alto = medio
    return baixo


def build_ed_prototype(
    data_path: str,
    ann_path: str,
    split: str = "train",
    label: str = "ed",
    grid_h: int = 16,
    grid_w: int = 16,
    min_duration: float = 2.0,
    recordings: set[str] | None = None,
) -> np.ndarray:
    with open(ann_path) as f:
        ann = json.load(f)

    with h5py.File(data_path, "r") as hf:
        split_recs = (
            set(recordings)
            if recordings is not None
            else {r for r in hf if hf[r].attrs.get("split") == split}
        )
        missing = split_recs - set(hf.keys())
        if missing:
            raise ValueError(f"Prototype recordings are missing from HDF5: {sorted(missing)}")

        grids: list[np.ndarray] = []
        n_skipped = 0

        for rec, v in ann["database"].items():
            if rec not in split_recs or rec not in hf:
                continue
            for roi_str, roi_anns in v["annotations"].items():
                if roi_str == "null":
                    continue
                matching_annotations = [
                    item
                    for item in roi_anns
                    if item["label"] == label
                    and (item["segment"][1] - item["segment"][0]) >= min_duration
                ]
                n_skipped += sum(
                    item["label"] == label
                    and (item["segment"][1] - item["segment"][0]) < min_duration
                    for item in roi_anns
                )
                if not matching_annotations:
                    continue
                roi_id = f"N{int(roi_str):02d}"
                if roi_id not in hf[rec]:
                    continue

                roi_height = int(hf[rec][roi_id].attrs["height"])
                roi_width = int(hf[rec][roi_id].attrs["width"])
                dataset = hf[rec][roi_id]["events"]

                for a in matching_annotations:
                    t_start, t_end = a["segment"]

                    # Antes cargabase a gravacion enteira con np.array() e
                    # aplicabase unha mascara booleana para quedar cuns poucos
                    # segundos. No corpus real hai gravacions de 1.467.653.571
                    # eventos: 23,5 GB para extraer uns megas, e o 2026-09-06
                    # iso levou build_prototypes a 25,8 GB e a maquina a swap.
                    #
                    # Os timestamps son non decrecentes (validate compróbao en
                    # todo o corpus), asi que a mascara selecciona un tramo
                    # CONTIGUO. Localizase por busca binaria sobre o dataset:
                    # ~31 lecturas dun elemento en vez de 1.470 millons. Os
                    # limites son os mesmos que daba a mascara: inicio no
                    # primeiro t >= t_start, fin despois do ultimo t <= t_end.
                    ini = _limite_binario(dataset, t_start * 1e6, dereita=False)
                    fin = _limite_binario(dataset, t_end * 1e6, dereita=True)
                    events = np.asarray(dataset[ini:fin])

                    if len(events) < 10:
                        n_skipped += 1
                        continue

                    grids.append(_events_to_grid(events, roi_height, roi_width, grid_h, grid_w))

    print(f"[prototipo] Instancias usadas: {len(grids)}  |  Descartadas: {n_skipped}")

    if not grids:
        print("[prototipo] AVISO: non se atoparon instancias. Retornando prototipo cero.")
        return np.zeros((grid_h, grid_w), dtype=np.float64)

    prototype = np.mean(grids, axis=0)
    norm = np.linalg.norm(prototype)
    return prototype / norm if norm > 0 else prototype


def get_prototype_score(
    events: np.ndarray,
    bins: np.ndarray,
    prototype: np.ndarray,
    roi_height: int,
    roi_width: int,
    min_events_per_bin: int = 5,
) -> np.ndarray:
    """Similitude coseno por bin co prototipo ED.

    Bins con menos de min_events_per_bin eventos reciben 0 (datos insuficientes).
    Como o prototipo está normalizado L2, a similitude coseno é o produto escalar.
    """
    grid_h, grid_w = prototype.shape
    bin_num = len(bins) - 1

    n_cells = grid_h * grid_w
    proto_flat = prototype.ravel()

    # Acumulacion por bloques. Cada temporal intermedio (bin_idx, gy, gx,
    # flat_idx) e do tamano dos eventos: con 493 M son ~35 GB, e o OOM killer
    # mataba o proceso nas gravacions grandes de THUMOS14-E. Aqui os temporais
    # limitanse ao bloque. Resultado identico.
    CHUNK = 25_000_000
    n = len(events)
    grid = np.zeros((bin_num, n_cells), dtype=np.float64)
    counts_per_bin = np.zeros(bin_num, dtype=np.int64)

    for ini in range(0, n, CHUNK):
        sl = slice(ini, min(ini + CHUNK, n))
        # O bloque lese UNHA vez. Antes facianse tres slices por columna
        # (events[sl, 2], [sl, 1], [sl, 0]); cando `events` e un dataset HDF5
        # sen ler, cada slice descomprime os mesmos chunks outra vez. A
        # aritmetica non cambia: as columnas saense do bloque xa lido.
        blq = np.asarray(events[sl])
        bi = np.searchsorted(bins[1:], blq[:, 2], side="right")
        np.clip(bi, 0, bin_num - 1, out=bi)
        gy = np.clip((blq[:, 1] / roi_height * grid_h).astype(np.int64), 0, grid_h - 1)
        gx = np.clip((blq[:, 0] / roi_width * grid_w).astype(np.int64), 0, grid_w - 1)
        comb = bi * n_cells + gy * grid_w + gx
        grid += np.bincount(comb, minlength=bin_num * n_cells).reshape(bin_num, n_cells)
        counts_per_bin += np.bincount(bi, minlength=bin_num)
        del blq, bi, gy, gx, comb

    scores = np.zeros(bin_num, dtype=np.float64)
    normas = np.linalg.norm(grid, axis=1)
    validos = (counts_per_bin >= min_events_per_bin) & (normas > 0)
    scores[validos] = np.clip((grid[validos] @ proto_flat) / normas[validos], 0.0, 1.0)

    return scores
    combinado = bin_idx.astype(np.int64) * n_cells + flat_idx
    grid = np.bincount(combinado, minlength=bin_num * n_cells).astype(np.float64)
    grid = grid.reshape(bin_num, n_cells)
    normas = np.linalg.norm(grid, axis=1)
    validos = np.zeros(bin_num, dtype=bool)
    validos[active_bins] = True
    validos &= normas > 0
    scores[validos] = np.clip(
        (grid[validos] @ proto_flat) / normas[validos], 0.0, 1.0
    )

    return scores
