"""Converte un .aedat4 real (DVXplorer 640x480) ao esquema que consome o pipeline.

Transformacions, e son SO estas duas:
  1. Reescalado espacial a 346x260 con aritmetica enteira exacta:
     xr = (x*346)//640  e  yr = (y*260)//480. Mapean [0,639]->[0,345] e
     [0,479]->[0,259] cubrindo o rango COMPLETO.
  2. Timestamps a base cero en microsegundos. O aedat4 dáos como microsegundos
     absolutos desde a epoca Unix (~1.7e15): restase o primeiro evento como
     int64 ANTES de calquera conversion a float, porque float32 non pode
     representar esa magnitude e colapsaria todos os eventos nun instante.
     Gardase t_orixe_us para que a operacion sexa reversible.

NON HAI DEDUPLICACION, e a razon esta medida. Unha version anterior deste
ficheiro quedaba cun evento por (pixel, bin de 33 ms, polaridade). Descartouse
por tres motivos, o primeiro deles fatal:

  a) `get_event_rate` (src/proposals.py:38-42) fai np.histogram sobre os
     timestamps: o sinal de actionness E un reconto cru de eventos. Limitar
     cada pixel a 30 eventos/s comprime precisamente as rexions de alta taxa,
     que son onde vive a accion. Dana SHAPE e a completitude C(p) directamente.
  b) Sobre unha xanela RECORTADA o dedup non e exacto: se o ultimo evento dun
     pixel cae no bin que a xanela corta, o superviviente dese bin queda fora e
     o pixel retrocede. O erro depende de onde caia o bordo dentro do bin. A
     rama continua ten bordos en multiplos de 500.000 us e 500000 = 15*33333+5,
     asi que o desfase e de 5k us e o efecto e desprezable; pero a rama do
     clasificador ATSN sobre propostas (src/classification.py:218-222) ten
     bordos arbitrarios e cae no rexime malo. Iso corrompe MOITO mais o brazo
     reTAG que o noso: unha asimetria que favorece o noso metodo, e a primeira
     cousa que atacaria un revisor.
  c) A xustificacion que tiña era compensar que dividir coordenadas multiplica
     a densidade por pixel por (640/346)*(480/260) = 3,42. Medido, iso non
     ocorre: a gravacion real da 1,26 / 0,60 / 0,57 Mev/s en tres videos fronte
     a 2,32 / 0,74 / 0,44 do corpus v2e dos mesmos. A densidade real esta na
     mesma orde, e de mediana por baixo. Non habia nada que compensar.

Saida: un .h5 por video coa MESMA xerarquia que o corpus v2e:
  /recording                 attrs: split, official_subset, cv_fold, duration_s,
                                    labels_json, protocol_id, t_orixe_us
  /recording/N01             attrs: width=346, height=260, duration_s
  /recording/N01/events      (N,4) uint32 [x, y, t_us, p]
Sen os attrs de /recording, extract_continuous_features.py:149 le split="",
descarta todas as gravacions e o pipeline peta con "No ROI sequences matched".
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import aedat
import h5py
import numpy as np

ANCHO, ALTO = 346, 260
ANCHO_FONTE, ALTO_FONTE = 640, 480
BLOQUE = 10_000_000     # eventos por bloque; fixa o pico de RAM (~0,8 GB/worker)
FILAS_CHUNK = 65_536    # 1 MB por chunk, para que caiba na cache por defecto de h5py
PROTOCOLO = "uibk-dvxplorer-real-v1"


def converter(
    path_entrada: Path | str,
    path_saida: Path | str,
    fila_manifesto: dict | None = None,
) -> tuple[int, float]:
    """Converte un ficheiro e devolve (eventos escritos, duracion en segundos)."""
    dec = aedat.Decoder(str(path_entrada))
    streams = dec.id_to_stream()
    fluxo = next((s for s in streams.values() if s.get("type") == "events"), None)
    if fluxo is None:
        raise ValueError(f"{path_entrada} non ten fluxo de eventos")
    if (fluxo["width"], fluxo["height"]) != (ANCHO_FONTE, ALTO_FONTE):
        raise ValueError(f"{path_entrada}: resolucion {fluxo} distinta de 640x480")

    tmp = Path(str(path_saida) + ".parcial")
    if tmp.exists():
        tmp.unlink()
    total = 0
    t_orixe: int | None = None
    t_ultimo = 0
    buf_x: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    buf_t: list[np.ndarray] = []
    buf_p: list[np.ndarray] = []
    n_buf = 0

    with h5py.File(tmp, "w") as fout:
        rec = fout.create_group("recording")
        g = rec.create_group("N01")
        dset = g.create_dataset(
            "events",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.uint32,
            chunks=(FILAS_CHUNK, 4),
            compression="lzf",
        )

        def descargar() -> None:
            nonlocal total, n_buf, buf_x, buf_y, buf_t, buf_p
            if not n_buf:
                return
            x = np.concatenate(buf_x)
            y = np.concatenate(buf_y)
            t = np.concatenate(buf_t)
            p = np.concatenate(buf_p)
            buf_x, buf_y, buf_t, buf_p, n_buf = [], [], [], [], 0
            # Sen dedup non hai costura entre bloques: cada evento escribese unha vez.
            # As dúas invariantes que o resto do pipeline asume, comprobadas aqui
            # porque un uint32 negativo colaria en silencio como 4,29e9 us.
            if t[0] < 0:
                raise ValueError(f"{Path(path_entrada).stem}: timestamp negativo tras "
                                 f"restar t_orixe; o primeiro evento non e o minimo")
            if not np.all(np.diff(t) >= 0):
                raise ValueError(f"{Path(path_entrada).stem}: timestamps non monotonos")
            xr = ((x.astype(np.int64) * ANCHO) // ANCHO_FONTE).astype(np.uint32)
            yr = ((y.astype(np.int64) * ALTO) // ALTO_FONTE).astype(np.uint32)
            n = len(t)
            dset.resize(total + n, axis=0)
            saida = np.empty((n, 4), dtype=np.uint32)
            saida[:, 0] = xr
            saida[:, 1] = yr
            saida[:, 2] = t
            saida[:, 3] = p
            dset[total : total + n] = saida
            total += n

        for packet in dec:
            if "events" not in packet:
                continue
            e = packet["events"]
            if not len(e):
                continue
            t = e["t"].astype(np.int64)  # int64 antes de restar: nunca float
            if t_orixe is None:
                t_orixe = int(t[0])
            buf_t.append(t - t_orixe)
            buf_x.append(e["x"].astype(np.int32))
            buf_y.append(e["y"].astype(np.int32))
            buf_p.append(e["on"].astype(np.uint8))
            n_buf += len(t)
            t_ultimo = int(t[-1] - t_orixe)
            if n_buf >= BLOQUE:
                descargar()
        descargar()

        duracion = t_ultimo / 1e6
        rec.attrs["duration_s"] = np.float64(duracion)
        rec.attrs["protocol_id"] = PROTOCOLO
        rec.attrs["fonte"] = "UIBK DVXplorer 640x480 -> 346x260, sen deduplicar"
        rec.attrs["eventos"] = np.int64(total)
        rec.attrs["t_orixe_us"] = np.int64(t_orixe if t_orixe is not None else 0)
        rec.attrs["t_ultimo_us"] = np.int64(
            (t_orixe or 0) + t_ultimo
        )
        if fila_manifesto is not None:
            rec.attrs["split"] = str(fila_manifesto["split"])
            rec.attrs["official_subset"] = str(fila_manifesto["official_subset"])
            rec.attrs["cv_fold"] = str(fila_manifesto["cv_fold"])
            rec.attrs["labels_json"] = str(fila_manifesto["labels_json"])
        g.attrs["width"] = np.int64(ANCHO)
        g.attrs["height"] = np.int64(ALTO)
        g.attrs["duration_s"] = np.float64(duracion)

    tmp.replace(path_saida)
    return total, duracion


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("saida")
    args = ap.parse_args()
    n, d = converter(args.entrada, args.saida)
    print(
        f"{Path(args.entrada).stem}: {n:,} eventos, {d:.2f} s, "
        f"{os.path.getsize(args.saida) / 2**20:.1f} MB"
    )
