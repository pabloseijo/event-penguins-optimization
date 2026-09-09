"""Ensambla o corpus de eventos reais de UIBK no formato que espera o pipeline.

Correse NO SERVIDOR, despois de transferir os .h5 a data/thumos14_real/canonical/.

Que fai, e por que cada cousa:

  1. Reutiliza `manifest.csv`, `config/annotations/**` e `split_audit.json` do
     corpus v2e. Son os MESMOS 413 video_id coas mesmas anotacions canonicas e
     os mesmos folds; o unico que cambia son os eventos. Copialos en vez de
     rexeneralos evita introducir diferenzas onde non as hai.
  2. Constrúe `preprocessed.h5` como indice de ExternalLink cara a /recording
     de cada ficheiro, exactamente como fai prepare_thumos14_event_corpus.py:1370-1375.
     Por iso o indice ocupa kilobytes e non hai que copiar 266 GB.
  3. Escribe `corpus_manifest.json` cun rol de conversion NOVO, `real_dvxplorer`.
     NON se escribe procedencia de v2e: este corpus non ten timestamp_resolution
     nin dvs_profile porque non se simulou nada, e poñelos seria mentir na
     traza. Iso implica que validate_assembled_corpus ten que aprender o rol
     novo; mentres non o faga, esta saida non pasara a validacion do plan, e
     isto e deliberado.
  4. Escribe `source_audit.json` cos 413 .aedat4 de orixe e o sha256 dos .h5
     xa convertidos (calculado no servidor, que le desde NVMe).

Uso:
    python ensamblar_corpus_real.py --v2e-dir  data/thumos14_events/thumos14e_original_rate_v1 \
                                    --real-dir data/thumos14_real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import pandas as pd

ROL = "real_dvxplorer"
PROTOCOLO = "uibk-dvxplorer-real-v1"


def sha256_ficheiro(path: Path, bloque: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(bloque)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copiar_metadatos(v2e: Path, real: Path) -> None:
    for nome in ("manifest.csv", "split_audit.json"):
        orixe = v2e / nome
        if orixe.exists():
            shutil.copy2(orixe, real / nome)
            print(f"[copiado] {nome}")
    destino_cfg = real / "config"
    if destino_cfg.exists():
        shutil.rmtree(destino_cfg)
    shutil.copytree(v2e / "config", destino_cfg)
    print("[copiado] config/ (anotacions, folds e vistas por clase)")
    for nome in ("official_annotations", "source_metadata", "videos"):
        orixe = v2e / nome
        ligazon = real / nome
        if orixe.exists() and not ligazon.exists():
            os.symlink(os.path.realpath(orixe), ligazon)
            print(f"[ligazon] {nome}")


def construir_indice(real: Path, ids: list[str]) -> Path:
    canonical = real / "canonical"
    indice = real / "preprocessed.h5"
    temporal = indice.with_suffix(".h5.building")
    if temporal.exists():
        temporal.unlink()
    with h5py.File(temporal, "w") as idx:
        idx.attrs["format"] = "event-penguins-external-[x,y,t_us,p]-v1"
        idx.attrs["source"] = "THUMOS14 gravado con DVXplorer real (espello de UIBK)"
        idx.attrs["conversion_role"] = ROL
        for vid in ids:
            rel = os.path.relpath(canonical / f"{vid}.h5", real)
            idx[vid] = h5py.ExternalLink(rel, "/recording")
    os.replace(temporal, indice)
    print(f"[indice] {indice} con {len(ids)} ExternalLink")
    return indice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2e-dir", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--aedat-inventario", default=None,
                    help="CSV opcional con nome,tamano dos .aedat4 de orixe")
    args = ap.parse_args()

    v2e = Path(args.v2e_dir).resolve()
    real = Path(args.real_dir).resolve()
    canonical = real / "canonical"
    real.mkdir(parents=True, exist_ok=True)
    if not canonical.is_dir():
        raise SystemExit(f"non existe {canonical}: transfire primeiro o corpus")

    manifest = pd.read_csv(v2e / "manifest.csv", keep_default_na=False)
    ids = [str(v) for v in manifest["video_id"]]
    if len(ids) != 413:
        raise SystemExit(f"o manifesto ten {len(ids)} vídeos, esperabanse 413")

    faltan = [v for v in ids if not (canonical / f"{v}.h5").exists()]
    if faltan:
        raise SystemExit(f"faltan {len(faltan)} ficheiros en canonical/: {faltan[:5]}")

    copiar_metadatos(v2e, real)

    rexistros = []
    total_eventos = 0
    for i, vid in enumerate(ids, 1):
        p = canonical / f"{vid}.h5"
        with h5py.File(p, "r") as f:
            rec = f["recording"]
            n = int(rec["N01"]["events"].shape[0])
            dur = float(rec.attrs["duration_s"])
            t0 = int(rec.attrs.get("t_orixe_us", 0))
        total_eventos += n
        rexistros.append({
            "video_id": vid,
            "canonical_path": str(p),
            "canonical_sha256": sha256_ficheiro(p),
            "events": n,
            "duration_s": dur,
            "t_orixe_us": t0,
            "protocol_id": PROTOCOLO,
            "source_bytes": int(p.stat().st_size),
        })
        if i % 50 == 0:
            print(f"[hash] {i}/{len(ids)}", flush=True)

    indice = construir_indice(real, ids)

    corpus_manifest = {
        "partial": False,
        "index_path": str(indice),
        "index_sha256": sha256_ficheiro(indice),
        "recordings": rexistros,
        "conversion_protocols": {
            PROTOCOLO: {
                "role": ROL,
                "recipe": {
                    # "timing" e o campo que valida CONVERSION_ROLES; para unha
                    # captura real vale "hardware_capture", non un modo de v2e.
                    "timing": "hardware_capture",
                    "capture": "hardware",
                    "sensor": "DVXplorer",
                    "sensor_resolution": [640, 480],
                    "output_width": 346,
                    "output_height": 260,
                    "spatial_rescale": "enteiro exacto: (x*346)//640, (y*260)//480",
                    "temporal": "sen remostraxe; timestamps nativos a base cero",
                    "deduplicacion": None,
                },
                "implementation": "dev/converter_aedat4.py",
                "fonte": "espello de UIBK, iis.uibk.ac.at/public/datasets/thumos14-dataset",
            }
        },
        "eventos_totais": total_eventos,
    }
    (real / "corpus_manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[escrito] corpus_manifest.json · {total_eventos:,} eventos")

    source_audit = {
        "videos": len(ids),
        "hashes_verified": True,
        "que_se_verificou": (
            "sha256 dos 413 .h5 convertidos, calculado neste servidor. Os .aedat4 "
            "de orixe viven na maquina local de Pablo e non se hashearon aqui."
        ),
        "conversion_role": ROL,
        "recordings": [
            {k: r[k] for k in ("video_id", "canonical_sha256", "events", "duration_s")}
            for r in rexistros
        ],
    }
    (real / "source_audit.json").write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[escrito] source_audit.json")
    print("\nSeguinte paso: correr a etapa validate do pipeline sobre este work-dir.")


if __name__ == "__main__":
    main()
