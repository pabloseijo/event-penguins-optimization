"""Avalia as clases que xa esten listas, sen agardar polas vinte.

Por que fai falla: dev/evaluate_thumos14e_ovr.py esixe as 20 clases (aserta
len(label_to_id) == 20 e peta con FileNotFoundError se falta unha prediccion).
Iso e correcto para o resultado oficial, e NON se toca. Pero durante a corrida
interesa ver o que hai.

Por que os numeros son comparables: ANETdetection calcula o AP de CADA clase
por separado, sobre a mesma ground truth canonica e os mesmos limiares de tIoU.
Este script le evaluator.ap[:, columna] soamente das clases dispoñibles, asi
que **cada AP por clase e identico** ao que daria a corrida completa. O unico
que cambia e sobre cantas clases se promedia, e iso vai declarado no resumo.

Uso:
    avaliar_parcial.py --plan <plan> --seed 1337 --actionformer-root <ruta>
                       [--out-dir <ruta>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.evaluate_thumos14e_ovr import canonical_test_protocol  # noqa: E402
from dev.run_thumos14e_supervised import load_plan  # noqa: E402

LIMIARES = np.asarray([0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)

# As dúas ramas: onde vive a predicion de cada unha, relativa a out_root.
RAMAS = {
    "reTAG": "predictions/retag/seed_{seed}",
    "CoTAD": "eventpenguins_full/seed_{seed}",
}


def filas_de_predicion(raiz: Path, nome: str, videos_test: set, label_to_id: dict):
    """Como prediction_rows do avaliador oficial, pero SALTANDO o que falte."""
    filas = []
    presentes = []
    for label in sorted(label_to_id, key=label_to_id.get):
        ruta = raiz / label / nome
        if not ruta.exists():
            continue
        payload = json.loads(ruta.read_text(encoding="utf-8"))
        declarado = payload.get("target_class")
        if declarado not in {None, label}:
            raise ValueError(f"{ruta} declara target_class={declarado!r}, esperabase {label!r}")
        presentes.append(label)
        for video_id, rois in payload.get("results", {}).items():
            if video_id not in videos_test:
                continue
            for deteccions in rois.values():
                for d in deteccions:
                    ini, fin = map(float, d["segment"])
                    if not 0 <= ini < fin:
                        raise ValueError(f"Deteccion invalida en {ruta}: {d}")
                    filas.append(
                        {
                            "video-id": str(video_id),
                            "t-start": ini,
                            "t-end": fin,
                            "label": int(label_to_id[label]),
                            "score": float(d.get("score", 0.0)),
                        }
                    )
    return pd.DataFrame(filas), presentes


def avaliar_rama(nome_rama, raiz, anotacions, videos_test, label_to_id, af_root, nworkers):
    pred, presentes = filas_de_predicion(raiz, "predictions.json", videos_test, label_to_id)
    if not presentes:
        return None
    sys.path.insert(0, str(af_root))
    from libs.utils import ANETdetection  # type: ignore

    ev = ANETdetection(
        str(anotacions),
        split="test",
        tiou_thresholds=LIMIARES,
        num_workers=nworkers,
        dataset_name=f"THUMOS14 real · parcial · {nome_rama}",
    )
    ev.evaluate(pred, verbose=False)

    filas = []
    for label in presentes:
        lid = label_to_id[label]
        col = ev.activity_index[lid]
        fila = {"clase": label}
        fila.update(
            {f"AP@{t:.1f}": float(ev.ap[i, col]) for i, t in enumerate(LIMIARES)}
        )
        fila["AP_medio"] = float(np.mean(ev.ap[:, col]))
        filas.append(fila)
    taboa = pd.DataFrame(filas)
    resumo = {
        "rama": nome_rama,
        "clases_avaliadas": len(presentes),
        "clases": presentes,
        "limiares_tiou": LIMIARES.tolist(),
        "mAP_parcial_por_tiou": {
            f"{t:.1f}": float(taboa[f"AP@{t:.1f}"].mean()) for t in LIMIARES
        },
        "mAP_parcial_medio": float(taboa["AP_medio"].mean()),
        "aviso": (
            "mAP promediado SO sobre as clases listadas. Cada AP por clase e "
            "identico ao da corrida completa; o promedio non o e."
        ),
    }
    return taboa, resumo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--actionformer-root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    # Permite avaliar un directorio alternativo de predicions (por exemplo o
    # reordenado co prior de duracion) sen tocar a estrutura oficial.
    ap.add_argument("--ruta-predicions", default=None,
                    help="ruta relativa a out_root cun directorio por clase")
    args = ap.parse_args()

    plan = load_plan(Path(args.plan))
    out_root = Path(plan["paths"]["out_root"])
    anotacions = Path(plan["inputs"]["canonical_annotations"]["path"])
    out_dir = Path(args.out_dir) if args.out_dir else out_root / "evaluation" / "parcial"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos_test, label_to_id, _ = canonical_test_protocol(anotacions)
    print(f"protocolo canonico: {len(videos_test)} videos de test, {len(label_to_id)} clases\n")

    ramas = dict(RAMAS)
    if args.ruta_predicions:
        ramas = {"reTAG-reordenado": args.ruta_predicions}

    resumos = {}
    for nome_rama, patron in ramas.items():
        raiz = out_root / patron.format(seed=args.seed)
        r = avaliar_rama(
            nome_rama, raiz, anotacions, videos_test, label_to_id,
            Path(args.actionformer_root), args.num_workers,
        )
        if r is None:
            print(f"{nome_rama}: ainda non hai predicions en {raiz}")
            continue
        taboa, resumo = r
        taboa.to_csv(out_dir / f"ap_por_clase_{nome_rama}.csv", index=False)
        resumos[nome_rama] = resumo
        print(f"== {nome_rama} · {resumo['clases_avaliadas']} clases ==")
        print(taboa.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"  mAP parcial medio: {resumo['mAP_parcial_medio']:.4f}\n")

    if len(resumos) == 2 and not args.ruta_predicions:
        comuns = sorted(set(resumos["reTAG"]["clases"]) & set(resumos["CoTAD"]["clases"]))
        print(f"== comparacion sobre as {len(comuns)} clases comuns ==")
        for nome in ("reTAG", "CoTAD"):
            t = pd.read_csv(out_dir / f"ap_por_clase_{nome}.csv")
            t = t[t["clase"].isin(comuns)]
            print(f"  {nome:6s} mAP medio {t['AP_medio'].mean():.4f}")

    (out_dir / "resumo_parcial.json").write_text(
        json.dumps(resumos, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nescrito en {out_dir}")


if __name__ == "__main__":
    main()
