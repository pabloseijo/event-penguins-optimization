"""Comproba que build_ed_prototype por busca binaria da o MESMO prototipo.

Os 19 prototipos que xa hai en disco construironse coa version vella (a que
cargaba a gravacion enteira con np.array) sobre ESTE mesmo corpus. Polo tanto,
reconstruilos coa version nova e comparalos bit a bit e unha proba de
regresion directa: se coinciden, o cambio non altera resultados.

Uso: probar_identidade_prototipo.py <plan> <Clase> [n_folds]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.run_thumos14e_supervised import (  # noqa: E402
    annotation_config_dir,
    fold_recordings,
    load_plan,
    target_prototype_path,
    target_recordings,
)


def cargar_novo(ruta: Path):
    """Importa a version nova sen tocar src/prototype.py."""
    spec = importlib.util.spec_from_file_location("prototipo_novo", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prototipo_novo"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    plan_path, label = sys.argv[1], sys.argv[2]
    n_folds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    novo = cargar_novo(ROOT / "dev" / "prototype_novo.py")

    plan = load_plan(Path(plan_path))
    work_dir = Path(plan["paths"]["work_dir"])
    out_root = Path(plan["paths"]["out_root"])
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    manifest = pd.read_csv(plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False)
    folds = pd.read_csv(plan["inputs"]["fold_manifest"]["path"], keep_default_na=False)
    validation = set(target_recordings(manifest, "validation"))

    ann_path = (
        annotation_config_dir(work_dir) / "by_class" / label / "annotations_trainable.json"
    )

    pools = [(f, fold_recordings(folds, f, "train")) for f in range(n_folds)]
    pools.append((None, validation))

    fallos = 0
    for fold, recordings in pools:
        gardado = target_prototype_path(out_root, label, fold)
        if not gardado.exists():
            print(f"  fold={fold}: non hai prototipo gardado, sáltase")
            continue
        vello = np.load(gardado)
        calculado = novo.build_ed_prototype(
            str(data_path),
            str(ann_path),
            split="train",
            min_duration=0.0,
            recordings=recordings,
        )
        igual = np.array_equal(vello, calculado)
        dif = float(np.abs(vello - calculado).max()) if vello.shape == calculado.shape else -1
        print(f"  fold={str(fold):>4}  {'IDENTICO' if igual else 'DIFIRE'}"
              f"  (max |dif| {dif:.3e}, norma {np.linalg.norm(calculado):.6f})", flush=True)
        if not igual:
            fallos += 1

    print()
    if fallos:
        print(f"FALLO: {fallos} prototipos difiren. NON despregar.")
        sys.exit(1)
    print("TODOS OS PROTOTIPOS COINCIDEN. Seguro despregar.")


if __name__ == "__main__":
    main()
