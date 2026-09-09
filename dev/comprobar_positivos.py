"""Conta, por clase, cantas ANOTACIONS caen dentro do subconxunto escollido.

Fai falla porque escoller gravacions que "conteñan" unha clase non garante que
haxa propostas positivas para adestrar a cabeza binaria de reTAG: Billiards
fallou con counts=[2907, 0], cero positivos.

Non replica o etiquetado exacto do adestramento (que depende do solapamento
proposta-anotacion), pero o numero de instancias anotadas no subconxunto e a
cota superior: se hai poucas, non pode haber moitos positivos.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.run_thumos14e_supervised import load_plan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--minimo", type=int, default=10)
    args = ap.parse_args()

    plan = load_plan(Path(args.plan))
    out_root = Path(plan["paths"]["out_root"])
    manifest = pd.read_csv(plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False)
    sub = json.loads((out_root / "subconxunto.json").read_text(encoding="utf-8"))

    for split in ("validation", "test"):
        elixidas = set(sub["splits"][split]["gravacions"])
        pool = manifest[manifest["official_subset"] == split]
        conta_sub: Counter = Counter()
        conta_total: Counter = Counter()
        for _, fila in pool.iterrows():
            etiquetas = json.loads(fila["labels_json"]) if fila["labels_json"] else []
            anot = json.loads(fila["annotations_json"]) if fila.get("annotations_json") else []
            # numero de instancias por clase nesta gravacion
            por_clase: Counter = Counter()
            for item in anot if isinstance(anot, list) else []:
                lab = item.get("label") if isinstance(item, dict) else None
                if lab:
                    por_clase[lab] += 1
            if not por_clase:
                for e in etiquetas:
                    por_clase[e] += 1
            for c, n in por_clase.items():
                conta_total[c] += n
                if fila["video_id"] in elixidas:
                    conta_sub[c] += n

        print(f"\n== {split} ==")
        febles = []
        for c in sorted(conta_total):
            n_sub = conta_sub.get(c, 0)
            marca = ""
            if n_sub < args.minimo:
                marca = "  <-- POUCAS"
                febles.append(c)
            print(f"  {c:22s} {n_sub:5d} de {conta_total[c]:5d}{marca}")
        print(f"  clases por debaixo de {args.minimo}: {len(febles)}")
        if febles:
            print("  ", ", ".join(febles))


if __name__ == "__main__":
    main()
