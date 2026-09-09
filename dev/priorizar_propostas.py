"""Reordena as propostas de reTAG para que o subconxunto se extraia PRIMEIRO.

Non reduce nada: o ficheiro segue tendo as mesmas filas. So cambia a orde, de
xeito que a extraccion produce antes as gravacions do subconxunto declarado e
despois continua co resto. Asi hai cifras das duas ramas en horas, sen
renunciar ao protocolo completo nin tirar traballo.

Por que funciona: o extractor procesa as propostas en orde e garda o progreso
por indice (`completed`), asi que o que se extrae primeiro e o que estea
arriba. As features gardanse na mesma orde que o CSV, e o extractor escribe
unha copia de proposals.csv xunto delas, polo que a correspondencia mantense.

Garda o ficheiro orixinal como proposals_orixinal.csv.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.run_thumos14e_supervised import load_plan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--subconxunto", default=None)
    args = ap.parse_args()

    plan = load_plan(Path(args.plan))
    out_root = Path(plan["paths"]["out_root"])
    sub_path = Path(args.subconxunto) if args.subconxunto else out_root / "subconxunto.json"
    sub = json.loads(sub_path.read_text(encoding="utf-8"))

    for split in ("validation", "test"):
        p_path = out_root / "proposals" / "retag" / split / "proposals.csv"
        if not p_path.exists():
            print(f"{split}: non hai propostas, sáltase")
            continue
        orixinal = p_path.parent / "proposals_orixinal.csv"
        if not orixinal.exists():
            shutil.copy2(p_path, orixinal)

        p = pd.read_csv(orixinal)
        prioritarias = set(sub["splits"][split]["gravacions"])
        p["_pri"] = (~p["rec_name"].isin(prioritarias)).astype(int)
        # kind="stable" para non alterar a orde relativa dentro de cada grupo:
        # so se adianta o subconxunto, o resto queda exactamente como estaba
        p = p.sort_values("_pri", kind="stable").drop(columns="_pri")
        p.to_csv(p_path, index=False)

        n_pri = int((p["rec_name"].isin(prioritarias)).sum())
        print(f"{split}: {n_pri:,} propostas prioritarias de {len(p):,} "
              f"({n_pri/len(p)*100:.1f}%) postas ao principio")
        print(f"  orixinal gardado en {orixinal.name}")


if __name__ == "__main__":
    main()
