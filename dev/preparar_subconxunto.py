"""Prepara un subconxunto DECLARADO de gravacions para avaliar as duas ramas.

Non e unha optimizacion: e unha reducion de alcance, e como tal ten que quedar
escrita. Escolle gravacions de test e de validation cubrindo as 20 clases, e
filtra as propostas de reTAG a esas gravacions para que a extraccion de
features so procese o que se vai avaliar.

O criterio de escolla:
  - cobertura: todas as clases teñen que aparecer no subconxunto de test, se
    non o AP desa clase non se pode calcular
  - as gravacions mais PEQUENAS primeiro dentro de cada clase, porque o custo
    da extraccion e proporcional aos eventos e o obxectivo e que caiba no
    calendario. Isto e un nesgo declarado: os videos curtos poden ser mais
    faciles, e hai que dicilo no artigo, non agochalo.
  - o mesmo criterio nas duas ramas, para que a comparacion sexa xusta.

Escribe:
  subconxunto.json          a lista, con motivo e cifras
  proposals_sub.csv         as propostas de reTAG filtradas, por split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dev.run_thumos14e_supervised import load_plan

    plan = load_plan(Path(args.plan))
    out_root = Path(plan["paths"]["out_root"])
    data_path = Path(plan["inputs"]["event_hdf5"]["path"])
    manifest = pd.read_csv(plan["inputs"]["corpus_manifest"]["path"], keep_default_na=False)

    with h5py.File(data_path, "r") as f:
        tam = {r: int(f[r]["N01"]["events"].shape[0]) for r in f.keys()}

    manifest["eventos"] = manifest["video_id"].map(tam)
    manifest["clases"] = manifest["labels_json"].apply(
        lambda s: json.loads(s) if s else []
    )

    escollidas = {}
    for subset, n in (("test", args.n_test), ("validation", args.n_val)):
        pool = manifest[manifest["official_subset"] == subset].copy()
        pool = pool.sort_values("eventos")
        elixidas: list[str] = []
        cubertas: set[str] = set()
        # primeira volta: a gravacion mais pequena de cada clase ainda sen cubrir
        for _, fila in pool.iterrows():
            novas = set(fila["clases"]) - cubertas
            if novas:
                elixidas.append(fila["video_id"])
                cubertas |= set(fila["clases"])
        # segunda volta: completar ata n coas mais pequenas que falten
        for _, fila in pool.iterrows():
            if len(elixidas) >= n:
                break
            if fila["video_id"] not in elixidas:
                elixidas.append(fila["video_id"])
        sel = pool[pool["video_id"].isin(elixidas)]
        escollidas[subset] = {
            "gravacions": sorted(elixidas),
            "n": len(elixidas),
            "de": int(len(pool)),
            "clases_cubertas": sorted(cubertas),
            "eventos": int(sel["eventos"].sum()),
            "eventos_totais_do_split": int(pool["eventos"].sum()),
        }
        frac = sel["eventos"].sum() / pool["eventos"].sum()
        print(f"{subset}: {len(elixidas)} de {len(pool)} gravacions · "
              f"{len(cubertas)}/20 clases · {frac*100:.1f}% dos eventos")

    # filtrar as propostas de reTAG
    for subset in ("validation", "test"):
        p_in = out_root / "proposals" / "retag" / subset / "proposals.csv"
        if not p_in.exists():
            print(f"  aviso: non hai propostas para {subset}")
            continue
        p = pd.read_csv(p_in)
        sub = p[p["rec_name"].isin(escollidas[subset]["gravacions"])]
        p_out = p_in.parent / "proposals_sub.csv"
        sub.to_csv(p_out, index=False)
        escollidas[subset]["propostas"] = int(len(sub))
        escollidas[subset]["propostas_totais"] = int(len(p))
        print(f"  {subset}: {len(sub):,} de {len(p):,} propostas "
              f"({len(sub)/len(p)*100:.1f}%) -> {p_out.name}")

    destino = Path(args.saida) if args.saida else out_root / "subconxunto.json"
    destino.write_text(
        json.dumps(
            {
                "motivo": (
                    "Reducion de alcance declarada para que as DUAS ramas se poidan "
                    "avaliar sobre os mesmos videos dentro do calendario. A extraccion "
                    "de features de reTAG sobre as 200 gravacions de validation "
                    "estimabase en ~50-100 h por particion."
                ),
                "criterio": (
                    "cobertura das 20 clases primeiro; despois as gravacions mais "
                    "pequenas. NESGO DECLARADO: favorece videos curtos."
                ),
                "splits": escollidas,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nescrito {destino}")


if __name__ == "__main__":
    main()
