#!/usr/bin/env python3
"""Comparación estatística entre dúas ramas sobre os mesmos folds.

Implementa o *corrected resampled t-test* de Nadeau e Bengio (2003). Os folds
comparten datos de adestramento, así que non son independentes: o t-test pareado
estándar subestima a varianza e dispara o erro de tipo I. A corrección engade o
termo `n_test/n_train` á varianza.

    t = mean(d) / sqrt( (1/k + n_test/n_train) * var(d) )

Uso:
    python dev/compare_folds.py --a resultados_retag.json --b resultados_noso.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def corrected_t(diffs: list[float], n_train: int, n_test: int) -> tuple[float, float, float]:
    """Devolve (t, gl, varianza corrixida) para diferenzas pareadas por fold."""
    k = len(diffs)
    media = sum(diffs) / k
    var = sum((d - media) ** 2 for d in diffs) / (k - 1) if k > 1 else 0.0
    correccion = 1.0 / k + n_test / n_train
    var_corr = correccion * var
    t = media / math.sqrt(var_corr) if var_corr > 0 else 0.0
    return t, k - 1, var_corr


def p_bilateral(t: float, gl: int) -> float:
    """p bilateral da t de Student, por integración numérica da densidade."""
    if gl <= 0:
        return float("nan")
    t = abs(t)
    # densidade da t
    c = math.gamma((gl + 1) / 2) / (math.sqrt(gl * math.pi) * math.gamma(gl / 2))
    f = lambda x: c * (1 + x * x / gl) ** (-(gl + 1) / 2)
    # Simpson ata t, e a cola é 0.5 - integral
    n = 20000
    h = t / n
    s = f(0) + f(t)
    for i in range(1, n):
        s += f(i * h) * (4 if i % 2 else 2)
    return max(0.0, min(1.0, 2 * (0.5 - s * h / 3)))


def cargar(ruta: Path) -> dict[str, float]:
    d = json.loads(ruta.read_text())
    return {str(k): float(v) for k, v in (d.get("por_fold") or d).items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="JSON da rama A (ex. reTAG)")
    ap.add_argument("--b", required=True, help="JSON da rama B (ex. sistema completo)")
    ap.add_argument("--nome-a", default="reTAG")
    ap.add_argument("--nome-b", default="sistema completo")
    ap.add_argument("--n-train", type=int, default=18, help="gravacións de adestramento por fold")
    ap.add_argument("--n-test", type=int, default=1, help="gravacións de avaliación por fold")
    args = ap.parse_args()

    a, b = cargar(Path(args.a)), cargar(Path(args.b))
    folds = sorted(set(a) & set(b), key=lambda x: str(x))
    if not folds:
        print("sen folds comúns entre os dous ficheiros")
        return 1

    diffs = [b[f] - a[f] for f in folds]
    ma = sum(a[f] for f in folds) / len(folds)
    mb = sum(b[f] for f in folds) / len(folds)

    print(f"{'fold':<10} {args.nome_a:>12} {args.nome_b:>18} {'dif':>10}")
    for f in folds:
        print(f"{f:<10} {a[f]:>12.4f} {b[f]:>18.4f} {b[f]-a[f]:>+10.4f}")
    print(f"{'media':<10} {ma:>12.4f} {mb:>18.4f} {mb-ma:>+10.4f}")
    print(f"{'peor':<10} {min(a[f] for f in folds):>12.4f} "
          f"{min(b[f] for f in folds):>18.4f}")

    t, gl, var_c = corrected_t(diffs, args.n_train, args.n_test)
    p = p_bilateral(t, gl)
    print(f"\ncorrected resampled t-test (Nadeau e Bengio, 2003)")
    print(f"  corrección  1/{len(folds)} + {args.n_test}/{args.n_train} = "
          f"{1/len(folds) + args.n_test/args.n_train:.4f}")
    print(f"  t = {t:.4f}  ·  gl = {gl}  ·  p = {p:.4f}")
    print(f"  {'diferenza significativa a 0,05' if p < 0.05 else 'NON significativa a 0,05'}")

    # o t-test inxenuo, só para ensinar canto esaxera
    k = len(diffs)
    media = sum(diffs) / k
    var = sum((d - media) ** 2 for d in diffs) / (k - 1)
    t_naive = media / math.sqrt(var / k) if var > 0 else 0.0
    print(f"\n  (t-test pareado sen corrixir: t = {t_naive:.4f}, "
          f"p = {p_bilateral(t_naive, gl):.4f} — infla a significancia)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
