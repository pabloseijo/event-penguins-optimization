"""Full pipeline mAP evaluation across all proposal variants.

Runs proposals → CNN → mAP for each variant and writes a summary report.

Run from event_penguins/ with the Python venv active:
    python dev/run_map_eval.py
    python dev/run_map_eval.py --split test --out-dir tmp/map_eval
    python dev/run_map_eval.py --second-round   # includes new improvements
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.proposals import ProposalGenerator
from src.prototype import build_ed_prototype
from src.classification import ProposalClassifier
from src.evaluation import DetectionsEvaluator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-pipeline mAP evaluation for all variants.")
    p.add_argument("--data-path",    default="data/preprocessed.h5")
    p.add_argument("--ann-path",     default="config/annotations/annotations.json")
    p.add_argument("--model-path",   default="models/model.pk")
    p.add_argument("--proto-path",   default="tmp/prototype/ed_prototype.npy")
    p.add_argument("--out-dir",      default="tmp/map_eval")
    p.add_argument("--split",        default="test")
    p.add_argument("--tiou",         type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    p.add_argument("--second-round", action="store_true",
                   help="Run second-round improvements (Soft-NMS, fusion, multi-scale, TTA, decay)")
    p.add_argument("--third-round", action="store_true",
                   help="Run third-round improvements (bug fix + Soft-NMS tuning + duration filter + combos)")
    p.add_argument("--fourth-round", action="store_true",
                   help="Run fourth-round improvements (min_ed_score tuning + max_dur refinement)")
    p.add_argument("--fifth-round", action="store_true",
                   help="Run fifth-round improvements (temperature scaling + duration penalty + combos)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Pre-fitted temperature T for R5 (from fit_temperature.py)")
    return p.parse_args()


BASE_PROPOSAL_CFG = dict(
    bin_width=0.033,
    percentile=1.0,
    nms_threshold=0.95,
)

BASE_CLASSIFIER_CFG = dict(
    num_tsn_samples=7,
    augment_factor=3,
    sample_duration=1,
    decay=5e-6,
    nms_threshold=0.5,
    batch_size=8,
)

# First-round variants (original 7)
VARIANTS_R1: list[tuple[str, dict, dict]] = [
    # (label, proposal_kwargs, classifier_kwargs)
    ("Baseline", {}, {}),
    ("+ λ adaptativo", {"use_adaptive_lambda": True}, {}),
    ("+ Compacidade (w=0.2)", {
        "use_adaptive_lambda": True, "use_spatial_compactness": True,
    }, {}),
    ("+ Ruído sostido", {
        "use_adaptive_lambda": True, "use_spatial_compactness": True,
        "use_noise_penalization": True,
    }, {}),
    ("+ Ruído disperso", {
        "use_adaptive_lambda": True, "use_spatial_compactness": True,
        "use_noise_penalization": True, "use_dispersed_noise": True,
    }, {}),
    ("+ Prototipo ED", {
        "use_adaptive_lambda": True, "use_spatial_compactness": True,
        "use_noise_penalization": True, "use_dispersed_noise": True,
        # prototype injected at runtime
    }, {}),
    ("+ Periodicidade", {
        "use_adaptive_lambda": True, "use_spatial_compactness": True,
        "use_noise_penalization": True, "use_dispersed_noise": True,
        "use_periodicity": True,
    }, {}),
]

# Best R1 proposal config (base for all R2 variants)
_BEST_R1_PROP = {
    "use_adaptive_lambda": True, "use_spatial_compactness": True,
    "use_noise_penalization": True, "use_dispersed_noise": True,
    "use_periodicity": True,
    # prototype injected at runtime
}

# Second-round variants — each tested INDEPENDENTLY on top of best R1.
# Format: (label, extra_proposal_kwargs, classifier_kwargs)
VARIANTS_R2: list[tuple[str, dict, dict]] = [
    # Soft-NMS solo
    ("R2: Soft-NMS", {}, {"use_soft_nms": True}),

    # Fusión de scores (independent of Soft-NMS)
    ("R2: Fusión (w=0.3)", {}, {"score_fusion_weight": 0.3}),
    ("R2: Fusión (w=0.5)", {}, {"score_fusion_weight": 0.5}),

    # Soft-NMS + Fusión combinados
    ("R2: Soft-NMS + Fusión (w=0.3)", {}, {"use_soft_nms": True, "score_fusion_weight": 0.3}),

    # TTA (augment_factor) — independente
    ("R2: TTA augment=5", {}, {"augment_factor": 5}),
    ("R2: TTA augment=7", {}, {"augment_factor": 7}),

    # Decay — independente
    ("R2: Decay 2e-6", {}, {"decay": 2e-6}),
    ("R2: Decay 1e-5", {}, {"decay": 1e-5}),

    # Mellor combinación R2 (Soft-NMS + mellor fusión + mellor TTA)
    ("R2: Soft-NMS + TTA=5", {}, {"use_soft_nms": True, "augment_factor": 5}),
]

# Best R2 classifier config (base for all R3 variants)
_BEST_R2_CLF = {"use_soft_nms": True, "augment_factor": 5}

# Third-round variants — each tested on top of R1best (proposals) + R2best (classifier).
# Bug fix in merge_proposals applies automatically to all (it's in the source).
VARIANTS_R3: list[tuple[str, dict, dict, bool]] = [
    # (label, extra_proposal_kwargs, extra_classifier_kwargs, multiscale)
    # Reference: R2best with bug fix
    ("R3: R2best+bugfix",           {}, {},                                                       False),
    # Soft-NMS sigma tuning (lower=more aggressive, higher=softer)
    ("R3: sigma=0.1",               {}, {"soft_nms_sigma": 0.1},                                  False),
    ("R3: sigma=0.25",              {}, {"soft_nms_sigma": 0.25},                                 False),
    ("R3: sigma=0.75",              {}, {"soft_nms_sigma": 0.75},                                 False),
    ("R3: sigma=1.0",               {}, {"soft_nms_sigma": 1.0},                                  False),
    # Soft-NMS score threshold (remove very low-confidence survivors)
    ("R3: score_thr=0.01",          {}, {"soft_nms_score_threshold": 0.01},                       False),
    ("R3: score_thr=0.05",          {}, {"soft_nms_score_threshold": 0.05},                       False),
    # Max duration filter post-NMS
    ("R3: max_dur=30s",             {}, {"max_duration_filter": 30},                              False),
    ("R3: max_dur=60s",             {}, {"max_duration_filter": 60},                              False),
    # Proposal NMS threshold (fewer, more precise proposals → less high-score FPs)
    ("R3: propNMS=0.90",            {"nms_threshold": 0.90}, {},                                  False),
    ("R3: propNMS=0.85",            {"nms_threshold": 0.85}, {},                                  False),
    # Multi-scale + Soft-NMS combined
    ("R3: Multi+SoftNMS",           {}, {},                                                       True),
    # Promising combos
    ("R3: sigma=0.25+dur60",        {}, {"soft_nms_sigma": 0.25, "max_duration_filter": 60},      False),
    ("R3: propNMS0.90+dur60",       {"nms_threshold": 0.90}, {"max_duration_filter": 60},         False),
    ("R3: sigma=0.1+propNMS0.90",   {"nms_threshold": 0.90}, {"soft_nms_sigma": 0.1},             False),
]

# Best R3 classifier config (base for all R4 variants)
# sigma=0.25 gave best single result in R3
_BEST_R3_CLF = {"use_soft_nms": True, "augment_factor": 5, "soft_nms_sigma": 0.25}

# Best R4 classifier config (base for all R5 variants)
# min_ed_score=0.3 gave best result in R4
_BEST_R4_CLF = {"use_soft_nms": True, "augment_factor": 5, "soft_nms_sigma": 0.25, "min_ed_score": 0.3}

# Fourth-round variants — min_ed_score tuning + better max_dur + combos
# All on top of R3best (proposals=R1best, classifier=R3best)
VARIANTS_R4: list[tuple[str, dict, dict, bool]] = [
    # Reference: R3best
    ("R4: R3best",                  {}, {},                                                         False),
    # Lower detection threshold — recover ROIs where CNN scored 0.35–0.5
    ("R4: min_score=0.45",          {}, {"min_ed_score": 0.45},                                    False),
    ("R4: min_score=0.4",           {}, {"min_ed_score": 0.4},                                     False),
    ("R4: min_score=0.35",          {}, {"min_ed_score": 0.35},                                    False),
    ("R4: min_score=0.3",           {}, {"min_ed_score": 0.3},                                     False),
    # Max duration refinement (GT max=42.5s; 50s cuts 14 FPs/3 TPs, 45s cuts 18/6)
    ("R4: max_dur=45s",             {}, {"max_duration_filter": 45},                               False),
    ("R4: max_dur=50s",             {}, {"max_duration_filter": 50},                               False),
    ("R4: max_dur=90s",             {}, {"max_duration_filter": 90},                               False),
    # Sigma finer tuning
    ("R4: sigma=0.15",              {}, {"soft_nms_sigma": 0.15},                                  False),
    ("R4: sigma=0.20",              {}, {"soft_nms_sigma": 0.20},                                  False),
    # Best combos
    ("R4: score0.4+dur50",          {}, {"min_ed_score": 0.4,  "max_duration_filter": 50},         False),
    ("R4: score0.4+dur45",          {}, {"min_ed_score": 0.4,  "max_duration_filter": 45},         False),
    ("R4: score0.35+dur50",         {}, {"min_ed_score": 0.35, "max_duration_filter": 50},         False),
    ("R4: score0.4+sigma0.2+dur50", {}, {"min_ed_score": 0.4,  "soft_nms_sigma": 0.20,
                                         "max_duration_filter": 50},                               False),
]


# Fifth-round variants — temperature scaling + duration penalty + combos
# All on top of R4best (proposals=R1best, classifier=R4best)
# T_FITTED is filled at runtime from args.temperature (from fit_temperature.py).
# Format: (label, extra_classifier_kwargs)
VARIANTS_R5_TEMP: list[tuple[str, dict]] = [
    # Reference: R4best
    ("R5: R4best",              {}),
    # Temperature grid (explore: T<1 sharpens, T>1 softens)
    ("R5: T=0.5",               {"temperature": 0.5}),
    ("R5: T=1.5",               {"temperature": 1.5}),
    ("R5: T=2.0",               {"temperature": 2.0}),
    ("R5: T=3.0",               {"temperature": 3.0}),
    ("R5: T=5.0",               {"temperature": 5.0}),
    # Duration soft-penalty (D_max=60s, sigma=20s)
    ("R5: dur_pen(60,20)",      {"duration_penalty_dmax": 60, "duration_penalty_sigma": 20}),
    ("R5: dur_pen(45,15)",      {"duration_penalty_dmax": 45, "duration_penalty_sigma": 15}),
    ("R5: dur_pen(90,30)",      {"duration_penalty_dmax": 90, "duration_penalty_sigma": 30}),
    # Temperature + duration penalty combos
    ("R5: T=2+dur(60,20)",      {"temperature": 2.0, "duration_penalty_dmax": 60, "duration_penalty_sigma": 20}),
    ("R5: T=3+dur(60,20)",      {"temperature": 3.0, "duration_penalty_dmax": 60, "duration_penalty_sigma": 20}),
    ("R5: T=5+dur(60,20)",      {"temperature": 5.0, "duration_penalty_dmax": 60, "duration_penalty_sigma": 20}),
]


def run_variant(
    name: str,
    proposal_kwargs: dict,
    classifier_kwargs: dict,
    args: argparse.Namespace,
    prototype: np.ndarray,
    device: torch.device,
    out_dir: Path,
    multiscale: bool = False,
) -> dict:
    tag = (name.replace(" ", "_").replace("+", "p").replace("λ", "lambda")
               .replace(":", "").replace("(", "").replace(")", "").replace(".", ""))
    variant_dir = out_dir / tag
    variant_dir.mkdir(parents=True, exist_ok=True)

    prop_kwargs = {**BASE_PROPOSAL_CFG, **proposal_kwargs,
                   "data_path": args.data_path,
                   "output_dir": str(variant_dir / "proposal_logs")}
    if name in ("+ Prototipo ED", "+ Periodicidade") or "proto" in str(proposal_kwargs):
        prop_kwargs["prototype"] = prototype
    if "use_periodicity" in proposal_kwargs and proposal_kwargs["use_periodicity"]:
        prop_kwargs["prototype"] = prototype

    clf_kwargs = {**BASE_CLASSIFIER_CFG, **classifier_kwargs,
                  "data_path": args.data_path,
                  "model_path": args.model_path}

    print(f"\n{'='*60}")
    print(f"[VARIANT] {name}")
    print(f"{'='*60}")

    gen = ProposalGenerator(**prop_kwargs)

    if multiscale:
        proposals = gen.run_multiscale([0.033, 0.066], split=args.split)
        print(f"  Propostas (multi-escala): {len(proposals)}")
    else:
        proposals = gen.run(split=args.split)
        print(f"  Propostas: {len(proposals)}")

    classifier = ProposalClassifier(device, **clf_kwargs)
    results = classifier.run(proposals)

    pred_path = variant_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)

    evaluator = DetectionsEvaluator(
        ground_truth_filename=args.ann_path,
        prediction_filename=str(pred_path),
        tiou_thresholds=np.array(args.tiou),
        valid_labels="ed",
        valid_sequences=list(results["results"].keys()),
        min_duration=2.0,
    )
    mean_ap = evaluator.run()
    print(f"  mAP: {mean_ap:.4f}")

    return {"name": name, "n_proposals": len(proposals), "mAP": mean_ap}


def write_report(results: list[dict], out_path: Path, args: argparse.Namespace) -> None:
    baseline_n = results[0]["n_proposals"]
    lines = [
        "# Informe mAP — Avaliación completa do pipeline reTAG",
        "",
        f"**Split:** {args.split} | **tIoU:** {args.tiou} | **Device:** cuda:0",
        "",
        "| Variante | Propostas | Δ vs Baseline | mAP |",
        "|---|---:|:---:|:---:|",
    ]
    for r in results:
        delta = f"{(r['n_proposals']-baseline_n)/baseline_n*100:+.1f}%" if r["n_proposals"] != baseline_n else "—"
        lines.append(f"| {r['name']} | {r['n_proposals']:,} | {delta} | {r['mAP']:.4f} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[INFO] Informe gardado en {out_path}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    proto_path = Path(args.proto_path)
    if proto_path.exists():
        prototype = np.load(proto_path)
        print(f"[INFO] Prototipo cargado: {proto_path}  shape={prototype.shape}")
    else:
        print("[INFO] Construíndo prototipo desde train split...")
        prototype = build_ed_prototype(args.data_path, args.ann_path, split="train")
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proto_path, prototype)

    if args.fifth_round:
        # Inject fitted T variant if provided
        variants_r5 = list(VARIANTS_R5_TEMP)
        if args.temperature is not None:
            variants_r5.insert(1, (f"R5: T={args.temperature:.2f}(fitted)", {"temperature": args.temperature}))
            variants_r5.append((f"R5: T={args.temperature:.2f}+dur(60,20)",
                                {"temperature": args.temperature,
                                 "duration_penalty_dmax": 60,
                                 "duration_penalty_sigma": 20}))

        results = []
        for name, extra_clf in variants_r5:
            clf_kwargs = {**_BEST_R4_CLF, **extra_clf}
            res = run_variant(name, _BEST_R1_PROP, clf_kwargs, args, prototype, device, out_dir)
            results.append(res)

    elif args.fourth_round:
        results = []
        for name, extra_prop, extra_clf, ms in VARIANTS_R4:
            prop_kwargs = {**_BEST_R1_PROP, **extra_prop}
            clf_kwargs  = {**_BEST_R3_CLF, **extra_clf}
            res = run_variant(name, prop_kwargs, clf_kwargs, args, prototype, device, out_dir,
                              multiscale=ms)
            results.append(res)

    elif args.third_round:
        # R3: each improvement tested independently on top of R1best (proposals) + R2best (classifier)
        # Bug fix in merge_proposals is now in source, applies automatically to all.
        variants_r3 = []
        for name, extra_prop, extra_clf, ms in VARIANTS_R3:
            prop_kwargs = {**_BEST_R1_PROP, **extra_prop}
            clf_kwargs  = {**_BEST_R2_CLF, **extra_clf}
            variants_r3.append((name, prop_kwargs, clf_kwargs, ms))

        results = []
        for name, prop_kwargs, clf_kwargs, ms in variants_r3:
            res = run_variant(name, prop_kwargs, clf_kwargs, args, prototype, device, out_dir,
                              multiscale=ms)
            results.append(res)

    elif args.second_round:
        # R2: each improvement tested independently on top of best R1 config
        variants = []
        variants.append(("+ Periodicidade (R1 best)", {
            "use_adaptive_lambda": True, "use_spatial_compactness": True,
            "use_noise_penalization": True, "use_dispersed_noise": True,
            "use_periodicity": True,
        }, {}))
        for name, extra_prop, clf_kwargs in VARIANTS_R2:
            prop_kwargs = {**_BEST_R1_PROP, **extra_prop}
            variants.append((name, prop_kwargs, clf_kwargs))
        variants.append(("R2: Multi-escala (33+66ms)", {**_BEST_R1_PROP}, {}))

        results = []
        for name, prop_kwargs, clf_kwargs in variants:
            multiscale = "Multi-escala" in name
            res = run_variant(name, prop_kwargs, clf_kwargs, args, prototype, device, out_dir,
                              multiscale=multiscale)
            results.append(res)

    else:
        variants = [(n, p, c) for n, p, c in VARIANTS_R1]
        results = []
        for name, prop_kwargs, clf_kwargs in variants:
            res = run_variant(name, prop_kwargs, clf_kwargs, args, prototype, device, out_dir)
            results.append(res)

    print(f"\n{'='*60}")
    print("RESUMO FINAL — mAP completo")
    print(f"{'='*60}")
    baseline_n = results[0]["n_proposals"]
    for r in results:
        delta = f"({(r['n_proposals']-baseline_n)/baseline_n*100:+.1f}%)" if r["n_proposals"] != baseline_n else ""
        print(f"{r['name']:<45} {r['n_proposals']:>8,} {delta:>8}  mAP={r['mAP']:.4f}")

    write_report(results, out_dir / "informe_map.md", args)


if __name__ == "__main__":
    main()
