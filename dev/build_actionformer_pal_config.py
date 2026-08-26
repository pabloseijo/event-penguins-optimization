#!/usr/bin/env python3
"""Create a fixed PAL-consistency config from an OOF ActionFormer config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    config["dataset_name"] = "thumos_pal"
    config["model_name"] = "LocPointTransformerPAL"
    config.setdefault("dataset", {}).update(
        {
            "pal_probability": 0.5,
            "pal_margin": 2,
            "pal_blend_min": 1.0,
            "pal_blend_max": 1.0,
        }
    )
    config.setdefault("train_cfg", {}).update(
        {
            "pal_consistency_weight": 0.1,
            "pal_consistency_temperature": 0.07,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
