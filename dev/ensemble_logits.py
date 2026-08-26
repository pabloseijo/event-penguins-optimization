"""Build a deterministic ensemble from aligned two-class logit caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average aligned ATSN logit caches.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=["logit_mean", "probability_mean"],
        default="logit_mean",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    inputs = [Path(path) for path in args.inputs]
    arrays = [np.load(path, allow_pickle=False)["logits"].astype(np.float64) for path in inputs]
    reference_shape = arrays[0].shape
    if len(reference_shape) != 2 or reference_shape[1] != 2:
        raise ValueError(f"Expected two-class logits, got shape={reference_shape}")
    for path, array in zip(inputs[1:], arrays[1:]):
        if array.shape != reference_shape:
            raise ValueError(f"{path} shape={array.shape}, expected={reference_shape}")

    if args.mode == "logit_mean":
        ensemble = np.mean(arrays, axis=0)
    else:
        probabilities = np.mean([softmax(array, args.temperature) for array in arrays], axis=0)
        ensemble = np.log(np.clip(probabilities, 1e-12, 1.0)) * args.temperature

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": args.mode,
        "temperature": args.temperature,
        "inputs": [str(path) for path in inputs],
        "shape": list(reference_shape),
    }
    np.savez_compressed(
        out,
        logits=ensemble.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
