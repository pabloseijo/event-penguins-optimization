#!/usr/bin/env python3
"""Run ActionFormer inference in restartable chunks and evaluate once."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("infer", "evaluate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--actionformer-root", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--checkpoint", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)

    infer_parser = subparsers.choices["infer"]
    infer_parser.add_argument("--start", type=int, required=True)
    infer_parser.add_argument("--end", type=int, required=True)
    infer_parser.add_argument("--print-freq", type=int, default=20)
    return parser.parse_args()


def prepare_actionformer(args: argparse.Namespace):
    root = args.actionformer_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    os.chdir(root)

    from libs.core import load_config
    from libs.datasets import make_dataset

    cfg = load_config(str(config_path))
    dataset = make_dataset(
        cfg["dataset_name"],
        False,
        cfg["val_split"],
        **cfg["dataset"],
    )
    return cfg, dataset, checkpoint_path, output_dir


def checkpoint_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def infer_chunk(args: argparse.Namespace) -> None:
    cfg, dataset, checkpoint_path, output_dir = prepare_actionformer(args)
    if not 0 <= args.start < args.end <= len(dataset):
        raise ValueError(
            f"Invalid chunk [{args.start}, {args.end}) for {len(dataset)} videos"
        )

    from libs.datasets import make_data_loader
    from libs.modeling import make_meta_arch
    from libs.utils import fix_random_seed, valid_one_epoch

    _ = fix_random_seed(0, include_cuda=True)
    subset = torch.utils.data.Subset(dataset, range(args.start, args.end))
    loader = make_data_loader(
        subset,
        False,
        None,
        1,
        cfg["loader"]["num_workers"],
    )
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = torch.nn.DataParallel(model, device_ids=cfg["devices"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg["devices"][0],
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict_ema"])

    chunk_stem = f"chunk_{args.start:03d}_{args.end:03d}"
    output_file = output_dir / f"{chunk_stem}.pkl"
    valid_one_epoch(
        loader,
        model,
        -1,
        evaluator=None,
        output_file=str(output_file),
        ext_score_file=cfg["test_cfg"]["ext_score_file"],
        tb_writer=None,
        print_freq=args.print_freq,
    )
    metadata = {
        "start": args.start,
        "end": args.end,
        "dataset_length": len(dataset),
        "video_ids": [
            dataset.data_list[index]["id"]
            for index in range(args.start, args.end)
        ],
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": checkpoint_identity(checkpoint_path),
        "predictions": str(output_file),
    }
    (output_dir / f"{chunk_stem}.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


def load_chunks(
    output_dir: Path,
    expected_length: int,
    checkpoint: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata_files = sorted(glob.glob(str(output_dir / "chunk_*.json")))
    if not metadata_files:
        raise ValueError("No chunk metadata files found")

    metadata = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in metadata_files
    ]
    metadata.sort(key=lambda item: item["start"])
    cursor = 0
    merged = {
        "video-id": [],
        "t-start": [],
        "t-end": [],
        "label": [],
        "score": [],
    }
    for chunk in metadata:
        if chunk["start"] != cursor:
            raise ValueError(
                f"Chunk coverage gap or overlap at {cursor}: {chunk}"
            )
        if chunk["dataset_length"] != expected_length:
            raise ValueError("Chunk dataset length does not match")
        if chunk["checkpoint"] != checkpoint:
            raise ValueError("Chunks were not produced by the same checkpoint")
        with Path(chunk["predictions"]).open("rb") as handle:
            predictions = pickle.load(handle)
        merged["video-id"].extend(predictions["video-id"])
        for key in ("t-start", "t-end", "label", "score"):
            merged[key].append(np.asarray(predictions[key]))
        cursor = chunk["end"]

    if cursor != expected_length:
        raise ValueError(
            f"Incomplete chunk coverage: ended at {cursor}, expected {expected_length}"
        )
    for key in ("t-start", "t-end", "label", "score"):
        merged[key] = np.concatenate(merged[key])
    return merged, metadata


def evaluate_chunks(args: argparse.Namespace) -> None:
    cfg, dataset, checkpoint_path, output_dir = prepare_actionformer(args)
    from libs.utils import ANETdetection

    checkpoint = checkpoint_identity(checkpoint_path)
    predictions, chunks = load_chunks(
        output_dir,
        len(dataset),
        checkpoint,
    )
    evaluator = ANETdetection(
        dataset.json_file,
        dataset.split[0],
        tiou_thresholds=dataset.get_attributes()["tiou_thresholds"],
    )
    map_by_tiou, average_map, recall = evaluator.evaluate(
        predictions,
        verbose=True,
    )

    label_by_id = {
        int(label_id): label for label, label_id in dataset.label_dict.items()
    }
    ap_by_class = {}
    for original_label_id, column_index in evaluator.activity_index.items():
        label = label_by_id[int(original_label_id)]
        ap_by_class[label] = evaluator.ap[:, column_index].tolist()

    metrics = {
        "checkpoint": checkpoint,
        "dataset_length": len(dataset),
        "chunk_ranges": [
            [chunk["start"], chunk["end"]] for chunk in chunks
        ],
        "number_predictions": len(predictions["score"]),
        "tiou_thresholds": evaluator.tiou_thresholds.tolist(),
        "map_by_tiou": map_by_tiou.tolist(),
        "average_map": float(average_map),
        "recall_shape": list(recall.shape),
        "ap_by_class": ap_by_class,
    }
    with (output_dir / "predictions.pkl").open("wb") as handle:
        pickle.dump(predictions, handle)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "infer":
        infer_chunk(args)
    else:
        evaluate_chunks(args)


if __name__ == "__main__":
    main()
