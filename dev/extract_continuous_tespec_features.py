"""Extract recurrent TESPEC features on the complete continuous ROI grid."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from dev.extract_tespec_features import stacked_histogram
from src.tespec_encoder import TespecEncoder


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser("index")
    index.add_argument("--base-feature-dir", required=True)
    index.add_argument("--out-dir", required=True)
    index.add_argument("--force", action="store_true")

    extract = subparsers.add_parser("extract")
    extract.add_argument("--out-dir", required=True)
    extract.add_argument("--checkpoint", default="tmp/pretrained/TESPEC_pretrained.ckpt")
    extract.add_argument("--data-path", default="data/preprocessed.h5")
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--sequence-batch-size", type=int, default=4)
    extract.add_argument("--image-size", type=int, default=224)
    extract.add_argument("--histogram-bins", type=int, default=10)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--smoke-steps", type=int, default=0)
    extract.add_argument(
        "--state-reset-steps",
        type=int,
        default=0,
        help="Reset ConvLSTM state at this fixed interval; 0 keeps full-history state.",
    )
    extract.add_argument(
        "--steps-per-process",
        type=int,
        default=0,
        help="Checkpoint recurrent state and exit 75 after this many grid steps.",
    )
    extract.add_argument("--overwrite-shard", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--out-dir", required=True)
    return parser.parse_args()


def build_index(args: argparse.Namespace) -> None:
    base_dir = resolve(args.base_feature_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "sequences.csv", out_dir / "metadata.json", out_dir / "frame_features.npy"]
    if any(path.exists() for path in paths) and not args.force:
        raise FileExistsError(f"TESPEC continuous cache already exists in {out_dir}")
    sequences = pd.read_csv(base_dir / "sequences.csv")
    metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    shutil.copyfile(base_dir / "sequences.csv", out_dir / "sequences.csv")
    output_metadata = {
        **metadata,
        "format_version": 1,
        "feature_dim": 768,
        "encoder": "TESPEC-pretrained-recurrent-continuous",
        "base_feature_dir": str(base_dir),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2), encoding="utf-8"
    )
    matrix = np.lib.format.open_memmap(
        out_dir / "frame_features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(int(metadata["num_points"]), 768),
    )
    matrix[:] = np.nan
    matrix.flush()
    if int(sequences.iloc[-1].offset + sequences.iloc[-1].length) != len(matrix):
        raise ValueError("Base sequence index does not cover the TESPEC matrix")
    print(f"[INDEX] sequences={len(sequences)} shape={matrix.shape}")


def roi_data(handle: h5py.File, rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    values = []
    for row in rows.itertuples(index=False):
        group = handle[row.rec_name][row.roi_key]
        events = np.asarray(group["events"])
        values.append(
            (
                events,
                events[:, 2],
                int(group.attrs["height"]),
                int(group.attrs["width"]),
            )
        )
    return values


def extract_shard(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")
    if args.histogram_bins != 10 or args.image_size != 224:
        raise ValueError("The official TESPEC checkpoint requires 10 bins and 224x224 inputs")
    out_dir = resolve(args.out_dir)
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    sequences = pd.read_csv(out_dir / "sequences.csv")
    sequences = sequences[sequences.sequence_index % args.num_shards == args.shard_index]
    status_path = out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    state_path = out_dir / f"recurrent_state_{args.shard_index:02d}_of_{args.num_shards:02d}.pt"
    if status_path.exists() and not args.overwrite_shard and args.smoke_steps == 0:
        if json.loads(status_path.read_text()).get("complete"):
            print(f"[SKIP] shard {args.shard_index} already complete")
            return
    device = torch.device(args.device)
    model = TespecEncoder(args.image_size)
    model.load_pretrained(resolve(args.checkpoint))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matrix = None
    if args.smoke_steps == 0:
        matrix = np.lib.format.open_memmap(out_dir / "frame_features.npy", mode="r+")
    stride_s = float(metadata["grid_stride_s"])
    window_us = float(metadata["window_duration_s"]) * 1e6
    completed_sequences = 0
    if len(sequences) > args.sequence_batch_size and args.steps_per_process > 0:
        raise ValueError("Resumable extraction requires one sequence batch per shard")
    with h5py.File(resolve(args.data_path), "r") as handle, torch.inference_mode():
        batches = range(0, len(sequences), args.sequence_batch_size)
        for start in tqdm(batches, desc=f"tespec-cont-{args.shard_index:02d}"):
            rows = sequences.iloc[start : start + args.sequence_batch_size]
            data = roi_data(handle, rows)
            states = None
            length = int(rows.iloc[0].length)
            start_step = 0
            if state_path.exists() and args.smoke_steps == 0:
                saved = torch.load(state_path, map_location="cpu", weights_only=False)
                expected_indices = rows.sequence_index.astype(int).tolist()
                if saved["sequence_indices"] != expected_indices:
                    raise ValueError("Saved TESPEC recurrent state belongs to another shard")
                if int(saved.get("state_reset_steps", 0)) != args.state_reset_steps:
                    raise ValueError("Saved TESPEC state uses another reset interval")
                start_step = int(saved["next_step"])
                states = [
                    (hidden.to(device), cell.to(device))
                    for hidden, cell in saved["states"]
                ]
            if args.smoke_steps > 0:
                end_step = min(length, args.smoke_steps)
            elif args.steps_per_process > 0:
                end_step = min(length, start_step + args.steps_per_process)
            else:
                end_step = length
            for time_index in range(start_step, end_step):
                if args.state_reset_steps > 0 and time_index % args.state_reset_steps == 0:
                    states = None
                center_us = (time_index + 0.5) * stride_s * 1e6
                left_us = center_us - 0.5 * window_us
                right_us = center_us + 0.5 * window_us
                images = []
                for events, timestamps, height, width in data:
                    left = int(np.searchsorted(timestamps, left_us, side="left"))
                    right = int(np.searchsorted(timestamps, right_us, side="right"))
                    images.append(
                        stacked_histogram(
                            events[left:right], height, width, args.histogram_bins, args.image_size
                        )
                    )
                batch = torch.from_numpy(np.stack(images)).to(
                    device, dtype=torch.float16, non_blocking=True
                )
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    pooled, states = model.encode_frame(batch, states)
                if matrix is not None:
                    values = pooled.float().cpu().numpy().astype(np.float16)
                    for row_index, row in enumerate(rows.itertuples(index=False)):
                        matrix[int(row.offset) + time_index] = values[row_index]
            if args.smoke_steps > 0:
                print(
                    f"[SMOKE] sequences={len(rows)} steps={end_step} "
                    f"feature={tuple(pooled.shape)}"
                )
                return
            matrix.flush()
            if end_step < length:
                torch.save(
                    {
                        "sequence_indices": rows.sequence_index.astype(int).tolist(),
                        "next_step": end_step,
                        "state_reset_steps": args.state_reset_steps,
                        "states": [
                            (hidden.detach().cpu(), cell.detach().cpu())
                            for hidden, cell in states
                        ],
                    },
                    state_path,
                )
                print(
                    f"[CHECKPOINT-EXIT] shard={args.shard_index} "
                    f"steps={start_step}:{end_step}/{length}",
                    flush=True,
                )
                raise SystemExit(75)
            completed_sequences += len(rows)
            state_path.unlink(missing_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "complete": True,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "num_sequences": completed_sequences,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] shard={args.shard_index}/{args.num_shards} sequences={completed_sequences}")


def verify(args: argparse.Namespace) -> None:
    out_dir = resolve(args.out_dir)
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    features = np.load(out_dir / "frame_features.npy", mmap_mode="r")
    expected = (int(metadata["num_points"]), int(metadata["feature_dim"]))
    if features.shape != expected:
        raise ValueError(f"TESPEC feature shape {features.shape} != {expected}")
    invalid = 0
    norms = []
    for start in tqdm(range(0, len(features), 8192), desc="verify-tespec"):
        values = np.asarray(features[start : start + 8192], dtype=np.float32)
        invalid += int((~np.isfinite(values).all(axis=1)).sum())
        norms.append(np.linalg.norm(values, axis=1))
    if invalid:
        raise ValueError(f"TESPEC cache contains {invalid} invalid rows")
    all_norms = np.concatenate(norms)
    print(f"[OK] shape={features.shape} norm_mean={all_norms.mean():.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "index":
        build_index(args)
    elif args.command == "extract":
        extract_shard(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
