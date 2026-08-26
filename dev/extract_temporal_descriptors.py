"""Append fixed temporal-order descriptors to cached ATSN representations."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmented_tsn import AugmentedTsn
from src.classification import ProposalDataset


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract temporal descriptors from ordered ATSN frame embeddings."
    )
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--base-representations", required=True)
    parser.add_argument("--out-representations", required=True)
    parser.add_argument("--data-path", default="data/preprocessed.h5")
    parser.add_argument("--model-path", default="models/model.pk")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-tsn-samples", type=int, default=7)
    parser.add_argument("--augment-factor", type=int, default=5)
    parser.add_argument("--sample-duration", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=5e-6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def expanded_tsn_samples(num_tsn_samples: int, augment_factor: int) -> int:
    num_augment = int(np.ceil(num_tsn_samples / augment_factor))
    return num_tsn_samples + 2 * num_augment


def proposal_fingerprint(proposals: pd.DataFrame) -> str:
    columns = ["rec_name", "roi_id", "t_start", "t_end"]
    hashed = pd.util.hash_pandas_object(
        proposals[columns], index=False
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def temporal_descriptor_names(num_segments: int) -> list[str]:
    names = [f"relative_norm_{index}" for index in range(num_segments)]
    names += [f"adjacent_cosine_{index}" for index in range(num_segments - 1)]
    names += [f"relative_delta_{index}" for index in range(num_segments - 1)]
    names += [f"center_cosine_{index}" for index in range(num_segments)]
    names += [f"spectral_power_{index}" for index in range(1, num_segments // 2 + 1)]
    return names


def temporal_descriptors(frame_features: torch.Tensor) -> torch.Tensor:
    """Summarize feature evolution without learning or recording-specific scales."""
    norms = torch.linalg.vector_norm(frame_features, dim=2)
    relative_norms = norms / norms.mean(dim=1, keepdim=True).clamp_min(1e-6)

    unit = F.normalize(frame_features, dim=2)
    adjacent_cosine = (unit[:, 1:] * unit[:, :-1]).sum(dim=2)
    delta_norm = torch.linalg.vector_norm(
        frame_features[:, 1:] - frame_features[:, :-1], dim=2
    )
    relative_delta = delta_norm / (norms[:, 1:] + norms[:, :-1]).clamp_min(1e-6)

    center = F.normalize(frame_features.mean(dim=1), dim=1)
    center_cosine = (unit * center[:, None]).sum(dim=2)

    centered = frame_features - frame_features.mean(dim=1, keepdim=True)
    spectral_power = torch.fft.rfft(centered, dim=1).abs().pow(2).mean(dim=2)[:, 1:]
    spectral_power = spectral_power / spectral_power.sum(dim=1, keepdim=True).clamp_min(1e-6)

    return torch.cat(
        (relative_norms, adjacent_cosine, relative_delta, center_cosine, spectral_power),
        dim=1,
    )


@torch.no_grad()
def extract_frame_features(model: AugmentedTsn, images: torch.Tensor) -> torch.Tensor:
    batch_size, num_segments = images.shape[:2]
    x = images.reshape((-1,) + images.shape[2:])
    x = model.backbone(x)["features"]
    x = model.avg_pool(x).flatten(1)
    return x.reshape(batch_size, num_segments, -1)


def main() -> None:
    args = parse_args()
    proposals = pd.read_csv(resolve(args.proposals)).reset_index(drop=True)
    cached = np.load(resolve(args.base_representations), allow_pickle=False)
    embeddings = cached["embeddings"]
    logits = cached["logits"]
    if len(proposals) != len(embeddings) or len(proposals) != len(logits):
        raise ValueError(
            "Proposals and base representations are not aligned: "
            f"{len(proposals)}, {len(embeddings)}, {len(logits)}"
        )

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model = AugmentedTsn(2, args.num_tsn_samples, args.augment_factor)
    try:
        state = torch.load(resolve(args.model_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(resolve(args.model_path), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    dataset = ProposalDataset(
        proposals,
        augment_fraction=1.0 / args.augment_factor,
        data_path=str(resolve(args.data_path)),
        num_tsn_samples=expanded_tsn_samples(args.num_tsn_samples, args.augment_factor),
        sample_duration=args.sample_duration * 1e6,
        decay=args.decay,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    batches = []
    progress = tqdm(loader, desc="temporal-descriptors", disable=args.quiet_progress)
    for images, *_ in progress:
        features = extract_frame_features(model, images.to(device, non_blocking=True))
        batches.append(temporal_descriptors(features).cpu().numpy().astype(np.float16))
    descriptors = np.concatenate(batches, axis=0)
    names = temporal_descriptor_names(dataset.num_tsn_samples)
    if descriptors.shape != (len(proposals), len(names)):
        raise RuntimeError(
            f"Unexpected descriptor shape {descriptors.shape}; expected {(len(proposals), len(names))}"
        )
    if not np.isfinite(descriptors).all():
        raise RuntimeError("Temporal descriptors contain non-finite values")

    augmented_embeddings = np.concatenate(
        (embeddings.astype(np.float16, copy=False), descriptors), axis=1
    )
    output = resolve(args.out_representations)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=augmented_embeddings,
        logits=logits,
        temporal_descriptors=descriptors,
        temporal_descriptor_names=np.asarray(names),
        proposal_fingerprint=np.asarray(proposal_fingerprint(proposals)),
    )
    print(
        f"[RESULTADO] proposals={len(proposals)} base={embeddings.shape} "
        f"descriptors={descriptors.shape} output={augmented_embeddings.shape}"
    )
    print(f"[RESULTADO] path={output}")


if __name__ == "__main__":
    main()
