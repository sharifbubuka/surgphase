from collections.abc import Callable
from pathlib import Path

import torch

from surgphase.data.encoding import EncodingSpec
from surgphase.data.surgery import Surgery
from surgphase.inference.encoding import encode_sequence
from surgphase.inference.frame_cache import CachedFrameDataset, build_frame_cache, frames_digest
from surgphase.models.base import BaseVideoEncoder
from surgphase.models.preprocessors import Letterbox


def embedding_cache_path(cache_dir: str | Path, spec: EncodingSpec, surgery: Surgery) -> Path:
    digest = frames_digest(surgery.frame_paths)
    name = f"{surgery.id}_fps{surgery.fps:g}_clip{spec.clip_length}x{spec.clip_stride}_{digest}.pt"
    return Path(cache_dir) / spec.model / spec.preprocessor / name


def load_embeddings(cache_dir: str | Path, spec: EncodingSpec, surgery: Surgery) -> torch.Tensor:
    path = embedding_cache_path(cache_dir, spec, surgery)
    if not path.is_file():
        raise FileNotFoundError(
            f"No cached {spec.model}/{spec.preprocessor} embeddings for {surgery.id} at "
            f"{surgery.fps:g} fps ({path}); run the encode-frames task first"
        )
    return torch.load(path, weights_only=True)


def load_or_encode(
    cache_dir: str | Path,
    spec: EncodingSpec,
    surgery: Surgery,
    load_encoder: Callable[[], BaseVideoEncoder],
    batch_size: int,
    num_workers: int,
    frame_cache_dir: str | Path | None = None,
) -> torch.Tensor:
    path = embedding_cache_path(cache_dir, spec, surgery)
    if path.is_file():
        return torch.load(path, weights_only=True)

    encoder = load_encoder()
    if encoder.name != spec.model:
        raise ValueError(f"load_encoder returned {encoder.name}, expected {spec.model}")
    frames = surgery.frame_paths
    if frame_cache_dir is not None and isinstance(encoder.preprocessor, Letterbox):
        cached = build_frame_cache(
            frame_cache_dir, encoder.preprocessor, surgery, num_workers=num_workers
        )
        frames = CachedFrameDataset(cached, encoder.preprocessor)
    embeddings = encode_sequence(
        encoder, frames, spec.clip_length, spec.clip_stride, batch_size, num_workers
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    torch.save(embeddings, partial)
    partial.replace(path)
    return embeddings
