import hashlib
from collections.abc import Callable
from pathlib import Path

import torch

from surgphase.data.encoding import EncodingSpec
from surgphase.data.surgery import Surgery
from surgphase.inference.encoding import encode_sequence
from surgphase.models.base import BaseVideoEncoder


def embedding_cache_path(cache_dir: str | Path, spec: EncodingSpec, surgery: Surgery) -> Path:
    digest = hashlib.sha1(
        "\n".join(Path(*path.parts[-3:]).as_posix() for path in surgery.frame_paths).encode()
    ).hexdigest()[:12]
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
) -> torch.Tensor:
    path = embedding_cache_path(cache_dir, spec, surgery)
    if path.is_file():
        return torch.load(path, weights_only=True)

    encoder = load_encoder()
    if encoder.name != spec.model:
        raise ValueError(f"load_encoder returned {encoder.name}, expected {spec.model}")
    embeddings = encode_sequence(
        encoder, surgery.frame_paths, spec.clip_length, spec.clip_stride, batch_size, num_workers
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    torch.save(embeddings, partial)
    partial.replace(path)
    return embeddings
