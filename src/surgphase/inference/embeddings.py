import hashlib
from collections.abc import Callable
from pathlib import Path

import torch

from surgphase.config import CLIP_ENCODING_BATCH_SIZE, ENCODING_BATCH_SIZE, PREPROCESSORS
from surgphase.data.surgery import Surgery
from surgphase.inference.encoding import encode_sequence
from surgphase.models.base import BaseVideoEncoder


def embedding_cache_path(
    cache_dir: str | Path,
    model_name: str,
    preprocessor: str,
    surgery: Surgery,
    clip_length: int,
    clip_stride: int,
) -> Path:
    digest = hashlib.sha1(
        "\n".join(Path(*path.parts[-3:]).as_posix() for path in surgery.frame_paths).encode()
    ).hexdigest()[:12]
    name = f"{surgery.id}_fps{surgery.fps:g}_clip{clip_length}x{clip_stride}_{digest}.pt"
    return Path(cache_dir) / model_name / preprocessor / name


def encoding_batch_size(clip_length: int) -> int:
    return ENCODING_BATCH_SIZE if clip_length == 1 else CLIP_ENCODING_BATCH_SIZE


def load_embeddings(
    cache_dir: str | Path,
    model_name: str,
    preprocessor: str,
    surgery: Surgery,
    clip_length: int,
    clip_stride: int,
) -> torch.Tensor:
    path = embedding_cache_path(
        cache_dir, model_name, preprocessor, surgery, clip_length, clip_stride
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"No cached {model_name}/{preprocessor} embeddings for {surgery.id} at "
            f"{surgery.fps:g} fps ({path}); run the encode-frames task first"
        )
    return torch.load(path, weights_only=True)


def load_or_encode(
    cache_dir: str | Path,
    model_name: str,
    preprocessor: str,
    surgery: Surgery,
    clip_length: int,
    clip_stride: int,
    load_encoder: Callable[[], BaseVideoEncoder],
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    if preprocessor not in PREPROCESSORS:
        raise ValueError(f"preprocessor must be one of {PREPROCESSORS}, got {preprocessor!r}")
    path = embedding_cache_path(
        cache_dir, model_name, preprocessor, surgery, clip_length, clip_stride
    )
    if path.is_file():
        return torch.load(path, weights_only=True)

    encoder = load_encoder()
    if encoder.name != model_name:
        raise ValueError(f"load_encoder returned {encoder.name}, expected {model_name}")
    embeddings = encode_sequence(
        encoder, surgery.frame_paths, clip_length, clip_stride, batch_size, num_workers
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    torch.save(embeddings, partial)
    partial.replace(path)
    return embeddings
