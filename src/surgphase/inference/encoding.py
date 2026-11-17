from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from surgphase.models.base import BaseVideoEncoder


class FrameDataset(Dataset):
    def __init__(
        self, frame_paths: Sequence[str | Path], preprocess: Callable[[Image.Image], torch.Tensor]
    ):
        self.frame_paths = frame_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.frame_paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


@torch.no_grad()
def encode_sequence(
    encoder: BaseVideoEncoder,
    frame_paths: Sequence[str | Path],
    clip_length: int,
    clip_stride: int,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    if min(clip_length, clip_stride, batch_size) < 1:
        raise ValueError("clip_length, clip_stride and batch_size must be >= 1")
    loader = DataLoader(
        FrameDataset(frame_paths, encoder.preprocessor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    history = (clip_length - 1) * clip_stride
    offsets = torch.arange(clip_length - 1, -1, -1) * clip_stride
    buffer, start, outputs = None, 0, []
    for frames in tqdm(loader, desc=f"Encoding with {encoder.name}", unit="batch"):
        buffer = frames if buffer is None else torch.cat([buffer, frames])
        end = start + len(buffer)
        positions = torch.arange(end - len(frames), end)
        indices = (positions[:, None] - offsets).clamp(min=0) - start
        outputs.append(encoder.encode_video(buffer[indices]).float())
        keep = min(history, len(buffer))
        buffer, start = buffer[len(buffer) - keep :], end - keep
    return torch.cat(outputs)
