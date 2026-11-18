import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from surgphase.data.surgery import Surgery
from surgphase.models.preprocessors import Letterbox


def frames_digest(frame_paths: Sequence[Path]) -> str:
    return hashlib.sha1(
        "\n".join(Path(*Path(path).parts[-3:]).as_posix() for path in frame_paths).encode()
    ).hexdigest()[:12]


def frame_cache_path(cache_dir: str | Path, letterbox: Letterbox, surgery: Surgery) -> Path:
    name = f"{surgery.id}_fps{surgery.fps:g}_{frames_digest(surgery.frame_paths)}.npy"
    return Path(cache_dir) / letterbox.cache_key / name


class ResizedFrameDataset(Dataset):
    def __init__(self, frame_paths: Sequence[str | Path], letterbox: Letterbox):
        self.frame_paths = frame_paths
        self.letterbox = letterbox

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> np.ndarray:
        with Image.open(self.frame_paths[index]) as image:
            return np.asarray(self.letterbox.resize(image))


class CachedFrameDataset(Dataset):
    def __init__(self, path: str | Path, letterbox: Letterbox):
        self.path = Path(path)
        self.letterbox = letterbox
        self.length = len(np.load(self.path, mmap_mode="r"))
        self.frames: np.ndarray | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.frames is None:
            self.frames = np.load(self.path, mmap_mode="r")
        return self.letterbox.finish(np.array(self.frames[index]))

    def __getstate__(self) -> dict:
        return {**self.__dict__, "frames": None}


def build_frame_cache(
    cache_dir: str | Path,
    letterbox: Letterbox,
    surgery: Surgery,
    batch_size: int = 64,
    num_workers: int = 16,
) -> Path:
    path = frame_cache_path(cache_dir, letterbox, surgery)
    if path.is_file():
        return path
    loader = DataLoader(
        ResizedFrameDataset(surgery.frame_paths, letterbox),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=np.stack,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")
    frames, start = None, 0
    for batch in tqdm(loader, desc=f"Caching {surgery.id} @ {surgery.fps:g} fps", unit="batch"):
        if frames is None:
            shape = (len(surgery.frame_paths), *batch.shape[1:])
            frames = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=shape)
        if batch.shape[1:] != frames.shape[1:]:
            raise ValueError(f"{surgery.id} has frames of different sizes")
        frames[start : start + len(batch)] = batch
        start += len(batch)
    frames.flush()
    del frames
    partial.replace(path)
    return path
