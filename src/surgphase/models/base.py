from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from PIL import Image


class BaseVideoEncoder(ABC):
    name: str
    device: str
    preprocessor: Callable[[Image.Image], torch.Tensor]

    @abstractmethod
    def encode_video(self, clips: torch.Tensor) -> torch.Tensor: ...


class BaseVisionLanguageModel(BaseVideoEncoder):
    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor: ...

    @abstractmethod
    def project_video(self, embeddings: torch.Tensor) -> torch.Tensor: ...


def check_clips(clips: torch.Tensor, name: str) -> None:
    if clips.dim() != 5:
        raise ValueError(f"{name} expects clips (B, T, C, H, W), got shape {tuple(clips.shape)}")
