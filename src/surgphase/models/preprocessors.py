from collections.abc import Callable

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from surgphase.config import PREPROCESSORS

Preprocessor = Callable[[Image.Image], torch.Tensor]
PreprocessorSpec = str | Preprocessor
ChannelStats = tuple[float, float, float]

IMAGENET_MEAN: ChannelStats = (0.485, 0.456, 0.406)
IMAGENET_STD: ChannelStats = (0.229, 0.224, 0.225)


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


class Letterbox:
    def __init__(
        self,
        size: int = 224,
        mean: ChannelStats = IMAGENET_MEAN,
        std: ChannelStats = IMAGENET_STD,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.size = size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        scale = self.size / max(image.size)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        image = TF.resize(image, [height, width], self.interpolation, antialias=True)
        tensor = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        left = (self.size - width) // 2
        top = (self.size - height) // 2
        padding = [left, top, self.size - width - left, self.size - height - top]
        return TF.pad(tensor, padding, fill=0.0)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, mean={self.mean}, std={self.std})"


def surgmotion_preprocess(size: int = 224) -> Preprocessor:
    return T.Compose(
        [
            T.Lambda(_to_rgb),
            T.Resize(
                int(size * 256 / 224), interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def resolve_preprocessor(
    spec: PreprocessorSpec,
    native: Preprocessor,
    size: int,
    mean: ChannelStats,
    std: ChannelStats,
) -> Preprocessor:
    if callable(spec):
        return spec
    if spec == "native":
        return native
    if spec == "letterbox":
        return Letterbox(size, mean, std)
    raise ValueError(
        f"Unknown preprocessor {spec!r}, expected one of {PREPROCESSORS} or a callable"
    )
