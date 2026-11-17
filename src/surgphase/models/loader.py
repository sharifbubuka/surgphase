from pathlib import Path

import torch

from surgphase.models.adapters import SurgCLIPAdapter, SurgMotionAdapter, SurgVLPAdapter
from surgphase.models.base import BaseVideoEncoder
from surgphase.models.preprocessors import PreprocessorSpec


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_weights(weights_dir: str | Path, model_name: str) -> Path | None:
    path = Path(weights_dir) / f"{model_name}.pth"
    return path if path.is_file() else None


def load_model(
    name: str,
    device: str = "cuda",
    weights: str | Path | None = None,
    preprocessor: PreprocessorSpec = "native",
) -> BaseVideoEncoder:
    if name in ("HecVL", "SurgVLP", "PeskaVLP"):
        return SurgVLPAdapter(name, device, weights, preprocessor)
    if name == "SurgCLIP":
        if weights is not None:
            raise ValueError("SurgCLIP weights are managed by the surgclip package")
        return SurgCLIPAdapter(device, preprocessor)
    if name == "SurgMotion300M":
        return SurgMotionAdapter("300M", device, weights, preprocessor)
    if name == "SurgMotion1B":
        return SurgMotionAdapter("1B", device, weights, preprocessor)
    raise ValueError(f"Unsupported model: {name}")
