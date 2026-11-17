from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from surgmotion.hub.backbones import _clean_backbone_key
from surgmotion.models import vision_transformer

from surgphase.config import MODEL_NUM_FRAMES
from surgphase.models.base import BaseVideoEncoder, check_clips
from surgphase.models.preprocessors import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessorSpec,
    resolve_preprocessor,
    surgmotion_preprocess,
)

HF_REPO = "CAIR-HKISI/SurgMotion"

VARIANTS = {
    "300M": ("vit_large", "SurgMotion-vitl.pt"),
    "1B": ("vit_giant_xformers", "SurgMotion-vitg-xformer.pt"),
}

IMAGE_SIZE = 224


class SurgMotionAdapter(BaseVideoEncoder):
    def __init__(
        self,
        variant: Literal["300M", "1B"] = "1B",
        device: str = "cuda",
        weights: str | Path | None = None,
        preprocessor: PreprocessorSpec = "native",
    ):
        self.name = f"SurgMotion{variant}"
        self.device = device
        self.preprocessor = resolve_preprocessor(
            preprocessor, surgmotion_preprocess(IMAGE_SIZE), IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
        )

        arch, filename = VARIANTS[variant]
        self.model = vision_transformer.__dict__[arch](
            patch_size=16,
            img_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_frames=MODEL_NUM_FRAMES[self.name],
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
        )
        if weights is None:
            weights = hf_hub_download(HF_REPO, filename)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(_clean_backbone_key(checkpoint["encoder"]), strict=False)
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_video(self, clips: torch.Tensor) -> torch.Tensor:
        check_clips(clips, self.name)
        tubelet_size = self.model.tubelet_size
        if clips.shape[1] % tubelet_size:
            raise ValueError(f"{self.name} needs T to be a multiple of {tubelet_size}")
        tokens = self.model(clips.permute(0, 2, 1, 3, 4).to(self.device), masks=None)
        return tokens.mean(dim=1).cpu()
