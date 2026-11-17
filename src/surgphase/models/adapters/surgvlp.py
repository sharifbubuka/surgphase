from pathlib import Path

import surgvlp
import torch

from surgphase.config import TEXT_ENCODER
from surgphase.models.base import BaseVisionLanguageModel, check_clips
from surgphase.models.preprocessors import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessorSpec,
    resolve_preprocessor,
)


def _model_config(model_name: str) -> dict:
    return {
        "type": model_name,
        "backbone_img": {
            "type": "img_backbones/ImageEncoder",
            "num_classes": 768,
            "pretrained": "imagenet",
            "backbone_name": "resnet_50",
            "img_norm": False,
        },
        "backbone_text": {
            "type": "text_backbones/BertEncoder",
            "text_bert_type": TEXT_ENCODER,
            "text_last_n_layers": 4,
            "text_aggregate_method": "sum",
            "text_norm": False,
            "text_embedding_dim": 768,
            "text_freeze_bert": False,
            "text_agg_tokens": True,
        },
    }


class SurgVLPAdapter(BaseVisionLanguageModel):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        weights: str | Path | None = None,
        preprocessor: PreprocessorSpec = "native",
    ):
        self.name = model_name
        self.device = device
        self.model, native = surgvlp.load(
            _model_config(model_name),
            device=device,
            pretrain=None if weights is None else str(weights),
        )
        self.preprocessor = resolve_preprocessor(
            preprocessor,
            native,
            surgvlp.surgvlp._INPUT_RES[model_name],
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

    @torch.no_grad()
    def encode_video(self, clips: torch.Tensor) -> torch.Tensor:
        check_clips(clips, self.name)
        batch, time = clips.shape[:2]
        frames = clips.flatten(0, 1).to(self.device)
        embeddings = self.model(inputs_img=frames, mode="video")["img_emb"]
        return embeddings.view(batch, time, -1).mean(dim=1).cpu()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = surgvlp.tokenize(texts, model_name=TEXT_ENCODER, device=self.device)
        return self.model(inputs_text=inputs, mode="text")["text_emb"].cpu()

    def project_video(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings
