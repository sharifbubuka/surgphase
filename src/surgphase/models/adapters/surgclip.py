import surgclip
import torch
from surgclip.transforms import _MEAN, _STD

from surgphase.config import MODEL_NUM_FRAMES
from surgphase.models.base import BaseVisionLanguageModel, check_clips
from surgphase.models.preprocessors import PreprocessorSpec, resolve_preprocessor


class SurgCLIPAdapter(BaseVisionLanguageModel):
    def __init__(self, device: str = "cuda", preprocessor: PreprocessorSpec = "native"):
        self.name = "SurgCLIP"
        self.device = device
        self.model, native, self.tokenizer = surgclip.load(
            "SurgCLIP-B", device=device, num_frames=MODEL_NUM_FRAMES[self.name]
        )
        size = self.model.config.inputs.image_res
        self.preprocessor = resolve_preprocessor(preprocessor, native, size, _MEAN, _STD)

    @torch.no_grad()
    def encode_video(self, clips: torch.Tensor) -> torch.Tensor:
        check_clips(clips, self.name)
        _, pooled = self.model.encode_vision(clips.to(self.device))
        return pooled.mean(dim=1).cpu()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = surgclip.tokenize(texts, self.tokenizer, device=self.device)
        _, pooled = self.model.encode_text(tokens)
        return self.model.text_proj(pooled).cpu()

    @torch.no_grad()
    def project_video(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.model.vision_proj(embeddings.to(self.device)).cpu()
