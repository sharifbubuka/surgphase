from collections.abc import Sequence
from pathlib import Path

import torch

from surgphase.config import MODEL_NUM_FRAMES
from surgphase.inference.encoding import encode_sequence
from surgphase.inference.result import PredictionResult
from surgphase.models.base import BaseVideoEncoder


class Predictor:
    def __init__(
        self,
        encoder: BaseVideoEncoder,
        phase_head: torch.nn.Module,
        clip_length: int | None = None,
        clip_stride: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.encoder = encoder
        self.phase_head = phase_head.to(encoder.device).eval()
        self.clip_length = clip_length or MODEL_NUM_FRAMES[encoder.name]
        self.clip_stride = clip_stride
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.no_grad()
    def predict(self, frame_paths: Sequence[str | Path]) -> PredictionResult:
        embeddings = encode_sequence(
            self.encoder,
            frame_paths,
            self.clip_length,
            self.clip_stride,
            self.batch_size,
            self.num_workers,
        )
        logits = self.phase_head(embeddings.to(self.encoder.device))
        confidences = logits.softmax(dim=1).cpu()
        return PredictionResult(
            predictions=confidences.argmax(dim=1).numpy(),
            confidences=confidences.numpy(),
        )
