from dataclasses import dataclass

from surgphase.config import (
    CLIP_ENCODING_BATCH_SIZE,
    ENCODING_BATCH_SIZE,
    MODEL_NUM_FRAMES,
    PREPROCESSORS,
)


@dataclass(frozen=True)
class EncodingSpec:
    model: str
    preprocessor: str
    clip_length: int
    clip_stride: int = 1

    def __post_init__(self):
        if self.preprocessor not in PREPROCESSORS:
            raise ValueError(
                f"preprocessor must be one of {PREPROCESSORS}, got {self.preprocessor!r}"
            )
        if min(self.clip_length, self.clip_stride) < 1:
            raise ValueError("clip_length and clip_stride must be >= 1")

    @classmethod
    def for_model(
        cls, model: str, preprocessor: str, clip_stride: int = 1, clip_length: int | None = None
    ) -> "EncodingSpec":
        return cls(model, preprocessor, clip_length or MODEL_NUM_FRAMES[model], clip_stride)

    @property
    def batch_size(self) -> int:
        return ENCODING_BATCH_SIZE if self.clip_length == 1 else CLIP_ENCODING_BATCH_SIZE
