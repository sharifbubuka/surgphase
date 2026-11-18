from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from surgphase.config import PROMPT_STYLES
from surgphase.data.phases import PhaseGroup, Prompts, ordered_phases
from surgphase.models.base import BaseVisionLanguageModel


def phase_prompts(
    phase_group: PhaseGroup, prompts: Mapping[str, Prompts], styles: Sequence[str]
) -> list[list[str]]:
    unknown = set(styles) - set(PROMPT_STYLES)
    if unknown or not styles:
        raise ValueError(f"styles must be a non-empty subset of {PROMPT_STYLES}, got {styles}")
    phases = ordered_phases(phase_group)
    missing = {member for phase in phases for member in phase.members} - set(prompts)
    if missing:
        raise KeyError(f"no prompts for phase ids {sorted(missing)}")
    return [
        [
            prompt.message
            for member in phase.members
            for style in styles
            for prompt in getattr(prompts[member], style)
        ]
        for phase in phases
    ]


@torch.no_grad()
def text_classifier(
    model: BaseVisionLanguageModel, prompts: Sequence[Sequence[str]]
) -> torch.Tensor:
    if any(not phase for phase in prompts):
        raise ValueError("every phase needs at least one prompt")
    texts = [text for phase in prompts for text in phase]
    embeddings = F.normalize(model.encode_text(texts).float(), dim=-1)
    per_phase = embeddings.split([len(phase) for phase in prompts])
    return F.normalize(torch.stack([phase.mean(dim=0) for phase in per_phase]), dim=-1)


@torch.no_grad()
def zero_shot_probabilities(
    model: BaseVisionLanguageModel,
    video_embeddings: torch.Tensor,
    classifier: torch.Tensor,
    temperature: float,
) -> np.ndarray:
    video = F.normalize(model.project_video(video_embeddings).float(), dim=-1)
    return (video @ classifier.T / temperature).softmax(dim=1).numpy()
