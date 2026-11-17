from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from surgphase.data.surgery import Surgery


@dataclass(frozen=True)
class LabelledSequence:
    embeddings: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self):
        if len(self.embeddings) != len(self.labels):
            raise ValueError(f"{len(self.embeddings)} embeddings but {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.labels)


Split = dict[str, LabelledSequence]


def labelled_frames(
    surgery: Surgery, embeddings: torch.Tensor, frame_stride: int = 1
) -> LabelledSequence:
    keep = np.flatnonzero(surgery.labelled)[::frame_stride]
    return LabelledSequence(embeddings[keep], torch.from_numpy(surgery.labels[keep]))


def full_sequence(surgery: Surgery, embeddings: torch.Tensor) -> LabelledSequence:
    return LabelledSequence(embeddings, torch.from_numpy(surgery.labels))


def frame_split(
    surgeries: Mapping[str, Surgery],
    embeddings: Mapping[str, torch.Tensor],
    ids: Iterable[str],
    frame_stride: int = 1,
) -> Split:
    return {
        surgery_id: labelled_frames(surgeries[surgery_id], embeddings[surgery_id], frame_stride)
        for surgery_id in ids
    }


def sequence_split(
    surgeries: Mapping[str, Surgery], embeddings: Mapping[str, torch.Tensor], ids: Iterable[str]
) -> Split:
    return {
        surgery_id: full_sequence(surgeries[surgery_id], embeddings[surgery_id])
        for surgery_id in ids
    }
