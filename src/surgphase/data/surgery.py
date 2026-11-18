from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surgphase.config import UNLABELLED
from surgphase.data.frames import frame_index, load_annotated_frames
from surgphase.data.phase_mapper import PhaseMapper
from surgphase.data.phases import PhaseGroup


@dataclass(frozen=True)
class Surgery:
    id: str
    fps: float
    frame_paths: tuple[Path, ...]
    labels: np.ndarray

    @property
    def labelled(self) -> np.ndarray:
        return self.labels != UNLABELLED


def load_surgeries(
    annotations_file: str | Path,
    phase_group: PhaseGroup,
    frames_root: str | Path,
    fps: float,
    surgery_ids: str | Iterable[str] | None = None,
    merges: Iterable[Sequence[str]] = (),
) -> dict[str, Surgery]:
    mapper = PhaseMapper(phase_group)
    frames = load_annotated_frames(
        annotations_file,
        surgery_id=surgery_ids,
        sample_rate=fps,
        phase_groups=mapper.phase_group.id,
        merges=merges,
    )
    if not frames:
        raise ValueError(
            f"No frames at fps={fps:g} for phase group {mapper.phase_group.id!r} "
            f"in {annotations_file}"
        )
    by_surgery = defaultdict(list)
    for frame in frames:
        by_surgery[frame.surgery_id].append(frame)

    surgeries = {}
    for surgery_id in sorted(by_surgery):
        ordered = sorted(by_surgery[surgery_id], key=lambda frame: frame_index(frame.path))
        surgeries[surgery_id] = Surgery(
            id=surgery_id,
            fps=float(fps),
            frame_paths=tuple(Path(frames_root) / frame.path for frame in ordered),
            labels=np.array(
                [
                    mapper.map_to_index(frame.phase_id)
                    if frame.phase_id is not None
                    else UNLABELLED
                    for frame in ordered
                ],
                dtype=np.int64,
            ),
        )
    return surgeries
