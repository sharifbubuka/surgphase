import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surgphase.config import UNLABELLED
from surgphase.data.frames import load_annotated_frames
from surgphase.data.phase_mapper import PhaseMapper


@dataclass(frozen=True)
class Surgery:
    id: str
    fps: float
    frame_paths: tuple[Path, ...]
    labels: np.ndarray

    @property
    def labelled(self) -> np.ndarray:
        return self.labels != UNLABELLED


def _frame_index(path: str) -> int:
    match = re.search(r"(\d+)$", Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot parse a frame index from {path}")
    return int(match.group(1))


def load_surgeries(
    annotations_file: str | Path,
    phase_group_file: str | Path,
    frames_root: str | Path,
    fps: float,
    surgery_ids: str | Iterable[str] | None = None,
) -> dict[str, Surgery]:
    mapper = PhaseMapper(phase_group_file)
    frames = load_annotated_frames(
        annotations_file,
        surgery_id=surgery_ids,
        sample_rate=fps,
        phase_groups=mapper.phase_group.id,
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
        ordered = sorted(by_surgery[surgery_id], key=lambda frame: _frame_index(frame.path))
        surgeries[surgery_id] = Surgery(
            id=surgery_id,
            fps=float(fps),
            frame_paths=tuple(Path(frames_root) / frame.path for frame in ordered),
            labels=np.array(
                [
                    mapper.map_to_index(frame.phase_id) if frame.phase_id else UNLABELLED
                    for frame in ordered
                ],
                dtype=np.int64,
            ),
        )
    return surgeries
