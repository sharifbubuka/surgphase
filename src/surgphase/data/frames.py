import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

ANNOTATION_COLUMNS = (
    "surgery_id",
    "frame_id",
    "frame_path",
    "sample_rate",
    "annotation_id",
    "phase_id",
)

FRAME_NAME = re.compile(r"frame_(\d+)")


@dataclass(frozen=True)
class AnnotatedFrame:
    id: str
    surgery_id: str
    path: str
    sample_rate: float
    phase_group: str
    phase_id: str | None

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "AnnotatedFrame":
        return cls(
            id=row["frame_id"],
            surgery_id=row["surgery_id"],
            path=row["frame_path"],
            sample_rate=float(row["sample_rate"]),
            phase_group=row["annotation_id"],
            phase_id=row["phase_id"] or None,
        )

    def to_row(self) -> dict[str, str | float | None]:
        return {
            "surgery_id": self.surgery_id,
            "frame_id": self.id,
            "frame_path": self.path,
            "sample_rate": self.sample_rate,
            "annotation_id": self.phase_group,
            "phase_id": self.phase_id,
        }


def frame_index(path: str | Path) -> int:
    match = FRAME_NAME.fullmatch(Path(path).stem)
    if match is None:
        raise ValueError(f"Invalid frame filename: {path}")
    return int(match.group(1))


def _normalize_filter(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def load_annotated_frames(
    file_path: str | Path,
    surgery_id: str | Iterable[str] | None = None,
    sample_rate: float | None = None,
    phase_groups: str | Iterable[str] | None = None,
    phase_ids: str | Iterable[str] | None = None,
    merges: Iterable[Sequence[str]] = (),
) -> list[AnnotatedFrame]:
    merged_into = {phase_id: ids[0] for ids in merges for phase_id in ids[1:]}
    surgery_ids = _normalize_filter(surgery_id)
    phase_groups = _normalize_filter(phase_groups)
    phase_ids = _normalize_filter(phase_ids)

    frames = []

    with Path(file_path).open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            frame = AnnotatedFrame.from_row(row)
            if frame.phase_id in merged_into:
                frame = replace(frame, phase_id=merged_into[frame.phase_id])
            if surgery_ids is not None and frame.surgery_id not in surgery_ids:
                continue
            if sample_rate is not None and frame.sample_rate != float(sample_rate):
                continue
            if phase_groups is not None and frame.phase_group not in phase_groups:
                continue
            if phase_ids is not None and frame.phase_id not in phase_ids:
                continue
            frames.append(frame)

    return frames


def save_annotated_frames(file_path: str | Path, frames: Iterable[AnnotatedFrame]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(frame.to_row() for frame in frames)
