import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnnotatedFrame:
    id: str
    surgery_id: str
    path: str
    sample_rate: float
    phase_group: str
    phase_id: str


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
) -> list[AnnotatedFrame]:
    surgery_ids = _normalize_filter(surgery_id)
    phase_groups = _normalize_filter(phase_groups)
    phase_ids = _normalize_filter(phase_ids)

    frames = []

    with Path(file_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if surgery_ids is not None and row["surgery_id"] not in surgery_ids:
                continue
            if sample_rate is not None and float(row["sample_rate"]) != float(sample_rate):
                continue
            if phase_groups is not None and row["annotation_id"] not in phase_groups:
                continue
            if phase_ids is not None and row["phase_id"] not in phase_ids:
                continue
            frames.append(
                AnnotatedFrame(
                    id=row["frame_id"],
                    surgery_id=row["surgery_id"],
                    path=row["frame_path"],
                    sample_rate=float(row["sample_rate"]),
                    phase_group=row["annotation_id"],
                    phase_id=row["phase_id"],
                )
            )

    return frames
