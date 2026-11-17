import argparse
import json
import math
import os
import sys
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from surgphase.config import ROOT_DIR
from surgphase.data import AnnotatedFrame, frame_index, save_annotated_frames

T = TypeVar("T")


@dataclass(frozen=True)
class FrameGroup:
    surgery_id: str
    annotation_id: str
    frames_dir: Path
    fps: float
    annotations_path: Path


@dataclass(frozen=True)
class PhaseInterval:
    phase_id: str
    start: float
    end: float


def parse_timestamp(value: str) -> float:
    parts = value.split(":")

    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid timestamp: {value}")

    seconds = 0.0

    for part in parts:
        seconds = seconds * 60 + float(part)

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {value}")

    return seconds


def load_annotations(path: Path) -> tuple[list[PhaseInterval], list[float]]:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)

    intervals = []

    for record in records:
        start = parse_timestamp(record["start_time"])
        end = parse_timestamp(record["end_time"])

        if end <= start:
            raise ValueError(f"Invalid interval: {record}")

        intervals.append(PhaseInterval(record["phase_id"], start, end))

    intervals.sort(key=lambda interval: interval.start)

    for previous, current in zip(intervals, intervals[1:], strict=False):
        if current.start < previous.end:
            raise ValueError("Overlapping phase intervals")

    starts = [interval.start for interval in intervals]

    return intervals, starts


def get_phase(
    timestamp: float,
    intervals: list[PhaseInterval],
    starts: list[float],
) -> str | None:
    index = bisect_right(starts, timestamp) - 1

    if index < 0:
        return None

    interval = intervals[index]

    if timestamp < interval.end:
        return interval.phase_id

    return None


def annotate_group(group: FrameGroup, root: Path) -> list[AnnotatedFrame]:
    if not math.isfinite(group.fps) or group.fps <= 0:
        raise ValueError(f"FPS must be a positive finite number: {group.fps}")

    if not group.frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {group.frames_dir}")

    intervals, starts = load_annotations(group.annotations_path)

    frames = [
        frame
        for frame in group.frames_dir.iterdir()
        if frame.is_file() and frame.suffix.lower() == ".png"
    ]

    if not frames:
        raise ValueError(f"No PNG frames found in {group.frames_dir}")

    frame_indices = sorted(
        ((frame_index(frame), frame) for frame in frames), key=lambda item: item[0]
    )

    annotated = []
    unannotated = []
    relative_frames_dir = Path(os.path.relpath(group.frames_dir.resolve(), root.resolve()))

    for index, frame in frame_indices:
        timestamp = index / group.fps

        phase_id = get_phase(timestamp, intervals, starts)

        if phase_id is None:
            unannotated.append(frame.name)

        frame_path = (relative_frames_dir / frame.name).as_posix()
        annotated.append(
            AnnotatedFrame(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, frame_path)),
                surgery_id=group.surgery_id,
                path=frame_path,
                sample_rate=group.fps,
                phase_group=group.annotation_id,
                phase_id=phase_id,
            )
        )

    if unannotated:
        print(
            f"Warning: {len(unannotated)} frames in {group.frames_dir} have no "
            f"matching phase; phase_id will be empty. Examples: {unannotated[:5]}",
            file=sys.stderr,
        )

    return annotated


def annotate_frames(groups: list[FrameGroup], root: Path, output_path: Path) -> None:
    if not groups:
        raise ValueError("At least one frame group is required")

    frames = []
    for group in groups:
        frames.extend(annotate_group(group, root))

    save_annotated_frames(output_path, frames)

    print(f"Annotated {len(frames)} frames")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assign surgical phases to sampled frames. Group options are matched "
            "by position; a single value is reused for every group."
        )
    )

    parser.add_argument("--surgery-id", nargs="+", required=True)
    parser.add_argument("--annotation-id", nargs="+", required=True)
    parser.add_argument("--frames-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--fps", type=float, nargs="+", required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="CSV file to write")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DIR,
        help="Frame paths in the CSV are written relative to this directory",
    )
    args = parser.parse_args()

    values = {
        "--surgery-id": args.surgery_id,
        "--annotation-id": args.annotation_id,
        "--frames-dir": args.frames_dir,
        "--fps": args.fps,
        "--annotations": args.annotations,
    }
    group_count = max(len(items) for items in values.values())
    for option, items in values.items():
        if len(items) not in (1, group_count):
            parser.error(
                f"{option} must have one value or {group_count} values (one per frame group)"
            )

    def value_at(items: list[T], index: int) -> T:
        return items[0] if len(items) == 1 else items[index]

    groups = [
        FrameGroup(
            surgery_id=value_at(args.surgery_id, index),
            annotation_id=value_at(args.annotation_id, index),
            frames_dir=value_at(args.frames_dir, index),
            fps=value_at(args.fps, index),
            annotations_path=value_at(args.annotations, index),
        )
        for index in range(group_count)
    ]
    annotate_frames(groups, args.root, args.output)


if __name__ == "__main__":
    main()
