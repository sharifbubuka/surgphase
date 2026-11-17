
import argparse
import math
import subprocess
from pathlib import Path


def sample_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    overwrite: bool = False,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be a positive finite number")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / "frame_%06d.png"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-start_number", "0",
        str(output_pattern),
    ]

    subprocess.run(command, check=True)

    print(f"Frames saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample frames from an MP4 video."
    )

    parser.add_argument(
        "--video",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to the input MP4 video(s)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root directory; frames go to <output>/<video stem>/fps<N>",
    )

    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=[1.0],
        help="Sampling rate(s) (default: 1 FPS)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames",
    )

    args = parser.parse_args()

    for video in args.video:
        for fps in args.fps:
            sample_frames(
                video_path=video,
                output_dir=args.output / video.stem / f"fps{fps:g}",
                fps=fps,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
