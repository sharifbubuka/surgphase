from collections.abc import Mapping
from pathlib import Path

import numpy as np

from surgphase.data.encoding import EncodingSpec


def predictions_file(spec: EncodingSpec, fps: float, source: str, variant: str) -> Path:
    return (
        Path(spec.model)
        / spec.preprocessor
        / f"fps{fps:g}_clip{spec.clip_length}x{spec.clip_stride}"
        / source
        / f"{variant}.npz"
    )


def save_predictions(file_path: str | Path, probabilities: Mapping[str, np.ndarray]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **probabilities)


def load_predictions(file_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(file_path)) as data:
        return {surgery_id: data[surgery_id] for surgery_id in data.files}
