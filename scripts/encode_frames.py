import argparse
import sys
import time
from functools import cache
from pathlib import Path

import torch

from surgphase.config import (
    EMBEDDINGS_DIR,
    MODEL_NUM_FRAMES,
    MODEL_WEIGHTS_DIR,
    MODELS,
    PHASE_GROUP_FILE,
    PREPROCESSORS,
    ROOT_DIR,
)
from surgphase.data import EncodingSpec, load_phase_group, load_surgeries
from surgphase.inference import embedding_cache_path, load_or_encode
from surgphase.models import default_device, find_weights, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode every annotated frame of each surgery with each model and cache the "
        "embeddings, so notebooks and training scripts never re-encode."
    )
    parser.add_argument("--annotations", type=Path, required=True, help="Annotated frames CSV")
    parser.add_argument("--fps", type=float, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_NUM_FRAMES), default=MODELS)
    parser.add_argument("--surgeries", nargs="+", help="Default: every surgery in the CSV")
    parser.add_argument("--preprocessor", choices=PREPROCESSORS, default="native")
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--frames-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--weights-dir", type=Path, default=MODEL_WEIGHTS_DIR)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--batch-size", type=int, help="Default: 64 for image models, 4 for clip models"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase_group = load_phase_group(PHASE_GROUP_FILE)
    surgeries_by_fps = {
        fps: load_surgeries(
            args.annotations,
            phase_group,
            args.frames_root,
            fps,
            args.surgeries,
        )
        for fps in args.fps
    }
    for fps, surgeries in surgeries_by_fps.items():
        for surgery in surgeries.values():
            missing = sum(not path.is_file() for path in surgery.frame_paths)
            if missing:
                print(
                    f"{surgery.id} @ {fps:g} fps: {missing} frames missing on disk", file=sys.stderr
                )
                return 1

    for model_name in args.models:
        spec = EncodingSpec.for_model(model_name, args.preprocessor, args.clip_stride)
        batch_size = args.batch_size or spec.batch_size
        weights = find_weights(args.weights_dir, model_name)

        @cache
        def encoder(model_name=model_name, weights=weights):
            print(f"Loading {model_name} on {args.device}")
            return load_model(model_name, args.device, weights, args.preprocessor)

        for fps, surgeries in surgeries_by_fps.items():
            for surgery in surgeries.values():
                path = embedding_cache_path(args.cache_dir, spec, surgery)
                if path.is_file():
                    print(f"{model_name} {surgery.id} @ {fps:g} fps: cached at {path}")
                    continue
                start = time.perf_counter()
                embeddings = load_or_encode(
                    args.cache_dir, spec, surgery, encoder, batch_size, args.num_workers
                )
                print(
                    f"{model_name} {surgery.id} @ {fps:g} fps: {tuple(embeddings.shape)} "
                    f"in {time.perf_counter() - start:.0f}s -> {path}"
                )
        encoder.cache_clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
