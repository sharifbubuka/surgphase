import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from surgphase.config import (
    MODEL_NUM_FRAMES,
    MODEL_WEIGHTS_DIR,
    PHASE_GROUP_FILE,
    PREPROCESSORS,
)
from surgphase.data import load_phase_group, merge_phases, parse_merge
from surgphase.finetuning import PhaseHead, SequenceHead, load_head
from surgphase.models import BaseVideoEncoder, default_device, find_weights, load_model


def load_encoder_and_head(
    model_name: str,
    device: str,
    weights_dir: Path,
    head_path: Path | None = None,
    weights: Path | None = None,
    preprocessor: str = "native",
) -> tuple[BaseVideoEncoder, PhaseHead | SequenceHead | None]:
    weights = weights or find_weights(weights_dir, model_name)
    encoder = load_model(model_name, device=device, weights=weights, preprocessor=preprocessor)

    head = load_head(head_path, device) if head_path else None
    return encoder, head


def _count_params(module: object) -> int | None:
    model = module if isinstance(module, nn.Module) else getattr(module, "model", None)
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters())
    return None


def _smoke_test(encoder: BaseVideoEncoder, model_name: str, image_size: int) -> torch.Tensor:
    batch = 2
    dummy = torch.rand(batch, MODEL_NUM_FRAMES[model_name], 3, image_size, image_size)

    embeddings = encoder.encode_video(dummy)
    print(f"  Input {tuple(dummy.shape)} -> {tuple(embeddings.shape)}")
    if embeddings.dim() != 2 or embeddings.shape[0] != batch:
        raise ValueError(f"expected ({batch}, D), got {tuple(embeddings.shape)}")
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a surgical phase model (encoder + optional fine-tuned head) "
        "and verify that it runs."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_NUM_FRAMES),
        help="Encoder to load",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Device to load on (cuda, cuda:1, mps, cpu). Default: best available",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Encoder checkpoint (default: <weights-dir>/<model>.pth if present, else the official download)",
    )
    parser.add_argument("--weights-dir", type=Path, default=MODEL_WEIGHTS_DIR)
    parser.add_argument(
        "--preprocessor",
        choices=PREPROCESSORS,
        default="native",
        help="Frame preprocessing: the model package's own transform, or letterbox (default: native)",
    )
    parser.add_argument(
        "--head",
        type=Path,
        help="Path to a fine-tuned phase head state dict (.pt)",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=parse_merge,
        default=[],
        metavar="P5+P6",
        help="Phases merged when the head was trained; its output size is checked against them",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip the forward pass with a random input",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Spatial size of the random smoke-test input (default: 224)",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    try:
        encoder, head = load_encoder_and_head(
            args.model, args.device, args.weights_dir, args.head, args.weights, args.preprocessor
        )
    except Exception as error:
        print(f"Failed to load model: {error}", file=sys.stderr)
        return 1
    print(f"Loaded encoder {args.model} on {args.device} in {time.perf_counter() - start:.1f}s")

    n_params = _count_params(encoder)
    if n_params is not None:
        print(f"  Encoder parameters: {n_params / 1e6:.1f}M")

    ok = True

    if head is not None:
        if isinstance(head, SequenceHead):
            shape = f"{head.config.num_stages}x{head.config.num_layers} TCN"
        else:
            shape = f"hidden {list(head.config.hidden_dims)}"
        print(
            f"Loaded head from {args.head}: {head.embedding_dim} -> {shape} -> "
            f"{head.num_phases} phases, {sum(p.numel() for p in head.parameters()):,} parameters"
        )
        phase_group = merge_phases(load_phase_group(PHASE_GROUP_FILE), args.merge)
        if head.num_phases != len(phase_group.phases):
            print(
                f"  ERROR: head predicts {head.num_phases} phases but "
                f"'{phase_group.name}' has {len(phase_group.phases)} (pass the training --merge)",
                file=sys.stderr,
            )
            ok = False
    else:
        print("No head given: embeddings only (phase predictions need a trained --head)")

    if not args.no_smoke_test:
        try:
            embeddings = _smoke_test(encoder, args.model, args.image_size)
        except Exception as error:
            print(f"Smoke test failed: {error}", file=sys.stderr)
            return 1
        embedding_dim = embeddings.shape[-1]
        print(f"Smoke test OK: embeddings shape {tuple(embeddings.shape)}")

        if head is not None:
            if head.embedding_dim != embedding_dim:
                print(
                    f"  ERROR: head expects {head.embedding_dim}-d embeddings but "
                    f"{args.model} produces {embedding_dim}-d",
                    file=sys.stderr,
                )
                ok = False
            else:
                with torch.no_grad():
                    logits = head(embeddings.to(args.device).float().reshape(-1, embedding_dim))
                print(f"  Head forward OK: logits shape {tuple(logits.shape)}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
