import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, fields
from functools import cache
from pathlib import Path

import numpy as np
import torch

from surgphase.config import (
    EMBEDDINGS_DIR,
    MODEL_NUM_FRAMES,
    MODEL_WEIGHTS_DIR,
    PHASE_GROUPS,
    PHASES_DIR,
    PREPROCESSORS,
    ROOT_DIR,
)
from surgphase.data import EncodingSpec, PhaseMapper, load_surgeries
from surgphase.finetuning import (
    HEAD_TYPES,
    Split,
    TrainingConfig,
    build_head,
    evaluate,
    fit,
    save_head,
)
from surgphase.inference import load_or_encode
from surgphase.models import default_device, find_weights, load_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def describe_split(name: str, split: Split, mapper: PhaseMapper) -> None:
    labels = torch.cat([labels for _, labels in split.values()])
    counts = Counter(labels.tolist())
    sizes = ", ".join(f"{surgery}={len(y)}" for surgery, (_, y) in split.items())
    print(f"{name}: {len(labels)} samples ({sizes})")
    for index in range(mapper.num_phases()):
        print(
            f"    {mapper.map_to_id(index):>3} {mapper.get_phase_name(index):<45} {counts.get(index, 0)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a phase classification head on top of a frozen encoder, "
        "using an annotated frames CSV produced by annotate_frames.py. "
        "Embeddings are computed once and cached, so repeated runs only train the head."
    )
    data = parser.add_argument_group("data")
    data.add_argument("--annotations", type=Path, required=True, help="Annotated frames CSV")
    data.add_argument("--phase-group", choices=PHASE_GROUPS, required=True)
    data.add_argument("--fps", type=float, required=True, help="Sample rate to use")
    data.add_argument(
        "--frames-root",
        type=Path,
        default=ROOT_DIR,
        help="Root the CSV frame paths are relative to",
    )
    data.add_argument(
        "--train-surgeries", nargs="+", help="Default: all surgeries not used for val/test"
    )
    data.add_argument("--val-surgeries", nargs="+", help="Default: the last surgery in the CSV")
    data.add_argument(
        "--test-surgeries", nargs="+", default=[], help="Evaluated once with the best head"
    )
    data.add_argument(
        "--frame-stride", type=int, default=1, help="Use every Nth labelled frame (default: 1)"
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model", required=True, choices=list(MODEL_NUM_FRAMES), help="Frozen encoder"
    )
    model.add_argument(
        "--weights",
        type=Path,
        help="Encoder checkpoint (default: <weights-dir>/<model>.pth if present, else the official download)",
    )
    model.add_argument("--weights-dir", type=Path, default=MODEL_WEIGHTS_DIR)
    model.add_argument(
        "--preprocessor",
        choices=PREPROCESSORS,
        default="native",
        help="Frame preprocessing: the model package's own transform, or letterbox (default: native)",
    )
    model.add_argument("--head-type", choices=HEAD_TYPES, default="simple")
    model.add_argument(
        "--dropout", type=float, help="Head dropout (default: linear 0, simple 0.1, robust 0.3)"
    )
    model.add_argument(
        "--clip-length",
        type=int,
        help="Frames per clip (default: 16 SurgCLIP, 64 SurgMotion, 1 otherwise)",
    )
    model.add_argument(
        "--clip-stride", type=int, default=1, help="Step between clip frames, in sampled frames"
    )

    defaults = TrainingConfig()
    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=defaults.epochs)
    training.add_argument("--batch-size", type=int, default=defaults.batch_size)
    training.add_argument("--lr", type=float, default=defaults.lr)
    training.add_argument(
        "--min-lr-ratio",
        type=float,
        default=defaults.min_lr_ratio,
        help="Cosine schedule floor as a fraction of --lr",
    )
    training.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    training.add_argument("--label-smoothing", type=float, default=defaults.label_smoothing)
    training.add_argument(
        "--class-weights", action="store_true", help="Weight the loss by inverse phase frequency"
    )
    training.add_argument(
        "--select-metric", choices=("f1_macro", "accuracy", "loss"), default=defaults.select_metric
    )
    training.add_argument(
        "--patience",
        type=int,
        default=defaults.patience,
        help="Early stopping patience; 0 disables it",
    )
    training.add_argument("--min-delta", type=float, default=defaults.min_delta)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--log-every", type=int, default=defaults.log_every)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default=default_device())
    runtime.add_argument(
        "--encode-batch-size", type=int, help="Default: 64 for image models, 4 for clip models"
    )
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--cache-dir", type=Path, default=EMBEDDINGS_DIR)
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument(
        "--name",
        help="Output file stem (default: <model>_<preprocessor>_<phase-group>_fps<N>_<head-type>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)

    if min(args.clip_length or 1, args.clip_stride, args.frame_stride) < 1:
        print("--clip-length, --clip-stride and --frame-stride must be >= 1", file=sys.stderr)
        return 1
    spec = EncodingSpec.for_model(args.model, args.preprocessor, args.clip_stride, args.clip_length)
    args.clip_length = spec.clip_length
    if args.encode_batch_size is None:
        args.encode_batch_size = spec.batch_size
    if args.model.startswith("SurgMotion") and spec.clip_length % 2:
        print("SurgMotion needs an even --clip-length (tubelet size 2)", file=sys.stderr)
        return 1

    mapper = PhaseMapper(PHASES_DIR / f"{args.phase_group}.json")
    num_phases = mapper.num_phases()
    surgeries = load_surgeries(
        args.annotations, PHASES_DIR / f"{args.phase_group}.json", args.frames_root, args.fps
    )

    surgery_ids = list(surgeries)
    val_surgeries = args.val_surgeries or surgery_ids[-1:]
    held_out = set(val_surgeries) | set(args.test_surgeries)
    train_surgeries = args.train_surgeries or [s for s in surgery_ids if s not in held_out]
    unknown = (set(train_surgeries) | held_out) - set(surgery_ids)
    if unknown:
        print(
            f"Surgeries not in the CSV: {sorted(unknown)} (available: {surgery_ids})",
            file=sys.stderr,
        )
        return 1
    overlap = (
        set(train_surgeries) & set(val_surgeries)
        | set(train_surgeries) & set(args.test_surgeries)
        | set(val_surgeries) & set(args.test_surgeries)
    )
    if overlap:
        print(f"Surgeries appear in more than one split: {sorted(overlap)}", file=sys.stderr)
        return 1
    if not train_surgeries:
        print("No training surgeries left; pass --train-surgeries", file=sys.stderr)
        return 1

    used = sorted(set(train_surgeries) | held_out)
    missing = {
        surgery_id: sum(not path.is_file() for path in surgeries[surgery_id].frame_paths)
        for surgery_id in used
    }
    missing = {surgery_id: count for surgery_id, count in missing.items() if count}
    if missing:
        details = ", ".join(f"{surgery}={count}" for surgery, count in missing.items())
        print(
            f"Frames listed in {args.annotations} are missing on disk ({details}). "
            "The CSV is probably stale: re-run annotate-frames.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Model {args.model}, phase group {args.phase_group} ({num_phases} phases), "
        f"clip length {args.clip_length}, clip stride {args.clip_stride}, frame stride {args.frame_stride}"
    )

    weights = args.weights or find_weights(args.weights_dir, args.model)

    @cache
    def encoder():
        print(f"Loading encoder {args.model} on {args.device}...")
        return load_model(args.model, args.device, weights, args.preprocessor)

    embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    start = time.perf_counter()
    for surgery_id in used:
        surgery = surgeries[surgery_id]
        if not surgery.labelled.any():
            print(f"{surgery_id} has no labelled frames", file=sys.stderr)
            return 1
        x = load_or_encode(
            args.cache_dir, spec, surgery, encoder, args.encode_batch_size, args.num_workers
        )
        keep = np.flatnonzero(surgery.labelled)[:: args.frame_stride]
        embeddings[surgery_id] = (x[keep], torch.from_numpy(surgery.labels[keep]))
    print(f"Embeddings ready in {time.perf_counter() - start:.1f}s")
    encoder.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_split = {s: embeddings[s] for s in train_surgeries}
    val_split = {s: embeddings[s] for s in val_surgeries}
    test_split = {s: embeddings[s] for s in args.test_surgeries}
    describe_split("Train", train_split, mapper)
    describe_split("Val", val_split, mapper)
    if test_split:
        describe_split("Test", test_split, mapper)

    embedding_dim = next(iter(embeddings.values()))[0].shape[1]
    head = build_head(args.head_type, embedding_dim, num_phases, args.dropout)
    print(
        f"Training {args.head_type} head: {embedding_dim} -> {num_phases}, "
        f"hidden {list(head.config.hidden_dims)}, dropout {head.config.dropout}, "
        f"{sum(p.numel() for p in head.parameters()):,} parameters"
    )
    config = TrainingConfig(
        **{f.name: getattr(args, f.name) for f in fields(TrainingConfig) if hasattr(args, f.name)}
    )
    result = fit(head, train_split, val_split, num_phases, config, args.device)

    results = {"val": evaluate(head, val_split, args.device)}
    if test_split:
        results["test"] = evaluate(head, test_split, args.device)

    print(f"\nBest epoch: {result.best_epoch} (selected on val {args.select_metric})")
    for split_name, split_result in results.items():
        overall = split_result["overall"]
        print(
            f"{split_name.capitalize():>5}: acc={overall['accuracy']:.4f} f1_macro={overall['f1_macro']:.4f} "
            f"iou={overall['iou']:.4f} precision={overall['precision']:.4f} recall={overall['recall']:.4f}"
        )
        for surgery_id, metrics in split_result["per_surgery"].items():
            print(
                f"       {surgery_id}: acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or (
        f"{args.model}_{args.preprocessor}_{args.phase_group}_fps{args.fps:g}_{args.head_type}"
    )
    head_path = args.output_dir / f"{stem}_head.pt"
    summary_path = args.output_dir / f"{stem}_summary.json"
    save_head(head, head_path)

    run_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    run_config.update(train_surgeries=train_surgeries, val_surgeries=val_surgeries)
    summary = {
        "config": run_config,
        "head": asdict(head.config),
        "embedding_dim": embedding_dim,
        "num_phases": num_phases,
        "phase_ids": mapper.get_all_phase_ids(),
        "phase_names": mapper.get_all_phase_names(),
        "best_epoch": result.best_epoch,
        "results": results,
        "history": result.history,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nSaved head to {head_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
