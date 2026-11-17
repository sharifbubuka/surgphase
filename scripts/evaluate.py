import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from surgphase.config import (
    EMBEDDINGS_DIR,
    HMM_PSEUDOCOUNT,
    MODEL_NUM_FRAMES,
    MODEL_WEIGHTS_DIR,
    MODELS,
    PHASE_GROUPS,
    PHASES_DIR,
    PREPROCESSORS,
    PROMPT_STYLES,
    ROOT_DIR,
    SMOOTHING_SECONDS,
    TEMPORAL_METHODS,
    ZERO_SHOT_TEMPERATURE,
)
from surgphase.data import EncodingSpec, PhaseGroup, Surgery, load_phase_group, load_surgeries
from surgphase.eval import (
    MetricsResult,
    leave_one_surgery_out,
    predictions_file,
    save_predictions,
    score,
)
from surgphase.finetuning import (
    HEAD_TYPES,
    SequenceHead,
    SequenceHeadConfig,
    TrainingConfig,
    build_head,
    fit,
    fit_sequence,
    frame_split,
    predict_logits,
    sequence_split,
)
from surgphase.inference import load_or_encode
from surgphase.models import BaseVisionLanguageModel, default_device, find_weights, load_model
from surgphase.temporal import HMM_METHODS, Decoding, decodings, leave_one_out_hmms
from surgphase.zeroshot import phase_prompts, text_classifier, zero_shot_probabilities

SOURCES = ("zeroshot", "frame_head", "sequence_head")


@dataclass(frozen=True)
class Prediction:
    source: str
    variant: str
    surgery_id: str
    probabilities: np.ndarray


@dataclass(frozen=True)
class EvaluationRecord:
    phase_group: str
    spec: EncodingSpec
    fps: float
    prediction: Prediction
    decoding: Decoding
    metrics: MetricsResult
    predictions_file: Path

    def to_row(self) -> dict:
        return {
            "phase_group": self.phase_group,
            "model": self.spec.model,
            "preprocessor": self.spec.preprocessor,
            "fps": self.fps,
            "clip_length": self.spec.clip_length,
            "clip_stride": self.spec.clip_stride,
            "source": self.prediction.source,
            "variant": self.prediction.variant,
            "surgery_id": self.prediction.surgery_id,
            **asdict(self.decoding),
            **asdict(self.metrics),
            "predictions_file": self.predictions_file.as_posix(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate models per surgery from cached embeddings: probabilities from each "
        "source, every temporal decoding, one CSV row per (model, fps, source, decoding, surgery). "
        "Writes <output-dir>/metrics.csv, <output-dir>/run.json and per-frame probabilities under "
        "<output-dir>/predictions (the predictions_file column points at them)."
    )
    parser.add_argument("--annotations", type=Path, required=True, help="Annotated frames CSV")
    parser.add_argument("--phase-group", choices=PHASE_GROUPS, required=True)
    parser.add_argument("--fps", type=float, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_NUM_FRAMES), default=MODELS)
    parser.add_argument("--surgeries", nargs="+", help="Default: every surgery in the CSV")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument(
        "--prompt-styles",
        nargs="+",
        choices=PROMPT_STYLES,
        default=PROMPT_STYLES,
        help="Zero-shot is run per style and with all of them combined",
    )
    parser.add_argument("--temperature", type=float, default=ZERO_SHOT_TEMPERATURE)
    parser.add_argument("--head-types", nargs="+", choices=HEAD_TYPES, default=["linear"])
    parser.add_argument(
        "--class-weights", action="store_true", help="Weight head losses by inverse phase frequency"
    )
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--sequence-stages", type=int, default=SequenceHeadConfig.num_stages)
    parser.add_argument("--sequence-layers", type=int, default=SequenceHeadConfig.num_layers)
    parser.add_argument(
        "--sequence-lr", type=float, default=5e-4, help="Learning rate for sequence heads"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temporal-methods", nargs="+", choices=TEMPORAL_METHODS, default=TEMPORAL_METHODS
    )
    parser.add_argument("--smoothing-seconds", type=float, nargs="+", default=SMOOTHING_SECONDS)
    parser.add_argument(
        "--centered", action="store_true", help="Also run centred (offline) windows"
    )
    parser.add_argument("--hmm-pseudocount", type=float, default=HMM_PSEUDOCOUNT)
    parser.add_argument("--preprocessor", choices=PREPROCESSORS, default="native")
    parser.add_argument("--clip-stride", type=int, default=1)
    parser.add_argument("--frames-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--weights-dir", type=Path, default=MODEL_WEIGHTS_DIR)
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory")
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--batch-size", type=int, help="Default: 64 for image models, 4 for clip models"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def zeroshot_predictions(
    args: argparse.Namespace,
    encoder: BaseVisionLanguageModel,
    phase_group: PhaseGroup,
    embeddings: dict[str, torch.Tensor],
) -> Iterator[Prediction]:
    variants = [[style] for style in args.prompt_styles]
    if len(args.prompt_styles) > 1:
        variants.append(list(args.prompt_styles))
    for styles in variants:
        classifier = text_classifier(encoder, phase_prompts(phase_group, styles))
        for surgery_id, surgery_embeddings in embeddings.items():
            yield Prediction(
                "zeroshot",
                "+".join(styles),
                surgery_id,
                zero_shot_probabilities(encoder, surgery_embeddings, classifier, args.temperature),
            )


def frame_head_predictions(
    args: argparse.Namespace,
    surgeries: dict[str, Surgery],
    embeddings: dict[str, torch.Tensor],
    num_phases: int,
) -> Iterator[Prediction]:
    config = TrainingConfig(epochs=args.epochs, class_weights=args.class_weights)
    embedding_dim = next(iter(embeddings.values())).shape[1]
    for head_type in args.head_types:
        for fold in leave_one_surgery_out(list(surgeries)):
            torch.manual_seed(args.seed)
            head = build_head(head_type, embedding_dim, num_phases)
            fit(
                head,
                frame_split(surgeries, embeddings, fold.train),
                frame_split(surgeries, embeddings, [fold.val]),
                num_phases,
                config,
                args.device,
            )
            logits = predict_logits(head, embeddings[fold.test], args.device)
            yield Prediction("frame_head", head_type, fold.test, logits.softmax(dim=1).numpy())


def sequence_head_predictions(
    args: argparse.Namespace,
    surgeries: dict[str, Surgery],
    embeddings: dict[str, torch.Tensor],
    num_phases: int,
) -> Iterator[Prediction]:
    config = TrainingConfig(
        epochs=args.epochs, lr=args.sequence_lr, class_weights=args.class_weights
    )
    head_config = SequenceHeadConfig(
        embedding_dim=next(iter(embeddings.values())).shape[1],
        num_phases=num_phases,
        num_stages=args.sequence_stages,
        num_layers=args.sequence_layers,
    )
    variant = f"tcn{args.sequence_stages}x{args.sequence_layers}"
    for fold in leave_one_surgery_out(list(surgeries)):
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        head = SequenceHead(head_config)
        fit_sequence(
            head,
            sequence_split(surgeries, embeddings, fold.train),
            sequence_split(surgeries, embeddings, [fold.val]),
            num_phases,
            config,
            args.device,
        )
        logits = predict_logits(head, embeddings[fold.test], args.device)
        yield Prediction("sequence_head", variant, fold.test, logits.softmax(dim=1).numpy())


def evaluate_predictions(
    args: argparse.Namespace,
    surgeries: dict[str, Surgery],
    predictions: Iterator[Prediction],
    num_phases: int,
    spec: EncodingSpec,
    fps: float,
    predictions_dir: Path,
) -> list[EvaluationRecord]:
    hmms = leave_one_out_hmms(
        {s.id: s.labels[s.labelled] for s in surgeries.values()}, num_phases, args.hmm_pseudocount
    )
    variants = decodings(args.temporal_methods, args.smoothing_seconds, args.centered)
    probabilities = defaultdict(dict)
    records = []
    for prediction in predictions:
        surgery = surgeries[prediction.surgery_id]
        file = predictions_file(spec, fps, prediction.source, prediction.variant)
        probabilities[file][surgery.id] = prediction.probabilities.astype(np.float32)
        for decoding in variants:
            hmm = hmms[surgery.id] if decoding.method in HMM_METHODS else None
            labels = decoding.apply(prediction.probabilities, surgery.fps, hmm)
            records.append(
                EvaluationRecord(
                    args.phase_group,
                    spec,
                    fps,
                    prediction,
                    decoding,
                    score(surgery.labels, labels),
                    file,
                )
            )
    for file, per_surgery in probabilities.items():
        save_predictions(predictions_dir / file, per_surgery)
    return records


def main() -> int:
    args = parse_args()
    phase_group_file = PHASES_DIR / f"{args.phase_group}.json"
    phase_group = load_phase_group(phase_group_file)
    num_phases = len(phase_group.phases)
    predictions_dir = args.output_dir / "predictions"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.json").open("w", encoding="utf-8") as file:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, file, indent=2
        )
    records = []
    for fps in args.fps:
        surgeries = load_surgeries(
            args.annotations, phase_group_file, args.frames_root, fps, args.surgeries
        )
        for model_name in args.models:
            spec = EncodingSpec.for_model(model_name, args.preprocessor, args.clip_stride)
            weights = find_weights(args.weights_dir, model_name)

            @cache
            def encoder(model_name=model_name, weights=weights):
                return load_model(model_name, args.device, weights, args.preprocessor)

            embeddings = {
                surgery_id: load_or_encode(
                    args.cache_dir,
                    spec,
                    surgery,
                    encoder,
                    args.batch_size or spec.batch_size,
                    args.num_workers,
                )
                for surgery_id, surgery in surgeries.items()
            }
            for source in args.sources:
                if source == "zeroshot":
                    if not isinstance(encoder(), BaseVisionLanguageModel):
                        print(f"{model_name} has no text encoder: skipping zeroshot")
                        continue
                    predictions = zeroshot_predictions(args, encoder(), phase_group, embeddings)
                elif source == "frame_head":
                    predictions = frame_head_predictions(args, surgeries, embeddings, num_phases)
                elif source == "sequence_head":
                    predictions = sequence_head_predictions(args, surgeries, embeddings, num_phases)
                new_records = evaluate_predictions(
                    args, surgeries, predictions, num_phases, spec, fps, predictions_dir
                )
                if not new_records:
                    continue
                records += new_records
                best = max(new_records, key=lambda record: record.metrics.f1_macro)
                print(
                    f"{model_name} @ {fps:g} fps {source}: {len(new_records)} rows, best f1_macro "
                    f"{best.metrics.f1_macro:.3f} ({best.prediction.variant}, "
                    f"{best.decoding.method}, {best.prediction.surgery_id})"
                )
            encoder.cache_clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_path = args.output_dir / "metrics.csv"
    pd.DataFrame([record.to_row() for record in records]).to_csv(metrics_path, index=False)
    print(f"Wrote {len(records)} rows to {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
