## Rules

- **No docstrings.** Do not add docstrings anywhere: not to modules, classes, functions, or methods.
- **Test every logic change, but don't commit tests.** After implementing any code logic, verify it by running tests (e.g. throwaway scripts or inline `python -c` checks). Do not add test files to the codebase; delete any temporary test scripts afterward.
- **Document new poe tasks with an example.** When adding a task to `poe_tasks.toml`, put a comment on the line directly after its `cmd = ...` line showing a full example invocation, e.g.:

  ```toml
  [tasks.train-head]
  cmd = "python scripts/train_head.py"
  # uv run poe train-head --model SurgVLP --annotations assets/annotations/default/annotated_frames.csv --fps 1
  ```

- **Commit messages are a single lowercase title.** Use conventional commits, `type(scope): summary`, e.g. `fix(eval): handle empty segments` or `refactor(data): add encoding spec`. Everything is lowercase, including names of classes and functions. No body, description or trailers.
- **No commit attribution.** Do not add `Co-Authored-By` or any other attribution lines to commit messages or PR descriptions. Commits carry only the user's git identity.

## Project map

Surgical phase recognition with surgical vision-language models (zero-shot, plus trained heads on frozen embeddings). The `uv run poe <task>` entry points are in `poe_tasks.toml`, and each one runs a script in `scripts/`.

- `src/surgphase/config/__init__.py`: every global constant and path (FPS, PATIENTS, MODELS, MODEL_NUM_FRAMES, *_DIR, *_FILE)
- `data/`: annotated-frame CSV schema (`AnnotatedFrame`, read and write), the single phase group (`assets/phases/default.json`; merge variants: `--merge P6+P7` relabels P7 frames as P6 in `load_annotated_frames` and `merge_phases` drops P7), zero-shot prompts keyed by phase id (`assets/prompts/default.json`), phase mapping, surgeries, `EncodingSpec` (model, preprocessor, clip length/stride: keys the embedding cache and prediction files)
- `models/`: `base.py` interface, `loader.py`, `preprocessors.py` (native/letterbox), `adapters/` (SurgVLP, SurgCLIP, SurgMotion)
- `inference/`: encoding frames, plus the on-disk embedding cache in `embeddings.py` (`load_or_encode`, under `results/embeddings`) and the predictor
- `zeroshot/`: text-prompt classifier
- `finetuning/`: frame head, sequence head, trainer, `split.py` (`LabelledSequence`/`Split` and the builders for frame and sequence splits)
- `temporal/`: smoothing, HMM filter/Viterbi decode, `Decoding` variants and leave-one-out HMMs
- `eval/`: metrics (sklearn/torchmetrics), scoring, leave-one-surgery-out splits, per-frame prediction files (`predictions.py`, written by the evaluate task)
- `viz/`: timelines, confusion matrices

## Working efficiently

- Don't read or list `assets/frames`, `assets/videos`, `assets/models` (2 TB of binaries) or `results/embeddings`. Paths follow `assets/frames/<surgery>/fps<N>/frame_000000.png`.
- Annotation CSVs are large. Check them with `head`/`wc -l`/pandas rather than reading them whole. Columns: `surgery_id,frame_id,frame_path,sample_rate,annotation_id,phase_id`.
- Use the project map above to open the file you need directly instead of searching broadly.
- Notebooks only analyse results: they load outputs from `results/` and produce tables and figures. Heavy compute (encoding, training, evaluation) belongs in `scripts/` run through poe tasks, never in notebooks.
- Keep test output short (`pytest -q`, `| tail`). Never run training or encoding without asking, since they are slow and print a lot.
