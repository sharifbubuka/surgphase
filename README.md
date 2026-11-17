# Sample video frames

Run the frame sampler through Poe, passing the script's CLI options as needed:

```sh
uv run poe sample-frames --video assets/videos/pat01.mp4 assets/videos/pat02.mp4 --output assets/frames --fps 1 2 30
```

`--video` and `--fps` accept multiple values. Every video is sampled at every FPS, and frames are written to `<output>/<video stem>/fps<N>`, e.g. `assets/frames/pat01/fps30`.

Add `--overwrite` to replace existing frames. The script's FPS defaults to `1` when `--fps` is omitted.

# Annotate sampled frames

Run the frame annotator through Poe, passing the surgery metadata and paths:

```sh
uv run poe annotate-frames --surgery-id pat01 --annotation-id default \
  --frames-dir assets/frames/pat01/fps1 --fps 1 \
  --annotations assets/annotations/default/pat01.json \
  --output-dir assets/annotations_csv
```

The CSV is written to the output directory as
`<surgery-id>_<annotation-id>_fps<fps>.csv`. To combine multiple frame groups
in one CSV, provide multiple values for the group options:

```sh
uv run poe annotate-frames \
  --surgery-id pat01 pat02 pat04 \
  --annotation-id default \
  --frames-dir assets/frames/pat01/fps1 assets/frames/pat02/fps1 assets/frames/pat04/fps1 \
  --fps 1 \
  --annotations assets/annotations/default/pat01.json assets/annotations/default/pat02.json assets/annotations/default/pat04.json \
  --output-dir assets/annotations_csv
```

Values are matched by position, not combined as a Cartesian product. Each option
must have either one value (reused for every group) or one value per group. To
include multiple FPS values for a surgery, list a group for each FPS, repeating
the other values as needed. A multi-group run writes `annotated_frames.csv` in
the output directory. Running the command again with the same output filename
replaces that CSV.

Frames outside all annotation intervals are included with an empty `phase_id`
field (CSV has no native null type). The script reports the number of unmatched
frames for each group on standard error; these rows can be filtered downstream.
The final intervals in the default and merged6&7 annotations extend to the
respective video stream durations, retaining the final phase through the end
of each video.

The `frame_path` column contains the path to each image relative to the project
root, for example `assets/frames/pat01/fps1/frame_000000.png`. Resolve it
against the project root when reading it. Frames outside the project root use
relative paths containing `..`.

Each `frame_id` is a randomly generated UUID4. Within a CSV, repeated references
to the same frame path share an ID and different frames have different IDs.
IDs are generated anew on each run and are not stable across CSVs; use
`frame_path` to match frames across runs. UUID4 collisions across separate runs
are extremely unlikely but cannot be ruled out without persistent state.

# Zip the project

To download the project directly to your Mac, run these commands **on the Mac**,
replacing `SERVER` with the SSH host or alias you use to connect:

```sh
scp SERVER:/home/bubuka/thesis/code/surgphase/scripts/download_project.sh ~/Downloads/
bash ~/Downloads/download_project.sh SERVER
```

The archive is streamed to `~/Downloads/surgphase_project.zip`; an existing file
at that location is not overwritten. If the project is elsewhere on the server,
pass its absolute path as a second argument to `download_project.sh`.

To create the archive on the server instead, run:

```sh
uv run poe zip-project
```

Both methods exclude dependencies, frames, videos, and model weights. The Poe
task creates `surgphase_project.zip` in the project root on the server.
# Encode frames

Encode every annotated frame of each surgery with each model once and cache the
embeddings under `results/embeddings/<model>/<preprocessor>/`:

```sh
uv run poe encode-frames --annotations assets/annotations/default/annotated_frames.csv \
  --phase-group default --fps 1 15 30 --models SurgVLP SurgCLIP SurgMotion300M
```

Each frame gets a causal clip ending at it (`MODEL_NUM_FRAMES` in the config), and
each frame is decoded once. Cached files are keyed by surgery, fps, clip length,
clip stride and the frame list, so changed frames are re-encoded. Training,
evaluation and notebooks read the same cache through
`surgphase.inference.load_or_encode`.

# Evaluate

Run zero-shot, frame-head and sequence-head predictions with every temporal
decoding and write one CSV row per (model, fps, source, decoding, surgery):

```sh
uv run poe evaluate --annotations assets/annotations/default/annotated_frames.csv \
  --phase-group default --fps 1 --models SurgVLP SurgCLIP \
  --sources zeroshot frame_head sequence_head --head-types linear simple \
  --output results/evaluation.csv
```

- `zeroshot` uses the phase prompts in `assets/phases/<group>.json` per prompt
  style and with all styles combined (vision-language models only).
- `frame_head` and `sequence_head` are trained with leave-one-surgery-out folds:
  each surgery is tested once, the next one validates, the rest train. They
  need at least three surgeries.
- Temporal decodings (`TEMPORAL_METHODS` in the config): moving average, moving
  median and exponential moving average over `SMOOTHING_SECONDS`, HMM filtering
  (online) and Viterbi (offline). The HMM is fitted on the labels of the other
  surgeries. Add `--centered` for offline, centred smoothing windows.

Metrics include accuracy, macro F1, Jaccard, edit score and segmental F1@{10,25,50};
unlabelled frames are skipped.

# Train a single head

```sh
uv run poe train-head --model SurgVLP --annotations assets/annotations/default/annotated_frames.csv \
  --phase-group default --fps 1 --train-surgeries pat01 pat02 --val-surgeries pat04 --head-type simple
```

# Library overview

| Module | Contents |
|---|---|
| `surgphase.data` | `load_surgeries` (frame paths and labels per surgery), phase groups and prompts |
| `surgphase.models` | `load_model`, adapters, preprocessors (`native` or `letterbox`) |
| `surgphase.inference` | `load_or_encode`, `encode_sequence`, `Predictor` |
| `surgphase.zeroshot` | `phase_prompts`, `text_classifier`, `zero_shot_probabilities` |
| `surgphase.finetuning` | frame heads (`build_head`, `fit`), causal MS-TCN `SequenceHead` (`fit_sequence`) |
| `surgphase.temporal` | smoothing, `PhaseHMM`, `decode` |
| `surgphase.eval` | `score`, `compute_metrics`, `leave_one_surgery_out` |
| `surgphase.viz` | `plot_timeline`, `plot_confusion_matrix`, `phase_colors` |
