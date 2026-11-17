<div align="center">

# SurgPhase

**Surgical phase recognition with surgical vision-language and video foundation models**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

## Overview

SurgPhase benchmarks pretrained surgical foundation models for frame-level phase recognition:

- **Zero-shot** classification from text prompts (vision-language models only).
- **Frame heads** (linear / MLP) and a causal **MS-TCN sequence head** trained on frozen embeddings.
- **Temporal decoding**: moving average, moving median, exponential moving average, HMM filtering (online) and Viterbi (offline).
- **Evaluation** with leave-one-surgery-out splits: accuracy, macro F1, Jaccard, edit score and segmental F1@{10, 25, 50}.

## Models

| Model | Type | Frames per clip | Source |
|---|---|---|---|
| SurgVLP | Vision-language | 1 | [CAMMA-public/SurgVLP](https://github.com/CAMMA-public/SurgVLP) |
| HecVL | Vision-language | 1 | [CAMMA-public/SurgVLP](https://github.com/CAMMA-public/SurgVLP) |
| PeskaVLP | Vision-language | 1 | [CAMMA-public/SurgVLP](https://github.com/CAMMA-public/SurgVLP) |
| SurgCLIP | Vision-language | 16 | `surgclip` package |
| SurgMotion-300M | Video encoder | 64 | [CAIR-HKISI/SurgMotion](https://huggingface.co/CAIR-HKISI/SurgMotion) |

SurgCLIP and SurgMotion weights download automatically. SurgVLP-family weights can be placed at `assets/models/weights/<model>.pth`.

## Installation

Requires Python 3.11, [uv](https://docs.astral.sh/uv/) and a CUDA GPU (recommended).

```sh
git clone https://github.com/sharifbubuka/surgphase.git
cd surgphase
uv sync                    # add --group notebook to run the notebooks
```

## Data

Videos, frames and annotations are not distributed. Use this layout:

```
assets/
├── videos/<surgery>.mp4
├── frames/<surgery>/fps<N>/frame_000000.png       # created by sample-frames
├── annotations/<group>/<surgery>.json             # phase intervals
├── annotations/<group>/annotated_frames.csv       # created by annotate-frames
└── phases/<group>.json                            # phase names and prompts
```

Phase groups: `default` and `merged6&7`.

## Usage

All entry points are [poe](https://poethepoet.natn.io/) tasks defined in `poe_tasks.toml`. Pass `--help` to any task to list its options.

**1. Sample frames**

```sh
uv run poe sample-frames --video assets/videos/pat01.mp4 assets/videos/pat02.mp4 assets/videos/pat04.mp4 \
  --output assets/frames --fps 1 15 30
```

**2. Annotate frames**

```sh
uv run poe annotate-frames --surgery-id pat01 pat02 pat04 --annotation-id default \
  --frames-dir assets/frames/pat01/fps1 assets/frames/pat02/fps1 assets/frames/pat04/fps1 --fps 1 \
  --annotations assets/annotations/default/pat01.json assets/annotations/default/pat02.json assets/annotations/default/pat04.json \
  --output assets/annotations/default/annotated_frames.csv
```

**3. Encode frames** (cached in `results/embeddings/`)

```sh
uv run poe encode-frames --annotations assets/annotations/default/annotated_frames.csv \
  --phase-group default --fps 1 15 30 --models SurgVLP SurgCLIP SurgMotion300M
```

**4. Evaluate** (zero-shot, frame heads and sequence heads with every temporal decoding)

```sh
uv run poe evaluate --annotations assets/annotations/default/annotated_frames.csv \
  --phase-group default --fps 1 --models SurgVLP SurgCLIP \
  --sources zeroshot frame_head sequence_head --head-types linear simple \
  --output-dir results/evaluation/default_fps1
```

Head training uses leave-one-surgery-out folds and needs at least three surgeries.

**Train a single head**

```sh
uv run poe train-head --model SurgVLP --annotations assets/annotations/default/annotated_frames.csv \
  --phase-group default --fps 1 --train-surgeries pat01 pat02 --val-surgeries pat04 \
  --head-type simple --class-weights --output-dir results/finetuning
```

**Load a model with a trained head**

```sh
uv run poe load-model --model SurgVLP --phase-group default \
  --head results/finetuning/SurgVLP_native_default_fps1_simple_head.pt
```

**Analyse results** in `notebooks/` (data, embeddings, zero-shot, results and error analysis).

## Project structure

```
├── scripts/            # entry points run by poe tasks
├── notebooks/          # analysis of outputs in results/
└── src/surgphase/
    ├── config/         # global constants and paths
    ├── data/           # annotations, phase groups and prompts
    ├── models/         # model loader, adapters, preprocessors
    ├── inference/      # encoding, embedding cache, predictor
    ├── zeroshot/       # text-prompt classifier
    ├── finetuning/     # frame and sequence heads, training
    ├── temporal/       # smoothing and HMM decoding
    ├── eval/           # metrics, splits, predictions
    └── viz/            # timelines and confusion matrices
```

## Citation

If you use this code, please cite the models it builds on:

```bibtex
@article{yuan2025surgvlp,
  title   = {Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures},
  author  = {Yuan, Kun and Srivastav, Vinkle and Yu, Tong and Lavanchy, Joel L. and Marescaux, Jacques and Mascagni, Pietro and Navab, Nassir and Padoy, Nicolas},
  journal = {Medical Image Analysis},
  year    = {2025}
}

@inproceedings{yuan2024hecvl,
  title     = {HecVL: Hierarchical Video-Language Pretraining for Zero-shot Surgical Phase Recognition},
  author    = {Yuan, Kun and Srivastav, Vinkle and Navab, Nassir and Padoy, Nicolas},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2024}
}

@inproceedings{yuan2024peskavlp,
  title     = {Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation},
  author    = {Yuan, Kun and Navab, Nassir and Padoy, Nicolas},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@inproceedings{farha2019mstcn,
  title     = {MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation},
  author    = {Farha, Yazan Abu and Gall, Juergen},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}

@inproceedings{alsentzer2019clinicalbert,
  title     = {Publicly Available Clinical BERT Embeddings},
  author    = {Alsentzer, Emily and Murphy, John and Boag, William and Weng, Wei-Hung and Jindi, Di and Naumann, Tristan and McDermott, Matthew},
  booktitle = {Proceedings of the 2nd Clinical Natural Language Processing Workshop},
  year      = {2019}
}
```

For SurgCLIP and SurgMotion, use the citations given in their repositories.

## Acknowledgements

Built on [SurgVLP](https://github.com/CAMMA-public/SurgVLP), SurgCLIP, [SurgMotion](https://huggingface.co/CAIR-HKISI/SurgMotion) and [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).
