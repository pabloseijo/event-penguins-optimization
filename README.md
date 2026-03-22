# Event-based Temporal Action Detection (TFG)

👉 **Galician version:** [README.gl.md](README.gl.md)

This repository contains a reproduction and extension of an event-based temporal action detection pipeline, based on the original *event_penguins* repository and the method published by Hamann et al. at CVPR 2024. The original repository is organized around `config/`, `docs/`, `scripts/`, `src/`, `requirements.txt`, and a practical workflow based on preprocessing and inference. :contentReference[oaicite:8]{index=8} The paper defines a two-stage pipeline: temporal proposal generation followed by CNN-based classification. :contentReference[oaicite:9]{index=9}

---

## Description

This project addresses **Temporal Action Detection (TAD)** on event camera data. The input is an event stream of the form:

E = {(x, y, t, p)}

where each event represents a brightness change at pixel location `(x, y)`, at time `t`, with polarity `p`.

Unlike standard frame-based video, event cameras provide properties that are especially useful for continuous monitoring scenarios:

- high temporal resolution
- low power consumption
- robustness to difficult lighting conditions
- motion-centered sensing by design

This work focuses on analyzing and improving the **temporal proposal generation stage**.

---

## Baseline pipeline

The original method follows, at a conceptual level, this sequence:

1. Event-rate computation r(t)
2. Robust normalization
3. Actionness estimation from activity magnitude
4. Temporal proposal generation with **reTAG**
5. Proposal classification with **ATSN**

In the original repository, the practical workflow consists of a preprocessing stage that produces `preprocessed.h5` and an inference stage that runs the full pipeline. :contentReference[oaicite:10]{index=10}

---

## Limitation of the baseline method

The key limitation identified in the baseline is that actionness depends almost entirely on activity magnitude.

This means the system is effective at answering:

“is there motion?”

but not necessarily:

- “is this motion relevant?”
- “does this motion belong to the target action?”
- “is this motion just environmental noise?”

In practice, this may lead to:

- false positives caused by noise
- low-discriminative temporal proposals
- confusion between actions with similar activity patterns

---

## Goal of this TFG

The goal of this thesis project is to improve proposal quality by redefining **actionness**.

Instead of using a score based only on activity magnitude, the proposed approach assigns each temporal proposal I = (t_a, t_b) a richer score combining:

- activity magnitude
- temporal consistency
- spatial structure of events
- noise indicators or penalties

The final CNN stage is not modified. The work is strictly focused on the proposal generation and proposal scoring stage.

---

## Research direction

The general strategy is:

1. reproduce the baseline pipeline
2. analyze the behavior of reTAG
3. incorporate additional descriptors into actionness
4. re-rank or filter proposals using the new score
5. evaluate the impact on proposal quality and final performance

The approach is intentionally classical and interpretable, avoiding additional deep learning components in the proposal stage.

---

## Repository structure

The repository remains close to the structure of the original project, with emphasis on the parts most relevant to this thesis work:

- `config/`
  - experiment and annotation configuration files

- `docs/`
  - auxiliary project documentation

- `scripts/`
  - `preprocess.py`: data preparation and restructuring
  - `inference.py`: full pipeline execution

- `src/`
  - core system implementation
  - modules related to proposal generation, descriptors, and classification

- `requirements.txt`
  - project dependencies

In the original repository, processed data is stored in a single `preprocessed.h5` file, organized by recording and ROI or nest. :contentReference[oaicite:11]{index=11}

---

## Installation

Create environment:

```bash
conda create --name eventpenguins python=3.8
conda activate eventpenguins
```

Install PyTorch according to the corresponding CUDA version.

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

In the original repository, PyTorch was tested with version 2.2.2. :contentReference[oaicite:12]{index=12}

---

## Data preparation

Create the data directory:

```bash
mkdir data
```

Download the base project data and place it inside `data/`.

Run preprocessing:

python scripts/preprocess.py --data_root data/EventPenguins --output_dir data --recording_info_path config/annotations/recording_info.csv

This generates:

`data/preprocessed.h5`

According to the original repository, preprocessing crops events according to pre-annotated nests and stores the recordings following the split defined by the method. :contentReference[oaicite:13]{index=13}

---

## Models

Create the models directory:

```bash
mkdir models
```

Download the pretrained weights from the base project and store them in `models/` if full baseline inference is required. :contentReference[oaicite:14]{index=14}

---

## Running inference

To run the full inference pipeline:

```bash
python scripts/inference.py --config config/exp/inference.yaml --verbose
```

---

## Scope of this repository

This work focuses specifically on:

- event-rate analysis
- actionness reformulation
- temporal proposal generation improvement
- false positive reduction
- improved robustness to noise and ambiguous motion patterns

This thesis does not aim to:

- redesign the ATSN architecture
- introduce additional deep learning models
- turn the system into a general multi-class TAD framework

---

## Dataset and annotations

The base project uses 24 ten-minute recordings with 16 annotated nests, and annotations follow an ActivityNet-like structure with an additional nest level. The method operates on ROI-cropped event streams rather than the full scene. :contentReference[oaicite:15]{index=15}

---

## References

If you use this repository or the baseline method, please consider citing the original paper:

Hamann, F., Ghosh, S., Juarez Martinez, I., Hart, T., Kacelnik, A., Gallego, G.
Low-power Continuous Remote Behavioral Localization with Event Cameras.
CVPR 2024.

Base repository:
tub-rip/event_penguins
