# Object Aware Room Re-Identification

This repository contains a PyTorch reimplementation of the pipeline for indoor room re-identification using global, object-level, and fine-grained matching cues.

## Project Overview

The pipeline follows a coarse-to-fine retrieval strategy:

1. **Reference preprocessing**
   - Select reference images for each room
   - Segment objects from reference images
   - Extract object, patch, and room-level embeddings
   - Save these embeddings for retrieval

2. **Inference**
   - Extract a global feature for the query image
   - Retrieve top candidate rooms using global similarity
   - Re-score candidates using object and patch features
   - Use fine matching between query and reference images for final selection

---

## Repository Structure

- `preprocess.py` — builds the reference database
- `inference.py` — runs coarse-to-fine room re-identification
- `models/build_model.py` — model builders and wrappers
- `data/query_dataset.py` — query dataset loader
- `data/reference_dataset.py` — reference dataset loader
- `utils/geometry.py` — geometry utilities for receptive field expansion
- `utils/scoring.py` — scoring functions for patch/object matching
- `config/preprocess.yaml` — preprocessing configuration
- `config/inference.yaml` — inference configuration

---

## Dataset Layout

Expected dataset structure:

```text
datasets/
  ReplicaReID/
    apartment_0/
      kitchen/
        rgb/
        depth/
      living/
        rgb/
        depth/
    apartment_1/
    apartment_2/
    office_0/
    office_1/
    ...
    room_label.txt
---
```
## Installation

### 1. Clone the repository in Google Colab

```python
%cd /content
!rm -rf cv_project
!GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Priya-Kumari-Chourasia/cv_project.git
%cd /content/cv_project
```

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Copy dataset into the project folder

```python
!pip install gdown

!gdown --id 1Nq7jFR9wqJPzd2n9qO1Rxf945hs4gVOF -O ReplicaReID.zip

!unzip ReplicaReID.zip

!mv ReplicaReID /content/cv_project/datasets/
```

### 4. Install requirements

```python
!pip install -r requirements.txt
```

### 5. Install LightGlue

```python
%cd /content
!rm -rf LightGlue
!git clone https://github.com/cvg/LightGlue.git
%cd /content/LightGlue
!python -m pip install -e .
%cd /content/cv_project
```

### 9. Run preprocessing and inference
```python

%cd /content/cv_project
!python preprocess.py
!python inference.py
```

