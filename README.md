# GestaltM³D-VL: Multimodal Vision–Language Diagnosis of Mendelian Rare Diseases

> **GestaltM³D-VL** (Gestalt Multimodal Model for Mendelian Disease Diagnosis with Vision–Language Integration)  
> A multimodal model leveraging **facial images** and **HPO-encoded clinical text** for **Mendelian rare disease diagnosis**.  
> We additionally train **StyleGAN3** on GMDB faces to generate **synthetic facial images** used for **evaluation and analysis**, **not** for privacy attacks.

---

### Overall pipeline

<p align="center">
  <img src="docs/figures/overview_pipeline.png" alt="Overall pipeline of GestaltM3D-VL" width="800">
</p>

### Example faces & HPO text

<p align="center">
  <img src="docs/figures/example_cases.png" alt="Example GMDB cases with facial images and HPO descriptions" width="800">
</p>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: GMDB Ten-Disease Subset](#2-dataset-gmdb-ten-disease-subset)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Part I – StyleGAN3 Synthesis for Evaluation](#4-part-i--stylegan3-synthesis-for-evaluation)
5. [Part II – GestaltM³D-VL for Multimodal Diagnosis](#5-part-ii--gestaltm³d-vl-for-multimodal-diagnosis)
6. [Repository Structure](#6-repository-structure)
7. [Installation](#7-installation)
8. [Data Preparation](#8-data-preparation)
9. [Training & Evaluation](#9-training--evaluation)
10. [Figures to Include](#10-figures-to-include)
11. [Citation](#11-citation)
12. [License & Disclaimer](#12-license--disclaimer)

---

## 1. Project Overview

This repository implements a pipeline for **Mendelian rare disease diagnosis** on the **GMDB ten-disease subset** (and optionally CHOP cohorts):

1. **Part I – StyleGAN3-based Image Synthesis for Evaluation**
   - Train **class-conditional StyleGAN3** on:
     - GMDB ten-disease facial images
     - (Optionally) a larger unaffected cohort
   - Generate **synthetic facial images** for:
     - Visual inspection of disease-specific facial gestalt
     - Stylized comparison between real and model-predicted cases
     - Additional **evaluation scenarios** (e.g., robustness tests, case studies)

2. **Part II – GestaltM³D-VL (Vision–Language Multimodal Diagnosis)**
   - Adapt **Qwen-2-VL / Qwen-2.5-VL / Qwen-3-VL** into a **multimodal classifier**.
   - Inputs:
     - **Facial image** (preprocessed 224×224 front-view face).
     - **Clinical text**: HPO-encoded phenotypes (+/- demographics).
   - Output:
     - **Disease label** among rare Mendelian syndromes.
   - Loss:
     - **Class-Balanced Focal Loss** to handle long-tail label distribution.

<p align="center">
  <img src="docs/figures/method_block_diagram.png" alt="High-level block diagram of the full method" width="800">
</p>

---

## 2. Dataset: GMDB Ten-Disease Subset

We use the **GestaltMatcher Database (GMDB)**, which pairs:

- **Facial images**
- **HPO-encoded clinical text**
- **Demographic information** (ethnicity, age, sex)

Overall label space: **528 syndromes/disorders** (244 frequent, 284 rare).  
In this repository we focus on the **ten diseases with the most distinctive facial phenotypes**, curated by clinical geneticists.

Key stats for the **ten-disease subset**:

- **1,847 cases** with patient-level splits.
- **Test set**: only **HPO-annotated cases** (≈ 818 cases).
- Non-HPO cases are used **only as data augmentation** during training.

> ⚠️ **Dataset access**  
> GMDB (and CHOP, if used) are **not included** in this repository.  
> You must obtain data access following your **institutional, IRB, and data-use agreements**, then place the files under `data/raw/`.

<p align="center">
  <img src="docs/figures/dataset_stats.png" alt="Label distribution and cohort statistics for the GMDB ten-disease subset" width="800">
</p>

---

## 3. Data Preprocessing Pipeline

Original facial images in GMDB/CHOP often suffer from:

- Low resolution, blur
- Grayscale or washed-out colors
- Cluttered backgrounds

We build a **three-stage preprocessing pipeline**:

1. **Color Restoration – DDColor**
   - Colorize grayscale / washed-out facial images.

2. **Face Restoration & Super-Resolution – GFPGAN**
   - Enhance facial details and upsample low-resolution faces.

3. **Face Detection & Normalization – MediaPipe Face Detection**
   - Detect faces and perform **tight cropping**.
   - Remove background and resize to **224×224** RGB.
   - Normalize input space and control token budget for the multimodal LLM.

The script `src/data/preprocess_faces.py` orchestrates DDColor, GFPGAN, and MediaPipe and writes processed images into `data/processed/images`.

<p align="center">
  <img src="docs/figures/preprocessing_pipeline.png" alt="DDColor + GFPGAN + MediaPipe facial preprocessing pipeline" width="800">
</p>

---

## 4. Part I – StyleGAN3 Synthesis for Evaluation

### 4.1 Goal

We use **StyleGAN3** as a **class-conditional face generator** trained on GMDB ten-disease faces.  
In this project, synthetic images are used for:

- **Qualitative evaluation** of disease-specific facial patterns.
- **Case studies** comparing:
  - real patient images, and
  - synthetic faces conditioned on the predicted disease.
- Potential **data augmentation / stress tests**, if desired.

We **do not** implement or focus on privacy attacks in this repository.

### 4.2 StyleGAN3 Overview

- Alias-free generation with improved geometric consistency.
- Reduced texture “sticking” compared to StyleGAN2.
- Separate **mapping network** and **synthesis network**, with class conditioning.

Training data:

- **Ten-disease cohort**: 1,847 patients.
- (Optional) **unaffected/healthy cohort** to balance age/ethnicity.

<p align="center">
  <img src="docs/figures/stylegan3_overview.png" alt="StyleGAN3 architecture and training overview" width="800">
</p>

### 4.3 Training & Sampling

Configuration sketch (see `configs/stylegan3/ten_disease_default.yaml`):

- Conditional labels: { ten-disease classes (+ optional unaffected class) }
- Training iterations: e.g., **25,000 kimg**
- Augmentations: StyleGAN3 default configuration

Run:

```bash
bash scripts/train_stylegan3.sh
bash scripts/sample_stylegan3.sh
```

This will:

1. Train StyleGAN3 on preprocessed GMDB faces.
2. Save synthetic samples for each disease class under `data/processed/stylegan3_samples/` (or a path you configure).

### 4.4 Using Synthetic Images for Evaluation

Synthetic images from StyleGAN3 can support analysis such as:

- Visual comparison between:
  - Real patient (input image)
  - Disease predicted by GestaltM³D-VL
  - Synthetic images conditioned on the same disease label
- Qualitative assessment of:
  - Model’s disease confusion patterns
  - Intra-class variability captured by StyleGAN3

You can load synthetic images as a small “gallery” in notebooks under `notebooks/` for interactive exploration.

<p align="center">
  <img src="docs/figures/stylegan3_synthetic_examples.png" alt="Synthetic samples per disease class generated by StyleGAN3" width="800">
</p>

---

## 5. Part II – GestaltM³D-VL for Multimodal Diagnosis

### 5.1 Motivation

We target **Mendelian rare disease diagnosis** from:

- **Facial gestalt** (front-view facial images).
- **HPO-encoded clinical text** (+/- demographics).

Challenges:

- GMDB is **long-tailed** (many rare diseases with few examples).
- Data is **noisy** (missing HPO terms, variable image quality).
- Naive cross-entropy tends to overfit **head classes**.

### 5.2 Backbone: Qwen-VL Family

We build on **Qwen-VL** models:

- **Qwen-2-VL-7B-Instruct**
- **Qwen-2.5-VL-7B-Instruct** (default backbone)
- (Optional) **Qwen-3-VL-8B-Instruct**

Advantages:

- Strong **vision–language alignment**.
- Good **parameter–compute trade-off** (7–8B).
- Handles **long clinical prompts** and **noisy faces**.

### 5.3 Model Architecture (Sequence Classification)

GestaltM³D-VL adapts Qwen-VL into **multimodal sequence classification**:

1. **Inputs**:
   - `image`: 224×224 preprocessed face.
   - `text`:
     - HPO-encoded phenotypes.
     - Optionally demographics (we often **drop** demographics to reduce bias and privacy concerns).

2. **Multimodal encoding**:
   - Construct an instruction-style prompt containing the HPO terms.
   - Feed image + text into Qwen-VL.
   - Take the hidden state at a special **`[CLS]`** (or equivalent pooling token).

3. **Classifier head**:
   - Apply a **Linear(D → #classes)** layer on the `[CLS]` representation.

4. **Loss**:
   - Use **Class-Balanced Focal Loss** to mitigate label imbalance.

Training strategy (example):

- Freeze most of the **text backbone**.
- Optionally partially freeze the **vision backbone**.
- Train:
  - Multimodal projector
  - Top transformer blocks (via LoRA)
  - Final classifier head

<p align="center">
  <img src="docs/figures/mm_llm_architecture.png" alt="GestaltM3D-VL multimodal sequence classification architecture" width="800">
</p>

### 5.4 Loss: Class-Balanced Focal Loss

Let \( n_y \) be the sample count for class \( y \).  
Compute class weights \( \alpha_y \) based on effective number of samples (e.g., Cui et al., CVPR 2019).  
Use focal loss with focusing parameter \( \gamma > 0 \):

- Addresses severe **class imbalance**.
- Focuses training on **hard** and **rare** classes.

Implementation details in:

- `src/models/mm_llm/losses.py`

### 5.5 Observations (from internal experiments)

- **Multimodal (image + text) > Image-only**.
- Removing demographics from the prompt:
  - Maintains or improves performance.
  - Reduces reliance on explicit ethnicity/age/sex.
- Upgrading backbone from Qwen-2-VL to **Qwen-2.5-VL** improves overall performance.

---

## 6. Repository Structure

Recommended repository layout:

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── configs/
│   ├── stylegan3/
│   │   └── ten_disease_default.yaml
│   └── mm_llm/
│       └── qwen2_5_vl_ten_disease.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── stylegan3_samples/   # synthetic images per class
│   └── splits/
│       └── ten_disease/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
├── docs/
│   └── figures/
│       ├── overview_pipeline.png
│       ├── example_cases.png
│       ├── dataset_stats.png
│       ├── preprocessing_pipeline.png
│       ├── stylegan3_overview.png
│       ├── stylegan3_synthetic_examples.png
│       ├── mm_llm_architecture.png
│       └── method_block_diagram.png
├── src/
│   ├── data/
│   │   ├── gmdb_dataset.py
│   │   ├── preprocess_faces.py
│   │   └── build_splits.py
│   ├── models/
│   │   ├── stylegan3/
│   │   │   ├── train_stylegan3.py
│   │   │   └── sample_stylegan3.py
│   │   └── mm_llm/
│   │       ├── mm_classifier.py
│   │       ├── losses.py
│   │       ├── train_mm_llm.py
│   │       └── evaluate_mm_llm.py
│   └── utils/
│       ├── config.py
│       ├── logging_utils.py
│       ├── metrics.py
│       └── seed.py
└── scripts/
    ├── prepare_gmdb.sh
    ├── train_stylegan3.sh
    ├── sample_stylegan3.sh
    ├── train_mm_llm.sh
    └── eval_mm_llm.sh
```

---

## 7. Installation

### 7.1 Conda environment

```bash
conda env create -f environment.yml
conda activate gestaltm3d-vl
```

### 7.2 Pip

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (sketch):

- `torch`, `torchvision`, `torchaudio`
- `transformers`, `accelerate`, `bitsandbytes` (optional)
- `timm`
- `opencv-python`, `mediapipe`
- `gfpgan`, `ddcolor` (or equivalent)
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `rich`, `pyyaml`

---

## 8. Data Preparation

### 8.1 Place raw data

Place GMDB (and optional CHOP) metadata and images under:

```text
data/raw/
  gmdb_metadata.csv
  images/
    patient_*.jpg
```

> The exact file naming depends on your internal export.  
> Adjust in `src/data/build_splits.py` and `src/data/preprocess_faces.py` accordingly.

### 8.2 Build patient-level splits

```bash
python -m src.data.build_splits \
  --input_meta data/raw/gmdb_metadata.csv \
  --output_dir data/splits/ten_disease
```

This creates:

- `train.csv`
- `val.csv`
- `test.csv`

with typical columns:

- `image_path`
- `disease_label`
- `hpo_text`
- `demographics` (optional)

### 8.3 Run facial preprocessing

```bash
python -m src.data.preprocess_faces \
  --input_dir data/raw/images \
  --meta_csv data/splits/ten_disease/train.csv \
  --output_dir data/processed/images
```

Repeat or adapt for val/test splits.

---

## 9. Training & Evaluation

### 9.1 StyleGAN3 (Synthesis for Evaluation)

```bash
# Train StyleGAN3 on GMDB ten-disease faces
bash scripts/train_stylegan3.sh

# Sample synthetic faces per disease class
bash scripts/sample_stylegan3.sh
```

This will populate `data/processed/stylegan3_samples/` with class-conditional synthetic images, which you can use for:

- qualitative inspection,
- generating figures for papers/presentations,
- additional robustness experiments.

### 9.2 GestaltM³D-VL (Multimodal Diagnosis)

```bash
# Train GestaltM3D-VL on GMDB ten-disease subset
bash scripts/train_mm_llm.sh

# Evaluate on held-out test set
bash scripts/eval_mm_llm.sh
```

`configs/mm_llm/qwen2_5_vl_ten_disease.yaml` controls:

- Backbone (Qwen-2-VL / Qwen-2.5-VL / Qwen-3-VL)
- Learning rate, batch size
- Modules to freeze / finetune
- Whether demographics are included in the text prompt

---

## 10. Figures to Include

To make the README visually informative on GitHub, we recommend preparing at least:

1. `overview_pipeline.png`  
   - Overall flow: GMDB → preprocessing → StyleGAN3 (synthesis) → GestaltM³D-VL (diagnosis).

2. `example_cases.png`  
   - Example patient faces (de-identified or synthetic) and their HPO descriptions.

3. `dataset_stats.png`  
   - Label distribution and summary statistics for the ten-disease subset.

4. `preprocessing_pipeline.png`  
   - Before/after examples of DDColor + GFPGAN + MediaPipe cropping.

5. `stylegan3_overview.png`  
   - StyleGAN3 training sketch (class-conditional, synthesis examples).

6. `stylegan3_synthetic_examples.png`  
   - A grid of synthetic faces per disease class.

7. `mm_llm_architecture.png`  
   - GestaltM³D-VL architecture diagram: image + HPO text → Qwen-VL → classifier.

8. `method_block_diagram.png`  
   - High-level schematic showing Part I (synthesis) and Part II (diagnosis).

---

## 11. Citation

If you build on this repository, please cite your own work and related foundations.  
A placeholder BibTeX entry:

```bibtex
@article{your_gestaltm3dvl_paper,
  title   = {GestaltM3D-VL: Multimodal Vision--Language Diagnosis of Mendelian Rare Diseases},
  author  = {Your Name and Collaborators},
  journal = {To appear},
  year    = {2025}
}
```

Also consider citing:

- GestaltMatcher / GestaltMML papers
- Qwen2-VL / Qwen2.5-VL / Qwen3-VL technical reports
- StyleGAN3, DDColor, GFPGAN, MediaPipe

---

## 12. License & Disclaimer

- **License**: To be decided (e.g., MIT / Apache-2.0 / custom research license).  
- **Data**: This repository does **not** include patient-identifiable GMDB/CHOP data.  
- **Usage**:
  - Any use of real patient data must follow appropriate **IRB**, **data-use agreements**, and **local regulations**.
  - Synthetic images are intended for research and evaluation only.

> This repository is for **research** and **method development** only.  
> It is **not** a certified medical device and should not be used directly for clinical decision-making without thorough validation and regulatory approval.
